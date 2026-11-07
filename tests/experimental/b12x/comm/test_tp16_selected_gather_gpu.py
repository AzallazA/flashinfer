"""Byte-exact TP16 gather profiles share replay banks across changing row counts."""

import ctypes
import gc
from dataclasses import replace
from datetime import timedelta
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="requires sixteen explicitly assigned peer-accessible SM120 GPUs",
)


def _worker(rank, directory):
    import torch
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.comm.pcie import _dcp_preparation
    from b12x.comm.pcie._dcp_a2a_cute import _GRAPH_ARRIVED_INDEX, _GRAPH_EPOCH_INDEX
    from b12x.comm.pcie.pcie_dcp_a2a import _SIGNAL_BYTES
    from b12x.preparation import (
        CollectiveRequirement,
        FrozenMapping,
        PreparationSession,
        PreparedCall,
    )

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=16,
        timeout=timedelta(minutes=10),
    )
    profiles = (
        ("ordinary", 512),
        ("selected_phased_barrier", 128),
        ("selected_phased_barrier", 512),
        ("selected_ring", 128),
        ("selected_recursive", 128),
        ("selected_recursive_ready", 128),
    )
    try:
        for origin in (0, 1):
            snapshot = None
            owner = pcie.DcpAllToAll.from_exchange_group(
                exchange_group=dist.group.WORLD,
                device=device,
                max_batch_size=32,
                total_heads=16,
                head_dim=8192,
                stream_affine=False,
            )
            assert owner._threads_override == 0
            assert owner._block_limit_override in (0, 8)
            session = PreparationSession(
                device=device, autotune=False, compile_workers=4, prime=False
            )
            cases, requests, graphs = {}, [], {}
            try:
                for rows in (1, 3, 31, 32):
                    source = torch.empty(
                        (rows, 1, 8192), dtype=torch.bfloat16, device=device
                    )
                    guard = torch.full(
                        (rows * 16 * 8192 * 2 + 32,),
                        0xA5,
                        dtype=torch.uint8,
                        device=device,
                    )
                    output = guard[16:-16].view(torch.bfloat16).view(rows, 16, 8192)
                    operands, expected = [], []
                    columns = torch.arange(rows * 8192 * 2, device=device).view(
                        rows, 1, -1
                    )
                    for variation in range(3):
                        operands.append(
                            ((columns + rank * 37 + variation * 53) % 256).to(
                                torch.uint8
                            )
                        )
                        expected.append(
                            torch.cat(
                                [
                                    ((columns + peer * 37 + variation * 53) % 256).to(
                                        torch.uint8
                                    )
                                    for peer in range(16)
                                ],
                                dim=1,
                            ).reshape(rows, 16, -1)
                        )
                    for transport, threads in profiles:
                        query = _dcp_preparation.query_from_runtime(
                            owner,
                            surface="DcpAllToAll.all_gather_heads",
                            call=dict(
                                local_input=source,
                                out=output,
                                threads=threads,
                                block_limit=8,
                            ),
                        )
                        if transport != "ordinary":
                            query = replace(
                                query,
                                call=FrozenMapping(
                                    dict(
                                        query.call,
                                        gather_transport=transport,
                                        submode="barrier",
                                        active_blocks=8,
                                    )
                                ),
                            )
                        plan = _dcp_preparation.plan(query, runtime=owner)
                        key = rows, transport, threads
                        name = f"rows{rows}-{transport}-{threads}"
                        requests.append(
                            plan.request(
                                name=name,
                                prepare_call=lambda state,
                                source=source,
                                output=output: PreparedCall(
                                    run=lambda: state.run(
                                        local_input=source, out=output
                                    ),
                                    output=output,
                                    owners=(owner, source, output),
                                ),
                                collective=CollectiveRequirement(
                                    name, tuple(range(16))
                                ),
                            )
                        )
                        cases[key] = source, output, guard, operands, expected, plan
                session.prepare(
                    tuple(requests),
                    coordinator=lambda progress: (
                        progress.ready_collectives[0].key
                        if progress.ready_collectives
                        else None
                    ),
                )
                source, output, _, operands, _, plan = cases[1, "ordinary", 512]
                source.view(torch.uint8).copy_(operands[0])
                for _ in range(1 if origin else 2):
                    owner.all_gather_heads(source, out=output, plan=plan)
                torch.cuda.synchronize()
                dist.barrier()
                for key, (source, output, _, _, _, plan) in cases.items():
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        owner.all_gather_heads(source, out=output, plan=plan)
                    graphs[key] = graph
                assert owner._device_slot_selection and owner._graph_base_slot == origin
                session.freeze()
                schedule = [
                    (key, variation)
                    for variation in range(3)
                    for key in (
                        list(cases) if variation % 2 == 0 else list(reversed(cases))
                    )
                ]
                saved = [torch.empty_like(cases[key][1]) for key, _ in schedule]
                stream = torch.cuda.current_stream(device)

                def signal_words(offset, values=None):
                    host = (ctypes.c_uint32 * 1)(*(values or [0]))
                    address = owner._signal_ptrs[rank] + offset * 4
                    owner._ipc.cudaMemcpyAsync(
                        ctypes.addressof(host) if values is None else address,
                        address if values is None else ctypes.addressof(host),
                        4,
                        stream.cuda_stream,
                    )
                    stream.synchronize()
                    return host[0]

                assert signal_words(_GRAPH_ARRIVED_INDEX) == 0
                state_words = (ctypes.c_uint32 * (_SIGNAL_BYTES // 4))()
                for index in range(64 * 16):
                    state_words[index] = 0xFFFFFFFD
                for index in range(64 * 16, len(state_words), 32):
                    state_words[index] = 0xFFFFFFFD
                state_words[_GRAPH_EPOCH_INDEX] = 0xFFFFFFFD
                state_words[_GRAPH_ARRIVED_INDEX] = 0
                owner._ipc.cudaMemcpyAsync(
                    owner._signal_ptrs[rank],
                    ctypes.addressof(state_words),
                    _SIGNAL_BYTES,
                    stream.cuda_stream,
                )
                stream.synchronize()
                dist.barrier()
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats(device)
                if rank == origin:
                    torch.cuda._sleep(200000)
                for (key, variation), snapshot in zip(schedule, saved, strict=True):
                    source, output, _, operands, _, _ = cases[key]
                    source.view(torch.uint8).copy_(operands[variation])
                    output.view(torch.uint8).fill_(0xA5)
                    graphs[key].replay()
                    snapshot.copy_(output)
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for counter in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[counter] == after[counter], (
                        counter,
                        before[counter],
                        after[counter],
                    )
                for (key, variation), snapshot in zip(schedule, saved, strict=True):
                    source, _, guard, operands, expected, _ = cases[key]
                    assert torch.equal(
                        snapshot.view(torch.uint8), expected[variation]
                    ), key
                    assert torch.all(guard[:16] == 0xA5) and torch.all(
                        guard[-16:] == 0xA5
                    )
                for source, _, _, operands, _, _ in cases.values():
                    assert torch.equal(source.view(torch.uint8), operands[-1])
                assert (
                    signal_words(_GRAPH_EPOCH_INDEX)
                    == (0xFFFFFFFD + len(schedule)) & 0xFFFFFFFF
                )
                assert signal_words(_GRAPH_ARRIVED_INDEX) == 0
                dist.barrier()
            except BaseException:
                import sys
                import traceback

                traceback.print_exc()
                sys.stderr.flush()
                raise
            finally:
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()
                session.close()
                owner.close()
    finally:
        dist.destroy_process_group()


def test_tp16_selected_gather_profiles_queued_replay(tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() != 16 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires sixteen assigned SM120 GPUs")
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=16, join=True)
