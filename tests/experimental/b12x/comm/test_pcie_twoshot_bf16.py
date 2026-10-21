"""Prepared BF16 collectives and all-reduce dispatch on four CUDA devices.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
        --nproc-per-node=4 tests/experimental/b12x/comm/test_pcie_twoshot_bf16.py

Add --benchmark to print all-rank graph times, or --benchmark-only to time the
same dispatch path in another checkout without running its correctness checks.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie import _twoshot_preparation as preparation
from b12x.comm.pcie._twoshot_bf16_cute import get_twoshot_bf16_launcher
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.comm.pcie.pcie_twoshot_bf16 import _make_layout
from b12x.preparation import CollectiveRequirement, PreparationSession


@pytest.mark.parametrize("shape", ((128, 1024), (32, 4096), (16, 8192), (131072,)))
def test_prepared_all_reduce_counts_native_rows(monkeypatch, shape):
    payload = torch.empty(shape, dtype=torch.bfloat16)
    out = torch.empty_like(payload)
    launches = []
    layout = _make_layout(32, 4096, 4)
    runtime = SimpleNamespace(
        world_size=4,
        rank=0,
        row_elems=4096,
        device=torch.device("cpu"),
        _pack_stride=layout.pack_stride,
        _slot_bytes=layout.slot_bytes,
        _push_shard=torch.empty((8, 4096), dtype=torch.bfloat16),
        _capture_context_depth=0,
        _device_slot_selection=False,
        _device_slot_bias=0,
        _slot=0,
        _staging_ptrs=((1, 2, 3, 4), (5, 6, 7, 8)),
        _signal_ptrs=(9, 10, 11, 12),
    )
    state = SimpleNamespace(
        query=SimpleNamespace(
            call={"operation": "all_reduce", "threads": 512, "block_limit": 64}
        ),
        launcher=lambda *key: lambda *args: launches.append((key, args)),
    )
    import b12x.comm.pcie.pcie_twoshot_bf16 as runtime_module

    monkeypatch.setattr(runtime_module, "_is_current_stream_capturing", lambda _: False)
    PCIeTwoShotBF16._launch_prepared(
        runtime, payload, None, out, state=state, threads=512, block_limit=64
    )
    assert len(launches) == 2
    assert [entry[0][-1] for entry in launches] == ["reduce_scatter", "all_gather"]
    assert launches[0][1][0] == payload.data_ptr()
    assert launches[0][1][3] == runtime._push_shard.data_ptr()
    assert launches[1][1][0] == runtime._push_shard.data_ptr()
    assert launches[1][1][3] == out.data_ptr()
    assert launches[0][1][1] == runtime._staging_ptrs[0]
    assert launches[1][1][1] == runtime._staging_ptrs[1]
    assert all(entry[1][-2] == 8 for entry in launches)


def test_layout_has_only_payload_slots():
    for world in (2, 8):
        with pytest.raises(ValueError, match="supports only world size 4"):
            _make_layout(64, 4096, world)
    layout = _make_layout(131072, 8, 4)
    assert layout.slot_bytes == 2 << 20
    assert layout.slab_bytes == layout.signal_bytes + 2 * layout.slot_bytes
    assert _make_layout(128, 4096, 4).slab_bytes > _make_layout(64, 4096, 4).slab_bytes


@pytest.mark.parametrize("world", (2, 8))
def test_private_launchers_reject_unsupported_world_sizes(world):
    with pytest.raises(ValueError, match="require world size 4"):
        get_twoshot_bf16_launcher("reduce_scatter", world, 0, False, 0, 512, 8, 0)


def test_preparation_accounts_for_persistent_shard():
    layout = _make_layout(131072, 8, 4)
    runtime = SimpleNamespace(
        rank=0,
        world_size=4,
        row_elems=8,
        max_rows=131072,
        device=torch.device("cpu"),
        _pack_stride=layout.pack_stride,
        _slot_bytes=layout.slot_bytes,
        _signal_ptrs=(100,) * 8,
        _staging_ptrs=(
            (100 + layout.signal_bytes,) * 8,
            (100 + layout.signal_bytes + layout.slot_bytes,) * 8,
        ),
    )
    inp = torch.empty(32768, dtype=torch.bfloat16)
    query = preparation.query_from_runtime(
        runtime, surface="PCIeTwoShotBF16.all_reduce", call={"inp": inp}
    )
    metadata = preparation.query_from_metadata(
        runtime,
        surface="PCIeTwoShotBF16.all_reduce",
        shape=tuple(inp.shape),
        dtype=inp.dtype,
    )
    assert query.setup == metadata.setup
    assert query.setup["push_shard_nbytes"] == (2 << 20) // 4
    plan = preparation.plan(query, runtime=runtime)
    from b12x.preparation.types import _plan_scope

    with _plan_scope(plan):
        memory = plan._memory_requirements(None, None)
    assert sorted(entry.required_nbytes for entry in memory.persistent) == [
        1 << 19,
        layout.slab_bytes,
    ]
    assert all(
        entry.required_nbytes == entry.resident_nbytes for entry in memory.persistent
    )


def _prepare(session, runtime, operation, payload, out):
    call = {"inp" if operation == "all_reduce" else "payload": payload, "out": out}
    if isinstance(runtime, PCIeAllReduce):
        plan = runtime.plan(payload, out=out)
    else:
        query = preparation.query_from_runtime(
            runtime, surface=f"PCIeTwoShotBF16.{operation}", call=call
        )
        plan = preparation.plan(query, runtime=runtime)
    name = f"{operation}-{payload.numel()}"
    if plan.query.surface.startswith("PCIeTwoShotBF16."):
        prime = lambda state: preparation.prepared_call(state, payload=payload, out=out)
    else:
        from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call

        prime = lambda state: _prepare_plain_call(state, inp=payload, out=out)
    session.prepare(
        (
            plan.request(
                name=name,
                collective=CollectiveRequirement(key=name, ranks=tuple(range(4))),
                prepare_call=prime,
            ),
        ),
        coordinator=lambda progress: (
            progress.ready_collectives[0].key if progress.ready_collectives else None
        ),
    )
    return plan


def _fill(tensor, rank, step):
    generator = torch.Generator(device=tensor.device).manual_seed(10007 * step + rank)
    tensor.copy_(
        torch.randn(tensor.shape, generator=generator, device=tensor.device) * 3
    )


def _reference(inp):
    values = [torch.empty_like(inp) for _ in range(4)]
    dist.all_gather(values, inp)
    shard = inp.numel() // 4
    result = torch.empty_like(inp).view(-1)
    for owner in range(4):
        interval = slice(owner * shard, (owner + 1) * shard)
        acc = torch.zeros(shard, dtype=torch.float32, device=inp.device)
        for offset in range(4):
            acc.add_(values[(owner + offset) % 4].view(-1)[interval].float())
        result[interval] = acc.to(torch.bfloat16)
    return result.view_as(inp)


def _allocation_state(device):
    return (
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_stats(device).get("allocation.all.allocated"),
    )


def _epoch(runtime):
    from b12x.comm.pcie._twoshot_cute import _GRAPH_EPOCH_OFFSET

    value = ctypes.c_uint32()
    runtime._ipc.cudaMemcpyAsync(
        ctypes.addressof(value),
        runtime._signal_ptrs[runtime.rank] + _GRAPH_EPOCH_OFFSET,
        4,
        int(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()
    return value.value


def _check_dispatch(session, manager, rank, *, benchmark=False, benchmark_only=False):
    sizes = (
        65536 - 64,
        65536,
        65536 + 64,
        (640 << 10) - 64,
        640 << 10,
        (640 << 10) + 64,
        (2 << 20) - 64,
        2 << 20,
        (2 << 20) + 64,
    )
    cases = []
    for size in sizes:
        inp = torch.empty((size // 64, 32), dtype=torch.bfloat16, device=manager.device)
        _fill(inp, rank, size)
        out = torch.empty_like(inp)
        plan = _prepare(session, manager, "all_reduce", inp, out)
        push = plan.query.surface == "PCIeTwoShotBF16.all_reduce"
        if not benchmark_only:
            assert push == (65536 <= size <= 2 << 20), (size, plan.query.surface)
        with session.capture():
            manager.all_reduce(inp, plan=plan, out=out)
        torch.cuda.synchronize()
        if not benchmark_only:
            assert torch.equal(out, _reference(inp)), ("eager", size)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with session.capture(), manager.capture(), torch.cuda.graph(graph):
            manager.all_reduce(inp, plan=plan, out=out)
        cases.append((size, inp, out, plan, graph))
    if not benchmark_only:
        fused_input = torch.empty(
            (1, 32768), device=manager.device, dtype=torch.bfloat16
        )
        fused_plan = manager.plan(
            fused_input, operation="all_reduce_fused_add_rms_norm"
        )
        assert (
            fused_plan.query.surface == "OneshotAllReduce.all_reduce_fused_add_rms_norm"
        )
        over_capacity = torch.empty(
            ((2 << 20) + 128) // 2, device=manager.device, dtype=torch.bfloat16
        )
        with pytest.raises(ValueError, match="requirements"):
            manager.plan(over_capacity)
        # Alternating sizes and live inputs share one persistent shard and epoch.
        for step in range(5):
            for size, inp, out, plan, graph in cases[:: 1 if step % 2 else -1]:
                _fill(inp, rank, 100 + step + size)
                expected = _reference(inp)
                out.fill_(float("nan"))
                torch.cuda.synchronize()
                dist.barrier()
                before = _allocation_state(manager.device)
                with (
                    session.capture(),
                    kernel_resolution_guard("prepared all-reduce replay"),
                ):
                    graph.replay()
                torch.cuda.synchronize()
                assert _allocation_state(manager.device) == before
                assert torch.equal(out, expected), ("graph", size, step)
    if benchmark:
        for size, inp, out, plan, graph in cases:
            durations = []
            for sample in range(5):
                for _ in range(8):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                local = torch.tensor(
                    [start.elapsed_time(end) * 1000 / 100], device=manager.device
                )
                dist.all_reduce(local, op=dist.ReduceOp.MAX)
                durations.append(round(local.item(), 3))
            if rank == 0:
                print(
                    f"allreduce bytes={size} surface={plan.query.surface} graph_us_max_rank={durations}",
                    flush=True,
                )
    for _, _, _, _, graph in cases:
        graph.reset()


def _check_mixed(session, pool, rank):
    # Native 8-element rows match the dispatcher's geometry, including odd slots.
    cases = []
    for rows in (4, 4096, 4092, pool.max_rows):
        inp = torch.empty(
            (rows, pool.row_elems), dtype=torch.bfloat16, device=pool.device
        )
        _fill(inp, rank, rows)
        out, shard, gathered = (
            torch.empty_like(inp),
            torch.empty_like(inp[: rows // 4]),
            torch.empty_like(inp),
        )
        ar = _prepare(session, pool, "all_reduce", inp, out)
        rs = _prepare(session, pool, "reduce_scatter", inp, shard)
        ag = _prepare(session, pool, "all_gather", shard, gathered)
        with session.capture():
            pool.reduce_scatter(inp, plan=rs, out=shard)  # Offset the initial slot.
            pool.all_reduce(inp, plan=ar, out=out)
            pool.all_gather(shard, plan=ag, out=gathered)
        torch.cuda.synchronize()
        expected = _reference(inp)
        assert torch.equal(out, expected) and torch.equal(gathered, expected)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), pool.capture(plan=ar), torch.cuda.graph(graph):
            pool.all_reduce(inp, plan=ar, out=out)
            pool.reduce_scatter(inp, plan=rs, out=shard)
            pool.all_gather(shard, plan=ag, out=gathered)
            pool.reduce_scatter(
                inp, plan=rs, out=shard
            )  # Five protocol epochs per replay.
        cases.append((inp, out, shard, gathered, ar, graph))
    epoch = _epoch(pool)
    shard_address = pool._push_shard.data_ptr()
    owners = tuple((b.local_ptr, tuple(b.remote_ptrs)) for b in pool._owned_buffers)
    for step in range(8):
        for inp, out, shard, gathered, ar, graph in cases[:: 1 if step % 2 else -1]:
            _fill(inp, rank, step + 1000 + inp.numel())
            expected = _reference(inp)
            for destination in (out, shard, gathered):
                destination.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
            before = _allocation_state(pool.device)
            with (
                session.capture(),
                kernel_resolution_guard("mixed BF16 collective replay"),
            ):
                graph.replay()
            torch.cuda.synchronize()
            assert _allocation_state(pool.device) == before
            epoch += 5
            assert _epoch(pool) == epoch
            assert pool._push_shard.data_ptr() == shard_address
            assert (
                tuple((b.local_ptr, tuple(b.remote_ptrs)) for b in pool._owned_buffers)
                == owners
            )
            assert torch.equal(out, expected) and torch.equal(gathered, expected)
            local = expected.view(4, -1)[rank].view_as(shard)
            assert torch.equal(shard, local)
    for *_, graph in cases:
        graph.reset()
    inp, out, shard, gathered, ar, _ = cases[0]
    rs = _prepare(session, pool, "reduce_scatter", inp, shard)
    ag = _prepare(session, pool, "all_gather", shard, gathered)
    for operation, payload, plan, invalid, error, message in (
        (pool.all_reduce, inp, ar, out.view(-1), ValueError, "output shape"),
        (pool.all_reduce, inp, ar, inp, ValueError, "must not overlap input"),
        (pool.reduce_scatter, inp, rs, inp[:1], ValueError, "must not overlap payload"),
        (pool.all_gather, inp[:1], ag, inp, ValueError, "must not overlap payload"),
        (
            pool.reduce_scatter,
            inp,
            rs,
            shard.to(torch.float16),
            TypeError,
            "must be bfloat16",
        ),
    ):
        with pytest.raises(error, match=message):
            operation(payload, plan=plan, out=invalid)
    assert not pool.accepts(
        torch.empty(
            pool.max_rows * pool.row_elems + 32,
            device=pool.device,
            dtype=torch.bfloat16,
        )
    )
    assert not pool.accepts(torch.empty(32, device=pool.device, dtype=torch.float16))
    storage = torch.empty(33, device=pool.device, dtype=torch.bfloat16)
    assert not pool.accepts(storage[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    assert dist.get_world_size() == 4
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=(2 << 20) + 64,
        single_channel=True,
    )
    manager.prepare_channels(("test:bf16",))
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    try:
        _check_dispatch(
            session,
            manager,
            rank,
            benchmark=args.benchmark or args.benchmark_only,
            benchmark_only=args.benchmark_only,
        )
        if not args.benchmark_only:
            _check_mixed(session, manager._bf16_twoshot, rank)
            if rank == 0:
                print(
                    "TP4 BF16 prepared dispatch and mixed eager/graph correctness OK",
                    flush=True,
                )
    finally:
        torch.cuda.synchronize()
        session.close()
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
