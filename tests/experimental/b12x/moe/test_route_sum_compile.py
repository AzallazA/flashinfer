"""Deterministic route reduction remains present in compiled CUDA graphs."""

import pytest
import torch

from b12x.moe.fused_moe._impl import _launch_dynamic_topk_sum
from ..conftest import require_b12x


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_route_sum_preserves_route_id_storage(monkeypatch, ids_dtype):
    from b12x.moe.fused_moe import _impl
    from b12x.moe._shared.kernels.w4a16.kernel import sum_launch

    routes = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    ids = torch.tensor([[0, -1, 1], [2, 1, -1]], dtype=ids_dtype)
    routes[ids.reshape(-1) < 0] = float("nan")
    output = torch.empty(2, 4, dtype=torch.bfloat16)
    launcher = object()

    def compile_sum(**kwargs):
        assert kwargs["route_ids_dtype"] == ids_dtype
        return launcher

    def launch_sum(
        values, result, m, topk, hidden, dtype, stream, *, route_expert_ids, launcher
    ):
        assert route_expert_ids.data_ptr() == ids.data_ptr()
        assert route_expert_ids.dtype == ids_dtype
        valid = route_expert_ids.reshape(m, topk) >= 0
        result.copy_(
            values.reshape(m, topk, hidden)
            .float()
            .masked_fill(~valid[..., None], 0)
            .sum(1)
        )

    monkeypatch.setattr(sum_launch, "compile_w4a16_topk_sum", compile_sum)
    monkeypatch.setattr(sum_launch, "_w4a16_topk_sum_launch_flat", launch_sum)
    monkeypatch.setattr(_impl, "current_cuda_stream", lambda: 0)
    _launch_dynamic_topk_sum(
        route_output=routes,
        output=output,
        m=2,
        num_topk=3,
        k=4,
        route_expert_ids=ids,
        stream=None,
    )
    torch.testing.assert_close(
        output, torch.tensor([[8, 10, 12, 14], [28, 30, 32, 34]], dtype=output.dtype)
    )


@pytest.mark.parametrize("tokens", [1, 4, 512])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_compiled_route_sum(tokens, output_dtype, ids_dtype):
    device = require_b12x()
    routes = torch.randn(tokens * 10, 2560, dtype=torch.bfloat16, device=device)
    output = torch.empty(tokens, 2560, dtype=output_dtype, device=device)
    ids = torch.zeros(tokens, 10, dtype=ids_dtype, device=device)

    def run():
        _launch_dynamic_topk_sum(
            route_output=routes,
            output=output,
            m=tokens,
            num_topk=10,
            k=2560,
            route_expert_ids=ids,
            stream=None,
        )

    run()
    expected = output.clone()
    torch.testing.assert_close(
        output, routes.reshape(tokens, 10, 2560).float().sum(1).to(output_dtype)
    )
    compiled = torch.compile(run, fullgraph=True, options={"triton.cudagraphs": False})
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(2):
        routes.normal_()
        run()
        expected.copy_(output)
        output.fill_(float("nan"))
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
