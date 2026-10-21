"""Output compatibility and coordinated setup for the PCIe dispatcher."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import pcie_allreduce as dispatch
from b12x.comm.pcie import pcie_oneshot as oneshot
from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16


def _runtime():
    runtime = object.__new__(PCIeTwoShotBF16)
    runtime.device = torch.device("cpu")
    runtime.world_size = 4
    runtime.row_elems = 8
    runtime.max_rows = (2 << 20) // 16
    runtime._closed = False
    return runtime


@pytest.mark.parametrize(
    "nbytes", (65536 - 64, 65536, 65536 + 64, 2 << 20, (2 << 20) + 64)
)
def test_output_aware_twoshot_size_band(nbytes):
    primary = SimpleNamespace(
        rank=0, world_size=4, device=torch.device("cpu"), max_size=(2 << 20) + 64
    )
    manager = dispatch.PCIeAllReduce(
        primary, "oneshot", bf16_twoshot=_runtime(), algorithm_override="auto"
    )
    inp = torch.empty(nbytes // 2, dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    expected = 65536 <= nbytes <= 2 << 20
    assert manager._use_bf16_twoshot(inp, out) == expected
    assert manager._use_bf16_twoshot(inp, None) == expected
    assert not manager._use_bf16_twoshot(inp, inp)


def test_twoshot_rejects_partially_overlapping_output():
    primary = SimpleNamespace(
        rank=0, world_size=4, device=torch.device("cpu"), max_size=2 << 20
    )
    manager = dispatch.PCIeAllReduce(
        primary, "oneshot", bf16_twoshot=_runtime(), algorithm_override="auto"
    )
    storage = torch.empty(32768 + 8, dtype=torch.bfloat16)
    assert not manager._use_bf16_twoshot(storage[:-8], storage[8:])
    assert not manager._use_bf16_twoshot(
        storage[:-8], torch.empty(128, 256, dtype=torch.bfloat16)
    )
    assert not manager._use_bf16_twoshot(
        storage[:-8], torch.empty(32768, dtype=torch.float16)
    )


@pytest.mark.parametrize("failure", ("local", "peer", "construction"))
def test_setup_failure_closes_existing_runtime(monkeypatch, failure):
    closed = []
    runtime = SimpleNamespace(
        device=torch.device("cuda:0"), close=lambda: closed.append(True)
    )
    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(
        dispatch.PCIeOneshotAllReducePool,
        "from_exchange_group",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: None
    )

    def properties(device):
        if failure == "local":
            raise RuntimeError("injected capability failure")
        return SimpleNamespace(multi_processor_count=64)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

    def exchange(local_error, **kwargs):
        if failure == "local":
            assert str(local_error) == "injected capability failure"
            return ((str(local_error),), (), (), ())
        assert local_error is None
        return (
            ((), ("injected peer capability failure",), (), ())
            if failure == "peer"
            else ((),) * 4
        )

    monkeypatch.setattr(oneshot, "_exchange_setup_failures", exchange)

    def gather(count, group):
        assert failure == "construction", (
            "SM exchange must follow successful capability setup"
        )
        return [count] * 4

    monkeypatch.setattr(dispatch, "_broadcast_gather_object", gather)

    def construct(**kwargs):
        raise RuntimeError("injected construction failure")

    monkeypatch.setattr(dispatch.PCIeTwoShotBF16, "from_exchange_group", construct)
    with pytest.raises(RuntimeError, match="injected"):
        dispatch.PCIeAllReduce.from_exchange_group(
            exchange_group=object(), device="cuda:0", single_channel=True
        )
    assert closed == [True]


@pytest.mark.parametrize(
    "counts, expected", (([64] * 4, True), ([64, 63, 64, 64], False))
)
def test_capability_requires_resident_grid_on_every_rank(monkeypatch, counts, expected):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=64),
    )
    monkeypatch.setattr(
        oneshot, "_exchange_setup_failures", lambda *args, **kwargs: ((),) * 4
    )
    monkeypatch.setattr(dispatch, "_broadcast_gather_object", lambda *args: counts)
    assert (
        dispatch._bf16_twoshot_supported(
            exchange_group=object(), device="cuda:0", world_size=4
        )
        == expected
    )
