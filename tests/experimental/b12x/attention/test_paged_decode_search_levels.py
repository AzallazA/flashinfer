"""Host-only paged decode head-split legality and search effort."""

from dataclasses import replace

import pytest

from b12x.attention.paged_decode import _preparation, _tuning
from b12x.preparation import DeviceIdentity, FrozenMapping, autotune_effort


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "Synthetic SM120")


def query(**changes):
    return replace(
        _tuning.PagedDecodeQuery(
            kv_dtype="float8_e4m3fn",
            num_q_heads=64,
            num_kv_heads=1,
            head_dim_qk=128,
            head_dim_vo=128,
            page_size=64,
            max_batch=1,
            max_q_per_req=1,
            window_left=-1,
            causal=True,
            has_sinks=False,
            descale_layout="tensor",
            sm_count=188,
            exhaustive=False,
            autotune_effort="medium",
        ),
        **changes,
    )


@pytest.mark.parametrize("heads", [3, 8, 64, 256])
@pytest.mark.parametrize("max_q", [1, 8])
@pytest.mark.parametrize("window", [-1, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_nested_search_and_natural_default_membership(heads, max_q, window, causal):
    q = query(num_q_heads=heads, max_q_per_req=max_q, window_left=window, causal=causal)
    default = _tuning.default_config(q)
    previous = set()
    for effort in ("off", "low", "medium", "high", "max"):
        active = replace(q, autotune_effort=effort)
        space = _tuning.TUNING.parameter_space(active, DEVICE)
        assignment = FrozenMapping({"head_splits": default.head_splits})
        space.validate(assignment)
        assert assignment in tuple(space.configurations())
        selected = {cfg for _, cfg in _tuning.TUNING.choices(active, DEVICE)}
        assert default in selected
        assert previous <= selected
        if effort in ("off", "low", "medium"):
            assert selected == {default}
        else:
            assert {cfg.head_splits for cfg in selected} == {
                hs for hs in _tuning._head_split_choices(q) if _tuning._rows_fit(q, hs)
            }
        for cfg in selected:
            _tuning.validate_config(active, cfg)
        previous = selected


def test_high_restores_legal_slices_but_keeps_row_capacity():
    q = query(max_q_per_req=8, autotune_effort="high")
    space = _tuning.TUNING.parameter_space(q, DEVICE)
    assert {choice["head_splits"] for choice in space.configurations()} == {4, 8, 16}
    with pytest.raises(ValueError, match="correctness"):
        space.validate(FrozenMapping({"head_splits": 1}))
    exhaustive = replace(q, autotune_effort="low", exhaustive=True)
    assert tuple(
        _tuning.TUNING.parameter_space(exhaustive, DEVICE).configurations()
    ) == tuple(space.configurations())


def test_effort_is_captured_in_query_and_excluded_from_compile_key():
    base = query()
    config = _tuning.default_config(base)
    with autotune_effort("high"):
        captured = _tuning.PagedDecodeQuery(
            **{
                k: v
                for k, v in _tuning.TUNING.encode_query(base).items()
                if k not in ("exhaustive", "autotune_effort")
            }
        )
    assert captured.autotune_effort == "high"
    assert _tuning.TUNING.encode_query(base) != _tuning.TUNING.encode_query(captured)
    assert _preparation._compile_key(base, config) == _preparation._compile_key(
        captured, config
    )
    assert _preparation._compile_key(base, config) == _preparation._compile_key(
        replace(captured, exhaustive=True), config
    )
