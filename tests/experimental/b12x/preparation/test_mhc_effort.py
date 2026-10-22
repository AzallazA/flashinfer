"""MHC effort tiers preserve heuristic geometry and kernel legality."""

from dataclasses import replace

import pytest

from b12x.norm.mhc import _preparation, _tuning
from b12x.preparation import DeviceIdentity, FrozenMapping


def query(tokens=8192, hidden=5120, effort="medium", pin=None):
    controls = {"B12X_AUTOTUNE_EFFORT": effort}
    if pin is not None:
        controls["B12X_MHC_PREFILL_TF32_MMA"] = str(int(pin))
    return _tuning.MhcQuery(
        dtype="bfloat16",
        max_tokens=tokens,
        hidden_size=hidden,
        split_k=hidden // 64,
        operation="post_pre",
        has_norm_weight=True,
        lagged_mix=True,
        rms_eps=1e-20,
        norm_eps=1e-20,
        smem_limit=100 << 10,
        controls=FrozenMapping(controls),
    )


def assignment(config):
    return {k: v for k, v in config.to_dict().items() if k != "projection_tile_m"}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize(
    "tokens", [1, 64, 128, 384, 1024, 2048, 2304, 3072, 3584, 4096, 8192]
)
@pytest.mark.parametrize("pin", [None, True, False])
def test_heuristic_naturally_survives(effort, hidden, tokens, pin):
    q = query(tokens, hidden, effort, pin)
    device = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    config = _tuning.TUNING.default_config(q, device)
    _tuning.TUNING.parameter_space(q, device).validate(assignment(config))


@pytest.mark.parametrize(
    "changes",
    [
        {"projection_num_n_warps": 3},
        {"projection_tile_k": 128, "projection_num_stages": 3},
        {"projection_k_splits": 16},
    ],
)
def test_high_explores_broad_prefill_tradeoffs(changes):
    q = query(tokens=4096)
    config = replace(_tuning.TUNING.default_config(q, None), **changes)
    choice = assignment(config)
    with pytest.raises(ValueError, match="efficiency predicates"):
        _tuning.TUNING.parameter_space(q, None).validate(choice)
    space = _tuning.TUNING.parameter_space(query(tokens=4096, effort="high"), None)
    space.validate(choice)


def test_low_shortlist_is_lifted_at_medium():
    q = query(tokens=64, effort="low")
    config = replace(
        _tuning.TUNING.default_config(q, None),
        backend="tf32_tma",
        lagged_prepare=False,
        projection_tile_n=24,
        projection_tile_k=64,
        projection_num_stages=2,
        projection_k_splits=20,
    )
    choice = assignment(config)
    with pytest.raises(ValueError, match="efficiency predicates"):
        _tuning.TUNING.parameter_space(q, None).validate(choice)
    _tuning.TUNING.parameter_space(query(tokens=64), None).validate(choice)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_effort_preserves_hard_constraints(effort):
    q = query(effort=effort)
    config = replace(_tuning.TUNING.default_config(q, None), projection_tile_k=16)
    with pytest.raises(ValueError, match="correctness predicates"):
        _tuning.TUNING.parameter_space(q, None).validate(assignment(config))


def test_effort_captured_and_removed_from_compile_cache_key(monkeypatch):
    from b12x.norm import mhc

    invocation = FrozenMapping({"operation": "collapse"})
    caps = mhc.Caps(device="cpu", max_tokens=1, hidden_size=5120)
    plans = []
    for effort in ("low", "high"):
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", effort)
        plans.append(mhc.plan(caps, invocation=invocation))
    assert plans[0].query.controls["B12X_AUTOTUNE_EFFORT"] == "low"
    assert plans[1].query.controls["B12X_AUTOTUNE_EFFORT"] == "high"
    assert _tuning.TUNING.encode_query(plans[0].query) != _tuning.TUNING.encode_query(
        plans[1].query
    )
    monkeypatch.setattr(_preparation, "_compile_mhc", lambda *args: args)
    keys = [
        _preparation.compile_mhc(_tuning.TUNING.encode_query(plan.query), {}, {}, 0)
        for plan in plans
    ]
    assert keys[0] == keys[1]


def test_off_selects_only_heuristic():
    q = query(effort="off")
    configuration = _tuning.TUNING.configure(q, device=None)
    candidates = list(_tuning.TUNING.iterate(configuration))
    assert len(candidates) == 1
    assert candidates[0][1] == configuration.default


def test_explicit_effort_overrides_legacy_exhaustive():
    q = query(effort="low")
    q = replace(
        q,
        controls=FrozenMapping(
            {**q.controls.to_dict(), "B12X_AUTOTUNE_EXHAUSTIVE": "1"}
        ),
    )
    assert _tuning.TUNING.parameter_space(q, None).autotune_effort.value == "low"
    q = replace(q, controls=FrozenMapping({"B12X_AUTOTUNE_EXHAUSTIVE": "1"}))
    assert _tuning.TUNING.parameter_space(q, None).autotune_effort.value == "max"


def test_max_explores_padding_while_retaining_resource_limits():
    q = query(tokens=4096, effort="high")
    config = replace(_tuning.TUNING.default_config(q, None), projection_tile_n=48)
    with pytest.raises(ValueError, match="efficiency predicates"):
        _tuning.TUNING.parameter_space(q, None).validate(assignment(config))
    exhaustive = query(tokens=4096, effort="max")
    _tuning.TUNING.parameter_space(exhaustive, None).validate(assignment(config))
    config = replace(config, projection_tile_k=256, projection_num_stages=4)
    with pytest.raises(ValueError, match="correctness predicates"):
        _tuning.TUNING.parameter_space(exhaustive, None).validate(assignment(config))


def test_off_returns_lowerable_assignment_for_materialized_default():
    q = query(tokens=128, effort="off")
    configuration = _tuning.TUNING.configure(q, device=None)
    [(choice, config)] = list(_tuning.TUNING.iterate(configuration))
    assert config == configuration.default
    assert "projection_tile_m" not in choice
    assert _tuning.TUNING.lower(q, None, choice) == config


def test_explicit_effort_overrides_invalid_legacy_control():
    q = query(tokens=128, effort="high")
    q = replace(
        q,
        controls=FrozenMapping(
            {**q.controls.to_dict(), "B12X_AUTOTUNE_EXHAUSTIVE": "invalid"}
        ),
    )
    configuration = _tuning.TUNING.configure(q, device=None)
    assert configuration.space.autotune_effort == "high"
