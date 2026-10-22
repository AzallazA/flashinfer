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


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_unsliceable_projection_layout_is_always_illegal(effort):
    q = query(tokens=4096, effort=effort)
    config = replace(_tuning.TUNING.default_config(q, None), projection_tile_n=48)
    with pytest.raises(ValueError, match="correctness predicates"):
        _tuning.TUNING.parameter_space(q, None).validate(assignment(config))
    with pytest.raises(ValueError, match="cannot slice tile_n=48"):
        _tuning.TUNING.validate_config(q, config, None)


def test_projection_constructor_rejects_unsliceable_layout():
    from b12x.norm.mhc._kernels import MHCPrefillTf32ProjectTmaKernel

    with pytest.raises(ValueError, match="cannot slice tile_n=48"):
        MHCPrefillTf32ProjectTmaKernel(
            hidden_size=5120,
            tile_m=16,
            tile_n=48,
            tile_k=32,
            num_stages=1,
            num_m_warps=1,
            num_n_warps=1,
            k_splits=2,
        )


def test_worker_compiles_legal_mhc_projection_tiles():
    from .test_compile_pool_fake_indexing import _run_offline_worker

    _run_offline_worker(
        """
        from dataclasses import replace
        from tempfile import TemporaryDirectory
        import os
        from b12x.norm import mhc
        from b12x.preparation import autotune_effort
        from b12x.preparation.device import detect_device
        from b12x._lib.compile_plan import compiled_program_available
        from b12x._lib.compile_pool import CompilePool

        compile_pool._initialize_worker(
            0, (12, 0), "synthetic", "RTX PRO 6000", 188, 101376, 102400,
            multiprocessing.get_context("spawn").Array("q", (0, 0)),
        )
        device = detect_device("cuda:0")
        jobs = []
        for operation in ("pre", "post_pre"):
            with autotune_effort("max"):
                plan = mhc.plan(
                    mhc.Caps(device="cuda:0", max_tokens=4096, hidden_size=5120),
                    invocation=dict(operation=operation, has_norm_weight=True,
                                    lagged_mix=True, expanded_residual=operation == "pre",
                                    rms_eps=1e-20, hc_eps=1e-6, norm_eps=1e-20,
                                    sinkhorn_iters=20),
                )
            default = plan.contract.default_config(plan.query, device.identity)
            for tile_n in (24, 32, 64):
                config = replace(default, projection_tile_m=16,
                                 projection_num_m_warps=1, projection_num_n_warps=1,
                                 projection_tile_n=tile_n, projection_tile_k=32,
                                 projection_num_stages=1, projection_k_splits=2)
                plan.contract.validate_config(plan.query, config, device.identity)
                jobs.extend(plan._compile_jobs(config, device))
        with TemporaryDirectory() as cache:
            os.environ["B12X_COMPILE_CACHE_DIR"] = cache
            with CompilePool(device_ordinal=0, compute_capability=(12, 0),
                             device_uuid="synthetic", product_name="RTX PRO 6000",
                             sm_count=188, max_shared_memory_per_block=101376,
                             max_shared_memory_per_multiprocessor=102400,
                             workers=1) as engine:
                descriptions = engine.plan(jobs)
                programs = engine.submit_plans(descriptions)
                engine.wait(programs)
                assert all(compiled_program_available(p) for p in programs)
                assert engine.summary().cute_compilations > 0
        assert not torch.cuda.is_initialized()
        """
    )


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
    config = replace(_tuning.TUNING.default_config(q, None), projection_tile_n=64)
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
