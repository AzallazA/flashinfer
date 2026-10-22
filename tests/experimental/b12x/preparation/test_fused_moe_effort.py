"""Fused MoE search effort preserves launch legality and heuristic defaults."""

from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity
from b12x.moe.fused_moe._tuning import TUNING
from .test_precision_choices import _nvfp4_query


DEVICES = (
    DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10"),
    DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000 Blackwell"),
)


def query(tokens=4, experts=384, top_k=6, recipe="w4a8_mx", **kwargs):
    return replace(
        _nvfp4_query(),
        quant_mode=recipe,
        quant_modes=(recipe,),
        source_format="fp4_e8m0_k32" if recipe == "w4a8_mx" else "modelopt_nvfp4",
        num_experts=experts,
        route_num_experts=experts,
        hidden_size=5120,
        intermediate_size=2304,
        num_tokens=tokens,
        top_k=top_k,
        routed_rows=tokens * top_k,
        shared_input_scales=recipe == "nvfp4",
        **kwargs,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("experts,top_k", ((384, 6), (128, 3)))
def test_nested_candidates_and_default_membership(device, experts, top_k):
    sets = []
    for effort in ("low", "medium", "high", "max"):
        q = query(experts=experts, top_k=top_k, autotune_effort=effort)
        configuration = TUNING.configure(q, device=device)
        default_payload = TUNING.config_payload(configuration.default)
        configuration.space.validate(default_payload)
        assert TUNING.lower(q, device, default_payload) == configuration.default
        candidates = {config for _, config in TUNING.iterate(configuration)}
        assert configuration.default in candidates
        sets.append(candidates)
    assert sets[0] <= sets[1] <= sets[2] < sets[3]
    assert any(c.max_active_clusters == 3 for c in sets[3])
    q = query(experts=experts, top_k=top_k, autotune_effort="off")
    configuration = TUNING.configure(q, device=device)
    assert list(TUNING.iterate(configuration)) == [
        (TUNING.config_payload(configuration.default), configuration.default)
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("experts,top_k,tokens", ((384, 6, 1025), (128, 3, 683)))
def test_inline_density_is_efficiency_only(device, experts, top_k, tokens):
    q = query(tokens, experts, top_k, csf_inline_scales=True)
    config = replace(
        TUNING.default_config(q, device),
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
        w4a16_route_mode=None,
    )
    payload = TUNING.config_payload(config)
    for effort in ("low", "medium"):
        with pytest.raises(ValueError, match="efficiency"):
            TUNING.lower(replace(q, autotune_effort=effort), device, payload)
    for effort in ("high", "max"):
        assert (
            TUNING.lower(replace(q, autotune_effort=effort), device, payload) == config
        )
    invalid = replace(config, dynamic_tile_m=64)
    with pytest.raises(ValueError, match="Inline scales"):
        TUNING.lower(
            replace(q, autotune_effort="max"), device, TUNING.config_payload(invalid)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_max_retains_residency_and_determinism_constraints(device):
    q = query(autotune_effort="max")
    config = replace(
        TUNING.default_config(q, device),
        backend="dynamic",
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        max_active_clusters=2 * device.sm_count + 1,
    )
    with pytest.raises(ValueError, match="outside|resident"):
        TUNING.lower(q, device, TUNING.config_payload(config))
    q = replace(q, deterministic_output=True)
    config = replace(config, max_active_clusters=None, dynamic_route_mode="direct")
    with pytest.raises(ValueError, match="deterministic"):
        TUNING.lower(q, device, TUNING.config_payload(config))


def test_effort_is_captured_and_not_a_program_identity(monkeypatch):
    from b12x.moe.fused_moe._preparation import _decode_query, _program_query_payload
    from b12x.preparation._efficiency import capture_autotune_effort

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "high")
    q = query(autotune_effort=capture_autotune_effort().value)
    assert _decode_query(TUNING.encode_query(q)).autotune_effort == "high"
    assert _program_query_payload(q) == _program_query_payload(
        replace(q, autotune_effort="max")
    )
    assert _decode_query(_program_query_payload(q)).autotune_effort == "medium"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("recipe", ("nvfp4", "w4a8_mx", "w4a16"))
def test_filters_naturally_preserve_defaults_across_shapes(device, recipe):
    for tokens in (1, 4, 8, 16, 64, 683, 1025, 2049):
        for experts, top_k in ((384, 6), (128, 3)):
            for intermediate in (2304, 1152, 576):
                for effort in ("low", "medium", "high", "max"):
                    q = replace(
                        query(tokens, experts, top_k, recipe),
                        intermediate_size=intermediate,
                        autotune_effort=effort,
                    )
                    if recipe == "w4a16":
                        q = replace(
                            q,
                            source_format="fp4_e8m0_k32",
                            w4a16_weight_layout="modelopt",
                            w4a16_scale_format="e8m0_k32",
                        )
                    config = TUNING.default_config(q, device)
                    payload = TUNING.config_payload(config)
                    TUNING.parameter_space(q, device).validate(payload)
                    assert TUNING.lower(q, device, payload) == config
