"""Packed precision effort widens performance choices without weakening legality."""

from dataclasses import replace

import pytest

from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING
from b12x.preparation import DeviceIdentity


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
LEVELS = ("low", "medium", "high", "max")


def query(recipe="nvfp4", m=8, n=128, k=256, **kwargs):
    return BlockscaledQuery(
        recipe=recipe,
        num_tokens=m,
        in_features=k,
        padded_in_features=k,
        out_features=n,
        **kwargs,
    )


def candidates(q, level):
    q = replace(q, autotune_effort=level)
    return {
        tuple(sorted(TUNING.equivalence_key(q, DEVICE, config).items()))
        for _, config in TUNING.choices(q, DEVICE)
    }


@pytest.mark.parametrize(
    "recipe", ("nvfp4", "mxfp4", "mxfp8", "iq2_xs", "iq2_xxs", "q8_0")
)
@pytest.mark.parametrize(
    "m,n,k",
    ((1, 8, 256), (8, 64, 256), (17, 128, 256), (17, 32768, 5120), (65, 64, 256)),
)
def test_levels_are_nested_and_naturally_include_heuristic(recipe, m, n, k):
    q = query(recipe, m, n, k)
    default = TUNING.default_config(q, DEVICE)
    TUNING.validate_config(q, default, DEVICE)
    key = tuple(sorted(TUNING.equivalence_key(q, DEVICE, default).items()))
    previous = set()
    for level in LEVELS:
        choices = candidates(q, level)
        assert previous <= choices
        assert key in choices
        previous = choices


@pytest.mark.parametrize(
    "q,tile_m,tile_n,tile_k,mma_warps,minimum",
    (
        (query(m=8), 64, 64, 64, None, "high"),
        (query("mxfp8", m=16, n=64), 16, 128, 64, 4, "high"),
        (query("iq2_xs", m=16), 4, 4, 256, None, "max"),
    ),
)
def test_performance_filters_lift_at_declared_effort(
    q, tile_m, tile_n, tile_k, mma_warps, minimum
):
    choice = dict(
        mode="a16",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=1,
        mma_warps=mma_warps,
    )
    for level in LEVELS:
        active = replace(q, autotune_effort=level)
        if LEVELS.index(level) < LEVELS.index(minimum):
            with pytest.raises(ValueError, match="efficiency"):
                TUNING.lower(active, DEVICE, choice)
        else:
            TUNING.lower(active, DEVICE, choice)


@pytest.mark.parametrize("level", LEVELS)
def test_mxfp8_row_support_precision_and_workspace_stay_mandatory(level):
    q = query("mxfp8", m=64, autotune_effort=level)
    choice = dict(mode="a16", tile_m=32, tile_n=64, tile_k=64, split_k=1, mma_warps=4)
    with pytest.raises(ValueError):
        TUNING.lower(q, DEVICE, choice)
    with pytest.raises(ValueError):
        TUNING.lower(
            replace(q, activation_mode="quantized"), DEVICE, choice | dict(tile_m=16)
        )
    with pytest.raises(ValueError, match="workspace"):
        TUNING.lower(
            replace(q, workspace_nbytes=0), DEVICE, choice | dict(tile_m=16, split_k=4)
        )


def test_effort_is_captured_in_declaration_but_excluded_from_program_key(monkeypatch):
    from b12x.gemm.blockscaled._preparation import _compile_query_payload, _decode_query

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "high")
    q = query()
    assert q.autotune_effort == "high"
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "low")
    assert q.autotune_effort == "high"
    assert TUNING.encode_query(q) != TUNING.encode_query(
        replace(q, autotune_effort="low")
    )
    payload = _compile_query_payload(q)
    assert payload == _compile_query_payload(replace(q, autotune_effort="max"))
    assert "autotune_effort" not in payload
    assert _decode_query(payload).autotune_effort == "medium"
