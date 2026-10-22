"""Host checks for compressed MLA split geometry search and effort identity."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla._tuning import TUNING, contract_for_query
from b12x.preparation import DeviceIdentity
from b12x.preparation._efficiency import AutotuneEffort
from b12x.preparation.session import _declaration_key

DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")


def declaration(*, rows=64, width=640, mode="decode", arithmetic="auto"):
    q = torch.empty((rows, 64, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    caps = mla.Caps(
        device="cpu",
        num_q_heads=64,
        max_q_rows=rows,
        max_width=width,
        swa_width=128,
        indexed_width=width - 128,
        cache_format="deepseek_v41",
        mode=mode,
        compute_mode=arithmetic,
    )
    return mla.plan(
        caps,
        invocation=mla.invocation_from_tensors(
            q=q,
            swa_k_cache=cache,
            indexed_k_cache=cache if width > 128 else None,
        ),
    )


def configurations(query):
    return {config for _, config in contract_for_query(query).choices(query, DEVICE)}


@pytest.mark.parametrize(
    "rows,width", [(1, 128), (64, 640), (257, 640), (1, 4096), (1, 270000)]
)
def test_effort_domains_are_nested_and_contain_heuristic(rows, width):
    query = declaration(rows=rows, width=width).query
    previous = set()
    for effort in AutotuneEffort:
        candidate_query = replace(query, autotune_effort=effort)
        configs = configurations(candidate_query)
        assert previous <= configs
        assert TUNING.default_config(candidate_query, DEVICE) in configs
        for config in configs:
            TUNING.validate_config(candidate_query, config, DEVICE)
        previous = configs


def test_medium_restores_legal_one_split_without_broadening_arithmetic():
    query = declaration().query
    low = configurations(replace(query, autotune_effort=AutotuneEffort.LOW))
    medium = configurations(replace(query, autotune_effort=AutotuneEffort.MEDIUM))
    assert {(c.max_chunks_per_row, c.split_chunk_size) for c in low} == {(54, 12)}
    assert {(c.max_chunks_per_row, c.split_chunk_size) for c in medium} == {
        (1, 1024),
        (10, 64),
        (54, 12),
    }
    assert {(c.v41_compute_mode, c.v41_heads_per_block) for c in medium} == {
        ("fp8", 8),
        ("fp8", 16),
        ("bf16", 16),
    }
    invalid = asdict(next(iter(medium)))
    invalid.update(max_chunks_per_row=1, split_chunk_size=12)
    assert not TUNING.parameter_space(query, DEVICE)._passes_predicates(invalid)


def test_max_covers_every_distinct_supported_split_contract():
    from b12x.attention.compressed_sparse_mla._tuning import _split_config

    query = replace(declaration(width=270000).query, autotune_effort=AutotuneEffort.MAX)
    supported = {
        (_split_config(query, n).num_chunks, _split_config(query, n).chunk_size)
        for n in range(1, 257)
    }
    configs = configurations(query)
    assert {(c.max_chunks_per_row, c.split_chunk_size) for c in configs} == supported


def test_pins_keep_legal_geometry_below_effort_filter():
    query = replace(declaration().query, autotune_effort=AutotuneEffort.LOW)
    config = replace(
        TUNING.default_config(query, DEVICE),
        max_chunks_per_row=1,
        split_chunk_size=1024,
    )
    assert config not in configurations(query)
    assert TUNING.configure(query, device=DEVICE, override=config).pinned == config
    with pytest.raises(ValueError, match="split contract"):
        TUNING.validate_config(query, replace(config, split_chunk_size=12), DEVICE)


@pytest.mark.parametrize("mode", ["extend", "verify", "draft_extend"])
def test_single_pass_geometry_remains_fixed(mode):
    query = replace(declaration(mode=mode).query, autotune_effort=AutotuneEffort.MAX)
    assert {
        (c.max_chunks_per_row, c.split_chunk_size, c.single_pass)
        for c in configurations(query)
    } == {(1, 1, True)}


def test_effort_is_captured_keyed_and_absent_from_compile_payload(monkeypatch):
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "low")
    low = declaration()
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "max")
    maximum = declaration()
    assert low.query.autotune_effort is AutotuneEffort.LOW
    assert maximum.query.autotune_effort is AutotuneEffort.MAX
    assert _declaration_key(low) != _declaration_key(maximum)
    assert low.contract.encode_query(low.query)["autotune_effort"] == "low"
    assert low._compile_jobs(
        TUNING.default_config(low.query, DEVICE), SimpleNamespace(ordinal=0)
    ) == maximum._compile_jobs(
        TUNING.default_config(maximum.query, DEVICE), SimpleNamespace(ordinal=0)
    )


def test_bf16_constraint_still_excludes_fp8_at_max():
    plan = declaration(arithmetic="bf16")
    query = replace(plan.query, autotune_effort=AutotuneEffort.MAX)
    assert all(c.v41_compute_mode == "bf16" for c in configurations(query))
