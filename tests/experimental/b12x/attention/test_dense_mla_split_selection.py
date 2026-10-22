"""Selected dense MLA splits determine scratch and launch geometry."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import dense_mla
from b12x.attention.dense_mla._scratch import plan_dense_mla_scratch
from b12x.attention.dense_mla.planner import choose_num_splits


def _caps(**kwargs):
    return dense_mla.Caps(
        device="cpu",
        mode="decode",
        kv_dtype=torch.bfloat16,
        num_q_heads=64,
        page_size=64,
        max_total_q=1,
        max_batch=1,
        max_cache_tokens=4096,
        max_page_table_width=64,
        num_cache_pages=64,
        **kwargs,
    )


def test_selected_splits_bypass_occupancy_default():
    heuristic = choose_num_splits(
        max_cache_tokens=4096,
        max_total_q=1,
        num_q_heads=64,
        query_tile=1,
        sm_count=148,
        budget=None,
    )
    assert heuristic == 16
    caps = _caps()
    assert plan_dense_mla_scratch(caps).num_splits == 1
    selected = plan_dense_mla_scratch(caps, num_splits=64)
    assert selected.num_splits == 64
    assert selected.chunks_per_split == 1


@pytest.mark.parametrize("window_size", [None, 128])
def test_selected_splits_match_preparation_memory_and_materialization(window_size):
    caps = _caps(window_size=window_size)
    declaration = dense_mla.plan(caps)
    config = dense_mla.DenseMlaConfig(max_splits=64)
    expected = plan_dense_mla_scratch(caps, num_splits=64)
    memory = declaration._memory_requirements(config, None)
    state = declaration._materialize(SimpleNamespace(config=config), None)
    assert state.layout.num_splits == 64
    assert state.layout.scratch_specs() == expected.scratch_specs()
    assert memory.scratch == expected.scratch_specs()


@pytest.mark.parametrize("splits", [0, 65, True, 1.5])
def test_selected_splits_reject_invalid_capacity(splits):
    with pytest.raises(ValueError, match="cache chunk count"):
        plan_dense_mla_scratch(_caps(), num_splits=splits)


@pytest.mark.parametrize(
    "budget,match",
    [
        (dense_mla.Budget(max_splits=16), "split budget"),
        (dense_mla.Budget(max_partial_rows=16), "partial rows"),
    ],
)
def test_selected_splits_respect_caller_budgets(budget, match):
    caps = _caps(budget=budget)
    with pytest.raises(ValueError, match=match):
        plan_dense_mla_scratch(caps, num_splits=64)
    declaration = dense_mla.plan(caps)
    config = dense_mla.DenseMlaConfig(max_splits=64)
    with pytest.raises(ValueError, match=match):
        declaration._memory_requirements(config, None)
    assert (
        plan_dense_mla_scratch(replace(caps, budget=None), num_splits=1).num_splits == 1
    )
