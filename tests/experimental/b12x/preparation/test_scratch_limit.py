"""Preparation admits configurations by their declared scratch requirements."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import (
    CandidateResourceIneligibleError,
    CollectiveRequirement,
    FinalistPolicy,
    MemoryRequirements,
    PersistentMemory,
    PreparedCall,
    current_plan,
)
from .test_defaults import Config, contract
from .test_session import _deterministic_timer, declaration, session


def _request(name="scratch", *, scale=256, materialized=None, **kwargs):
    def memory(config, device):
        return MemoryRequirements(
            scratch=(
                ScratchBufferSpec(
                    name="workspace",
                    shape=(config.width * scale,),
                    dtype=torch.uint8,
                    device=torch.device("cpu"),
                ),
            ),
            persistent=(PersistentMemory(current_plan(), 16),),
        )

    def materialize(selection, device):
        if materialized is not None:
            materialized.append(selection.config.width)
        return SimpleNamespace(value=selection.config.width * 3)

    plan = replace(
        declaration(**kwargs), _memory_requirements=memory, _materialize=materialize
    )
    return plan.request(
        name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: state.value),
        benchmark_call=lambda state: PreparedCall(
            run=lambda: state.value, produce=lambda: None
        ),
    )


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "256"])
def test_scratch_limit_requires_a_nonnegative_byte_count(tmp_path, invalid):
    with session(tmp_path) as engine:
        with pytest.raises(ValueError, match="nonnegative byte count"):
            engine.configure_scratch_limit(invalid)


def test_scratch_limit_is_fixed_before_preparation_begins(tmp_path):
    with session(tmp_path) as engine:
        item = _request(tuning=contract(default=1, values=(1,)))
        assert engine.candidate_memory_envelope((item,)).scratch_nbytes == 256
        engine.configure_scratch_limit(256)
        job = engine.begin((item,))
        engine.configure_scratch_limit(256)
        with pytest.raises(RuntimeError, match="before preparation begins"):
            engine.configure_scratch_limit(512)
        job.close()
        with pytest.raises(RuntimeError, match="before preparation begins"):
            engine.configure_scratch_limit(None)


def test_scratch_envelope_and_race_admit_the_same_configurations(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    materialized = []
    item = _request(tuning=contract(), materialized=materialized)
    with session(tmp_path) as engine:
        assert engine.candidate_memory_envelope((item,)).scratch_nbytes == 1792
        engine.configure_scratch_limit(512)
        envelope = engine.candidate_memory_envelope((item,))
        assert envelope.scratch_nbytes == 512
        assert envelope.pending_persistent_nbytes == 16
        result = engine.prepare((item,))
        assert result.selections["scratch"].config == Config(2)
        assert result.benchmarked_candidates == 2
        assert result.coverage["scratch"] == {
            "cartesian_count": 3,
            "legal_count": 3,
            "effective_count": 3,
            "measured_count": 2,
            "scratch_ineligible_count": 1,
        }
    assert set(materialized) == {1, 2}


@pytest.mark.parametrize("mode", ["override", "default", "effort_off"])
def test_required_selections_fail_before_materializing(tmp_path, monkeypatch, mode):
    if mode == "effort_off":
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "off")
    materialized = []
    item = _request(
        tuning=contract(),
        pin=Config(4) if mode == "override" else None,
        materialized=materialized,
    )
    with session(tmp_path, autotune=mode != "default") as engine:
        engine.configure_scratch_limit(512)
        with pytest.raises(CandidateResourceIneligibleError, match="scratch limit"):
            engine.prepare((item,))
    assert materialized == []


def test_scratch_limit_reports_an_empty_eligible_catalog(tmp_path):
    item = _request(tuning=contract())
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(0)
        with pytest.raises(CandidateResourceIneligibleError, match="no eligible"):
            engine.candidate_memory_envelope((item,))
        with pytest.raises(CandidateResourceIneligibleError, match="no eligible"):
            engine.prepare((item,))


def test_cache_identity_follows_admitted_catalog_not_byte_limit(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    for limit, source, measured in (
        (512, "tuned", 2),
        (768, "cached", 0),
        (1024, "tuned", 3),
        (None, "cached", 0),
    ):
        with session(tmp_path) as engine:
            engine.configure_scratch_limit(limit)
            result = engine.prepare((_request(tuning=contract()),))
            assert result.selections["scratch"].source == source
            assert result.selections["scratch"].config == Config(2)
            assert result.benchmarked_candidates == measured


def test_coalesced_bindings_all_constrain_the_catalog(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    first = _request("first", tuning=contract(), scale=128)
    materialized = []
    second = _request("second", tuning=contract(), scale=256, materialized=materialized)
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(512)
        assert engine.candidate_memory_envelope((first, second)).scratch_nbytes == 512
        result = engine.prepare((first, second))
        assert result.benchmarked_candidates == 2
        assert result.selections["first"].config == Config(2)
        assert result.selections["second"].config == Config(2)
    assert materialized == [2]


def test_shared_bindings_count_persistent_storage_once(tmp_path):
    first = _request("first", tuning=contract(), shared=True)
    second = _request("second", tuning=contract(), shared=True)
    with session(tmp_path) as engine:
        assert (
            engine.candidate_memory_envelope((first, second)).pending_persistent_nbytes
            == 16
        )
        engine.configure_scratch_limit(512)
        envelope = engine.candidate_memory_envelope((first, second))
        assert envelope.scratch_nbytes == 512
        assert envelope.pending_persistent_nbytes == 16


def test_shared_alias_checks_scratch_before_installing(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    first = _request("first", tuning=contract(), shared=True)
    second = _request("second", tuning=contract(), scale=1024, shared=True)
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(512)
        engine.prepare((first,))
        prepared = first.plan.prepared
        with pytest.raises(CandidateResourceIneligibleError, match="scratch limit"):
            engine.prepare((second,))
        assert second.plan.prepared is None
        assert prepared.users == [first.plan]


@pytest.mark.parametrize("frozen", [False, True])
def test_prepared_selections_still_require_admissible_scratch(
    tmp_path, monkeypatch, frozen
):
    _deterministic_timer(monkeypatch)
    sizes = {1: 256, 2: 512, 4: 1024, 7: 1792}
    item = _request(tuning=contract())
    original = item.plan._memory_requirements
    item = replace(
        item,
        plan=replace(
            item.plan,
            _memory_requirements=lambda config, device: original(
                Config(sizes[config.width] // 256), device
            ),
        ),
    )
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(512)
        engine.prepare((item,))
        prepared = item.plan.prepared
        if frozen:
            engine.freeze()
        sizes[2] = 1024
        with pytest.raises(CandidateResourceIneligibleError, match="scratch limit"):
            engine.prepare((item,))
        assert item.plan.prepared is prepared


def test_finalist_cache_never_restores_an_excluded_family(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    for limit, source, widths in (
        (1024, "tuned", {1, 2, 4}),
        (512, "tuned", {1, 2}),
        (768, "cached", {1, 2}),
    ):
        item = replace(
            _request(tuning=contract()),
            finalist_policy=FinalistPolicy(family_fields=("width",)),
        )
        with session(tmp_path) as engine:
            engine.configure_scratch_limit(limit)
            result = engine.prepare((item,))
            assert result.selections["scratch"].source == source
            assert {
                finalist.selection.config.width
                for finalist in result.finalists["scratch"]
            } == widths


def test_collective_catalog_contains_only_admitted_configurations(tmp_path):
    item = _request(tuning=replace(contract(), collective_signature=lambda query: ()))
    item = replace(item, collective=CollectiveRequirement("scratch:0", (0, 1)))
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(512)
        job = engine.begin((item,))
        state = job.advance()
        while state.ready_collective_tuning is None:
            state = job.advance()
        requirement = state.ready_collective_tuning
        assert requirement.step == "catalog"
        assert [entry[0]["width"] for entry in requirement.payload["candidates"]] == [
            1,
            2,
        ]
        job.close()


def test_scratch_admission_preserves_sharded_counts_and_finalist_indices(
    tmp_path, monkeypatch
):
    _deterministic_timer(monkeypatch)
    engines, jobs, states = [], [], []
    try:
        for rank in (0, 1):
            engine = session(tmp_path / str(rank))
            engines.append(engine)
            engine.configure_scratch_limit(768)
            engine.configure_tuning_shard(rank, (0, 1))
            item = replace(
                _request(tuning=contract(values=(1, 2, 3, 4))),
                finalist_policy=FinalistPolicy(family_fields=("width",)),
            )
            job = engine.begin((item,))
            jobs.append(job)
            states.append(job.advance())
        snapshots = tuple(state.ready_cache for state in states)
        states = [job.advance(cache=snapshots) for job in jobs]
        for index, job in enumerate(jobs):
            while not states[index].ready_tuning:
                states[index] = job.advance()
        contributions = tuple(state.ready_tuning[0] for state in states)
        assert [
            {item.candidate_index for item in contribution.finalists}
            for contribution in contributions
        ] == [{0, 2}, {1}]
        for job in jobs:
            state = job.advance(tuning=contributions)
            while not state.done:
                state = job.advance()
            result = job.result()
            assert result.coverage["scratch"]["effective_count"] == 4
            assert result.coverage["scratch"]["scratch_ineligible_count"] == 1
            assert result.coverage["scratch"]["measured_count"] == 3
            assert {
                item.selection.config.width for item in result.finalists["scratch"]
            } == {1, 2, 3}
            assert state.completed_candidates == state.total_candidates
    finally:
        for engine in engines:
            engine.close()


def _install_payload(item, configuration, config, device):
    from b12x.preparation.types import Selection, _Prepared

    selection = Selection(
        item.plan.component_id, configuration.encoded_query, config, "tuned"
    )
    item.plan._install(
        _Prepared(
            SimpleNamespace(value=getattr(config, "width", 1) * 3),
            selection,
            frozenset(),
            None,
            (),
            (),
            device,
            users=[item.plan],
        )
    )
    return selection


def test_installation_checks_real_glm_bindings_once(tmp_path):
    from b12x.attention.dsa_indexer import glm_pool
    from b12x.preparation.session import PreparationJob, _Obligation

    calls, requests = [], []
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(0)
        for index in range(64):
            plan = glm_pool._plan(
                glm_pool.PoolQuery(mode="quantize", rows=64), torch.device("cpu")
            )
            original = plan._memory_requirements

            def memory(config, device, original=original):
                calls.append(config)
                return original(config, device)

            plan = replace(plan, _memory_requirements=memory)
            item = plan.request(
                name=f"quantize{index}", prepare_call=lambda state: None
            )
            configuration = plan.contract.configure(plan.query, device=None)
            _install_payload(item, configuration, configuration.default, engine.device)
            requests.append(item)
        obligation = _Obligation(
            requests[0],
            configuration,
            requests=tuple(requests),
            selection=requests[0].plan.selection,
            ready=True,
        )
        job = PreparationJob(engine, tuple(requests), autotune=True)
        calls.clear()
        selections = {}
        assert list(job._install_obligation(obligation, selections)) == []
        assert len(selections) == len(requests)
        assert len(calls) == len(requests)


def test_warmup_checks_each_retained_configuration_across_all_bindings(tmp_path):
    from b12x.preparation.session import PreparationJob, _Obligation

    tuning = contract(default=1)
    first = _request("first", tuning=tuning, scale=256)
    second = _request("second", tuning=tuning, scale=128)
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(256)
        configuration = tuning.configure(first.plan.query, device=None)
        selected = _install_payload(first, configuration, Config(1), engine.device)
        _install_payload(second, configuration, Config(2), engine.device)
        obligation = _Obligation(
            first,
            configuration,
            requests=(first, second),
            selection=selected,
            ready=True,
        )
        job = PreparationJob(engine, (first, second), autotune=False)
        with pytest.raises(CandidateResourceIneligibleError, match="scratch limit"):
            list(job._install_obligation(obligation, {}))
        assert first.plan.selection.config == Config(1)
        assert second.plan.selection.config == Config(2)


def test_shared_alias_configuration_rechecks_the_entire_installation_group(tmp_path):
    from b12x.preparation.session import PreparationJob, _Obligation, _declaration_key

    tuning = contract(default=1)
    first = _request("first", tuning=tuning, scale=256)
    alias = _request("alias", tuning=tuning, scale=128, shared=True)
    last = _request("last", tuning=tuning, scale=128)
    owner = _request("owner", tuning=tuning, scale=128, shared=True)
    with session(tmp_path) as engine:
        engine.configure_scratch_limit(256)
        configuration = tuning.configure(first.plan.query, device=None)
        selected = _install_payload(first, configuration, Config(1), engine.device)
        _install_payload(last, configuration, Config(1), engine.device)
        _install_payload(owner, configuration, Config(2), engine.device)
        engine._shared[_declaration_key(owner.plan)] = owner.plan
        requests = (first, alias, last)
        obligation = _Obligation(
            first,
            configuration,
            requests=requests,
            selection=selected,
            ready=True,
        )
        job = PreparationJob(engine, requests, autotune=True)
        with pytest.raises(CandidateResourceIneligibleError, match="scratch limit"):
            list(job._install_obligation(obligation, {}))
        assert alias.plan.prepared is owner.plan.prepared
        assert last.plan.selection.config == Config(1)


def test_warmup_default_binding_checks_scale_linearly(tmp_path):
    from b12x.preparation.session import PreparationJob, _Obligation
    from b12x.preparation.types import Selection

    observed = []
    for count in (4, 16):
        calls, requests = [], []
        tuning = contract(default=1)
        for index in range(count):
            item = _request(f"binding{index}", tuning=tuning)
            original = item.plan._memory_requirements

            def memory(config, device, original=original):
                calls.append(config)
                return original(config, device)

            requests.append(
                replace(item, plan=replace(item.plan, _memory_requirements=memory))
            )
        with session(tmp_path / str(count), prime=False) as engine:
            engine.configure_scratch_limit(256)
            configuration = tuning.configure(requests[0].plan.query, device=None)
            selected = Selection(
                requests[0].plan.component_id,
                configuration.encoded_query,
                configuration.default,
                "default",
            )
            obligation = _Obligation(
                requests[0],
                configuration,
                requests=tuple(requests),
                selection=selected,
            )
            job = PreparationJob(engine, tuple(requests), autotune=False)
            assert all(
                step == "gpu" for step in job._install_obligation(obligation, {})
            )
            assert all(item.plan.selection.config == Config(1) for item in requests)
            observed.append(len(calls))
    assert observed[1] <= 4 * observed[0]
