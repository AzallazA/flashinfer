"""Offline compiler jobs release executable owners after publishing artifacts."""

from .test_compile_pool_fake_indexing import _run_offline_worker


def test_repeated_worker_jobs_release_unretained_programs_and_preserve_artifacts():
    _run_offline_worker(
        """
        import gc
        import os
        import pickle
        import weakref
        from dataclasses import replace
        from tempfile import TemporaryDirectory

        from b12x.norm import mhc
        from b12x.preparation.device import detect_device
        from b12x._lib import compile_plan, compiler
        from b12x._lib.compile_pool import describe_compilation
        from b12x._lib.program_cache import evict_unretained

        initialize_worker()
        device = detect_device("cuda:0")
        plan = mhc.plan(mhc.Caps(device="cuda:0", max_tokens=1, hidden_size=5120),
                        invocation=dict(operation="collapse"))
        config = plan.contract.default_config(plan.query, device.identity)
        (job,) = plan._compile_jobs(config, device)
        description = describe_compilation(job)
        compile_plan.evict_planning_artifacts()
        factory = compile_pool._factory(job.factory)
        references = []

        def observe_factory(*args, **kwargs):
            result = factory(*args, **kwargs)
            references.extend(weakref.ref(compile_plan._RESIDENT_PROGRAMS[program])
                              for program in compile_plan.program_keys(result))
            return result

        observed_job = replace(job, factory="__main__:observe_factory")
        payload = pickle.dumps(observed_job)
        with TemporaryDirectory() as cache:
            os.environ["B12X_COMPILE_CACHE_DIR"] = cache
            before = compiler.compile_cache_info()["compile_misses"]
            first = compile_pool._run_job(payload, description.programs)
            assert first[0] == os.getpid()
            assert first[1] > 0 and first[2] == 0
            assert first[3] == description.programs
            assert references and all(owner() is None for owner in references), (
                "a completed compiler job retained an executable owner"
            )
            assert all(compile_plan.compiled_program_available(program)
                       for program in description.programs)
            for _ in range(3):
                completed = compile_pool._run_job(payload, description.programs)
                assert completed == (os.getpid(), 0, 0, description.programs)
                assert all(owner() is None for owner in references)
            assert compiler.compile_cache_info()["compile_misses"] == before + first[1]

            references.clear()
            with compile_plan.retain_compiled_programs() as retained:
                completed = compile_pool._run_job(payload, description.programs)
                assert completed == (os.getpid(), 0, 0, description.programs)
                assert references and all(owner() is not None for owner in references)
            assert all(owner() is not None for owner in references)
            del retained
            evict_unretained(())
            gc.collect()
            assert all(owner() is None for owner in references)

            unexpected = compile_plan.ProgramKey("cute", "unexpected", "unexpected")
            try:
                compile_pool._run_job(payload, (unexpected,))
            except RuntimeError as error:
                assert observed_job.factory in str(error)
                assert "factory program keys changed" in str(error)
                assert "planned-only" in str(error) and "compiled-only" in str(error)
            else:
                raise AssertionError("compiler job accepted different program keys")
            assert compiler.compile_cache_info()["compile_misses"] == before + first[1]
        assert not torch.cuda.is_initialized()
        """
    )
