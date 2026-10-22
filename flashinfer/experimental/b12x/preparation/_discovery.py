"""Persistent program discovery from metadata factories and their source closure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path

from b12x._lib.compile_plan import ProgramKey
from b12x._lib.compile_pool import CompilationPlan
from b12x._lib.source_dependencies import canonical_module, source_dependencies
from ._cache import digest
from .types import FrozenMapping


def _metadata(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, FrozenMapping):
        return ["frozen_mapping", value.to_dict()]
    if isinstance(value, Enum):
        return [
            "enum",
            canonical_module(type(value).__module__),
            type(value).__qualname__,
            _metadata(value.value),
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return [
            "dataclass",
            canonical_module(type(value).__module__),
            type(value).__qualname__,
            [
                [item.name, _metadata(getattr(value, item.name))]
                for item in fields(value)
            ],
        ]
    if isinstance(value, Mapping):
        if all(type(key) is str for key in value):
            return ["mapping", {key: _metadata(item) for key, item in value.items()}]
        return [
            "mapping",
            sorted(
                [(_metadata(key), _metadata(item)) for key, item in value.items()],
                key=lambda item: json.dumps(item[0], sort_keys=True),
            ),
        ]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_metadata(item) for item in value]]
    import torch

    if isinstance(value, torch.dtype):
        return ["dtype", str(value)]
    if isinstance(value, torch.device):
        if value.index is not None:
            raise ValueError("compile job placement requires CompileJob.DEVICE_ORDINAL")
        return ["device", value.type]
    raise TypeError(f"unsupported discovery metadata: {type(value).__name__}")


@lru_cache(maxsize=None)
def _factory_sources(factories):
    dependencies = source_dependencies(Path(__file__).resolve().parents[1])
    modules = sorted(
        {
            *(canonical_module(factory.partition(":")[0]) for factory in factories),
            "b12x.preparation._discovery",
            "b12x._lib.compile_pool",
            "b12x._lib.compile_plan",
            "b12x._lib.compiler",
        }
    )
    roots = tuple((module, dependencies.module_path(module)) for module in modules)
    if any(path is None for _, path in roots):
        return None
    return dependencies.fingerprint(roots)


@lru_cache(maxsize=None)
def _compiler_context(compute_capability):
    from cutlass import cute
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import get_cache_invalidating_env_vars, make_backend
    from triton.runtime.cache import triton_key
    from b12x._lib.compiler import _static_compile_cache_context
    from b12x._lib.compile_options import resolve_compile_options

    major, minor = compute_capability
    backend = make_backend(GPUTarget("cuda", major * 10 + minor, 32))
    hook = knobs.runtime.add_stages_inspection_hook
    return (
        _static_compile_cache_context(resolve_compile_options(cute.compile)),
        triton_key(),
        backend.hash(),
        backend.parse_options({}).hash(),
        tuple(sorted(get_cache_invalidating_env_vars().items())),
        None if hook is None else hook()[0],
    )


def discovery_key(candidates, device):
    """Identify an ordered candidate set without invoking any kernel factory.

    Source and toolchain fingerprints share the compiler's process-local memos.
    Factories outside the b12x source tree use uncached discovery.
    """
    if device.identity is None:
        return None
    factories = tuple(
        sorted({job.factory for _, _, jobs in candidates for job in jobs})
    )
    sources = _factory_sources(factories)
    if sources is None:
        return None
    identity = device.identity
    return digest(
        {
            "schema": 1,
            "target": (
                identity.vendor,
                identity.compute_capability,
                identity.sm_count,
                device.max_shared_memory_per_block,
                device.max_shared_memory_per_multiprocessor,
            ),
            "compiler": _compiler_context(identity.compute_capability),
            "sources": sources,
            "candidates": [
                (
                    assignment.to_dict(),
                    payload.to_dict(),
                    [
                        (job.factory, _metadata(job.args), _metadata(job.kwargs))
                        for job in jobs
                    ],
                )
                for assignment, payload, jobs in candidates
            ],
        }
    )


def discovery_record(candidates, compiled, identity):
    return {
        "identity": identity,
        "programs": [
            [
                [(p.dialect, p.key, p.name) for p in item.programs]
                for item in compiled[payload]
            ]
            for _, payload, _ in candidates
        ],
    }


def restore_discovery(candidates, record):
    compiled, programs = {}, {}
    for (_, payload, jobs), descriptions in zip(
        candidates, record["programs"], strict=True
    ):
        compiled[payload] = tuple(
            CompilationPlan(job, tuple(ProgramKey(*p) for p in keys))
            for job, keys in zip(jobs, descriptions, strict=True)
        )
        programs[payload] = frozenset(
            p for item in compiled[payload] for p in item.programs
        )
    return record["identity"], compiled, programs
