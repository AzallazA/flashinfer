"""Durable publication for the CuTe object cache.

Objects are written to a temporary file, fsynced, renamed into place, and the
directory entry is persisted, so a visible object file is always complete.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


def _fsync_directory(path: Path) -> None:
    # Windows has no O_DIRECTORY: directories cannot be opened or fsynced there.
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    if path.is_dir():
        return
    _mkdir_durable(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    # Persist new shard directories as well as files within those shards.
    _fsync_directory(path.parent)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish a complete, fsynced file, then persist its directory entry.

    Writers must hold the compiler's per-key lock across object and manifest
    publication. Publish the manifest last; readers reject incomplete pairs.
    """
    _mkdir_durable(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            with suppress(OSError):
                os.unlink(temporary)
