"""Transactional tuning progress survives process loss and concurrent ranks."""

import multiprocessing
import os

import pytest

from b12x.preparation._race_cache import RaceCache, RaceCheckpoint


def _crash_writer(path):
    cache = RaceCache(path)
    cache.save("race", RaceCheckpoint({0: 3.0, 1: 1.0}, 1))
    cache.connection.execute("BEGIN IMMEDIATE")
    cache.connection.execute("DELETE FROM races_v1")
    os._exit(17)


def _rank_writer(path, rank):
    cache = RaceCache(path)
    for count in range(1, 6):
        cache.save(str(rank), RaceCheckpoint({i: 1.0 + i for i in range(count)}, 0))
    cache.close()


def test_committed_batch_survives_process_exit_and_uncommitted_transaction(tmp_path):
    path = tmp_path / "progress.sqlite3"
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_writer, args=(path,)
    )
    process.start()
    process.join(30)
    try:
        assert process.exitcode == 17
        cache = RaceCache(path)
        try:
            assert cache.load("race") == RaceCheckpoint({0: 3.0, 1: 1.0}, 1)
            assert cache.load("other-race") is None
        finally:
            cache.close()
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def test_concurrent_ranks_keep_independent_checkpoints(tmp_path):
    path = tmp_path / "progress.sqlite3"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_rank_writer, args=(path, rank)) for rank in range(4)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        cache = RaceCache(path)
        try:
            for rank in range(4):
                assert cache.load(str(rank)) == RaceCheckpoint(
                    {i: 1.0 + i for i in range(5)}, 0
                )
            cache.save("0", RaceCheckpoint({0: 2.0}, 0))
            assert len(cache.load("0").measurements) == 5
        finally:
            cache.close()
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


@pytest.mark.parametrize(
    "checkpoint",
    [
        RaceCheckpoint({}, 0),
        RaceCheckpoint({0: float("nan")}, 0),
        RaceCheckpoint({0: -1.0}, 0),
        RaceCheckpoint({0: 1.0}, 1),
    ],
)
def test_invalid_measurements_cannot_replace_a_checkpoint(tmp_path, checkpoint):
    cache = RaceCache(tmp_path / "progress.sqlite3")
    try:
        good = RaceCheckpoint({0: 1.0}, 0)
        cache.save("race", good)
        with pytest.raises(ValueError):
            cache.save("race", checkpoint)
        assert cache.load("race") == good
    finally:
        cache.close()
