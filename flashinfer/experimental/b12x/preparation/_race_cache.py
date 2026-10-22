"""Durable tournament checkpoints for independently progressing tuning ranks."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock


@dataclass
class RaceCheckpoint:
    measurements: dict[int, float]
    champion: int

    def validate(self):
        if not self.measurements or any(
            type(index) is not int
            or index < 0
            or type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
            for index, latency in self.measurements.items()
        ):
            raise ValueError("race checkpoints require finite candidate measurements")
        if type(self.champion) is not int or self.champion not in self.measurements:
            raise ValueError("race checkpoint champion must have a measurement")


class RaceCache:
    """Commit one rank's completed batches without a tuning collective."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            # Serialize WAL/schema initialization across simultaneous rank startups.
            with FileLock(str(self.path) + ".lock", timeout=30):
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS races_v1 ("
                    "key TEXT PRIMARY KEY, measured_count INTEGER NOT NULL, "
                    "payload TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS discoveries_v1 ("
                    "key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
        except BaseException:
            self.connection.close()
            raise

    def load(self, key: str) -> RaceCheckpoint | None:
        row = self.connection.execute(
            "SELECT payload FROM races_v1 WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        pairs = payload["measurements"]
        checkpoint = RaceCheckpoint(dict(pairs), payload["champion"])
        if len(pairs) != len(checkpoint.measurements):
            raise ValueError("race checkpoint repeats a candidate")
        checkpoint.validate()
        return checkpoint

    def save(self, key: str, checkpoint: RaceCheckpoint):
        checkpoint.validate()
        payload = json.dumps(
            {
                "measurements": sorted(checkpoint.measurements.items()),
                "champion": checkpoint.champion,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        # Concurrent identical exports may have different progress. A slower writer
        # must not replace a checkpoint containing more completed candidates.
        self.connection.execute(
            "INSERT INTO races_v1 (key, measured_count, payload) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "measured_count=excluded.measured_count, payload=excluded.payload "
            "WHERE excluded.measured_count >= races_v1.measured_count",
            (key, len(checkpoint.measurements), payload),
        )

    def delete(self, key: str):
        self.connection.execute("DELETE FROM races_v1 WHERE key = ?", (key,))

    def discovery_lock(self, key: str):
        directory = self.path.with_suffix(".discovery-locks")
        directory.mkdir(exist_ok=True)
        return FileLock(directory / (key + ".lock"))

    def load_discovery(self, key: str):
        row = self.connection.execute(
            "SELECT payload FROM discoveries_v1 WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def save_discovery(self, key: str, record):
        self.connection.execute(
            "INSERT INTO discoveries_v1 (key, payload) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
            (key, json.dumps(record, separators=(",", ":"), allow_nan=False)),
        )

    def close(self):
        self.connection.close()
