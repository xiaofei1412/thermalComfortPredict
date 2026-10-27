"""Versioned repair: every mutable JSON reader and writer shares one OS lock."""

from contextlib import contextmanager
import errno
import json
import os
import time

from .rl_budget import BudgetLedger as FrozenLedger, atomic_json


LOCK_WAIT_SECONDS = 10.


@contextmanager
def locked(path):
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
        else:
            import fcntl
        deadline, acquired = time.monotonic()+LOCK_WAIT_SECONDS, False
        try:
            while not acquired:
                try:
                    if os.name == "nt":
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as error:
                    if error.errno not in [errno.EACCES, errno.EAGAIN, errno.EDEADLK]:
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Snapshot mutex acquisition timed out") from error
                    time.sleep(.005)  # Retry only a mutex, never a write or simulator request.
            yield
        finally:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_snapshot(path):
    with locked(path.with_suffix(".lock")):
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)


def write_snapshot(path, value):
    with locked(path.with_suffix(".lock")):
        atomic_json(path, value)


class BudgetLedger(FrozenLedger):
    def _read_under_lock(self):
        # Only mutation methods below call this; their surrounding OS lock is held.
        return super().read()

    def read(self):
        with locked(self.lock):
            return self._read_under_lock()

    # Frozen mutation semantics, with only the internal read entry changed.
    # Kept explicit so lock acquisition is visible and never nested.
    def start_seed(self, algorithm, seed, started=None):
        if algorithm not in self.plan["algorithms"] or seed not in self.plan["seeds"]:
            raise ValueError("Unregistered algorithm or seed")
        key = f"{algorithm}-{seed}"
        with locked(self.lock):
            state = self._read_under_lock()
            order = [f"{a}-{s}" for a in self.plan["algorithms"] for s in self.plan["seeds"]]
            if key in state["seeds"] or order[len(state["seeds"])] != key:
                raise ValueError("Duplicate/restarted or out-of-order seed")
            active = [k for k, v in state["seeds"].items() if v["status"] == "running"]
            if len(active) >= 3 or any(not k.startswith(algorithm+"-") for k in active):
                raise ValueError("Concurrency/algorithm order violated")
            start = self.now() if started is None else started
            if any(start[k] > self.now()[k] for k in start):
                raise ValueError("Seed clock cannot start in the future")
            state["seeds"][key] = {"status": "running", "start": start,
                "elapsed_seconds": 0., "counts": dict.fromkeys(state["counts"], 0)}
            atomic_json(self.path, state)
        return key

    def reserve(self, key, kind="check"):
        with locked(self.lock):
            state = self._read_under_lock()
            entry = state["seeds"][key]
            if kind == "stop":
                return state
            if entry["status"] != "running":
                raise RuntimeError("Seed has already stopped")
            entry["elapsed_seconds"] = max(entry["elapsed_seconds"], self.elapsed(entry["start"]))
            stage_elapsed = self.elapsed(state["stage_start"]) if state["stage_start"] else 0
            exhausted = (entry["elapsed_seconds"] > self.plan["budget_per_algorithm_seed"]["wall_seconds_max"]
                or stage_elapsed > self.plan["budget_total_all_algorithms_seeds"]["stage_wall_seconds_max"])
            if kind in state["counts"]:
                exhausted |= (entry["counts"][kind] >= self.plan["budget_per_algorithm_seed"][kind+"_max"]
                    or state["counts"][kind] >= self.plan["budget_total_all_algorithms_seeds"][kind+"_max"])
            if exhausted:
                atomic_json(self.path, state)
                raise RuntimeError("Original clock or cumulative interaction budget exhausted")
            if kind == "select" and state["stage_start"] is None:
                state["stage_start"] = self.now()
            if kind in state["counts"]:
                entry["counts"][kind] += 1
                state["counts"][kind] += 1
            atomic_json(self.path, state)
            return state

    def finish(self, key, status):
        if status not in ["completed", "failed"]:
            raise ValueError("Invalid final status")
        with locked(self.lock):
            state = self._read_under_lock()
            entry = state["seeds"][key]
            if entry["status"] != "running" and status != "failed":
                raise RuntimeError("Stopped seed cannot be restarted")
            entry["elapsed_seconds"] = max(entry["elapsed_seconds"], self.elapsed(entry["start"]))
            stage_elapsed = self.elapsed(state["stage_start"]) if state["stage_start"] else 0.
            invalid = status == "completed" and (
                entry["elapsed_seconds"] > self.plan["budget_per_algorithm_seed"]["wall_seconds_max"]
                or stage_elapsed > self.plan["budget_total_all_algorithms_seeds"]["stage_wall_seconds_max"]
                or any(entry["counts"][k] != self.plan["budget_per_algorithm_seed"][k+"_max"] for k in entry["counts"]))
            entry["status"] = "failed" if invalid else status
            atomic_json(self.path, state)
            if invalid:
                raise RuntimeError("Final serialization exceeded original clock or run is incomplete")
