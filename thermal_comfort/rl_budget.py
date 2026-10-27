"""Process-shared attempt budgets and original clocks; no restart or refund."""

from contextlib import contextmanager
import json
import hashlib
import os
from pathlib import Path
import tempfile
import time


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name+".", suffix=".tmp", delete=False, newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        name = stream.name
    os.replace(name, path)


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
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class BudgetLedger:
    def __init__(self, directory, plan, clock=time):
        self.path = Path(directory)/"budget.json"
        self.lock = self.path.with_suffix(".lock")
        self.plan, self.clock = plan, clock
        self.plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()

    def read(self):
        state = json.loads(self.path.read_text())
        current = hashlib.sha256(json.dumps(self.plan, sort_keys=True).encode()).hexdigest()
        if state["plan_sha256"] != self.plan_hash or current != self.plan_hash:
            raise ValueError("Campaign budget protocol changed")
        return state

    def create(self):
        with locked(self.lock):
            if self.path.exists():
                raise FileExistsError("Existing campaign cannot be reset")
            atomic_json(self.path, {"plan_sha256": self.plan_hash, "stage_start": None, "counts": dict.fromkeys(
                ["select", "initialize", "advance"], 0), "seeds": {}})

    def start_seed(self, algorithm, seed, started=None):
        if algorithm not in self.plan["algorithms"] or seed not in self.plan["seeds"]:
            raise ValueError("Unregistered algorithm or seed")
        key = f"{algorithm}-{seed}"
        with locked(self.lock):
            state = self.read()
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

    def now(self):
        return {"wall": self.clock.time(), "monotonic": self.clock.monotonic()}

    def elapsed(self, start):
        now = self.now()
        if now["monotonic"] < start["monotonic"]:
            raise RuntimeError("Clock origin changed; restarting is forbidden")
        return max(now[k]-start[k] for k in now)

    def reserve(self, key, kind="check"):
        with locked(self.lock):
            state = self.read()
            entry = state["seeds"][key]
            if kind == "stop":
                return state  # Cleanup stays possible after failure/deadline.
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
            state = self.read()
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


class BudgetTransport:
    """Place inside the frozen scoped client, after ownership/permission checks."""
    def __init__(self, client, ledger, key):
        self.client, self.ledger, self.key = client, ledger, key

    def call(self, method, path, data=None):
        kind = "select" if path.endswith("/select") else path.split("/")[0]
        self.ledger.reserve(self.key, kind)
        return self.client.call(method, path, data)
