"""Offline OS-process reader/writer/monitor and killed-handle-holder regression."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from thermal_comfort.rl_budget import BudgetTransport
from thermal_comfort.rl_budget_safe import BudgetLedger, locked, read_snapshot, write_snapshot


def load_plan(directory):
    return json.loads((directory/"fixture-plan.json").read_text())


def check_state(state):
    if state["counts"]["advance"] != sum(s["counts"]["advance"] for s in state["seeds"].values()):
        raise ValueError("Counter snapshot is inconsistent")


def worker(directory, role, key):
    ledger = BudgetLedger(directory, load_plan(directory))
    write_snapshot(directory/(key+".ready.json"), {"ready": True, "pid": os.getpid()})
    while not (directory/"go.json").exists():
        time.sleep(.005)
    prior, reads, progress_reads = -1, 0, 0
    if role == "writer":
        for _ in range(400):
            ledger.reserve(key, "advance")
            time.sleep(.001)
    else:
        for index in range(600):
            state = ledger.read()
            check_state(state)
            if state["counts"]["advance"] < prior:
                raise ValueError("Snapshot counter moved backwards")
            prior = state["counts"]["advance"]
            reads += 1
            if role == "monitor":
                write_snapshot(directory/"progress.json", {"index": index, "budget": state})
            elif (directory/"progress.json").exists():
                check_state(read_snapshot(directory/"progress.json")["budget"])
                progress_reads += 1
            # Processing the returned value must not retain a file handle or mutex.
            time.sleep(.001)
    write_snapshot(directory/(key+".result.json"), {"status": "passed", "role": role,
        "iterations": 400 if role == "writer" else 600, "budget_reads": reads, "progress_reads": progress_reads})


def crash_worker(directory, role):
    ledger = BudgetLedger(directory, load_plan(directory))
    if role == "holder":
        with locked(ledger.lock), ledger.path.open("r"):
            (directory/"uncommitted-fixture.tmp").write_text('{"advance":999999}')
            write_snapshot(directory/"holder.ready.json", {"pid": os.getpid()})
            time.sleep(30)
    else:
        write_snapshot(directory/(role+".intent.json"), {"pid": os.getpid()})
        state = ledger.read() if role == "waiting-reader" else ledger.reserve("SAC-11", "advance")
        check_state(state)
        write_snapshot(directory/(role+".result.json"), {"status": "passed", "counts": state["counts"]})


def wait_for(path, processes, seconds=30):
    end = time.monotonic()+seconds
    while not path.exists():
        if any(p.poll() not in [None, 0] for p in processes):
            raise RuntimeError("Fixture process failed; inspect its preserved log")
        if time.monotonic() > end:
            raise TimeoutError("Fixture handshake timed out")
        time.sleep(.01)


def verify(directory):
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent.parent
    plan = json.loads((root/"configs/v2_comparison.json").read_text())
    write_snapshot(directory/"fixture-plan.json", plan)
    ledger = BudgetLedger(directory, plan)
    ledger.create()
    for seed in [11, 23, 37]:
        key = ledger.start_seed("SAC", seed)
        ledger.reserve(key, "select")
        ledger.reserve(key, "initialize")
    initial = ledger.read()
    children, streams = [], []
    def launch(role, key):
        log = (directory/(key+".log")).open("wb")
        streams.append(log)
        process = subprocess.Popen([sys.executable, "-m", "scripts.verify_v2_io_repair", "worker",
            "--output", str(directory), "--role", role, "--key", key], stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        children.append(process)
        return process
    try:
        jobs = [("writer", f"SAC-{seed}") for seed in [11, 23, 37]]+[
            ("reader", "reader-0"), ("reader", "reader-1"), ("monitor", "monitor-0")]
        for role, key in jobs:
            launch(role, key)
        for _, key in jobs:
            wait_for(directory/(key+".ready.json"), children)
        write_snapshot(directory/"go.json", {"go": True})
        for child in children:
            if child.wait(timeout=60) != 0:
                raise RuntimeError("Concurrent regression failed")
        state = ledger.read()
        if state["counts"] != {"select": 3, "initialize": 3, "advance": 1200}:
            raise ValueError("Concurrent counts were lost")
        if state["stage_start"] != initial["stage_start"] or any(
                state["seeds"][k]["start"] != initial["seeds"][k]["start"] for k in state["seeds"]):
            raise ValueError("Original clocks changed")
        results = {key: read_snapshot(directory/(key+".result.json")) for _, key in jobs}
        if sum(results[k]["progress_reads"] for k in ["reader-0", "reader-1"]) == 0:
            raise ValueError("Monitor read/write overlap was not exercised")
        class Broken:
            def call(self, *args):
                raise TimeoutError("injected transport failure")
        try:
            BudgetTransport(Broken(), ledger, "SAC-11").call("POST", "advance/fixture", {})
        except TimeoutError:
            pass
        else:
            raise ValueError("Injected failed transport was hidden")
        if ledger.read()["counts"]["advance"] != 1201:
            raise ValueError("Failed attempt was refunded")
        holder = launch("holder", "holder")
        wait_for(directory/"holder.ready.json", [holder])
        blocked = [launch(role, role) for role in ["waiting-reader", "waiting-writer"]]
        for role in ["waiting-reader", "waiting-writer"]:
            wait_for(directory/(role+".intent.json"), blocked)
        time.sleep(.1)
        if any(p.poll() is not None for p in blocked):
            raise ValueError("Reader/writer bypassed the holder mutex")
        holder.kill()  # Only this owned, socket-free file fixture; no training/FMUs.
        holder.wait(timeout=10)
        for child in blocked:
            if child.wait(timeout=10) != 0:
                raise RuntimeError("Lock/handle was not released after fixture process exit")
        final = ledger.read()
        if final["counts"]["advance"] != 1202:
            raise ValueError("Uncommitted file leaked into the ledger")
        return {"status": "passed_offline_io_regression", "platform": sys.platform,
            "concurrent_processes": 6, "concurrent_writers": 3, "concurrent_readers": 2, "concurrent_monitors": 1,
            "writer_attempts": 1200, "budget_snapshots": 1800, "monitor_snapshots": 600,
            "worker_results": results, "original_clocks_preserved": True,
            "injected_failed_attempt_charged": True, "forced_fixture_exit_released_handles": True,
            "final_fixture_counts": final["counts"], "uncommitted_temp_preserved": True,
            "snapshot_or_sharing_errors": 0, "research_training_steps": 0, "FMU_calls": 0,
            "heldout_reads_or_initializations": 0, "old_trial_reopened": False}
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role")
    parser.add_argument("--key")
    args = parser.parse_args()
    with patch("socket.socket.connect", side_effect=AssertionError("No network/FMUs authorized")):
        if args.mode == "verify":
            result = verify(args.output)
            write_snapshot(args.output/"audit.json", result)
            print(json.dumps(result, indent=2))
        elif args.role in ["writer", "reader", "monitor"]:
            worker(args.output, args.role, args.key)
        else:
            crash_worker(args.output, args.role)
