"""Read-only replica preflight and two-slot dispatch of the frozen nine seeds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_v2_formal import PARENTS, REPLICAS
from thermal_comfort.rl_budget import BudgetLedger, atomic_json
from thermal_comfort.rl_runner import append_json
from thermal_comfort.simulation import Client
from thermal_comfort.v2_protocol import ROOT, read, verify_files


DOCKER = ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--", "docker"]


def preflight(output):
    if output.exists():
        raise FileExistsError("Preserve the original preflight receipt")
    baseline = read(ROOT/PARENTS[0])["v1_baseline"]
    started = time.perf_counter()
    findings = []
    for replica in REPLICAS:
        client = Client(f"http://127.0.0.1:{replica['port']}")
        version = client.call("GET", "version")
        cases = client.call("GET", "testcases")
        if version["version"] != "0.9.0" or "bestest_hydronic_heat_pump" not in {r["testcaseid"] for r in cases}:
            raise ValueError("Replica version/testcase differs")
        worker = replica["project"]+"-worker-1"
        inspected = json.loads(subprocess.check_output(DOCKER+["inspect", worker]))[0]
        if inspected["Config"]["Image"] != "ghcr.io/ibpsa/boptest-worker:v0.9.0@sha256:08ae9e6f49ff05c38847174fb776e547888c962f2d0b283ae45ebc6f972f0bb7":
            raise ValueError("Worker image differs")
        code = ("import hashlib,json,urllib.request; "
            "data=urllib.request.urlopen('http://minio:9000/boptest/testcases/ibpsa/bestest_hydronic_heat_pump/bestest_hydronic_heat_pump.fmu').read(); "
            "print(json.dumps({'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}))")
        stored = json.loads(subprocess.check_output(DOCKER+["exec", worker, "python3", "-c", code]))
        if stored["sha256"] != baseline["fmu_sha256"]:
            raise ValueError("Provisioned FMU object differs from frozen local asset")
        findings.append({**replica, "version": version, "testcases": cases, "stored_FMU": stored,
            "worker_image": inspected["Config"]["Image"], "worker_image_id": inspected["Image"],
            "worker_started_at": inspected["State"]["StartedAt"]})
    for name in PARENTS:
        verify_files(ROOT, read(ROOT/name)["files_sha256"])
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(), "replicas": REPLICAS,
        "findings": findings, "FMU_select_or_initialize_calls": 0, "wall_seconds": time.perf_counter()-started,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    atomic_json(output, result)
    return result


def dispatch(output):
    registration = read(output/"registration.json")
    verify_files(ROOT, registration["files_sha256"])
    if not registration["formal_training_authorized"]:
        raise PermissionError("Explicit owner authorization required")
    plan = registration["protocol"]
    ledger = BudgetLedger(output, plan)
    dispatch_log = output/"dispatch.jsonl"
    if dispatch_log.exists():
        raise FileExistsError("This dispatcher cannot be resumed or duplicated")
    ports = [r["port"] for r in REPLICAS]
    for algorithm in plan["algorithms"]:
        pending = list(plan["seeds"])
        active = {}
        while pending or active:
            if pending and ports:
                seed, port = pending.pop(0), ports.pop(0)
                key = f"{algorithm}-{seed}"
                log = (output/(key+".log")).open("wb")
                started = ledger.now()
                process = subprocess.Popen([sys.executable, "-m", "scripts.run_v2_formal", "seed", "--output", str(output),
                    "--algorithm", algorithm, "--seed", str(seed), "--port", str(port)], stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                append_json(dispatch_log, {"event": "launch", "key": key, "port": port,
                    "pid": process.pid, "clock": started})
                active[key] = (process, log, port, started)
                # Preserve the exact registered start order without consuming any FMU calls.
                while key not in ledger.read()["seeds"] and process.poll() is None:
                    time.sleep(.2)
            for key, (process, log, port, started) in list(active.items()):
                if process.poll() is not None:
                    log.close()
                    if key not in ledger.read()["seeds"]:
                        seed = int(key.rsplit("-", 1)[1])
                        ledger.start_seed(algorithm, seed, started)
                        ledger.finish(key, "failed")
                        root = output/key
                        root.mkdir(exist_ok=True)
                        atomic_json(root/"failure.json", {"error": "launcher failed before runner registration",
                            "exit_code": process.returncode, "log": key+".log", "budget": ledger.read()})
                    elif ledger.read()["seeds"][key]["status"] == "running":
                        ledger.finish(key, "failed")
                        atomic_json(output/key/"dispatcher-failure.json", {"exit_code": process.returncode,
                            "error": "process exited without completing or closing its seed", "budget": ledger.read()})
                    append_json(dispatch_log, {"event": "exit", "key": key, "exit_code": process.returncode})
                    ports.append(port)
                    del active[key]
            atomic_json(output/"progress.json", {"algorithm": algorithm, "pending": pending,
                "active": {k: p[0].pid for k, p in active.items()}, "budget": ledger.read()})
            if active:
                time.sleep(2)
    atomic_json(output/"dispatch-result.json", {"status": "finished_no_retries", "budget": ledger.read()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "dispatch"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.output)
    else:
        dispatch(args.output)
