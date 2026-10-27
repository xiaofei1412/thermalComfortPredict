"""Versioned I/O repair entry; all newly registered campaigns remain unauthorized."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import time

PROCESS_START = {"wall": time.time(), "monotonic": time.monotonic()}

import joblib
import torch

from scripts.run_sac import reference
from scripts.run_v2_comparison import register as frozen_register
from scripts.run_v2_formal import PARENTS, REPLICAS
from thermal_comfort.rl_budget import BudgetTransport
from thermal_comfort.rl_budget_safe import BudgetLedger, read_snapshot, write_snapshot
from thermal_comfort.rl_comparison_protocol import validate
from thermal_comfort.rl_failure_export import FirstGateRunner
from thermal_comfort.rl_runner import ComparisonEnv, append_json
from thermal_comfort.sac_env import CampaignClient
from thermal_comfort.simulation import Client, digest
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read, verify_files


REVISION = "serialized_snapshot_and_fragment_v1"
PARENT_LOCKS = [*PARENTS, "reports/v2_formal/lock.json"]
REPAIR_SOURCES = ["thermal_comfort/rl_budget_safe.py", "thermal_comfort/rl_failure_export.py", "scripts/run_v2_safe.py"]


def register(output):
    for name in PARENT_LOCKS:
        verify_files(ROOT, read(ROOT/name)["files_sha256"])
    frozen_register(output)  # Exclusive new directory, false authorization, zero attempts.
    registration = read(output/"registration.json")
    registration.update(io_revision=REVISION, formal_training_authorized=False,
        parent_locks_sha256={name: digest(ROOT/name) for name in PARENT_LOCKS})
    registration["files_sha256"].update({name: digest(ROOT/name) for name in REPAIR_SOURCES})
    write_snapshot(output/"registration.json", registration)


def registration_for(output, require_authorization):
    # Registration is immutable during dispatch. Reject old trials before sidecar creation.
    registration = read(output/"registration.json")
    if registration.get("io_revision") != REVISION:
        raise PermissionError("Old trials cannot be resumed with the repaired runner")
    if require_authorization and registration["formal_training_authorized"] is not True:
        raise PermissionError("This offline repair does not authorize formal training")
    verify_files(ROOT, registration["files_sha256"])
    verify_files(ROOT, registration["parent_locks_sha256"])
    validate(registration["protocol"])
    return registration


def status(output):
    registration = registration_for(output, False)
    ledger = BudgetLedger(output, registration["protocol"])
    result = {"budget": ledger.read(), "formal_training_authorized": registration["formal_training_authorized"]}
    if (output/"progress.json").exists():
        result["progress"] = read_snapshot(output/"progress.json")
    return result


def seed(output, algorithm, number, port):
    registration = registration_for(output, True)
    if port not in [replica["port"] for replica in REPLICAS]:
        raise ValueError("Unregistered replica")
    baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
    verify_files(ROOT, {"outputs/solar-selected/model.joblib": baseline["model_sha256"],
        "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu": baseline["fmu_sha256"]})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = registration["protocol"]
    key, root = f"{algorithm}-{number}", output/f"{algorithm}-{number}"
    ledger = BudgetLedger(output, plan)
    adapter = {"train_ids": plan["train_ids"], "development_ids": plan["development_ids"],
        "budget_per_seed": plan["budget_per_algorithm_seed"], "fixed_observation_scaling": plan["shared"]["observation_scaling"],
        "reward": plan["shared"]["reward"], "guard": plan["shared"]["guard"]}
    client = CampaignClient(BudgetTransport(Client(f"http://127.0.0.1:{port}"), ledger, key), adapter, root)
    guard = None
    def factory(spec, role):
        nonlocal guard
        if guard is None:
            guard = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
        return ComparisonEnv(SolarController(guard, baseline["effective_config"], "guarded_rule", "perfect"),
            client, spec, role, adapter, reference(spec), algorithm=algorithm)
    FirstGateRunner(root, algorithm, number, plan, ledger, factory,
        registration["files_sha256"], started=PROCESS_START).run()


def dispatch(output):
    registration = registration_for(output, True)
    ledger = BudgetLedger(output, registration["protocol"])
    log_path = output/"dispatch.jsonl"
    if log_path.exists():
        raise FileExistsError("No dispatcher restart is permitted")
    ports = [replica["port"] for replica in REPLICAS]
    active = {}
    with ExitStack() as handles:
        try:
            for algorithm in registration["protocol"]["algorithms"]:
                pending = list(registration["protocol"]["seeds"])
                while pending or active:
                    if pending and ports:
                        number, port = pending.pop(0), ports.pop(0)
                        key = f"{algorithm}-{number}"
                        log = handles.enter_context((output/(key+".log")).open("wb"))
                        started = ledger.now()
                        process = subprocess.Popen([sys.executable, "-m", "scripts.run_v2_safe", "seed", "--output", str(output),
                            "--algorithm", algorithm, "--seed", str(number), "--port", str(port)], stdout=log,
                            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        active[key] = (process, log, port, started)
                        append_json(log_path, {"event": "launch", "key": key, "port": port, "pid": process.pid})
                        while key not in ledger.read()["seeds"] and process.poll() is None:
                            time.sleep(.2)
                    for key, (process, log, port, started) in list(active.items()):
                        if process.poll() is not None:
                            state = ledger.read()
                            if key not in state["seeds"]:
                                ledger.start_seed(algorithm, int(key.rsplit("-", 1)[1]), started)
                                ledger.finish(key, "failed")
                                root = output/key
                                root.mkdir(exist_ok=True)
                                write_snapshot(root/"failure.json", {"error": "launcher failed before runner registration",
                                    "exit_code": process.returncode, "budget": ledger.read()})
                            elif state["seeds"][key]["status"] == "running":
                                ledger.finish(key, "failed")
                                write_snapshot(output/key/"dispatcher-failure.json", {"error": "worker exited without closing seed",
                                    "exit_code": process.returncode, "budget": ledger.read()})
                            append_json(log_path, {"event": "exit", "key": key, "exit_code": process.returncode})
                            log.close()
                            ports.append(port)
                            del active[key]
                    write_snapshot(output/"progress.json", {"algorithm": algorithm, "pending": pending,
                        "active": {k: v[0].pid for k, v in active.items()}, "budget": ledger.read()})
                    if active:
                        time.sleep(2)
            write_snapshot(output/"dispatch-result.json", {"status": "finished_no_retries", "budget": ledger.read()})
        except BaseException:
            # Cooperative stop only: never reissue a possibly applied advance.
            for key in active:
                if key in ledger.read()["seeds"]:
                    ledger.finish(key, "failed")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "status", "seed", "dispatch"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["SAC", "PPO", "TD3"])
    parser.add_argument("--seed", type=int, choices=[11, 23, 37])
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.mode == "register":
        register(args.output)
    elif args.mode == "status":
        print(status(args.output))
    elif args.mode == "dispatch":
        dispatch(args.output)
    else:
        seed(args.output, args.algorithm, args.seed, args.port)
