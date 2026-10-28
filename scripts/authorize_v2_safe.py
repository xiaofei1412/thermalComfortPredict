"""Explicit owner authorization of a fresh repaired campaign; never resume an old one."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_v2_safe import PARENT_LOCKS, REPLICAS, register
from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
from thermal_comfort.simulation import digest
from thermal_comfort.v2_protocol import ROOT, read, verify_files


PARENTS = [*PARENT_LOCKS, "reports/v2_io_repair/lock.json"]
SOURCES = ["scripts/authorize_v2_safe.py", "scripts/audit_v2_safe_formal.py", "docs/experiments/v2_safe_formal.md"]


def authorize(output, preflight):
    if (preflight["status"] != "passed" or preflight["replicas"] != REPLICAS
            or preflight["FMU_select_or_initialize_calls"] != 0):
        raise ValueError("Read-only replica preflight did not pass")
    for name in PARENTS:
        verify_files(ROOT, read(ROOT/name)["files_sha256"])
    baseline = read(ROOT/PARENTS[0])["v1_baseline"]
    assets = {"outputs/solar-selected/model.joblib": baseline["model_sha256"],
        "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu": baseline["fmu_sha256"]}
    verify_files(ROOT, assets)
    register(output)
    (output/"registration-before-authorization.json").write_bytes((output/"registration.json").read_bytes())
    write_snapshot(output/"preflight.json", preflight)
    auth = {"utc": datetime.now(timezone.utc).isoformat(), "basis": "explicit_owner_request_in_current_chat",
        "request": "修复安全入口全新登记SAC/PPO/TD3正式比较；冻结预算；首回合门；失败停止；不恢复旧试次，不触碰留出",
        "protocol_sha256": digest(ROOT/"configs/v2_comparison.json"),
        "parent_locks_sha256": {p: digest(ROOT/p) for p in PARENTS}, "frozen_assets_sha256": assets,
        "replicas": REPLICAS, "actual_max_parallel": 2, "old_trial_resumption_authorized": False,
        "heldout_initializations_authorized": 0, "first_episode_gate_extra_FMU_calls": 0,
        "initial_registration_sha256": digest(output/"registration-before-authorization.json"),
        "preflight_sha256": digest(output/"preflight.json"),
        "launch_sources_sha256": {p: digest(ROOT/p) for p in SOURCES}}
    write_snapshot(output/"authorization.json", auth)
    registration = read(output/"registration.json")
    registration.update(formal_training_authorized=True, authorization_sha256=digest(output/"authorization.json"),
        parent_locks_sha256=auth["parent_locks_sha256"])
    registration["files_sha256"].update(auth["launch_sources_sha256"])
    write_snapshot(output/"registration.json", registration)
    (output/"registration-authorized.json").write_bytes((output/"registration.json").read_bytes())


def close(output):
    registration = read(output/"registration.json")
    if registration.get("formal_training_authorized") is not True or (output/"closure.json").exists():
        raise PermissionError("Only one active fresh authorization can be closed")
    ledger = BudgetLedger(output, registration["protocol"])
    state = ledger.read()
    if any(v["status"] == "running" for v in state["seeds"].values()):
        raise RuntimeError("Wait for workers to release before closing authorization")
    registration["formal_training_authorized"] = False
    write_snapshot(output/"registration.json", registration)
    write_snapshot(output/"closure.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed_no_restart_or_replacement", "counts": state["counts"],
        "authorized_registration_sha256": digest(output/"registration-authorized.json"),
        "closed_registration_sha256": digest(output/"registration.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["authorize", "close"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.mode == "authorize":
        authorize(args.output, read(args.preflight))
    else:
        close(args.output)
