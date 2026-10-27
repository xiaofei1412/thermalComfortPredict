"""Owner-authorized formal launch; frozen runner plus an in-budget first-episode gate."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

PROCESS_START = {"wall": time.time(), "monotonic": time.monotonic()}

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.run_sac import export, reference
from scripts.run_v2_comparison import register
from thermal_comfort.rl_budget import BudgetLedger, BudgetTransport, atomic_json
from thermal_comfort.rl_campaign import TrainingRunner
from thermal_comfort.rl_comparison_protocol import validate
from thermal_comfort.rl_interface import native_score
from thermal_comfort.rl_runner import ComparisonEnv
from thermal_comfort.sac_env import CampaignClient
from thermal_comfort.simulation import Client, digest
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read, verify_files


PARENTS = ["reports/v2_0/lock.json", "reports/v2_1/lock.json", "reports/v2_2/lock.json",
    "reports/v2_3_numerics/lock.json", "reports/v2_sac/lock.json", "reports/v2_comparison/lock.json",
    "reports/v2_runner/lock.json"]
REPLICAS = [{"project": "thermal-sac-11", "port": 8083}, {"project": "thermal-sac-23", "port": 8084}]


def authorize(output, preflight):
    for name in PARENTS:
        verify_files(ROOT, read(ROOT/name)["files_sha256"])
    baseline = read(ROOT/PARENTS[0])["v1_baseline"]
    assets = {"outputs/solar-selected/model.joblib": baseline["model_sha256"],
        "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu": baseline["fmu_sha256"]}
    verify_files(ROOT, assets)
    if preflight["status"] != "passed" or preflight["replicas"] != REPLICAS or preflight["FMU_select_or_initialize_calls"] != 0:
        raise ValueError("Read-only replica preflight did not pass")
    register(output)
    original = (output/"registration.json").read_bytes()
    (output/"registration-before-authorization.json").write_bytes(original)
    atomic_json(output/"preflight.json", preflight)
    auth = {"utc": datetime.now(timezone.utc).isoformat(), "basis": "explicit_owner_request_in_current_chat",
        "request": "正式启动冻结预算下的SAC、PPO、TD3比较；原预算内核查首回合；失败停止，不触碰留出",
        "protocol_sha256": digest(ROOT/"configs/v2_comparison.json"),
        "parent_locks_sha256": {p: digest(ROOT/p) for p in PARENTS},
        "frozen_assets_sha256": assets, "replicas": REPLICAS, "actual_max_parallel": 2,
        "reason_for_parallel_limit": "preserve memory for other active applications; within frozen maximum three",
        "heldout_initializations_authorized": 0, "first_episode_gate_extra_FMU_calls": 0,
        "initial_registration_sha256": digest(output/"registration-before-authorization.json"),
        "preflight_sha256": digest(output/"preflight.json"),
        "launch_sources_sha256": {p: digest(ROOT/p) for p in ["scripts/run_v2_formal.py", "scripts/dispatch_v2_formal.py", "docs/experiments/v2_formal.md"]}}
    atomic_json(output/"authorization.json", auth)
    registration = read(output/"registration.json")
    registration.update(formal_training_authorized=True, authorization_sha256=digest(output/"authorization.json"))
    registration["files_sha256"].update(auth["launch_sources_sha256"])
    atomic_json(output/"registration.json", registration)


def check_first(env, summary, ledger, key, model, episode_seconds):
    score = summary["outcome"]["score"]
    budget = ledger.reserve(key)
    entry = budget["seeds"][key]
    if (env.spec["id"] != "train014" or env.role != "train" or model.num_timesteps != 664
            or summary["transitions"] != 672 or summary["actor_transitions"] != 664
            or score["native_samples"] != 20161 or not score["kpis"]
            or entry["counts"] != {"select": 1, "initialize": 1, "advance": 672}
            or any(not np.isfinite(v) or v < 0 or v > 1e-7 for v in score["kpi_absolute_difference"].values())):
        raise ValueError("First-episode frozen count/KPI gate failed")
    return {"status": "passed", "episode": "train014", "prefix": "frozen_reset_compared_all_native_prefix_fields",
        "counts": entry["counts"], "native_samples": 20161, "kpi_absolute_difference": score["kpi_absolute_difference"],
        "episode_wall_seconds_including_reset_and_updates": episode_seconds,
        "original_seed_elapsed_seconds": entry["elapsed_seconds"],
        "env_step_seconds_mean_excluding_policy_updates": float(np.mean(env.step_seconds)),
        "cost_gate": "unchanged_original_seed_and_stage_clocks_no_projected_threshold", "extra_FMU_calls": 0}


class FirstGateRunner(TrainingRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, exporter=self.export_with_gate, **kwargs)
        self.first_checked = False

    def export_with_gate(self, env, folder, wall_seconds):
        summary = export(env, folder, wall_seconds)
        if not self.first_checked:
            result = check_first(env, summary, self.ledger, self.key, self.model, wall_seconds)
            native = pd.read_csv(folder/"native.csv.gz", float_precision="round_trip")
            schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
            recomputed = native_score(native, schedule)
            maximum = max(abs(value-summary["outcome"]["score"]["metrics"][name]) for name, value in recomputed.items())
            if not np.isfinite(maximum) or maximum > 1e-7:
                raise ValueError("First exported native score failed independent recomputation")
            result["max_exported_native_metric_difference"] = maximum
            atomic_json(self.root/"first-gate.json", result)
            self.first_checked = True
            print(self.key+": first real episode gate passed", flush=True)
        return summary


def seed(output, algorithm, number, port):
    registration, auth = read(output/"registration.json"), read(output/"authorization.json")
    if not registration["formal_training_authorized"] or digest(output/"authorization.json") != registration["authorization_sha256"]:
        raise PermissionError("No intact owner authorization")
    verify_files(ROOT, registration["files_sha256"])
    verify_files(ROOT, auth["parent_locks_sha256"])
    verify_files(ROOT, auth["frozen_assets_sha256"])
    if port not in [r["port"] for r in auth["replicas"]]:
        raise ValueError("Unregistered replica endpoint")
    plan = read(ROOT/"configs/v2_comparison.json")
    validate(plan)
    if registration["protocol"] != plan:
        raise ValueError("Frozen comparison protocol changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    key, root = f"{algorithm}-{number}", output/f"{algorithm}-{number}"
    ledger = BudgetLedger(output, plan)
    adapter = {"train_ids": plan["train_ids"], "development_ids": plan["development_ids"],
        "budget_per_seed": plan["budget_per_algorithm_seed"], "fixed_observation_scaling": plan["shared"]["observation_scaling"],
        "reward": plan["shared"]["reward"], "guard": plan["shared"]["guard"]}
    client = CampaignClient(BudgetTransport(Client(f"http://127.0.0.1:{port}"), ledger, key), adapter, root)
    guard = None
    def factory(spec, role):
        nonlocal guard
        baseline = read(ROOT/PARENTS[0])["v1_baseline"]
        if guard is None:
            guard = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
        return ComparisonEnv(SolarController(guard, baseline["effective_config"], "guarded_rule", "perfect"),
            client, spec, role, adapter, reference(spec), algorithm=algorithm)
    FirstGateRunner(root, algorithm, number, plan, ledger, factory,
        registration["files_sha256"], started=PROCESS_START).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["authorize", "seed"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--algorithm", choices=["SAC", "PPO", "TD3"])
    parser.add_argument("--seed", type=int, choices=[11, 23, 37])
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.mode == "authorize":
        authorize(args.output, read(args.preflight))
    else:
        seed(args.output, args.algorithm, args.seed, args.port)
