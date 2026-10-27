"""Register the shared runner; seed command fails closed until future authorization."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

PROCESS_START = {"wall": time.time(), "monotonic": time.monotonic()}

import joblib
import torch

from thermal_comfort.rl_budget import BudgetLedger, BudgetTransport, atomic_json
from thermal_comfort.rl_campaign import TrainingRunner
from thermal_comfort.rl_comparison_protocol import validate
from thermal_comfort.rl_runner import ComparisonEnv
from thermal_comfort.sac_env import CampaignClient
from thermal_comfort.simulation import Client, digest
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read, verify_files
from scripts.run_sac import reference


SOURCES = ["configs/v2_comparison.json", "thermal_comfort/rl_budget.py", "thermal_comfort/rl_runner.py",
    "thermal_comfort/rl_checkpoint.py", "thermal_comfort/rl_campaign.py", "scripts/run_v2_comparison.py",
    "thermal_comfort/rl_random_state.py", "thermal_comfort/rl_comparison_protocol.py",
    "thermal_comfort/sac_env.py", "thermal_comfort/rl_interface.py", "scripts/run_sac.py",
    "requirements/sac-windows-py314.lock.txt", "reports/v2_comparison/lock.json"]


def register(output):
    plan = read(ROOT/"configs/v2_comparison.json")
    validate(plan)
    verify_files(ROOT, read(ROOT/"reports/v2_comparison/lock.json")["files_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/"registration.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "formal_training_authorized": False, "protocol": plan,
        "files_sha256": {p: digest(ROOT/p) for p in SOURCES}})
    BudgetLedger(output, plan).create()


def run_seed(output, algorithm, seed, url):
    registration = read(output/"registration.json")
    # Checked before ledger mutation, Client construction, model loading or select.
    if registration["formal_training_authorized"] is not True:
        raise PermissionError("Formal training has not been authorized; this stage is engineering-only")
    plan = read(ROOT/"configs/v2_comparison.json")
    validate(plan)
    verify_files(ROOT, registration["files_sha256"])
    if registration["protocol"] != plan:
        raise ValueError("Registered comparison protocol changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, key = output/f"{algorithm}-{seed}", f"{algorithm}-{seed}"
    ledger = BudgetLedger(output, plan)
    adapter = {"train_ids": plan["train_ids"], "development_ids": plan["development_ids"],
        "budget_per_seed": plan["budget_per_algorithm_seed"], "fixed_observation_scaling": plan["shared"]["observation_scaling"],
        "reward": plan["shared"]["reward"], "guard": plan["shared"]["guard"]}
    client = CampaignClient(BudgetTransport(Client(url), ledger, key), adapter, root)
    guard_model = None

    def env_factory(spec, role):
        nonlocal guard_model
        baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
        if guard_model is None:
            if digest(ROOT/"outputs/solar-selected/model.joblib") != baseline["model_sha256"]:
                raise ValueError("Frozen guard model changed")
            guard_model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
        return ComparisonEnv(SolarController(guard_model, baseline["effective_config"], "guarded_rule", "perfect"),
            client, spec, role, adapter, reference(spec), algorithm=algorithm)

    TrainingRunner(root, algorithm, seed, plan, ledger, env_factory, registration["files_sha256"],
        started=PROCESS_START).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "seed"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["SAC", "PPO", "TD3"])
    parser.add_argument("--seed", type=int, choices=[11, 23, 37])
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    args = parser.parse_args()
    if args.mode == "register":
        register(args.output)
    else:
        run_seed(args.output, args.algorithm, args.seed, args.url)
