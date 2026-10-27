"""One synthetic six-episode cycle per algorithm plus publication/clock faults; no FMU."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from scripts.rl_runner_fixture import FixtureEnv, export_fixture
from scripts.run_v2_comparison import SOURCES, register
from thermal_comfort.rl_budget import BudgetLedger, atomic_json
from thermal_comfort.rl_campaign import TrainingRunner
from thermal_comfort.rl_checkpoint import inspect_bundle, save_bundle
from thermal_comfort.rl_runner import optimizer_steps
from thermal_comfort.simulation import digest
from thermal_comfort.v2_protocol import ROOT, read


class OffsetClock:
    offset = 0.
    def time(self):
        return time.time()+self.offset
    def monotonic(self):
        return time.monotonic()+self.offset


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError, OSError) as error:
        return type(error).__name__+": "+str(error)
    raise AssertionError("Fault was accepted")


def verify(output):
    register(output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read(ROOT/"configs/v2_comparison.json")
    sources = {p: digest(ROOT/p) for p in [*SOURCES, "scripts/rl_runner_fixture.py", "scripts/verify_rl_runner.py"]}
    atomic_json(output/"engineering-sources.json", sources)
    for name in sources:
        destination = output/"registered-source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    reports = {}
    for algorithm in plan["algorithms"]:
        case = output/algorithm
        case.mkdir()
        fixture_plan = copy.deepcopy(plan)
        fixture_plan["algorithms"], fixture_plan["seeds"] = [algorithm], [11]
        clock = OffsetClock()
        ledger = BudgetLedger(case, fixture_plan, clock)
        ledger.create()
        key = algorithm+"-11"
        def factory(spec, role):
            return FixtureEnv(spec, role, ledger, key, plan["shared"])
        runner = TrainingRunner(case/key, algorithm, 11, plan, ledger, factory, sources, export_fixture)
        tick = time.perf_counter()
        final = runner.run_cycle()
        duration = time.perf_counter()-tick
        pin = digest(final/"manifest.json")
        _, _, callback = inspect_bundle(final, runner.env, pin)
        if runner.model.num_timesteps != 3984 or callback["rows"] != 3984:
            raise AssertionError("Synthetic cycle boundary differs")
        rows = [json.loads(s) for s in (runner.root/"actions.jsonl").read_text().splitlines()]
        for row in rows:
            if abs(row["library_action"][0]) > 1 or not 18 <= row["control"]["actual_command_C"] <= 23.25:
                raise AssertionError("Action bounds differ")
        faults = {"early_development": rejected(runner.evaluate_final)}
        for name in ["rng.pkl", "callback.json", "budget.json"]:
            path, hidden = final/name, final/(name+".hidden")
            path.rename(hidden)
            try:
                faults["missing_"+name] = rejected(lambda: inspect_bundle(final, runner.env, pin))
            finally:
                hidden.rename(path)
        actions = runner.root/"actions.jsonl"
        raw = actions.read_bytes()
        actions.write_bytes(b"!"+raw[1:])
        try:
            faults["modified_action_prefix"] = rejected(lambda: inspect_bundle(final, runner.env, pin))
        finally:
            actions.write_bytes(raw)
        if algorithm == "PPO":
            # Re-serialize this already-trained fixture only, never rerun an episode.
            preserved = final.with_name("cycle-01.preserved")
            final.rename(preserved)
            try:
                with patch("thermal_comfort.rl_checkpoint.inspect_bundle", side_effect=OSError("injected load validation failure")):
                    faults["atomic_validation_failure"] = rejected(lambda: save_bundle(runner.root, runner.model,
                        runner.env, algorithm, 1, runner.journal, ledger, key, plan, runner.folders, sources))
                if final.exists() or not final.with_name("cycle-01.partial").exists():
                    raise AssertionError("Incomplete bundle was published")
                final.with_name("cycle-01.partial").rename(final.with_name("cycle-01.failed-validation.partial"))
                original_rename = Path.rename
                def refuse_publish(path, destination):
                    if path.name == "cycle-01.partial":
                        raise OSError("injected atomic rename failure")
                    return original_rename(path, destination)
                with patch.object(Path, "rename", refuse_publish):
                    faults["atomic_rename_failure"] = rejected(lambda: save_bundle(runner.root, runner.model,
                        runner.env, algorithm, 1, runner.journal, ledger, key, plan, runner.folders, sources))
                if final.exists() or not final.with_name("cycle-01.partial").exists():
                    raise AssertionError("Rename failure published an incomplete bundle")
                final.with_name("cycle-01.partial").rename(final.with_name("cycle-01.failed-rename.partial"))
                original_save = runner.model.save
                def slow_save(path):
                    original_save(path)
                    clock.offset += 10801
                with patch.object(runner.model, "save", side_effect=slow_save):
                    faults["serialization_cost_exceeds_original_clock"] = rejected(lambda: save_bundle(runner.root,
                        runner.model, runner.env, algorithm, 1, runner.journal, ledger, key, plan, runner.folders, sources))
                if final.exists():
                    raise AssertionError("Over-deadline checkpoint was published")
            finally:
                preserved.rename(final)
        counts = ledger.read()["seeds"][key]["counts"]
        if counts != {"select": 6, "initialize": 6, "advance": 4032}:
            raise AssertionError("Physical-prefix attempt accounting differs")
        report = {"status": "passed_synthetic_engineering_only", "actor_steps": runner.model.num_timesteps,
            "gradient_or_epoch_updates": runner.model._n_updates, "optimizer_steps": optimizer_steps(runner.model),
            "synthetic_attempts": counts, "action_rows": len(rows), "callback_rewards": len(callback["rewards"]),
            "fallback_rows": sum(r["control"]["fallback"] for r in rows),
            "PPO_raw_outside_Box_rows": sum(abs(r["algorithm_raw"][0]) > 1 for r in rows) if algorithm == "PPO" else None,
            "manifest_sha256": pin, "faults_rejected": faults, "cycle_wall_seconds": duration,
            "no_formal_development_or_holdout": True}
        atomic_json(case/"audit.json", report)
        ledger.finish(key, "failed")  # Fixture deliberately stops after one cycle; never a formal completed seed.
        reports[algorithm] = report
        print(algorithm+": synthetic cycle, checkpoint and fault checks passed", flush=True)
    # Real separate processes share the same OS-locked counter file.
    parallel = output/"parallel-budget"
    parallel.mkdir()
    atomic_json(parallel/"plan.json", plan)
    shared = BudgetLedger(parallel, plan)
    shared.create()
    keys = [shared.start_seed("SAC", seed) for seed in [11, 23, 37]]
    workers = [subprocess.Popen([sys.executable, "-m", "scripts.verify_rl_runner", "--budget-worker",
        "--output", str(parallel), "--key", key]) for key in keys]
    if any(worker.wait() != 0 for worker in workers) or shared.read()["counts"]["advance"] != 60:
        raise AssertionError("Cross-process attempt updates were lost")
    result = {"status": "passed_engineering_only", "algorithms": reports, "synthetic_training_actor_steps": 11952,
        "synthetic_scored_advances": 12096, "cross_process_budget_workers": 3,
        "cross_process_budget_reservations": 60, "formal_training_steps": 0, "FMU_calls": 0,
        "development_or_holdout_initializations": 0,
        "scope": "Synthetic parameter updates and bundle engineering, not thermal performance or formal continuation"}
    atomic_json(output/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-worker", action="store_true")
    parser.add_argument("--key")
    args = parser.parse_args()
    with patch("socket.socket.connect", side_effect=AssertionError("No network/FMUs allowed")):
        if args.budget_worker:
            ledger = BudgetLedger(args.output, read(args.output/"plan.json"))
            for _ in range(20):
                ledger.reserve(args.key, "advance")
        else:
            verify(args.output)
