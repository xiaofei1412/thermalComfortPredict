"""Run the fixed V2-1 train014 engineering budget; never trains a policy."""

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd

from thermal_comfort.rl_interface import ResidentialEnv, START, PREFIX
from thermal_comfort.simulation import Client, digest, run_episode, save_json
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read, verify_files


class BudgetClient:
    def __init__(self, client, budget, directory):
        self.client, self.budget = client, budget
        self.started = time.perf_counter()
        self.counts = {"initialize": 0, "advance": 0}
        self.calls = []
        self.directory = directory

    def call(self, method, path, data=None):
        kind = path.split("/")[0]
        if kind != "stop" and time.perf_counter()-self.started > self.budget["wall_seconds_max"]:
            raise RuntimeError("V2-1 wall budget exhausted")
        if kind in self.counts:
            limit = "initializations_max" if kind == "initialize" else "advance_attempts_max"
            if self.counts[kind] >= self.budget[limit]:
                raise RuntimeError("V2-1 interaction budget exhausted")
            if kind == "initialize" and data != {"start_time": START, "warmup_period": 7*86400}:
                raise ValueError("Only the registered train014 initialization is allowed")
            self.counts[kind] += 1
        started = time.perf_counter()
        entry = {"path": kind, "request": data, "ok": False}
        # Persist the attempted mutation before the request: failures consume budget too.
        with (self.directory/"attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"path": kind, "counts": self.counts, "request": data})+"\n")
        try:
            result = self.client.call(method, path, data)
            entry["ok"] = True
            return result
        finally:
            entry["seconds"] = time.perf_counter()-started
            self.calls.append(entry)
            with (self.directory/"calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry)+"\n")


def export_env(env, destination, reset_info, final_info, elapsed, actor_times):
    destination.mkdir()
    with gzip.open(destination/"steps.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in env.records:
            entry = {k: v for k, v in record.items() if k not in ("native", "schedule", "forecast")}
            stream.write(json.dumps(entry, allow_nan=False)+"\n")
    with gzip.open(destination/"bounded_forecasts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for r in env.records:
            stream.write(json.dumps(r["forecast"].to_dict("list"), allow_nan=False)+"\n")
    for key in ["native", "schedule"]:
        frame = pd.concat([r[key].iloc[:-1] for r in env.records]+
                          [env.records[-1][key].iloc[-1:]], ignore_index=True)
        frame.to_csv(destination/(key+".csv.gz"), index=False,
                     compression={"method": "gzip", "mtime": 0})
    summary = {"reset": reset_info, "final": final_info, "wall_seconds": elapsed,
               "actor_steps": len(actor_times), "step_wall_seconds": {
                   "mean": float(np.mean(actor_times)), "median": float(np.median(actor_times)),
                   "p95": float(np.percentile(actor_times, 95)), "max": float(max(actor_times))},
               "sha256": {p.name: digest(p) for p in destination.iterdir()}}
    save_json(destination/"summary.json", summary)
    return summary


def execute(output):
    protocol = read(ROOT/"configs/v2_1.json")
    parent = read(ROOT/protocol["parent_lock"])
    verify_files(ROOT, parent["files_sha256"])
    verify_files(ROOT, parent["v1_baseline"]["files_sha256"])
    if digest(ROOT/"reports/v2_0/audit.json") != parent["audit_sha256"]:
        raise ValueError("Parent audit changed")
    if digest(ROOT/"outputs/solar-selected/model.joblib") != parent["v1_baseline"]["model_sha256"]:
        raise ValueError("Frozen model changed")
    if digest(ROOT/"outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu") != parent["v1_baseline"]["fmu_sha256"]:
        raise ValueError("Frozen FMU changed")
    if protocol["training_authorized"] or protocol["allowed_episode"] != "train014":
        raise ValueError("V2-1 scope changed")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/"registration.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "protocol": protocol,
        "sha256": {p: digest(ROOT/p) for p in ["configs/v2_1.json", "reports/v2_0/lock.json",
            "docs/experiments/v2_1.md", "thermal_comfort/rl_interface.py",
            "scripts/verify_rl_interface.py", "requirements/rl-interface.txt"]}})
    states = {}
    for port in [8768, 8770]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3) as response:
                states[str(port)] = json.load(response)
        except OSError as error:
            states[str(port)] = {"unavailable": type(error).__name__}
    save_json(output/"workbench-before.json", states)
    client = BudgetClient(Client(protocol["live_engine"]), protocol["live_budget"], output)
    config = parent["v1_baseline"]["effective_config"]
    model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")

    def rule():
        return SolarController(model, config, "guarded_rule", "perfect")

    legacy_config = {"testcase": "bestest_hydronic_heat_pump", "episode": {"start_day": 14},
                     "warmup_days": 7, "step_s": 900, "days": 7, "area_m2": 192}
    try:
        print("train014: legacy frozen rule, 672 steps", flush=True)
        run_episode(client, legacy_config, "train014", "guarded_rule", output/"legacy_rule", rule())
        summaries = {}
        for name, cap in [("gym_rule", None), ("gym_action_probe", 16)]:
            env = ResidentialEnv(rule(), client=client, max_agent_steps=cap)
            started = time.perf_counter()
            try:
                obs, reset_info = env.reset(seed=11)
                save_json(output/(name+"_initial_observation.json"), {k: v.tolist() for k, v in obs.items()})
                actor_times = []
                while True:
                    step_started = time.perf_counter()
                    if cap is None:
                        action, _ = env.policy.decide(env.index, env.before, env.view)
                        normalized = [(action["oveTSet_u"]-env.before["oveTSet_u"])/.5]
                    else:
                        # Upper saturation then a sustained ramp to the lower bound.
                        normalized = [1 if env.index-PREFIX < 4 else -1]
                    obs, reward, terminal, truncated, final_info = env.step(np.array(normalized, np.float32))
                    if abs(reward+env.records[-1]["metrics"]["energy_kWh"]) > 1e-12:
                        raise ValueError("Reward/energy mismatch")
                    if not env.observation_space.contains(obs):
                        raise ValueError("Observation violates Gymnasium space")
                    actor_times.append(time.perf_counter()-step_started)
                    if terminal or truncated:
                        final_info.update(terminated=terminal, truncated=truncated)
                        break
                    if len(actor_times) % 96 == 0:
                        print(f"{name}: {len(actor_times)} actor steps", flush=True)
                summaries[name] = export_env(env, output/name, reset_info, final_info,
                                            time.perf_counter()-started, actor_times)
            finally:
                env.close()
        legacy = pd.read_csv(output/"legacy_rule/native.csv.gz")
        gym_native = pd.read_csv(output/"gym_rule/native.csv.gz")
        if not np.array_equal(legacy.time, gym_native.time):
            raise ValueError("Legacy/Gym native time differs")
        differences = {c: float(np.max(np.abs(legacy[c]-gym_native[c])))
                       for c in ["reaTZon_y", "oveTSet_u"]}
        if max(differences.values()) > 1e-6:
            raise ValueError(f"Legacy/Gym native state/action mismatch: {differences}")
        old = read(output/"legacy_rule/summary.json")
        energy_difference = abs(old["metrics"]["energy_kWh"]-
                                summaries["gym_rule"]["final"]["score"]["metrics"]["energy_kWh"])
        if energy_difference > 1e-6:
            raise ValueError("Legacy/Gym energy differs")
        if not summaries["gym_rule"]["final"]["episode_complete"]:
            raise ValueError("Full rule episode did not terminate normally")
        if not summaries["gym_action_probe"]["final"]["truncated"]:
            raise ValueError("Probe did not truncate")
        result = {"status": "interface_passed_not_training_qualified", "training_runs": 0,
            "episodes_initialized": ["train014"]*client.counts["initialize"],
            "counts": client.counts, "wall_seconds": time.perf_counter()-client.started,
            "legacy": old, "gym": summaries, "equivalence_max_absolute_K": differences,
            "equivalence_energy_difference_kWh": energy_difference,
            "call_timing_seconds": {kind: {"count": len(values), "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)), "max": float(max(values))}
                for kind in ["initialize", "advance", "forecast", "results", "stop"]
                if (values := [c["seconds"] for c in client.calls if c["path"] == kind])}}
        save_json(output/"result.json", result)
        print(json.dumps({k: result[k] for k in ["status", "counts", "wall_seconds", "equivalence_max_absolute_K"]}), flush=True)
    except Exception as error:
        save_json(output/"failure.json", {"type": type(error).__name__, "error": str(error),
                                        "counts": client.counts, "wall_seconds": time.perf_counter()-client.started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args().output)
