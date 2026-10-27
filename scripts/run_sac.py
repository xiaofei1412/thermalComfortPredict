"""Register, run one independent SAC seed, or summarize its fixed-budget campaign."""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch

from thermal_comfort.sac_env import CampaignClient, SACEnv, load_plan, replay_hash, weights_hash
from thermal_comfort.simulation import Client, digest, save_json
from thermal_comfort.rl_interface import native_score
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_baselines import comparison, stitch
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def register(output):
    plan = load_plan()
    output.mkdir(parents=True, exist_ok=False)
    files = ["configs/v2_sac.json", "docs/experiments/v2_sac.md", "thermal_comfort/sac_env.py",
        "scripts/run_sac.py", "tests/test_sac.py", "requirements/sac.txt", "requirements/sac-windows-py314.lock.txt",
        *plan["parent_locks"]]
    baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
    if digest(ROOT/"outputs/solar-selected/model.joblib") != baseline["model_sha256"]:
        raise ValueError("Frozen baseline model changed")
    fmu = ROOT/"outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"
    if digest(fmu) != baseline["fmu_sha256"]:
        raise ValueError("FMU changed")
    save_json(output/"registration.json", {"utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"sb3": stable_baselines3.__version__, "torch": torch.__version__},
        "files_sha256": {p: digest(ROOT/p) for p in files},
        "v1_model_sha256": baseline["model_sha256"], "fmu_sha256": baseline["fmu_sha256"]})


class Curve(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.rewards = deque(maxlen=96)

    def _on_step(self):
        self.rewards.append(float(self.locals["rewards"][0]))
        if not np.isfinite(self.rewards).all():
            raise ValueError("Nonfinite training reward")
        if self.num_timesteps % 96 == 0:
            row = {"actor_steps": self.num_timesteps, "gradient_updates": self.model._n_updates,
                   "reward_mean_last96": float(np.mean(self.rewards))}
            for key in ["train/actor_loss", "train/critic_loss", "train/ent_coef"]:
                if key in self.model.logger.name_to_value:
                    row[key] = float(self.model.logger.name_to_value[key])
                    if not np.isfinite(row[key]):
                        raise ValueError("Nonfinite SAC loss")
            with (self.output/"learning_curve.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        return True


def export(env, folder, wall_seconds):
    folder.mkdir(parents=True, exist_ok=False)
    if not env.records:
        save_json(folder/"empty.json", {"records": 0})
        return None
    with gzip.open(folder/"steps.jsonl.gz", "wt", encoding="utf-8") as stream:
        for r in env.records:
            stream.write(json.dumps({k: v for k, v in r.items() if k not in ["native", "schedule", "forecast"]}, allow_nan=False)+"\n")
    with gzip.open(folder/"forecasts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for r in env.records:
            stream.write(json.dumps(r["forecast"].to_dict("list"), allow_nan=False)+"\n")
    for name in ["native", "schedule"]:
        stitch([r[name] for r in env.records]).to_csv(folder/(name+".csv.gz"), index=False,
            compression={"method": "gzip", "mtime": 0})
    np.savez_compressed(folder/"actor_observations.npz", observations=np.asarray(env.observations, dtype=np.float32),
                        terminal_observation=env.cached_obs)
    with gzip.open(folder/"terminal_forecast.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(env.view.to_dict("list"), stream, allow_nan=False)
    modes = Counter(r["control"].get("mode") for r in env.records[8:])
    summary = {"episode": env.spec, "role": env.role, "transitions": len(env.records),
        "actor_transitions": len(env.observations), "modes": dict(modes),
        "fallback_fraction_excluding_prefix": modes["fallback"]/max(1, len(env.observations)),
        "actor_return": sum(r.get("reward", 0) for r in env.records),
        "outcome": outcome_for_json(env.final_info),
        "wall_seconds": wall_seconds,
        "sha256": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}}
    b = env.before
    summary["end_state"] = {"operative_C": b["reaTZon_y"]-273.15, "return_C": b["reaTRet_y"]-273.15,
        "supply_C": b["reaTSup_y"]-273.15, "heat_kW": b["reaQFloHea_y"]/1000,
        "electric_kW": sum(b[p] for p in ["reaPHeaPum_y", "reaPFan_y", "reaPPumEmi_y"])/1000}
    save_json(folder/"summary.json", summary)
    return summary


def outcome_for_json(info):
    # DummyVecEnv mutates the returned info with this ndarray after env.step.
    # Its exact value is stored separately in actor_observations.npz.
    return None if info is None else {k: v for k, v in info.items() if k != "terminal_observation"}


def verify_registration(output, registration, resume):
    if not resume:
        verify_files(ROOT, registration["files_sha256"])
        return None
    repair = read(output/"engineering-repair.json")
    verify_files(ROOT, repair["files_sha256"])
    for name, expected in registration["files_sha256"].items():
        path = ROOT/name if digest(ROOT/name) == expected else output/"registered-source"/name
        if digest(path) != expected:
            raise ValueError("Original registered source was not preserved")
    return repair


def reference(spec):
    if spec["id"] == "train014":
        return ROOT/"outputs/v2-1/live-registered/gym_rule"
    if spec["role"] == "development":
        return ROOT/"outputs/v2-2/live-registered"/(spec["id"]+"__guarded_rule")
    return None


def run_seed(output, seed, resume=False):
    plan = load_plan()
    if seed not in plan["seeds"]:
        raise ValueError("Unregistered seed")
    registration = read(output/"registration.json")
    repair = verify_registration(output, registration, resume)
    if registration["plan"] != plan:
        raise ValueError("Learning plan changed after registration")
    torch.set_num_threads(plan["sac"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    root = output/f"seed{seed}"
    if not resume:
        root.mkdir(exist_ok=False)
    elif (root/"result.json").exists():
        raise ValueError("Cannot resume a completed seed")
    deadline = datetime.fromisoformat(registration["utc"]).timestamp()+plan["total_budget"]["wall_seconds_max"]
    if resume:
        stat = root.stat()
        deadline = min(deadline, getattr(stat, "st_birthtime", stat.st_ctime)+plan["budget_per_seed"]["wall_seconds_max"])
    client = CampaignClient(Client(f"http://127.0.0.1:{plan['replicas'][str(seed)]['port']}"), plan, root, deadline)
    baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
    if digest(ROOT/"outputs/solar-selected/model.joblib") != registration["v1_model_sha256"]:
        raise ValueError("Frozen guard model changed")
    guard_model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
    config = baseline["effective_config"]
    specs = {e["id"]: e for e in read(ROOT/"configs/v2_0.json")["episodes"]}
    curve = Curve(root)
    model, env = None, None
    summaries = {}
    started = time.perf_counter()
    try:
        train_ids = plan["train_ids"]
        if resume:
            preserved = repair["seeds"][str(seed)]
            if digest(root/"failure.json") != preserved["failure_sha256"]:
                raise ValueError("Original failure changed")
            for name in ["partial_checkpoint.zip", "partial_replay.pkl"]:
                if digest(root/name) != preserved[name+"_sha256"]:
                    raise ValueError("Recovery checkpoint changed")
            failure = read(root/"failure.json")
            if failure["counts"] != {"select": 1, "initialize": 1, "advance": 672} or failure["timesteps"] != 664:
                raise ValueError("Only the completed first-episode export failure may resume")
            with (root/"resume-marker.json").open("x", encoding="utf-8") as stream:
                json.dump({"started_utc": datetime.now(timezone.utc).isoformat(), "repair_sha256": digest(output/"engineering-repair.json")}, stream)
            client.counts = dict(failure["counts"])
            model = SAC.load(root/"partial_checkpoint.zip", device="cpu")
            model.load_replay_buffer(root/"partial_replay.pkl")
            if model.num_timesteps != 664 or replay_hash(model) != preserved["replay"] or weights_hash(model) != preserved["weights_sha256"]:
                raise ValueError("Partial learned state recovery mismatch")
            save_json(root/"resume-state.json", {"rng_reset_by_checkpoint_load": True,
                "torch_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "steps": model.num_timesteps, "gradient_updates": model._n_updates,
                "weights_sha256": weights_hash(model), "replay": replay_hash(model),
                "not_bitwise_continuation_of_uninterrupted_training": True})
            summaries["train014"] = read(root/"train/train014/summary.json")
            train_ids = train_ids[1:]
        for name in train_ids:
            spec = specs[name]
            env = SACEnv(SolarController(guard_model, config, "guarded_rule", "perfect"), client,
                         spec, "train", plan, reference(spec))
            if model is None:
                params = {k: v for k, v in plan["sac"].items() if k not in ["net_arch", "torch_threads"]}
                model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": plan["sac"]["net_arch"]},
                            seed=seed, verbose=0, **params)
            else:
                model.set_env(env)
            before = model.num_timesteps
            episode_started = time.perf_counter()
            model.learn(total_timesteps=664, reset_num_timesteps=False, callback=curve)
            summary = export(env, root/"train"/name, time.perf_counter()-episode_started)
            summaries[name] = summary
            if (model.num_timesteps-before != 664 or not env.final_info["episode_complete"]
                    or any(not torch.isfinite(p).all() for p in model.policy.parameters())):
                raise ValueError("Incomplete/nonfinite SAC training episode")
            env.close()
            env.records.clear()
        checkpoint = root/"checkpoint.zip"
        model.save(checkpoint)
        model.save_replay_buffer(root/"replay.pkl")
        training_state = {"steps": model.num_timesteps, "gradient_updates": model._n_updates,
                          "weights_sha256": weights_hash(model), "replay": replay_hash(model)}
        if (training_state["steps"] != 3984 or training_state["gradient_updates"] <= 0
                or training_state["replay"]["size"] != 3984 or training_state["replay"]["terminal_transitions"] != 6
                or training_state["replay"]["timeouts"] != 0):
            raise ValueError("Training buffer or terminal semantics differ")
        loaded = SAC.load(checkpoint, device="cpu")
        loaded.load_replay_buffer(root/"replay.pkl")
        if weights_hash(loaded) != training_state["weights_sha256"] or replay_hash(loaded) != training_state["replay"]:
            raise ValueError("Checkpoint/replay reload mismatch")
        save_json(root/"training_state.json", training_state)
        for name in plan["development_ids"]:
            spec = specs[name]
            env = SACEnv(SolarController(guard_model, config, "guarded_rule", "perfect"), client,
                         spec, "development", plan, reference(spec))
            episode_started = time.perf_counter()
            obs, _ = env.reset()
            while True:
                action, _ = loaded.predict(obs, deterministic=True)
                original, _ = model.predict(obs, deterministic=True)
                if not np.array_equal(action, original):
                    raise ValueError("Saved and loaded deterministic predictions differ")
                obs, _, terminal, truncated, info = env.step(action)
                if terminal or truncated:
                    break
            summaries[name] = export(env, root/"development"/name, time.perf_counter()-episode_started)
            if not info["episode_complete"]:
                raise ValueError("Development episode aborted")
            if (weights_hash(loaded) != training_state["weights_sha256"] or replay_hash(loaded) != training_state["replay"]
                    or loaded.num_timesteps != training_state["steps"] or loaded._n_updates != training_state["gradient_updates"]):
                raise ValueError("Evaluation updated learned state")
            env.close()
            env.records.clear()
        save_json(root/"result.json", {"status": "chain_completed_not_holdout_qualified", "seed": seed,
            "counts": client.counts, "training_state": training_state, "summaries": summaries,
            "checkpoint_sha256": digest(checkpoint), "replay_file_sha256": digest(root/"replay.pkl"),
            "wall_seconds": time.perf_counter()-started, "engineering_resume": resume})
        print(f"seed{seed}: SAC chain completed", flush=True)
    except Exception as error:
        if env is not None and env.records and not (root/env.role/env.spec["id"]).exists():
            export(env, root/env.role/env.spec["id"], time.perf_counter()-started)
        if model is not None:
            model.save(root/("resume_partial_checkpoint.zip" if resume else "partial_checkpoint.zip"))
            model.save_replay_buffer(root/("resume_partial_replay.pkl" if resume else "partial_replay.pkl"))
        save_json(root/("resume_failure.json" if resume else "failure.json"), {"error": str(error), "counts": client.counts,
            "timesteps": model.num_timesteps if model is not None else 0,
            "completed_episodes": list(summaries)})
        raise
    finally:
        if env is not None:
            env.close()


def summarize(output):
    plan = load_plan()
    old = read(ROOT/"reports/v2_2/summary.json")
    gates = read(ROOT/"configs/v2_0.json")["comparison_gates"]
    seeds = {}
    for seed in plan["seeds"]:
        root = output/f"seed{seed}"
        if not (root/"result.json").exists():
            seeds[str(seed)] = {"status": "failed", "failure": read(root/"failure.json")}
            continue
        result = read(root/"result.json")
        pairs = {}
        for name in plan["development_ids"]:
            trial = result["summaries"][name]
            metrics = trial["outcome"]["score"]["metrics"]
            pairings = {}
            active = old["runs"][name+"__guarded_rule"]["metrics"]["energy_kWh"] >= gates["active_reference_energy_kWh_min"]
            for kind in ["guarded_rule", "solar_mpc", "minimum_heat"]:
                baseline = old["runs"][name+"__"+kind]
                pair = comparison(baseline["metrics"], metrics, gates)
                pair["active"] = active
                energy = baseline["metrics"]["energy_kWh"]
                pair["energy_gate_passed"] = (pair["saving_kWh"] >= gates["saving_absolute_kWh_min"]
                    and energy > 0 and pair["saving_kWh"]/energy >= gates["saving_relative_min"] if active
                    else -pair["saving_kWh"] <= gates["low_load_energy_increase_kWh_tolerance"])
                pair["saving_fraction"] = pair["saving_kWh"]/energy if active and energy > 0 else None
                pair["joint_gate_passed"] = bool(pair["energy_gate_passed"] and pair["thermal_nonincrease"])
                pair["terminal_difference"] = {k: trial["end_state"][k]-baseline["end_state"][k]
                    for k in ["operative_C", "return_C", "supply_C", "heat_kW"]}
                pairings[kind] = pair
            pairs[name] = {"metrics": metrics, "fallback_fraction": trial["fallback_fraction_excluding_prefix"],
                "comparisons": pairings,
                "fallback_gate_passed": trial["fallback_fraction_excluding_prefix"] <= gates["rl_fallback_fraction_max_excluding_common_prefix"]}
        seeds[str(seed)] = {"status": result["status"], "counts": result["counts"],
            "training_state": result["training_state"], "development": pairs,
            "wall_seconds": result["wall_seconds"], "checkpoint_sha256": result["checkpoint_sha256"]}
    save_json(output/"summary.json", {"protocol": plan["protocol"], "seeds": seeds,
        "holdout_evaluations": 0, "new_configs_after_results": 0, "best_seed_selected": False})


def prepare_repair(output):
    plan = load_plan()
    if (output/"engineering-repair.json").exists():
        raise ValueError("A repair has already been registered")
    registration = read(output/"registration.json")
    preserved = {}
    for seed in plan["seeds"]:
        root = output/f"seed{seed}"
        failure = read(root/"failure.json")
        if (failure["error"] != "Object of type ndarray is not JSON serializable"
                or failure["counts"] != {"select": 1, "initialize": 1, "advance": 672}
                or failure["timesteps"] != 664 or (root/"development").exists()):
            raise ValueError("This is not the one permitted export-only recovery")
        calls = (root/"calls.jsonl").read_bytes()
        entries = [json.loads(s) for s in calls.decode("utf-8").splitlines()]
        if not all(c["ok"] for c in entries) or not entries[-1]["path"].startswith("stop/"):
            raise ValueError("The completed first worker was not released cleanly")
        model = SAC.load(root/"partial_checkpoint.zip", device="cpu")
        model.load_replay_buffer(root/"partial_replay.pkl")
        state = replay_hash(model)
        if model.num_timesteps != 664 or state["size"] != 664 or state["terminal_transitions"] != 1 or state["timeouts"] != 0:
            raise ValueError("First episode learned state is incomplete")
        preserved[str(seed)] = {"failure_sha256": digest(root/"failure.json"),
            "partial_checkpoint.zip_sha256": digest(root/"partial_checkpoint.zip"),
            "partial_replay.pkl_sha256": digest(root/"partial_replay.pkl"),
            "weights_sha256": weights_hash(model), "replay": state,
            "original_calls_prefix_bytes": len(calls), "original_calls_prefix_sha256": hashlib.sha256(calls).hexdigest()}
        folder = root/"train/train014"
        with gzip.open(folder/"steps.jsonl.gz", "rt", encoding="utf-8") as stream:
            records = [json.loads(s) for s in stream]
        native = pd.read_csv(folder/"native.csv.gz", float_precision="round_trip")
        schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
        if len(records) != 672 or len(native) != 20161:
            raise ValueError("Completed raw first episode missing")
        metrics = native_score(native, schedule)
        modes = Counter(r["control"].get("mode") for r in records[8:])
        b = records[-1]["after"]
        summary = {"episode": {"id": "train014", "role": "train", "start_day": 14}, "role": "train",
            "transitions": 672, "actor_transitions": 664, "modes": dict(modes),
            "fallback_fraction_excluding_prefix": modes["fallback"]/664,
            "actor_return": sum(r.get("reward", 0) for r in records), "wall_seconds": None,
            "outcome": {"metrics": records[-1]["metrics"], "control": records[-1]["control"],
                "aborted": False, "episode_complete": True, "end_reason": "finite_horizon",
                "terminated": True, "truncated": False, "score": {"metrics": metrics,
                    "kpis": None, "kpi_absolute_difference": None, "transitions": 672, "native_samples": 20161,
                    "live_validation_passed_before_export": True, "official_values_not_retained": True}},
            "end_state": {"operative_C": b["reaTZon_y"]-273.15, "return_C": b["reaTRet_y"]-273.15,
                "supply_C": b["reaTSup_y"]-273.15, "heat_kW": b["reaQFloHea_y"]/1000,
                "electric_kW": sum(b[p] for p in ["reaPHeaPum_y", "reaPFan_y", "reaPPumEmi_y"])/1000},
            "recovered_export": True, "terminal_observation_source": "replay_buffer.next_observations[663]",
            "sha256": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}}
        save_json(folder/"summary.json", summary)
    changed = [p for p, expected in registration["files_sha256"].items() if digest(ROOT/p) != expected]
    for filename in changed:
        if digest(output/"registered-source"/filename) != registration["files_sha256"][filename]:
            raise ValueError("Unpreserved source change")
    save_json(output/"engineering-repair.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "reason": "DummyVecEnv terminal_observation ndarray JSON export failure after the first complete training episode",
        "remaining_train_ids": plan["train_ids"][1:], "extra_physical_budget": 0,
        "RNG_reset_disclosed": True, "original_registration_sha256": digest(output/"registration.json"),
        "files_sha256": {p: digest(ROOT/p) for p in [*changed, "docs/experiments/v2_sac_repair.md"]}, "seeds": preserved})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "seed", "prepare-repair", "resume", "summarize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "register":
        register(args.output)
    elif args.mode == "prepare-repair":
        prepare_repair(args.output)
    elif args.mode in ["seed", "resume"]:
        run_seed(args.output, args.seed, resume=args.mode == "resume")
    else:
        summarize(args.output)
