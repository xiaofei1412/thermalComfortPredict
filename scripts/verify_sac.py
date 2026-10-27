"""Offline SAC data, reward, replay isolation, reload and development prediction audit."""

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
import torch

from scripts.run_sac import verify_registration
from thermal_comfort.rl_interface import bounded_forecast, native_score, project_action
from thermal_comfort.sac_env import load_plan, replay_hash, reward_value, scaled_observation, weights_hash
from thermal_comfort.simulation import POWER, digest, save_json, validate_kpis
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_baselines import check_action
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def row(before, previous_C, mean_heat, remaining):
    angle = 2*np.pi*(before["time"] % 86400)/86400
    return [before["reaTZon_y"]-273.15, before["oveTSet_u"]-273.15, previous_C,
        before["reaQFloHea_y"]/1000, sum(before[p] for p in POWER)/1000, mean_heat,
        np.sin(angle), np.cos(angle), remaining]


def verify(output):
    plan = load_plan()
    registration = read(output/"registration.json")
    repair = verify_registration(output, registration, resume=True)
    torch.set_num_threads(1)
    baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
    guard_model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
    inputs = read(ROOT/"outputs/v2-2/live-registered/development189__guarded_rule/inputs.json")
    checked, replayed, official, max_kpi, max_action, max_observation, max_prediction = 0, 0, 0, 0., 0., 0., 0.
    seeds = {}
    for seed in plan["seeds"]:
        root = output/f"seed{seed}"
        if not (root/"result.json").exists():
            seeds[str(seed)] = {"status": "failed", "details": read(root/("resume_failure.json" if (root/"resume_failure.json").exists() else "failure.json"))}
            continue
        result = read(root/"result.json")
        if result["counts"] != {"select": 9, "initialize": 9, "advance": 6048}:
            raise ValueError("Seed budget differs")
        original = repair["seeds"][str(seed)]
        call_bytes = (root/"calls.jsonl").read_bytes()
        if hashlib.sha256(call_bytes[:original["original_calls_prefix_bytes"]]).hexdigest() != original["original_calls_prefix_sha256"]:
            raise ValueError("Original failure-stage call ledger was altered")
        if digest(root/"failure.json") != original["failure_sha256"]:
            raise ValueError("Original engineering failure not preserved")
        calls = [json.loads(line) for line in call_bytes.decode("utf-8").splitlines()]
        if not all(c["ok"] for c in calls) or sum(c["path"].startswith("stop/") for c in calls) != 9:
            raise ValueError("API failures or unconfirmed worker release")
        for role, ids in [("train", plan["train_ids"]), ("development", plan["development_ids"])]:
            starts = [c["episode"]["id"] for c in calls if c["path"].startswith("initialize/") and c["episode"]["role"] == role]
            if starts != ids:
                raise ValueError("Unexpected date order/role or repeated physical episode")
        if digest(root/"checkpoint.zip") != result["checkpoint_sha256"] or digest(root/"replay.pkl") != result["replay_file_sha256"]:
            raise ValueError("Final checkpoint/replay changed")
        model = SAC.load(root/"checkpoint.zip", device="cpu")
        model.load_replay_buffer(root/"replay.pkl")
        state = result["training_state"]
        if weights_hash(model) != state["weights_sha256"] or replay_hash(model) != state["replay"]:
            raise ValueError("Saved learned state differs")
        if state["steps"] != 3984 or state["gradient_updates"] <= 0 or state["replay"]["terminal_transitions"] != 6 or state["replay"]["timeouts"]:
            raise ValueError("Gradient/terminal semantics incorrect")
        offset = 0
        for role, ids in [("train", plan["train_ids"]), ("development", plan["development_ids"])]:
            for name in ids:
                folder = root/role/name
                summary = read(folder/"summary.json")
                verify_files(folder, summary["sha256"])
                if summary != result["summaries"][name] or summary["role"] != role or summary["actor_transitions"] != 664:
                    raise ValueError("Episode summary differs")
                with gzip.open(folder/"steps.jsonl.gz", "rt", encoding="utf-8") as stream:
                    records = [json.loads(s) for s in stream]
                native = pd.read_csv(folder/"native.csv.gz", float_precision="round_trip")
                schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
                with np.load(folder/"actor_observations.npz") as data:
                    observations = data["observations"].copy()
                    terminal_observation = data["terminal_observation"].copy() if "terminal_observation" in data else model.replay_buffer.next_observations[offset+663, 0].copy()
                start = summary["episode"]["start_day"]*86400
                if len(records) != 672 or not np.array_equal(native.time, start+np.arange(20161)*30):
                    raise ValueError("Incomplete 7-day native trace")
                metrics = native_score(native, schedule)
                independent_energy = float(np.trapezoid(native[POWER].sum(axis=1), native.time)/3600000)
                if abs(independent_energy-metrics["energy_kWh"]) > 1e-10:
                    raise ValueError("Independent energy integration differs")
                for key, value in metrics.items():
                    if not np.isclose(value, summary["outcome"]["score"]["metrics"][key], rtol=1e-10, atol=1e-9):
                        raise ValueError("Native score mismatch")
                kpis = summary["outcome"]["score"]["kpis"]
                if kpis is not None:
                    max_kpi = max(max_kpi, *validate_kpis(metrics, kpis).values())
                    official += 1
                elif not (name == "train014" and summary.get("recovered_export")):
                    raise ValueError("Unexplained missing official KPI")
                if role == "development" and kpis is None:
                    raise ValueError("Development KPI must be fully retained")
                policy = SolarController(guard_model, baseline["effective_config"], "guarded_rule", "perfect")
                policy.reset(records[0]["before"], inputs)
                history = deque([row(records[0]["before"], records[0]["before"]["oveTSet_u"]-273.15, 0, 672)], maxlen=16)
                guard = None
                with gzip.open(folder/"forecasts.jsonl.gz", "rt", encoding="utf-8") as stream:
                    for index, (record, line) in enumerate(zip(records, stream, strict=True)):
                        before, after = record["before"], record["after"]
                        if record["index"] != index or before["time"] != start+index*900 or after["time"] != before["time"]+900:
                            raise ValueError("Transition timing differs")
                        view = bounded_forecast(json.loads(line), before["time"])
                        segment = native.iloc[index*30:index*30+31].reset_index(drop=True)
                        check_action(record["action"], before, segment)
                        shadow, shadow_info = policy.decide(index, before, view)
                        expected = shadow
                        if index >= 8:
                            actor_index = index-8
                            h = np.zeros((16, 9), np.float32)
                            mask = np.zeros(16, np.int8)
                            h[-len(history):], mask[-len(history):] = np.asarray(history, np.float32), 1
                            obs = scaled_observation({"history": h, "history_mask": mask,
                                "forecast": view.iloc[::30].drop(columns="time").to_numpy(np.float32)}, plan)
                            error = float(np.max(np.abs(obs-observations[actor_index])))
                            max_observation = max(max_observation, error)
                            if error > 1e-6:
                                raise ValueError("Causal observation reconstruction mismatch")
                            t, g = before["reaTZon_y"]-273.15, plan["guard"]
                            if guard == "low" and t >= g["low_release_C"] or guard == "high" and t <= g["high_release_C"]:
                                guard = None
                            if t <= g["low_enter_C"]:
                                guard = "low"
                            elif t >= g["high_enter_C"]:
                                guard = "high"
                            raw = record["control"]["raw_actor_normalized"]
                            proposal, _ = project_action([raw], before["oveTSet_u"]-273.15)
                            expected = shadow if guard else proposal
                            if record["control"]["fallback"] != (guard is not None):
                                raise ValueError("Fallback attribution differs")
                            local = native_score(segment, schedule.iloc[index*30:index*30+31].reset_index(drop=True))
                            reward, _ = reward_value(local, record["action"]["oveTSet_u"]-before["oveTSet_u"],
                                guard is not None, record["aborted"], 671-index, plan)
                            if not np.isclose(reward, record["reward"], rtol=1e-10, atol=1e-9):
                                raise ValueError("Reward differs")
                            if role == "development":
                                predicted = float(model.predict(observations[actor_index], deterministic=True)[0][0])
                                max_prediction = max(max_prediction, abs(predicted-raw))
                                if abs(predicted-raw) > 1e-6:
                                    raise ValueError("Frozen development policy replay mismatch")
                                replayed += 1
                        max_action = max(max_action, abs(expected["oveTSet_u"]-record["action"]["oveTSet_u"]))
                        if abs(expected["oveTSet_u"]-record["action"]["oveTSet_u"]) > 1e-7:
                            raise ValueError("Raw proposal/guard/actual command mismatch")
                        policy.update(before, after, segment, view.iloc[0])
                        mean_heat = float(np.trapezoid(segment.reaQFloHea_y, segment.time)/900000)
                        history.append(row(after, before["oveTSet_u"]-273.15, mean_heat, 671-index))
                        checked += 1
                if role == "train":
                    buffer = model.replay_buffer
                    np.testing.assert_array_equal(buffer.observations[offset:offset+664, 0], observations)
                    np.testing.assert_array_equal(buffer.next_observations[offset:offset+663, 0], observations[1:])
                    np.testing.assert_array_equal(buffer.next_observations[offset+663, 0], terminal_observation)
                    np.testing.assert_allclose(buffer.actions[offset:offset+664, 0, 0],
                        [r["control"]["raw_actor_normalized"] for r in records[8:]], rtol=0, atol=1e-6)
                    np.testing.assert_array_equal(buffer.rewards[offset:offset+664, 0], np.array([r["reward"] for r in records[8:]], np.float32))
                    expected_done = np.zeros(664, np.float32); expected_done[-1] = 1
                    np.testing.assert_array_equal(buffer.dones[offset:offset+664, 0], expected_done)
                    offset += 664
        if offset != 3984 or weights_hash(model) != state["weights_sha256"] or replay_hash(model) != state["replay"]:
            raise ValueError("Audit changed learned state or mixed data roles")
        seeds[str(seed)] = {"status": "verified", "counts": result["counts"], "training_steps": 3984,
            "gradient_updates": state["gradient_updates"], "replay": state["replay"], "engineering_resume": True}
        print(f"seed{seed}: all native/reward/replay/policy checks passed", flush=True)
    return {"status": "passed" if all(s["status"] == "verified" for s in seeds.values()) else "partial_failure",
        "seeds": seeds, "native_transitions_checked": checked, "development_predictions_replayed": replayed,
        "official_KPI_archives_checked": official, "training_KPI_payloads_missing_after_original_export": 3,
        "max_official_KPI_difference": max_kpi, "max_action_reconstruction_difference_K": max_action,
        "max_causal_observation_difference": max_observation, "max_development_prediction_difference": max_prediction,
        "heldout_initializations": 0, "old_strict_gates_changed": False, "additional_FMU_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, verify(args.directory))
