"""Recompute V2-2 artifacts and replay frozen decisions; no simulator calls."""

import argparse
import ast
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thermal_comfort.rl_interface import bounded_forecast
from thermal_comfort.simulation import FORECAST, POWER, digest, save_json, validate_kpis
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_baselines import analyze, check_action, command, protocol, score
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def independent_score(native, schedule):
    t = native.time.to_numpy()
    power = sum(native[p].to_numpy() for p in POWER)
    energy_ws = float(np.trapezoid(power, t))
    values = {"energy_kWh": energy_ws/3600000,
        "boptest_energy_kWh_m2": energy_ws*2.77778e-7/192,
        "heat_kWh": float(np.trapezoid(native.reaQFloHea_y, t)/3600000),
        "cost_EUR_m2": float(np.trapezoid(power*schedule.PriceElectricPowerConstant, t)*2.77778e-7/192),
        "emissions_kg_m2": float(np.trapezoid(power*schedule.EmissionsElectricPower, t)*2.77778e-7/192)}
    for kind, d in [("cold", np.maximum(schedule["LowerSetp[1]"]-native.reaTZon_y, 0)),
                    ("hot", np.maximum(native.reaTZon_y-schedule["UpperSetp[1]"], 0))]:
        values[kind+"_Kh"] = float(np.trapezoid(d, t)/3600)
        values[kind+"_duration_s"] = float(np.trapezoid((d > 0).astype(float), t))
        values[kind+"_max_K"] = float(max(d))
    values["discomfort_Kh"] = values["cold_Kh"]+values["hot_Kh"]
    return values


def verify(directory):
    plan, episodes, config = protocol()
    registration, result = read(directory/"registration.json"), read(directory/"result.json")
    repaired = "thermal_comfort/v2_baselines.py"
    verify_files(ROOT, {p: h for p, h in registration["sha256"].items() if p != repaired})
    if digest(ROOT/repaired) != registration["sha256"][repaired]:
        original = directory/"registered-source"/repaired
        if digest(original) != registration["sha256"][repaired]:
            raise ValueError("Original live-run source snapshot changed")
        repair = read(directory/"offline-repair.json")
        verify_files(ROOT, repair["files_sha256"])
        if repair["simulator_calls_added"] != 0:
            raise ValueError("Offline repair must not consume new FMU interactions")
        old_tree = ast.parse(original.read_text(encoding="utf-8"))
        new_tree = ast.parse((ROOT/repaired).read_text(encoding="utf-8"))
        helper = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "assert_same_values")
        expected = ast.parse('def assert_same_values(left, right):\n    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)')
        if ast.dump(helper) != ast.dump(expected.body[0]):
            raise ValueError("Unexpected numeric comparison repair")
        new_tree.body.remove(helper)
        for call in ast.walk(new_tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "assert_same_values":
                call.func = ast.parse("pd.testing.assert_frame_equal", mode="eval").body
                call.keywords = [ast.keyword(arg="check_exact", value=ast.Constant(value=True))]
        if ast.dump(old_tree) != ast.dump(new_tree):
            raise ValueError("Repair changed more than CSV dtype compatibility")
    if registration["protocol"] != plan or result["counts"] != {"initialize": 15, "advance": 6192}:
        raise ValueError("Protocol or budget differs")
    calls = [json.loads(s) for s in (directory/"calls.jsonl").read_text(encoding="utf-8").splitlines()]
    if not all(c["ok"] for c in calls):
        raise ValueError("Invalid API call in completed experiment")
    counts = {k: sum(c["path"].startswith(k+"/") for c in calls) for k in ["initialize", "advance", "stop", "forecast"]}
    if counts != {"initialize": 15, "advance": 6192, "stop": 15, "forecast": 6192}:
        raise ValueError("Actual API counts differ")
    initialized_days = [c["request"]["start_time"]/86400 for c in calls if c["path"].startswith("initialize/")]
    if sorted(initialized_days) != [84]*5+[168]*5+[189]*5:
        raise ValueError("Unexpected date or date count")
    for c in calls:
        if c["path"].startswith("forecast/") and c["request"] != {
                "point_names": FORECAST, "horizon": 43200, "interval": 30}:
            raise ValueError("Forecast permission exceeded")
    model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
    summaries, checks = {}, []
    max_action_difference, max_kpi_difference = 0., 0.
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir() or folder.name == "registered-source":
            continue
        summary = read(folder/"summary.json")
        verify_files(folder, summary["sha256"])
        episode = summary["episode"]
        if episode not in episodes:
            raise ValueError("Unexpected data role")
        records = [json.loads(s) for s in (folder/"steps.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = 672 if summary["target_C"] is None else 24
        if len(records) != expected:
            raise ValueError("Incomplete episode")
        native = pd.read_csv(folder/"native.csv.gz")
        schedule = pd.read_csv(folder/"schedule.csv.gz")
        start = episode["start_day"]*86400
        if not np.array_equal(native.time, start+np.arange(expected*30+1)*30):
            raise ValueError("Native time gaps")
        np.testing.assert_array_equal(native.time, schedule.time)
        metrics = independent_score(native, schedule)
        for name, value in metrics.items():
            if not np.isclose(value, summary["metrics"][name], rtol=1e-10, atol=1e-9):
                raise ValueError(f"Independent metric differs: {folder.name}/{name}")
        max_kpi_difference = max(max_kpi_difference, *validate_kpis(metrics, summary["kpis"]).values())
        policy = SolarController(model, config, summary["kind"], "perfect")
        policy.reset(summary["initial"], read(folder/"inputs.json"))
        previous_after = summary["initial"]
        interval_energy = 0.
        with gzip.open(folder/"forecasts.jsonl.gz", "rt", encoding="utf-8") as stream:
            for index, (record, line) in enumerate(zip(records, stream, strict=True)):
                before, after = record["before"], record["after"]
                if record["index"] != index or before != previous_after or after["time"] != before["time"]+900:
                    raise ValueError("Transition ordering differs")
                view = bounded_forecast(json.loads(line), before["time"])
                segment = native.iloc[index*30:index*30+31].reset_index(drop=True)
                local_schedule = schedule.iloc[index*30:index*30+31].reset_index(drop=True)
                np.testing.assert_allclose(local_schedule, view.iloc[:31][local_schedule.columns], rtol=1e-12, atol=1e-9)
                action, control = command(policy, index, before, view, summary["target_C"])
                difference = abs(action["oveTSet_u"]-record["action"]["oveTSet_u"])
                max_action_difference = max(max_action_difference, difference)
                if difference > 1e-7 or action["oveTSet_activate"] != record["action"]["oveTSet_activate"]:
                    raise ValueError("Frozen action replay differs")
                if control["mode"] != record["control"]["mode"] or control.get("reason") != record["control"].get("reason"):
                    raise ValueError("Frozen decision reason differs")
                check_action(record["action"], before, segment)
                measured = [name for name in segment if name in after]
                np.testing.assert_allclose(segment.iloc[-1][measured].to_numpy(),
                                           np.asarray([after[name] for name in measured]), rtol=1e-10, atol=1e-7)
                local_metrics = score(segment, local_schedule)
                for name, value in local_metrics.items():
                    if not np.isclose(value, record["metrics"][name], rtol=1e-10, atol=1e-9):
                        raise ValueError("Interval score differs")
                interval_energy += local_metrics["energy_kWh"]
                policy.update(before, after, segment, view.iloc[0])
                previous_after = after
        if abs(interval_energy-metrics["energy_kWh"]) > 1e-9:
            raise ValueError("Interval energies do not add up")
        summaries[folder.name] = summary
        checks.append({"run": folder.name, "transitions_replayed": expected, "native_points": len(native),
                       "summary_sha256": digest(folder/"summary.json")})
        print(f"verified {folder.name}: {expected} frozen decisions", flush=True)
    analysis = analyze(directory, summaries, plan)
    for key, value in analysis.items():
        if value != result[key]:
            raise ValueError(f"Recomputed stage gate differs: {key}")
    if set(summaries) != set(result["summaries"]):
        raise ValueError("Run list differs")
    return {"status": "passed", "calls": counts, "initialized_days": initialized_days,
        "holdout_initializations": 0, "training_runs": 0, "runs": checks,
        "frozen_action_max_absolute_difference_K": max_action_difference,
        "official_kpi_max_absolute_difference": max_kpi_difference,
        "stage_gates_recomputed": True, "paired_prefix_and_exogenous_equal": True,
        "source_result_sha256": digest(directory/"result.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, verify(args.directory))
