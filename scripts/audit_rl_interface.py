"""Read-only independent V2-1 artifact checks; no FMU or training calls."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_comfort.simulation import FORECAST, POWER, digest, save_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(directory):
    registration = read(directory/"registration.json")
    result = read(directory/"result.json")
    if result["counts"] != {"initialize": 3, "advance": 1368} or result["training_runs"] != 0:
        raise ValueError("Unexpected budget or training count")
    calls = [json.loads(line) for line in (directory/"calls.jsonl").read_text(encoding="utf-8").splitlines()]
    if not all(c["ok"] for c in calls) or sum(c["path"] == "stop" for c in calls) != 3:
        raise ValueError("API failure or worker release is unconfirmed")
    inits = [c["request"] for c in calls if c["path"] == "initialize"]
    if inits != [{"start_time": 14*86400, "warmup_period": 7*86400}]*3:
        raise ValueError("Unauthorized initialization")
    forecast_calls = [c["request"] for c in calls if c["path"] == "forecast"]
    if forecast_calls[0]["horizon"] != 7*86400+43200 or any(
            c != {"point_names": FORECAST, "horizon": 43200, "interval": 30}
            for c in forecast_calls[1:]):
        raise ValueError("Unexpected forecast permissions")
    reports = {}
    for name, expected in [("gym_rule", 672), ("gym_action_probe", 24)]:
        folder = directory/name
        summary = read(folder/"summary.json")
        for filename, expected_sha in summary["sha256"].items():
            if digest(folder/filename) != expected_sha:
                raise ValueError("Trajectory hash mismatch")
        with gzip.open(folder/"steps.jsonl.gz", "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        if len(records) != expected:
            raise ValueError("Incomplete trajectory")
        with gzip.open(folder/"bounded_forecasts.jsonl.gz", "rt", encoding="utf-8") as stream:
            count = 0
            for record, line in zip(records, stream, strict=True):
                frame = pd.DataFrame(json.loads(line))
                start = record["before"]["time"]
                if set(frame) != {"time", *FORECAST} or not np.array_equal(
                        frame.time, start+np.arange(1441)*30):
                    raise ValueError("Forecast exposes unauthorized fields/times")
                count += 1
        previous_end = 14*86400
        actions = []
        for i, record in enumerate(records):
            b, a, after = record["before"], record["action"], record["after"]
            value = a["oveTSet_u"]-273.15
            if (b["time"] != previous_end or after["time"] != previous_end+900
                    or not 18-1e-7 <= value <= 23.25+1e-7
                    or abs(a["oveTSet_u"]-b["oveTSet_u"]) > .5+1e-7
                    or a["oveTSet_activate"] != 1
                    or record["control"]["actor_transition"] != (i >= 8)
                    or record["control"]["common_prefix"] != (i < 8)):
                raise ValueError("Action, timing or prefix mismatch")
            actions.append(value)
            previous_end = after["time"]
        native = pd.read_csv(folder/"native.csv.gz")
        schedule = pd.read_csv(folder/"schedule.csv.gz")
        if not np.array_equal(native.time, 14*86400+np.arange(expected*30+1)*30):
            raise ValueError("Native gaps")
        np.testing.assert_array_equal(native.time, schedule.time)
        t = native.time.to_numpy()
        power = sum(native[p].to_numpy() for p in POWER)
        energy = float(np.trapezoid(power, t)/3600000)
        metrics = summary["final"]["score"]["metrics"]
        if abs(energy-metrics["energy_kWh"]) > 1e-10:
            raise ValueError("Independent energy mismatch")
        if abs(sum(r["metrics"]["energy_kWh"] for r in records)-energy) > 1e-10:
            raise ValueError("Prefix/actor energy does not add up")
        for kind, deviation in [("cold", np.maximum(schedule["LowerSetp[1]"]-native.reaTZon_y, 0)),
                                ("hot", np.maximum(native.reaTZon_y-schedule["UpperSetp[1]"], 0))]:
            if abs(float(np.trapezoid(deviation, t)/3600)-metrics[kind+"_Kh"]) > 1e-10:
                raise ValueError("Independent temperature score mismatch")
        reports[name] = {"transitions": expected, "native_points": len(native),
            "bounded_forecasts": count, "energy_kWh": energy,
            "prefix_energy_kWh": sum(r["metrics"]["energy_kWh"] for r in records[:8]),
            "actor_energy_kWh": sum(r["metrics"]["energy_kWh"] for r in records[8:]),
            "action_min_C": min(actions), "action_max_C": max(actions),
            "projected_actor_actions": sum(bool(r["control"].get("projected")) for r in records[8:])}
    old = pd.read_csv(directory/"legacy_rule/native.csv.gz")
    new = pd.read_csv(directory/"gym_rule/native.csv.gz")
    if set(old) != set(new):
        raise ValueError("Legacy/native fields differ")
    all_native_difference = float(np.max(np.abs(old[new.columns].to_numpy()-new.to_numpy())))
    if all_native_difference > 1e-6:
        raise ValueError("Legacy and Gym native trajectories differ")
    return {"status": "passed", "runs": reports, "all_native_fields_max_absolute_difference": all_native_difference,
        "parent_lock_sha256": registration["sha256"]["reports/v2_0/lock.json"],
        "initializations": inits, "successful_stops": 3, "bounded_forecast_api_calls": len(forecast_calls)-1,
        "raw_result_sha256": digest(directory/"result.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, audit(args.directory))
