"""Read-only audit of the fixed development189 numerical diagnostic."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_comfort.rl_interface import bounded_forecast
from thermal_comfort.simulation import FORECAST, digest, save_json, validate_kpis
from thermal_comfort.v2_baselines import assert_same_values, check_action, score
from thermal_comfort.v2_numerics import (
    ARCHIVE, analyze, check_zero, load_plan, raw_frame, source_metadata)
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def verify(directory):
    plan, actions = load_plan()
    registration, result = read(directory/"registration.json"), read(directory/"result.json")
    verify_files(ROOT, registration["files_sha256"])
    if registration["plan"] != plan or registration["source_metadata"] != source_metadata():
        raise ValueError("Registered scope or source metadata changed")
    for kind, expected in registration["archived_action_sha256"].items():
        if digest(ARCHIVE/f"development189__{kind}/steps.jsonl") != expected:
            raise ValueError("Original archived action source changed")
    calls = [json.loads(line) for line in (directory/"calls.jsonl").read_text(encoding="utf-8").splitlines()]
    if not all(c["ok"] for c in calls):
        raise ValueError("Failed API calls in completed diagnostic")
    counts = {k: sum(c["path"].startswith(k+"/") for c in calls) for k in ["initialize", "advance", "stop", "forecast", "results"]}
    counts["select"] = sum(c["path"] == "testcases/bestest_hydronic_heat_pump/select" for c in calls)
    if counts != {"initialize": 4, "advance": 2688, "stop": 4, "forecast": 2688, "results": 2688, "select": 4}:
        raise ValueError("Diagnostic budget counts differ")
    if result["counts"] != {"initialize": 4, "advance": 2688, "select": 4}:
        raise ValueError("Reported live budget differs")
    for call in calls:
        if call["path"].startswith("initialize/") and call["request"] != {
                "start_time": 189*86400, "warmup_period": 604800}:
            raise ValueError("Unexpected initialized date")
        if call["path"].startswith("forecast/") and call["request"] != {
                "point_names": FORECAST, "horizon": 43200, "interval": 30}:
            raise ValueError("Forecast permissions exceeded")
    reports, max_kpi = [], 0.
    for item in plan["runs"]:
        folder = directory/item["id"]
        summary = read(folder/"summary.json")
        verify_files(folder, summary["sha256"])
        records = [json.loads(line) for line in (folder/"steps.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(records) != 672:
            raise ValueError("Incomplete diagnostic episode")
        native = raw_frame(folder)
        schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
        if not np.array_equal(native.time, 189*86400+np.arange(20161)*30):
            raise ValueError("Native time gaps")
        np.testing.assert_array_equal(native.time, schedule.time)
        check_zero(native, plan)
        metrics = score(native, schedule)
        for key, value in metrics.items():
            if not np.isclose(value, summary["metrics"][key], rtol=1e-12, atol=1e-10):
                raise ValueError("Recomputed score differs")
        max_kpi = max(max_kpi, *validate_kpis(metrics, summary["kpis"]).values())
        previous = summary["initial"]
        with gzip.open(folder/"forecasts.jsonl.gz", "rt", encoding="utf-8") as stream:
            for index, (record, line) in enumerate(zip(records, stream, strict=True)):
                if record["index"] != index or record["before"] != previous:
                    raise ValueError("Transition sequence differs")
                if record["action"] != actions[item["archived_kind"]][index]:
                    raise ValueError("Archived action was not replayed")
                view = bounded_forecast(json.loads(line), record["before"]["time"])
                segment = native.iloc[index*30:index*30+31].reset_index(drop=True)
                check_action(record["action"], record["before"], segment)
                assert_same_values(view.iloc[:31].reset_index(drop=True), schedule.iloc[index*30:index*30+31].reset_index(drop=True))
                if record["after"]["time"] != record["before"]["time"]+900:
                    raise ValueError("Transition time differs")
                measured = list(segment)
                np.testing.assert_allclose(segment.iloc[-1][measured],
                    [record["after"][p] for p in measured], rtol=0, atol=1e-10)
                previous = record["after"]
        reports.append({"run": item["id"], "transitions_checked": 672, "native_points": len(native),
                        "summary_sha256": digest(folder/"summary.json")})
    analysis = analyze(directory, plan)
    for key, value in analysis.items():
        if value != result[key]:
            raise ValueError(f"Recomputed diagnostic conclusion differs: {key}")
    old = read(ROOT/"reports/v2_2/summary.json")
    if old["comparisons_vs_rule"]["development189"]["solar_mpc"]["joint_gate_passed"]:
        raise ValueError("Original strict failure was overwritten")
    return {"status": "passed", "counts": counts, "runs": reports,
        "official_kpi_max_absolute_difference": max_kpi, "diagnostic_recomputed": True,
        "old_failure_preserved": True, "training_runs": 0, "holdout_initializations": 0,
        "new_tolerance_authorized": False, "result_sha256": digest(directory/"result.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, verify(args.directory))
