"""Verify saved BOPTEST trajectories and compare a fresh short repeat to the full run."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_comfort.simulation import digest, integrate, save_json, transitions_frame, validate_kpis


def verify(root):
    summaries = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    result = []
    for summary in summaries:
        directory = root / f"{summary['scenario']}__{summary['controller']}"
        for name, expected in summary["sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Changed artifact: {directory / name}")
        native = pd.read_csv(directory / "native.csv.gz")
        forecast = pd.read_csv(directory / "exogenous.csv.gz")
        records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        config = summary["config"]
        expected_steps = int(config["days"] * 86400 / config["step_s"])
        if len(records) != expected_steps or len(records) != summary["transitions"]:
            raise ValueError("Unexpected trajectory length")
        if native.time.iloc[0] != summary["start_time_s"] or native.time.iloc[-1] != summary["end_time_s"]:
            raise ValueError("Incorrect native trajectory limits")
        if np.diff(native.time).max() > 30.000001:
            raise ValueError("Missing native samples")
        metrics = integrate(native, forecast, config["area_m2"])
        differences = validate_kpis(metrics, summary["kpis"])
        rebuilt = transitions_frame(records, native, forecast, config)
        saved = pd.read_csv(directory / "transitions.csv.gz")
        pd.testing.assert_frame_equal(rebuilt.astype(float), saved.astype(float), rtol=1e-12, atol=1e-12)
        if not np.isclose(saved.interval_energy_kWh.sum(), metrics["energy_kWh"], rtol=1e-12):
            raise ValueError("Interval energy sum mismatch")
        result.append({"scenario": summary["scenario"], "controller": summary["controller"],
                       "transitions": len(saved), "native_samples": len(native),
                       "max_native_gap_s": float(np.diff(native.time).max()),
                       "kpi_absolute_difference": differences, "artifact_hashes_match": True})
    return result


def compare_repeat(root, repeat):
    results = []
    for directory in sorted(p for p in repeat.iterdir() if p.is_dir()):
        short = pd.read_csv(directory / "native.csv.gz")
        full = pd.read_csv(root / directory.name / "native.csv.gz")
        full = full.loc[full.time <= short.time.iloc[-1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(full, short, check_exact=True, check_dtype=False)
        short_steps = pd.read_csv(directory / "transitions.csv.gz")
        full_steps = pd.read_csv(root / directory.name / "transitions.csv.gz").iloc[:len(short_steps)]
        pd.testing.assert_frame_equal(full_steps, short_steps, check_exact=True, check_dtype=False)
        results.append({"episode": directory.name, "native_samples": len(short),
                        "transitions": len(short_steps), "exact_prefix_match": True})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    checks = {"episodes": verify(args.output)}
    if args.repeat:
        checks["repeat_episodes"] = verify(args.repeat)
        checks["reproducibility"] = compare_repeat(args.output, args.repeat)
    save_json(args.output / "verification.json", checks)
    print(json.dumps(checks, indent=2))
