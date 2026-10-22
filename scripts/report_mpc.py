"""Archive the grey-box, intervention and guarded MPC evidence."""

import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_comfort.simulation import digest, save_json


if __name__ == "__main__":
    model_root, response_root, root = Path("outputs/greybox-mpc"), Path("outputs/action-response-final"), Path("outputs/mpc")
    report = Path("reports/mpc")
    report.mkdir(parents=True, exist_ok=True)
    checks = json.loads((root / "verification.json").read_text())
    if not checks["paired_action_response_gate_rebuilt"]:
        raise ValueError("Verify interventions and controllers before reporting")
    for source, prefix, files in [
        (model_root, "greybox_", ["metrics.csv", "selection.json", "manifest.json", "split_manifest.json", "margin_coverage.csv"]),
        (response_root, "response_", ["manifest.json", "summary.json", "response_metrics.csv", "response_gate.json"]),
        (root, "", ["manifest.json", "summary.json", "verification.json", "decision_audit.csv"])]:
        for name in files:
            shutil.copyfile(source / name, report / (prefix + name))
    summaries = json.loads((root / "summary.json").read_text())
    rows, samples = [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    scenarios = list(dict.fromkeys(s["scenario"] for s in summaries))
    for summary in summaries:
        scenario, controller = summary["scenario"], summary["controller"]
        frame = pd.read_csv(root / f"{scenario}__{controller}" / "transitions.csv.gz")
        row = scenarios.index(scenario)
        days = (frame.time_end_s - summary["start_time_s"]) / 86400
        axes[row, 0].plot(days, frame.operative_end_C, label=controller, linewidth=1)
        axes[row, 1].plot(days, frame.interval_energy_kWh.cumsum(), label=controller, linewidth=1.3)
        if controller == "guarded_rule":
            axes[row, 0].step(days, frame.lower_start_C, where="pre", label="comfort lower", color="gray", linestyle="--", linewidth=.6)
            axes[row, 0].step(days, frame.upper_start_C, where="pre", label="comfort upper", color="gray", linestyle=":", linewidth=.6)
        for ax in axes[row]:
            ax.set_title(scenario)
            ax.set_xlabel("Elapsed evaluation days")
            ax.grid(alpha=.2)
        axes[row, 0].set_ylabel("Operative temperature (°C)")
        axes[row, 1].set_ylabel("Cumulative electricity (kWh)")
        audit = next(c for c in checks["control_policy_replay"] if c["scenario"] == scenario and c["controller"] == controller)
        rows.append({"scenario": scenario, "controller": controller, **summary["metrics"],
                     "mpc_steps": audit["modes"].get("mpc",0), "total_steps": summary["transitions"],
                     "decision_p95_s": audit["decision_p95_s"]})
        sample = frame.iloc[:96].copy()
        sample.insert(0, "controller", controller)
        sample.insert(0, "scenario", scenario)
        samples.append(sample)
    axes[0, 0].legend(fontsize=7, ncols=2)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(report / "closed_loop.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(report / "comparison.csv", index=False, lineterminator="\n")
    pd.concat(samples, ignore_index=True).to_csv(report / "first_day_transitions.csv", index=False, lineterminator="\n")
    response = pd.read_csv(response_root / "response_metrics.csv")
    contrasts = response[response.actual_delta_temperature_C.notna()]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for column, day in enumerate(contrasts.start_day.unique()):
        part = contrasts[contrasts.start_day == day]
        for row, variable, label in [(0, "temperature_C", "High-minus-low temperature (°C)"),
                                     (1, "energy_kWh", "High-minus-low electricity (kWh)")]:
            ax = axes[row, column]
            ax.plot(part.horizon_steps / 4, part["actual_delta_" + variable], "o-", label="FMU intervention")
            ax.plot(part.horizon_steps / 4, part["predicted_delta_" + variable], "s--", label="grey-box prediction")
            ax.axvline(2, color="gray", linewidth=.6, label="MPC horizon" if row == 0 else None)
            ax.set_title(f"Calibration context / day {day:g}")
            ax.set_xlabel("Hours after intervention")
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(report / "action_response.png", dpi=150)
    plt.close(fig)
    files = [Path(p) for p in ["configs/mpc.json", "thermal_comfort/greybox.py", "thermal_comfort/mpc.py",
                              "thermal_comfort/simulation.py", "scripts/verify_mpc.py", "scripts/report_mpc.py"]]
    save_json(report / "provenance.json", {"python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
              "source_files_sha256": {str(p): digest(p) for p in files},
              "boptest_revision": "9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9",
              "fmu_sha256": digest(Path("outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"))})
    print(pd.DataFrame(rows).to_string(index=False))
