"""Archive dynamics metrics, collection evidence and scientific figures."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_comfort.simulation import digest, save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/dynamics-baseline"))
    parser.add_argument("--source", type=Path, default=Path("outputs/excitation"))
    parser.add_argument("--report", type=Path, default=Path("reports/dynamics"))
    args = parser.parse_args()
    checks = json.loads((args.output / "verification.json").read_text())
    if not checks["reloaded_model_scores_match"]:
        raise ValueError("Verify the model artifacts before archiving")
    args.report.mkdir(parents=True, exist_ok=True)
    for name in ["metrics.csv", "coverage.csv", "manifest.json", "selection.json", "split_manifest.json",
                 "evaluation_origins.json", "rollout_examples.csv", "verification.json"]:
        shutil.copyfile(args.output / name, args.report / name)
    for name in ["manifest.json", "summary.json", "safety.json", "verification.json"]:
        shutil.copyfile(args.source / name, args.report / ("collection_" + name))
    metrics = pd.read_csv(args.output / "metrics.csv")
    test_episodes = metrics.loc[metrics.split == "test", "episode"].unique()
    fig, axes = plt.subplots(2, len(test_episodes), figsize=(12, 6), squeeze=False, layout="constrained")
    colors = {"persistence": "gray", "ridge": "tab:blue", "extra_trees": "tab:orange"}
    for column, episode in enumerate(test_episodes):
        for model, color in colors.items():
            for weather, style in [("perfect", "-"), ("hold", "--")]:
                if model == "persistence" and weather == "hold":
                    continue
                part = metrics[(metrics.episode == episode) & (metrics.model == model) & (metrics.weather == weather)]
                for row, variable in [(0, "temperature_RMSE_C"), (1, "cumulative_energy_RMSE_kWh")]:
                    axes[row, column].plot(part.horizon_hours, part[variable], style, color=color,
                                            marker="o", markersize=3, label=f"{model} / {weather}")
        for row in [0, 1]:
            ax = axes[row, column]
            ax.set_title(episode)
            ax.set_xlabel("Forecast horizon (hours)")
            ax.set_xscale("log", base=2)
            ax.set_xticks([.25, 1, 4, 8, 24], ["0.25", "1", "4", "8", "24"])
            ax.grid(alpha=.2)
        axes[0, column].set_ylabel("Temperature RMSE (°C)")
        axes[1, column].set_ylabel("Cumulative energy RMSE (kWh)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(args.report / "forecast_errors.png", dpi=150)
    plt.close(fig)

    summaries = json.loads((args.source / "summary.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    samples, collection_rows = [], []
    for ax, summary in zip(axes.flat, summaries):
        plan = summary["config"]["episode"]
        frame = pd.read_csv(args.source / (plan["id"] + "__excitation") / "transitions.csv.gz")
        days = (frame.time_end_s - summary["start_time_s"]) / 86400
        ax.plot(days, frame.operative_end_C, label="operative temperature", linewidth=1)
        ax.plot(days, frame.setpoint_requested_C, label="applied setpoint", linewidth=.7, alpha=.7)
        ax.axhline(18, color="gray", linestyle="--", linewidth=.6, label="stop band")
        ax.axhline(28, color="gray", linestyle="--", linewidth=.6)
        ax.set_title(f"{plan['split']} / day {plan['start_day']} / seed {plan['seed']}", fontsize=10)
        ax.set_xlabel("Elapsed evaluation days")
        ax.set_ylabel("Temperature (°C)")
        ax.set_ylim(17, 29)
        ax.grid(alpha=.2)
        sample = frame.iloc[:96].copy()
        sample.insert(0, "episode", plan["id"])
        samples.append(sample)
        collection_rows.append({**plan, "transitions": summary["transitions"],
                                "native_samples": summary["native_samples"], **summary["metrics"]})
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.savefig(args.report / "excitation.png", dpi=150)
    plt.close(fig)
    pd.concat(samples, ignore_index=True).to_csv(args.report / "excitation_samples.csv", index=False, lineterminator="\n")
    pd.DataFrame(collection_rows).to_csv(args.report / "collection_metrics.csv", index=False, lineterminator="\n")
    files = [Path("configs/dynamics.json"), Path("thermal_comfort/dynamics.py"), Path("thermal_comfort/excitation.py"),
             Path("thermal_comfort/simulation.py"), Path("scripts/verify_dynamics.py"), Path(__file__)]
    save_json(args.report / "provenance.json", {
        "boptest_revision": "9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9",
        "fmu_sha256": digest(Path("outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu")),
        "source_files_sha256": {str(p): digest(p) for p in files}})
    print(metrics[(metrics.split == "test") & (metrics.weather == "perfect")].to_string(index=False))
