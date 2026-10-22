"""Archive compact, reviewable evidence from a verified local BOPTEST run."""

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
    parser.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    parser.add_argument("--report", type=Path, default=Path("reports/simulation"))
    args = parser.parse_args()
    summaries = json.loads((args.output / "summary.json").read_text())
    args.report.mkdir(parents=True, exist_ok=True)
    for name in ["manifest.json", "summary.json", "verification.json"]:
        shutil.copyfile(args.output / name, args.report / name)
    rows, samples = [], []
    scenarios = list(dict.fromkeys(s["scenario"] for s in summaries))
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(12, 6), squeeze=False, layout="constrained")
    for summary in summaries:
        scenario, controller = summary["scenario"], summary["controller"]
        directory = args.output / f"{scenario}__{controller}"
        data = pd.read_csv(directory / "transitions.csv.gz")
        rows.append({"scenario": scenario, "controller": controller,
                     "transitions": summary["transitions"], **summary["metrics"]})
        sample = data.iloc[:96].copy()
        sample.insert(0, "controller", controller)
        sample.insert(0, "scenario", scenario)
        samples.append(sample)
        ax_temp, ax_energy = axes[scenarios.index(scenario)]
        days = (data.time_end_s - summary["start_time_s"]) / 86400
        ax_temp.plot(days, data.operative_end_C, label=controller, linewidth=1)
        ax_energy.plot(days, data.interval_energy_kWh.cumsum(), label=controller, linewidth=1.5)
        if controller == "rule":
            ax_temp.step(days, data.lower_start_C, where="pre", label="comfort lower", color="gray", linestyle="--", linewidth=.7)
            ax_temp.step(days, data.upper_start_C, where="pre", label="comfort upper", color="gray", linestyle=":", linewidth=.7)
        for ax in [ax_temp, ax_energy]:
            ax.set_xlabel("Elapsed evaluation days")
            ax.set_title(scenario)
            ax.grid(alpha=.2)
        ax_temp.set_ylabel("Operative temperature (°C)")
        ax_energy.set_ylabel("Cumulative electricity (kWh)")
    axes[0, 0].legend(fontsize=8, ncols=2)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(args.report / "trajectories.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.report / "metrics.csv", index=False, lineterminator="\n")
    pd.concat(samples, ignore_index=True).to_csv(args.report / "first_day_transitions.csv", index=False, lineterminator="\n")
    source = Path("outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu")
    save_json(args.report / "provenance.json", {
        "boptest_revision": "9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9",
        "fmu_sha256": digest(source),
        "local_full_artifacts": str(args.output),
        "tracked_files_sha256": {str(p): digest(p) for p in [
            Path("infra/boptest.compose.yml"), Path("infra/storage.Dockerfile"),
            Path("scripts/setup_boptest.ps1"), Path("thermal_comfort/simulation.py"),
            Path("configs/simulation.json"), Path("scripts/verify_simulation.py"), Path(__file__)]}})
    print(pd.DataFrame(rows).to_string(index=False))
