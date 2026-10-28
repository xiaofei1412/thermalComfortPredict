"""Offline V2-4 diagnostic, restricted to the frozen development exports."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_v2_formal import lines
from thermal_comfort.rl_budget_safe import write_snapshot
from thermal_comfort.simulation import digest
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def diagnose(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(ROOT/"configs/v2_4.json")
    lock = read(ROOT/plan["parent_lock"])
    verify_files(ROOT, lock["files_sha256"])
    trial = ROOT/"outputs/v2-safe-formal/live-registered"
    if read(trial/"registration.json")["formal_training_authorized"]:
        raise ValueError("Parent study must remain closed")
    rows = []
    for algorithm in ["SAC", "PPO", "TD3"]:
        for seed in plan["seeds"]:
            for name in plan["development"]:
                folder = trial/f"{algorithm}-{seed}"/"development"/name
                summary = read(folder/"summary.json")
                verify_files(folder, summary["sha256"])
                records = lines(folder/"steps.jsonl.gz")[8:]
                schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
                reference = ROOT/"outputs/v2-2/live-registered"/(name+"__guarded_rule")/"native.csv.gz"
                native = pd.read_csv(reference, usecols=["time", "oveTSet_u"], float_precision="round_trip")
                commands = np.array([r["action"]["oveTSet_u"]-273.15 for r in records])
                reference_commands = native.oveTSet_u.to_numpy()[np.arange(8,672)*30+1]-273.15
                gap = commands-reference_commands
                cold = np.array([r["metrics"]["cold_Kh"] > 0 for r in records])
                hot = np.array([r["metrics"]["hot_Kh"] > 0 for r in records])
                before = np.array([r["before"]["reaTZon_y"]-273.15 for r in records])
                lower = schedule["LowerSetp[1]"].to_numpy()[np.arange(8,672)*30]-273.15
                shadow_recovery = np.array([r["control"].get("shadow_reason") in ["recover_heat","anticipatory_recovery"] for r in records])
                metrics = summary["outcome"]["score"]["metrics"]
                rows.append({"algorithm":algorithm,"seed":seed,"episode":name,
                    "energy_kWh":metrics["energy_kWh"],"cold_Kh":metrics["cold_Kh"],"hot_Kh":metrics["hot_Kh"],
                    "fallback_fraction":summary["fallback_fraction_excluding_prefix"],
                    "command_mean_C":float(commands.mean()),"at_min_fraction":float(np.mean(commands <= 18+1e-7)),
                    "at_max_fraction":float(np.mean(commands >= 23.25-1e-7)),
                    "same_clock_rule_command_gap_mean_K":float(gap.mean()),
                    "cold_actor_intervals":int(cold.sum()),"hot_actor_intervals":int(hot.sum()),
                    "cold_intervals_below_rule_fraction":float(np.mean(gap[cold] < -1e-7)) if cold.any() else None,
                    "hot_intervals_above_rule_fraction":float(np.mean(gap[hot] > 1e-7)) if hot.any() else None,
                    "below_current_lower_fraction":float(np.mean(before < lower)),
                    "shadow_recovery_fraction":float(shadow_recovery.mean()),
                    "actor_intervals":len(records),"summary_sha256":digest(folder/"summary.json"),
                    "reference_sha256":digest(reference)})
    supported = sum(r["algorithm"] == "SAC" and r["episode"] == "development084" and r["cold_Kh"] > 0
        and r["fallback_fraction"] == 0 for r in rows) >= 2
    result = {"status":"diagnosis_completed", "protocol_sha256":digest(ROOT/"configs/v2_4.json"),
        "parent_lock_sha256":digest(ROOT/plan["parent_lock"]), "hypothesis_gate_passed":supported,
        "rows":rows, "FMU_calls":0,"training_steps":0,"heldout_payload_reads":0,
        "limitation":"aligned rule controls belong to another state trajectory; gaps do not prove a causal intervention effect"}
    write_snapshot(output/"diagnostic.json",result)
    print(f"27 development exports checked; hypothesis gate={supported}; no training or FMU calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    diagnose(parser.parse_args().output)
