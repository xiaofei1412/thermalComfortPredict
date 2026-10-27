"""Read-only audit of the stopped formal campaign; incomplete energy is not compared."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_comfort.rl_budget import BudgetLedger, atomic_json, locked
from thermal_comfort.rl_interface import native_score
from thermal_comfort.simulation import FORECAST, digest, validate_kpis
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def lines(path):
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(s) for s in stream]


def audit(directory):
    auth, closed = read(directory/"authorization.json"), read(directory/"registration.json")
    registered = read(directory/"registration-authorized.json")
    closure = read(directory/"closure.json")
    assert digest(directory/"registration-authorized.json") == closure["authorized_registration_sha256"]
    assert digest(directory/"registration.json") == closure["closed_registration_sha256"]
    assert registered["formal_training_authorized"] and not closed["formal_training_authorized"]
    assert digest(directory/"authorization.json") == registered["authorization_sha256"]
    verify_files(ROOT, registered["files_sha256"])
    verify_files(ROOT, auth["parent_locks_sha256"])
    verify_files(ROOT, auth["frozen_assets_sha256"])
    plan = registered["protocol"]
    ledger = BudgetLedger(directory, plan)
    with locked(ledger.lock):
        state = ledger.read()
    assert all(v["status"] == "failed" for v in state["seeds"].values())
    valid_train = set(plan["train_ids"])
    result, total_calls = {}, dict.fromkeys(["select", "initialize", "advance", "stop"], 0)
    native_points, forecasts, archived_transitions, official = 0, 0, 0, 0
    max_kpi, first_pass = 0., []
    for algorithm in plan["algorithms"]:
        for seed in plan["seeds"]:
            key, root = f"{algorithm}-{seed}", directory/f"{algorithm}-{seed}"
            if key not in state["seeds"]:
                assert not root.exists()
                result[key] = {"status": "not_started_shared_infrastructure_stop", "counts": dict.fromkeys(state["counts"], 0)}
                continue
            calls = lines(root/"calls.jsonl")
            counts = dict.fromkeys(total_calls, 0)
            owned = set()
            for call in calls:
                kind, _, testid = call["path"].partition("/")
                assert call["episode"]["id"] in valid_train and call["episode"]["role"] == "train"
                if kind == "testcases":
                    assert call["path"] == "testcases/bestest_hydronic_heat_pump/select"
                if kind == "initialize":
                    assert call["request"] == {"start_time": call["episode"]["start_day"]*86400, "warmup_period": 604800}
                counter = "select" if kind == "testcases" else kind
                if call["ok"] and counter in counts:
                    counts[counter] += 1
                if kind == "inputs":
                    owned.add(testid)
                if kind == "stop" and call["ok"]:
                    owned.discard(testid)
            assert not owned, (key, owned)
            assert {k: counts[k] for k in state["counts"]} == state["seeds"][key]["counts"], key
            for kind, count in counts.items():
                total_calls[kind] += count
            journals = lines(root/"actions.jsonl")
            for index, action in enumerate(journals):
                assert action["actor_step"] == index+1 and action["episode"] in valid_train
                assert np.isfinite(action["algorithm_raw"]).all() and abs(action["library_action"][0]) <= 1
                control = action["control"]
                assert 18-1e-7 <= control["actual_command_C"] <= 23.25+1e-7
                assert abs(control["measured_setpoint_C"]-control["actual_command_C"]) < 1e-7
            folders, stored_steps = [], 0
            for summary_path in sorted(root.rglob("summary.json")):
                summary, folder = read(summary_path), summary_path.parent
                verify_files(folder, summary["sha256"])
                records = lines(folder/"steps.jsonl.gz")
                native = pd.read_csv(folder/"native.csv.gz", float_precision="round_trip")
                schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
                assert summary["episode"]["role"] == "train" and summary["episode"]["id"] in valid_train
                assert len(records) == summary["transitions"] and len(native) == len(records)*30+1
                assert np.all(np.diff(native.time) == 30)
                start = summary["episode"]["start_day"]*86400
                assert native.time.iloc[0] == start and native.time.iloc[-1] == start+900*len(records)
                views = lines(folder/"forecasts.jsonl.gz")
                assert len(views) == len(records)
                for index, (record, view) in enumerate(zip(records, views, strict=True)):
                    assert record["index"] == index and record["after"]["time"]-record["before"]["time"] == 900
                    command = record["action"]["oveTSet_u"]-273.15
                    assert 18-1e-7 <= command <= 23.25+1e-7 and abs(command-(record["before"]["oveTSet_u"]-273.15)) <= .5+1e-7
                    assert np.allclose(native.oveTSet_u.iloc[index*30+1:(index+1)*30+1], command+273.15, rtol=0, atol=1e-7)
                    assert set(view) == {"time", *FORECAST}
                    np.testing.assert_array_equal(view["time"], start+index*900+np.arange(1441)*30)
                metrics = native_score(native, schedule)
                complete = bool(summary.get("outcome", {}).get("episode_complete", False))
                if complete:
                    score = summary["outcome"]["score"]
                    assert summary["transitions"] == 672 and summary["actor_transitions"] == 664
                    assert max(abs(v-score["metrics"][k]) for k, v in metrics.items()) < 1e-7
                    differences = validate_kpis(metrics, score["kpis"])
                    max_kpi = max(max_kpi, *differences.values())
                    official += 1
                with np.load(folder/"actor_observations.npz") as obs:
                    assert obs["observations"].shape == (summary["actor_transitions"], 503)
                    assert obs["terminal_observation"].shape == (503,)
                    assert all(np.isfinite(obs[k]).all() for k in obs)
                stored_steps += len(records)
                native_points += len(native)
                forecasts += len(views)
                folders.append({"folder": str(folder.relative_to(root)), "episode": summary["episode"]["id"],
                    "stored_transitions": len(records), "actor_observations": summary["actor_transitions"],
                    "complete_official_KPI_retained": complete, "summary_sha256": digest(summary_path)})
            archived_transitions += stored_steps
            if (root/"first-gate.json").exists():
                gate = read(root/"first-gate.json")
                assert gate["status"] == "passed" and gate["extra_FMU_calls"] == 0
                first_pass.append(key)
            failure = read(root/"failure.json")
            assert not (root/"result.json").exists() and not list(root.glob("checkpoints/cycle-??"))
            result[key] = {"status": "failed_budget_io" if algorithm == "SAC" else "cooperatively_aborted_after_dispatcher_failure",
                "error": failure["error"], "counts": counts, "first_gate_passed": key in first_pass,
                "actor_steps_at_failure": failure["actor_steps"], "retained_action_rows": len(journals),
                "elapsed_seconds": state["seeds"][key]["elapsed_seconds"], "exports": folders,
                "successful_advances_without_retained_native_segment": counts["advance"]-stored_steps,
                "partial_export_error": failure.get("partial_export_error"),
                "failure_sha256": digest(root/"failure.json"), "calls_sha256": digest(root/"calls.jsonl")}
    assert {k: total_calls[k] for k in state["counts"]} == state["counts"]
    return {"status": "stopped_formal_trial_audit_passed_incomplete_evidence_disclosed", "seeds": result,
        "successful_service_calls": total_calls, "ledger_counts": state["counts"], "first_gates_passed": first_pass,
        "complete_training_episode_KPI_archives": official, "max_official_KPI_difference": max_kpi,
        "archived_transitions_checked": archived_transitions, "native_points_checked": native_points,
        "forecast_snapshots_checked": forecasts, "final_candidates": 0, "formal_development_episodes": 0,
        "heldout_initializations": 0, "additional_FMU_calls": 0, "performance_comparison_permitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, audit(args.directory))
