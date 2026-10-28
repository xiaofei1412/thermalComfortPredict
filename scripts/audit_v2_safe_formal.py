"""Read-only audit of a closed repaired formal trial, without simulator calls."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_v2_formal import lines
from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
from thermal_comfort.rl_interface import native_score
from thermal_comfort.simulation import FORECAST, digest, validate_kpis
from thermal_comfort.v2_baselines import comparison
from thermal_comfort.v2_protocol import ROOT, read, verify_files


def pairs(result):
    baselines = read(ROOT/"reports/v2_2/summary.json")["runs"]
    gates = read(ROOT/"configs/v2_0.json")["comparison_gates"]
    output = {}
    for name, trial in result["development"].items():
        metrics = trial["outcome"]["score"]["metrics"]
        active = baselines[name+"__guarded_rule"]["metrics"]["energy_kWh"] >= gates["active_reference_energy_kWh_min"]
        pairings = {}
        for kind in ["guarded_rule", "solar_mpc", "minimum_heat"]:
            base = baselines[name+"__"+kind]
            pair = comparison(base["metrics"], metrics, gates)
            energy = base["metrics"]["energy_kWh"]
            pair["energy_gate_passed"] = (pair["saving_kWh"] >= gates["saving_absolute_kWh_min"]
                and energy > 0 and pair["saving_kWh"]/energy >= gates["saving_relative_min"] if active
                else -pair["saving_kWh"] <= gates["low_load_energy_increase_kWh_tolerance"])
            pair.update(active=active, saving_fraction=pair["saving_kWh"]/energy if active and energy > 0 else None,
                joint_gate_passed=bool(pair["energy_gate_passed"] and pair["thermal_nonincrease"]),
                terminal_difference={k: trial["end_state"][k]-base["end_state"][k]
                    for k in ["operative_C", "return_C", "supply_C", "heat_kW"]})
            pairings[kind] = pair
        output[name] = {"metrics": metrics, "comparisons": pairings,
            "fallback_fraction": trial["fallback_fraction_excluding_prefix"],
            "fallback_gate_passed": trial["fallback_fraction_excluding_prefix"] <= gates["rl_fallback_fraction_max_excluding_common_prefix"]}
    return output


def audit(directory):
    auth, closed = read(directory/"authorization.json"), read(directory/"registration.json")
    registered, closure = read(directory/"registration-authorized.json"), read(directory/"closure.json")
    assert digest(directory/"registration-authorized.json") == closure["authorized_registration_sha256"]
    assert digest(directory/"registration.json") == closure["closed_registration_sha256"]
    assert registered["formal_training_authorized"] and not closed["formal_training_authorized"]
    assert digest(directory/"authorization.json") == registered["authorization_sha256"]
    assert digest(directory/"registration-before-authorization.json") == auth["initial_registration_sha256"]
    assert digest(directory/"preflight.json") == auth["preflight_sha256"]
    for mapping in [registered["files_sha256"], auth["parent_locks_sha256"], auth["frozen_assets_sha256"]]:
        verify_files(ROOT, mapping)
    for name in auth["parent_locks_sha256"]:
        verify_files(ROOT, read(ROOT/name)["files_sha256"])
    plan = registered["protocol"]
    state = BudgetLedger(directory, plan).read()
    assert all(v["status"] != "running" for v in state["seeds"].values())
    specs = {e["id"]: e for e in read(ROOT/"configs/v2_0.json")["episodes"] if e["role"] in ["train", "development"]}
    results, totals = {}, dict.fromkeys(["select", "initialize", "advance", "stop"], 0)
    points = transitions = forecasts = official = 0
    max_kpi = 0.
    for algorithm in plan["algorithms"]:
        for number in plan["seeds"]:
            key = f"{algorithm}-{number}"
            root = directory/key
            if key not in state["seeds"]:
                assert not root.exists()
                results[key] = {"status": "not_started"}
                continue
            entry = state["seeds"][key]
            calls, owned = lines(root/"calls.jsonl"), set()
            successful = dict.fromkeys(totals, 0)
            for call in calls:
                episode = call["episode"]
                assert episode == specs[episode["id"]]
                allowed = plan["train_ids"] if episode["role"] == "train" else plan["development_ids"]
                assert episode["id"] in allowed
                kind, _, testid = call["path"].partition("/")
                if kind == "initialize":
                    assert call["request"] == {"start_time": episode["start_day"]*86400, "warmup_period": 604800}
                counter = "select" if kind == "testcases" else kind
                if call["ok"] and counter in successful:
                    successful[counter] += 1
                if kind == "inputs":
                    owned.add(testid)
                if kind == "stop" and call["ok"]:
                    owned.discard(testid)
            assert not owned, (key, owned)
            assert all(successful[k] <= entry["counts"][k] for k in entry["counts"])
            for kind in totals:
                totals[kind] += successful[kind]
            for index, action in enumerate(lines(root/"actions.jsonl")):
                assert action["actor_step"] == index+1 and action["episode"] in plan["train_ids"]
                assert np.isfinite(action["algorithm_raw"]).all() and abs(action["library_action"][0]) <= 1
                assert abs(action["control"]["actual_command_C"]-action["control"]["measured_setpoint_C"]) < 1e-7
            stored, episodes = 0, []
            for path in sorted(root.rglob("summary.json")):
                summary, folder = read(path), path.parent
                verify_files(folder, summary["sha256"])
                assert summary["episode"] == specs[summary["episode"]["id"]]
                records, views = lines(folder/"steps.jsonl.gz"), lines(folder/"forecasts.jsonl.gz")
                assert len(records) == summary["transitions"] == len(views)
                start = summary["episode"]["start_day"]*86400
                if records:
                    native = pd.read_csv(folder/"native.csv.gz", float_precision="round_trip")
                    schedule = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
                    assert len(native) == len(records)*30+1 and np.all(np.diff(native.time) == 30)
                    assert native.time.iloc[0] == start and native.time.iloc[-1] == start+len(records)*900
                    for index, (record, view) in enumerate(zip(records, views, strict=True)):
                        assert record["index"] == index and record["after"]["time"]-record["before"]["time"] == 900
                        command = record["action"]["oveTSet_u"]-273.15
                        assert 18-1e-7 <= command <= 23.25+1e-7
                        assert abs(command-(record["before"]["oveTSet_u"]-273.15)) <= .5+1e-7
                        np.testing.assert_allclose(native.oveTSet_u.iloc[index*30+1:(index+1)*30+1], command+273.15, rtol=0, atol=1e-7)
                        assert set(view) == {"time", *FORECAST}
                        np.testing.assert_array_equal(view["time"], start+index*900+np.arange(1441)*30)
                    points += len(native)
                    if summary.get("outcome", {}).get("episode_complete"):
                        score, metrics = summary["outcome"]["score"], native_score(native, schedule)
                        assert len(records) == 672 and summary["actor_transitions"] == 664
                        assert max(abs(v-score["metrics"][k]) for k, v in metrics.items()) < 1e-7
                        max_kpi = max(max_kpi, *validate_kpis(metrics, score["kpis"]).values())
                        official += 1
                with np.load(folder/"actor_observations.npz") as values:
                    size = summary.get("actor_transitions", summary.get("actor_observations"))
                    assert values["observations"].shape == (size, 503)
                    assert all(np.isfinite(values[k]).all() for k in values)
                if summary.get("status") == "incomplete_failure_fragment":
                    assert summary["eligible_for_comparison_or_resume"] is False
                stored += len(records)
                episodes.append({"folder": str(folder.relative_to(root)), "sha256": digest(path), "transitions": len(records)})
            transitions += stored
            forecasts += stored
            details = {"status": entry["status"], "counts": entry["counts"], "successful_calls": successful,
                "elapsed_seconds": entry["elapsed_seconds"], "exports": episodes,
                "budgeted_advances_minus_archived_segments": entry["counts"]["advance"]-stored,
                "first_gate": read(root/"first-gate.json") if (root/"first-gate.json").exists() else None}
            if entry["status"] == "completed":
                result = read(root/"result.json")
                assert entry["counts"] == {k: plan["budget_per_algorithm_seed"][k+"_max"] for k in entry["counts"]}
                final = root/"checkpoints/cycle-12"
                manifest = read(final/"manifest.json")
                assert digest(final/"manifest.json") == read(root/"checkpoint-manifests/cycle-12.json")["manifest_sha256"]
                verify_files(final, manifest["files_sha256"])
                assert manifest["learning"]["actor_steps"] == 47808 and set(result["development"]) == set(plan["development_ids"])
                details["development"] = pairs(result)
                details["result_sha256"] = digest(root/"result.json")
            else:
                assert not (root/"result.json").exists()
                details["failure"] = read(root/"failure.json") if (root/"failure.json").exists() else read(root/"dispatcher-failure.json")
            results[key] = details
    return {"status": "closed_repaired_trial_audit_passed", "seeds": results, "ledger_counts": state["counts"],
        "successful_service_calls": totals, "archived_transitions_checked": transitions, "native_points_checked": points,
        "forecast_snapshots_checked": forecasts, "complete_episode_KPI_archives": official, "max_official_KPI_difference": max_kpi,
        "final_candidates": sum(v["status"] == "completed" for v in results.values()),
        "heldout_initializations": 0, "additional_FMU_calls": 0, "best_seed_selected": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_snapshot(args.output, audit(args.directory))
