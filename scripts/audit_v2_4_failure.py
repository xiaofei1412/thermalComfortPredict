"""Read-only closure audit: this trial failed before any physical select."""

import argparse
from pathlib import Path

from scripts.audit_v2_formal import lines
from scripts.audit_v2_safe_formal import audit
from thermal_comfort.rl_budget_safe import write_snapshot
from thermal_comfort.simulation import digest
from thermal_comfort.v2_protocol import read


def audit_failure(directory):
    base = audit(directory)
    assert base["ledger_counts"] == dict.fromkeys(["select","initialize","advance"],0)
    assert base["final_candidates"] == base["archived_transitions_checked"] == base["complete_episode_KPI_archives"] == 0
    assert read(directory/"budget.json")["stage_start"] is None
    assert digest(directory/"diagnostic.json") == read(directory/"authorization.json")["diagnostic_sha256"]
    attempts = {}
    for key, value in base["seeds"].items():
        if value["status"] == "not_started":
            continue
        root = directory/key
        failure = read(root/"failure.json")
        assert value["status"] == "failed" and failure["error"] == "'select_max'"
        assert failure["actor_steps"] == failure["cycle"] == 0 and not failure["completed_episode_folders"]
        assert not list(root.glob("checkpoints/*")) and not (root/"first-gate.json").exists()
        calls = lines(root/"calls.jsonl")
        assert len(calls) == 2 and calls[0]["path"] == "version" and calls[0]["ok"]
        assert calls[1]["path"] == "testcases/bestest_hydronic_heat_pump/select" and not calls[1]["ok"]
        assert calls[1]["counts"] == {"select":1,"initialize":0,"advance":0}
        attempts[key] = {"local_select_intents_rejected_before_transport_client":1,
            "physical_selects":0,"model_actor_steps":0,"error":failure["error"],
            "failure_sha256":digest(root/"failure.json"),"calls_sha256":digest(root/"calls.jsonl")}
    assert set(attempts) == {"SAC-11","SAC-23"}
    assert base["seeds"]["SAC-37"]["status"] == "not_started"
    return {"status":"closed_engineering_failure_before_physical_select",
        "archive_integrity":base["status"],"not_an_algorithm_negative_result":True,
        "base_audit":base,"local_attempts":attempts,"first_episode_gate_passed":False,
        "official_KPI_comparison_performed":False,"stage_clock_started":False,
        "wall_cost_source":"original per-seed elapsed clocks; zero physical work does not mean zero process cost",
        "additional_FMU_calls":0,"new_trial_or_restart":False,"heldout_initializations":0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    write_snapshot(args.output,audit_failure(args.directory))
