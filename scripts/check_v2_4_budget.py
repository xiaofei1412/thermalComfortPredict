"""Reproduce the closed V2-4 pre-reservation fault and test an offline correction."""

import argparse
from pathlib import Path
import tempfile

from scripts.run_v2_residual import plan
from thermal_comfort.rl_budget_contract import corrected_residual_plan, validate_budget
from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot


def check():
    failed = plan()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        ledger = BudgetLedger(folder,failed)
        ledger.create()
        key = ledger.start_seed("SAC",11)
        try:
            ledger.reserve(key,"select")
        except KeyError as error:
            assert error.args == ("select_max",)
        else:
            raise AssertionError("Registered failure was not reproduced")
        rejected = ledger.read()
        assert rejected["counts"] == {"select":0,"initialize":0,"advance":0} and rejected["stage_start"] is None
    try:
        validate_budget(failed)
    except ValueError as error:
        reason = str(error)
    else:
        raise AssertionError("Broken contract was accepted")
    fixed = corrected_residual_plan(failed)
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        ledger = BudgetLedger(folder,fixed)
        ledger.create()
        key = ledger.start_seed("SAC",11)
        for kind in ["select","initialize","advance"]:
            ledger.reserve(key,kind)
        state = ledger.read()
        assert state["counts"] == dict.fromkeys(["select","initialize","advance"],1)
        assert state["stage_start"] is not None
        ledger.finish(key,"failed")
    return {"status":"offline_candidate_contract_verified", "frozen_registered_fault_reproduced":"KeyError: select_max",
        "original_counts_after_rejected_reservation":rejected["counts"],"original_stage_start":None,
        "fail_fast_validation":reason,"candidate_budget":fixed["budget_total_all_algorithms_seeds"],
        "candidate_fixture_predebits":state["counts"],"fixture_only":True,"FMU_calls":0,"training_steps":0,
        "new_authorization_or_trial_created":False,"frozen_entry_modified":False,
        "required_next_step":"bind validated plan in a separately versioned entry; test full transport and dispatch before new owner-authorized trial"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    write_snapshot(args.output,check())
