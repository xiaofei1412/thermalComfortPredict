"""Offline budget-contract checks; no simulator client or training authorization."""

import copy


def validate_budget(plan):
    """Reject incomplete/mismatched accounting before a future trial creates files."""
    per_seed = plan["budget_per_algorithm_seed"]
    total = plan["budget_total_all_algorithms_seeds"]
    workers = len(plan["algorithms"])*len(plan["seeds"])
    for kind in ["select", "initialize", "advance"]:
        key = kind+"_max"
        if key not in total or key not in per_seed:
            raise ValueError("Missing ledger contract: "+key)
        if type(total[key]) is not int or total[key] != per_seed[key]*workers or total[key] <= 0:
            raise ValueError("Inconsistent total budget: "+key)
    if any(kind in total for kind in ["select","initialize","advance"]):
        raise ValueError("Ambiguous count names; ledger caps require _max suffix")
    for budget in [per_seed, total]:
        key = "wall_seconds_max" if budget is per_seed else "stage_wall_seconds_max"
        if budget[key] <= 0:
            raise ValueError("Nonpositive clock limit")
    return plan


def corrected_residual_plan(failed_plan):
    """Candidate correction only; caller must separately register a new runner/trial."""
    value = copy.deepcopy(failed_plan)
    total = value["budget_total_all_algorithms_seeds"]
    for kind in ["select","initialize","advance"]:
        total[kind+"_max"] = total.pop(kind)
    return validate_budget(value)
