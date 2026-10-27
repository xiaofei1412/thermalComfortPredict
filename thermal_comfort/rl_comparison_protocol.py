"""Read-only checks for the preregistered V2 comparison budget; no run entry point."""

import copy

from .simulation import digest
from .v2_protocol import ROOT, read, verify_files


def validate(plan):
    base = read(ROOT/"configs/v2_sac.json")
    if (plan["training_authorized"] is not False or plan["algorithms"] != ["SAC", "PPO", "TD3"]
            or plan["seeds"] != [11, 23, 37] or plan["formal_configs_per_algorithm"] != 1
            or plan["train_cycles"] != 12 or not plan["fresh_initialization"]
            or plan["old_SAC_engineering_trial_used_as_formal_seed"]
            or any(plan[k] != base[k] for k in ["train_ids", "development_ids"])):
        raise ValueError("Formal protocol roles, budget version or authorization changed")
    expected_parents = [*base["parent_locks"], "reports/v2_sac/lock.json"]
    if list(plan["parent_locks_sha256"]) != expected_parents:
        raise ValueError("Parent lock chain differs")
    for name, expected in plan["parent_locks_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("Parent lock hash differs")
        verify_files(ROOT, read(ROOT/name)["files_sha256"])
    episode = {"warmup_seconds": 604800, "control_seconds": 900, "native_seconds": 30,
        "scored_advances": 672, "common_rule_prefix": 8, "actor_steps": 664, "forecast_hours": 12}
    if plan["episode"] != episode:
        raise ValueError("Common episode or forecast semantics changed")
    trains, devs = 12*len(plan["train_ids"]), len(plan["development_ids"])
    budget = {"train_episodes": trains, "development_episodes": devs,
        "select_max": trains+devs, "initialize_max": trains+devs, "advance_max": (trains+devs)*672,
        "train_actor_steps": trains*664, "development_actor_steps": devs*664,
        "common_prefix_advances": (trains+devs)*8, "wall_seconds_max": 10800}
    if plan["budget_per_algorithm_seed"] != budget:
        raise ValueError("Per-seed budget arithmetic mismatch")
    total = {key: value*9 for key, value in budget.items() if key in ["select_max", "initialize_max",
        "advance_max", "train_actor_steps", "development_actor_steps", "common_prefix_advances"]}
    total["stage_wall_seconds_max"] = 43200
    if plan["budget_total_all_algorithms_seeds"] != total:
        raise ValueError("Total comparison budget arithmetic mismatch")
    shared = plan["shared"]
    if (shared["reward"] != base["reward"] or shared["guard"] != base["guard"]
            or shared["observation_scaling"] != base["fixed_observation_scaling"]
            or shared["state_statistics_fitted"] or shared["use_sde"] or shared["gamma"] != .995
            or shared["net_arch"] != [64, 64] or shared["activation"] != "ReLU"
            or shared["physical_action_bounds_C"] != [18, 23.25] or shared["action_rate_limit_K"] != .5):
        raise ValueError("Shared learning information, reward or actions differ")
    ppo = plan["parameters"]["PPO"]
    if ppo["n_steps"] != 664 or ppo["n_steps"] % ppo["batch_size"] or ppo["n_epochs"] != 10:
        raise ValueError("PPO would overshoot or change rollout/update budget")
    checkpoints = plan["checkpoints"]
    if (checkpoints["selection"] != "final_after_47808_actor_steps_only"
            or checkpoints["intermediate_development_evaluations"] != 0
            or checkpoints["fallback_to_earlier_checkpoint_if_final_missing"]
            or checkpoints["automatic_resume"] or checkpoints["planned_process_restarts"] != 0
            or checkpoints["mid_episode_FMU_resume"] or checkpoints["cycle_actor_steps"] != 3984
            or plan["evaluation"]["holdout_initializations_authorized"] != 0
            or plan["evaluation"]["extra_baseline_FMU_budget"] != 0):
        raise ValueError("Checkpoint selection, restart or evaluation permission changed")
    return {"status": "passed_preregistered_not_training", "budget_per_algorithm_seed": budget,
        "budget_total_all_algorithms_seeds": total, "parent_locks_checked": len(expected_parents),
        "training_authorized": False, "FMU_calls": 0, "heldout_reads_or_initializations": 0}


def parameters(plan, algorithm):
    params = copy.deepcopy(plan["parameters"][algorithm])
    policy = {"net_arch": copy.deepcopy(plan["shared"]["net_arch"])}
    if algorithm == "PPO":
        policy["ortho_init"] = params.pop("ortho_init")
    return params, policy
