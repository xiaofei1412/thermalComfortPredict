import copy
import unittest

from thermal_comfort.rl_comparison_protocol import parameters, validate
from thermal_comfort.v2_protocol import ROOT, read


class ComparisonProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(ROOT/"configs/v2_comparison.json")

    def test_registered_totals_and_no_training_permission(self):
        report = validate(self.plan)
        self.assertFalse(report["training_authorized"])
        self.assertEqual(report["budget_total_all_algorithms_seeds"]["advance_max"], 453600)
        self.assertEqual(report["budget_per_algorithm_seed"]["train_actor_steps"], 47808)

    def test_inconsistent_budget_or_opened_training_rejected(self):
        for name, key, value in [("budget_per_algorithm_seed", "advance_max", 50401),
                ("budget_total_all_algorithms_seeds", "train_actor_steps", 430273)]:
            plan = copy.deepcopy(self.plan)
            plan[name][key] = value
            with self.assertRaises(ValueError):
                validate(plan)
        self.plan["training_authorized"] = True
        with self.assertRaises(ValueError):
            validate(self.plan)

    def test_holdout_and_extra_development_selection_rejected(self):
        for group, key, value in [("evaluation", "holdout_initializations_authorized", 1),
                ("checkpoints", "intermediate_development_evaluations", 1),
                ("checkpoints", "automatic_resume", True)]:
            plan = copy.deepcopy(self.plan)
            plan[group][key] = value
            with self.assertRaises(ValueError):
                validate(plan)

    def test_role_and_reward_drift_rejected(self):
        self.plan["train_ids"][0] = "development084"
        with self.assertRaises(ValueError):
            validate(self.plan)
        self.plan = read(ROOT/"configs/v2_comparison.json")
        self.plan["shared"]["reward"]["hot_Kh"] = 0
        with self.assertRaises(ValueError):
            validate(self.plan)

    def test_ppo_rollout_cannot_overshoot_episode(self):
        self.plan["parameters"]["PPO"]["n_steps"] = 2048
        with self.assertRaises(ValueError):
            validate(self.plan)

    def test_parameters_do_not_mutate_registration(self):
        before = copy.deepcopy(self.plan)
        params, policy = parameters(self.plan, "PPO")
        self.assertTrue(policy["ortho_init"])
        self.assertNotIn("ortho_init", params)
        self.assertEqual(before, self.plan)


if __name__ == "__main__":
    unittest.main()
