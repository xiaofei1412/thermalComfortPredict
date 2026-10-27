import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from thermal_comfort.demands import submission
from thermal_comfort.demand_rule import BASELINE, common_action
from thermal_comfort.residential_c import (CONFIG, eligible_origins, historical_registry, merged_windows, physical_coverage,
                                          qualify_development, regime_masks, response_gate,
                                          semantic_coverage, validate_plan)
from thermal_comfort.workbench import Conflict


class ResidentialCContracts(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def request(self, resolution="unresolved", revision=1):
        return {"mode": "needs", "target_C": 21., "preferences": {"A": "hold", "B": "hold"},
                "resolution": resolution, "demand_revision": revision}

    def snapshot(self):
        demand = submission({"person": "A", "kind": "warmer", "base_C": 21., "ttl_steps": 1, "revision": 0}, 0, 0)
        return {"revision": 1, "time_s": 0, "demands": {"A": demand, "B": None}}

    def test_common_rule_adopts_demand_and_preserves_rate_limit_and_audit(self):
        snapshot = self.snapshot(); saved = copy.deepcopy(snapshot)
        action, d = common_action(self.request(), snapshot, {"time": 0, "reaTZon_y": 294.15, "oveTSet_u": 293.65})
        self.assertEqual(d["baseline"], BASELINE)
        self.assertAlmostEqual(action["oveTSet_u"]-273.15, 21.)
        self.assertEqual(d["target_C"], 21.5)
        self.assertEqual(d["evaluation"]["unmet_people"], ["A"])
        self.assertEqual(snapshot, saved)

    def test_common_rule_cannot_bypass_conflict_or_stale_resolution(self):
        s = self.snapshot()
        s["demands"]["B"] = submission({"person": "B", "kind": "cooler", "base_C": 21., "ttl_steps": 1, "revision": 0}, 0, 0)
        state = {"time": 0, "reaTZon_y": 294.15, "oveTSet_u": 294.15}
        with self.assertRaises(Conflict): common_action(self.request(), s, state)
        with self.assertRaises(Conflict): common_action(self.request("A", revision=0), s, state)
        _, d = common_action(self.request("A"), s, state)
        self.assertEqual(d["evaluation"]["unmet_people"], ["B"])
        with self.assertRaises(ValueError): common_action({"mode": "rule", "target_C": 21., "preferences": {"A": "hold", "B": "hold"}}, s, state)

    def test_deadline_does_not_renew_or_use_previous_frame(self):
        s = self.snapshot(); s["time_s"] = 900
        state = {"time": 900, "reaTZon_y": 294.15, "oveTSet_u": 294.15}
        _, d = common_action(self.request(), s, state)
        self.assertEqual(d["target_C"], 21.)
        self.assertFalse(d["demand"]["active_requests"])
        with self.assertRaises(ValueError): common_action(self.request(), self.snapshot(), state)

    def test_reserved_roles_are_ordered_and_all_warmups_isolated(self):
        rows = validate_plan(self.config, {"used_windows_days": [[7, 42], [308, 322], [336, 350]]})
        self.assertEqual([r["warmup_start_day"] for r in rows], [217, 280, 293, 322, 350])
        with self.assertRaises(ValueError): validate_plan(self.config, {"used_windows_days": [[216, 218]]})
        changed = copy.deepcopy(self.config);changed["reserved_episodes"][2]["start_day"] = 290
        with self.assertRaises(ValueError): validate_plan(changed, {"used_windows_days": []})

    def test_invalid_roles_ids_and_year_boundary_rejected(self):
        for index, field, value in [(0, "start_day", 5), (4, "start_day", 365), (2, "role", "train"), (1, "id", "intent_train224")]:
            changed = copy.deepcopy(self.config); changed["reserved_episodes"][index][field] = value
            with self.assertRaises(ValueError): validate_plan(changed, {"used_windows_days": []})

    def test_registry_uses_metadata_without_needing_old_result_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p = root/"outputs/old/run"; p.mkdir(parents=True)
            (p/"summary.json").write_text(json.dumps({"config": {"testcase": self.config["case"], "warmup_days": 7},
                "start_time_s": 98*86400, "end_time_s": 105*86400}), encoding="utf-8")
            result = historical_registry(root)
            self.assertEqual(result["used_windows_days"], [[91, 105]])
            (p/"summary.json").unlink()
            (p/"api_metadata.json").write_text("{}")
            with self.assertRaises(ValueError): historical_registry(root)

    def test_window_union_does_not_count_repeated_controllers_as_new_days(self):
        self.assertEqual(merged_windows([(10, 20), (10, 20), (18, 25), (30, 35)]), [[10, 25], [30, 35]])

    def test_regime_switch_only_uses_current_and_past_observations(self):
        heat = np.array([0., 0., 2., 2., 2., 2., 2., 0.])
        masks = regime_masks(heat, self.config)
        self.assertTrue(masks["transition"][2:6].all())
        self.assertTrue(masks["active"][6])
        self.assertTrue(np.all(sum(m.astype(int) for m in masks.values()) == 1))
        prefix = regime_masks(heat[:5], self.config)
        self.assertTrue(all(np.array_equal(prefix[k], masks[k][:5]) for k in masks))

    def test_rollout_origins_cannot_cross_episode_or_use_missing_history(self):
        origins = eligible_origins(192, 24, self.config)
        self.assertEqual(int(origins[0]), 16)
        self.assertEqual(int(origins[-1]), 168)
        self.assertFalse(eligible_origins(20, 24, self.config).size)
        with self.assertRaises(ValueError): eligible_origins(192, 96, self.config)

    def test_coverage_rejects_partial_native_bounds_and_nonfinite_values(self):
        times = np.arange(672)*900
        frame = pd.DataFrame({"time_start_s": times, "time_end_s": times+900, "setpoint_requested_C": 21.,
                              "occupancy_start": 1., "solar_start_W_m2": 100., "outdoor_start_C": 5., "operative_start_C": 21.})
        native = pd.DataFrame({"time": np.arange(672*30+1)*30, "reaQFloHea_y": 0.})
        self.assertEqual(physical_coverage(frame, native, self.config)["idle_steps"], 672)
        with self.assertRaises(ValueError): physical_coverage(frame, native.iloc[1:], self.config)
        frame.loc[0, "outdoor_start_C"] = np.nan
        with self.assertRaises(ValueError): physical_coverage(frame, native, self.config)

    def valid_scores(self):
        return [{"group": g, "horizon_steps": h, "origins": 12, "distinct_days": 2,
                 "temperature_RMSE_K": .1, "temperature_absolute_error_q95_K": .2,
                 "energy_RMSE_kWh": .1, "energy_absolute_bias_kWh": .01, "power_MAE_kW": .1,
                 "power_absolute_bias_kW": .01, "persistence_temperature_RMSE_K": .2,
                 "persistence_energy_RMSE_kWh": .2, "nonfinite_predictions": 0, "negative_power_predictions": 0}
                for g in self.config["minimum_coverage"]["groups"] for h in self.config["horizons_steps"]]

    def checks(self):
        return {k: True for k in ("causal", "native_integration", "action_limits", "response_direction_amplitude", "support_coverage", "provenance")}

    def test_gates_require_all_groups_horizons_and_coverage(self):
        rows = self.valid_scores()
        self.assertTrue(qualify_development(rows, self.checks(), self.config)["qualified"])
        self.assertFalse(qualify_development(rows[:-1], self.checks(), self.config)["qualified"])
        rows[0]["origins"] = 11
        self.assertFalse(qualify_development(rows, self.checks(), self.config)["qualified"])

    def test_single_energy_bias_or_structural_failure_blocks_qualification(self):
        for field, value in [("energy_absolute_bias_kWh", 1.), ("temperature_RMSE_K", float("nan")), ("negative_power_predictions", 1), ("nonfinite_predictions", -1), ("distinct_days", True)]:
            rows = self.valid_scores(); rows[0][field] = value
            self.assertFalse(qualify_development(rows, self.checks(), self.config)["qualified"])
        self.assertFalse(qualify_development(self.valid_scores(), {}, self.config)["qualified"])

    def test_idle_and_persistence_gates_cannot_hide_behind_average_rmse(self):
        rows = self.valid_scores(); next(r for r in rows if r["group"] == "idle")["power_absolute_bias_kW"] = .051
        self.assertFalse(qualify_development(rows, self.checks(), self.config)["qualified"])
        rows = self.valid_scores(); next(r for r in rows if r["group"] == "all" and r["horizon_steps"] == 24)["persistence_temperature_RMSE_K"] = .01
        self.assertFalse(qualify_development(rows, self.checks(), self.config)["qualified"])
        with self.assertRaises(ValueError): qualify_development(rows + rows[:1], self.checks(), self.config)

    def test_response_direction_alone_does_not_pass_amplitude_or_activity_gate(self):
        self.assertTrue(response_gate(.2, .25, 1., 1.2, self.config))
        self.assertIs(response_gate(np.float64(.2), np.float64(.25), 1., 1.2, self.config), True)
        for values in [(.2, .8, 1., 1.2), (.01, .01, .05, .05), (.2, -.2, 1., 1.), (.2, .2, 1., float("nan"))]:
            self.assertFalse(response_gate(*values, self.config))

    def test_passive_preferences_do_not_count_as_semantic_coverage(self):
        self.assertFalse(any(semantic_coverage({"events": [{"intent": {"preferences": {"A": "warmer"}}}]}).values()))


if __name__ == "__main__":
    unittest.main()
