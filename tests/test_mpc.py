import copy
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thermal_comfort.greybox import discretize, energy_prediction, episode_rollout, heat_prediction, thermal_step
from thermal_comfort.mpc import Controller, ResponseController, candidate_actions, limited_command


CONFIG = json.loads(Path("configs/mpc.json").read_text())
INPUTS = {"oveTSet_u": {"Minimum": 278.15, "Maximum": 308.15}}


def model_fixture():
    parameters = np.array([3., .15, 2., 10., 20., .1])
    a, b = discretize(parameters)
    return {"parameters": parameters, "A": a, "B": b, "observer_gain": .2,
            "residual_weight": 0., "actuator": np.array([.8, 1., 2., .01, .3]),
            "power": np.array([.2, .3, .1]), "heat_max_kW": 15., "margin_K": np.full(8, .1),
            "support_low": np.array([10., -30., 0.]), "support_high": np.array([40., 40., 2000.])}


def forecast_fixture():
    return pd.DataFrame({"time": np.arange(34) * 900, "LowerSetp[1]": 294.15,
                         "UpperSetp[1]": 297.15, "TDryBul": 278.15, "HGloHor": 0.,
                         "Occupancy[1]": 5., "PriceElectricPowerConstant": .2535})


def before_fixture(temperature=22.):
    return {"time": 0, "reaTZon_y": temperature + 273.15,
            "oveTSet_u": 295.15, "reaQFloHea_y": 4000.}


class GreyBoxTests(unittest.TestCase):
    def test_positive_rc_network_is_stable_and_preserves_equilibrium(self):
        model = model_fixture()
        self.assertTrue((model["A"] >= 0).all())
        self.assertTrue((model["B"] >= -1e-12).all())
        self.assertLess(max(abs(np.linalg.eigvals(model["A"]))), 1)
        exog = np.array([[20., 0., 0., 0., 1.]])
        equilibrium = thermal_step(model, np.array([[20., 20.]]), np.array([0.]), exog)
        heated = thermal_step(model, np.array([[20., 20.]]), np.array([1.]), exog)
        np.testing.assert_allclose(equilibrium, [[20., 20.]], atol=1e-12)
        self.assertTrue((heated > equilibrium).all())

    def test_energy_nonnegative_by_construction_and_correct_units(self):
        model = model_fixture()
        q = np.linspace(0, 15, 100)
        energy = energy_prediction(model, q, np.full(100, 5.))
        self.assertTrue((energy >= 0).all())
        self.assertTrue((np.diff(energy) >= 0).all())
        self.assertAlmostEqual(energy_prediction(model, np.array([2.]), np.array([5.]))[0], .2375)

    def test_actuator_has_positive_immediate_action_response_and_bounds(self):
        model = model_fixture()
        errors = np.linspace(-20, 20, 100)
        heat = heat_prediction(model, np.full(100, 4.), errors, np.zeros(100), np.full(100, 22.), np.full(100, 5.))
        self.assertTrue((np.diff(heat) >= 0).all())
        self.assertGreaterEqual(heat.min(), 0)
        self.assertLessEqual(heat.max(), 15)

    def test_future_measured_temperature_and_heat_do_not_leak(self):
        model = model_fixture()
        episode = {"states": np.tile([22., 0., 0., .2], (41, 1)), "heat": np.full(40, 4.),
                   "initial_heat": 4., "actions": np.full(40, 22.),
                   "exog": np.tile([5., 0., 5., 0., 1.], (40, 1))}
        prediction = episode_rollout(model, episode, np.array([8]), 8)
        changed = copy.deepcopy(episode)
        changed["states"][9:, 0] += 100
        changed["heat"][8:] += 100
        np.testing.assert_array_equal(prediction, episode_rollout(model, changed, np.array([8]), 8))

    def test_serialization_preserves_dynamics(self):
        model = model_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            reloaded = joblib.load(path)
        for name in ["A", "B", "parameters", "power", "actuator"]:
            np.testing.assert_array_equal(model[name], reloaded[name])


class MPCTests(unittest.TestCase):
    def controller(self, model=None, kind="mpc"):
        policy = Controller(model if model is not None else model_fixture(), CONFIG, kind)
        policy.reset(before_fixture(), INPUTS)
        return policy

    def test_enumeration_obeys_bounds_slew_and_blocking(self):
        plans = candidate_actions(20.5, CONFIG)
        self.assertEqual(plans.shape, (81, 8))
        changes = np.diff(np.column_stack([np.full(len(plans), 20.5), plans]), axis=1)
        self.assertLessEqual(abs(changes).max(), .5)
        self.assertTrue((plans >= 18).all() and (plans <= 23.25).all())
        np.testing.assert_array_equal(plans[:, ::2], plans[:, 1::2])
        edge = candidate_actions(23.25, CONFIG)
        self.assertLessEqual(edge.max(), 23.25)
        self.assertLess(len(edge), 81)  # Saturated duplicate plans are removed.

    def test_feasible_mpc_returns_checked_plan_and_json_log(self):
        action, info = self.controller().decide(8, before_fixture(), forecast_fixture())
        self.assertEqual(info["mode"], "mpc")
        self.assertGreater(info["feasible_candidates"], 0)
        self.assertTrue(all(21.1 <= t <= 23.9 for t in info["predicted_temperature_C"]))
        self.assertTrue(all(e >= 0 for e in info["predicted_energy_kWh"]))
        self.assertAlmostEqual(action["oveTSet_u"] - 273.15, info["planned_actions_C"][0])
        json.dumps(info, allow_nan=False)

    def test_impossible_margin_uses_rule_fallback(self):
        model = model_fixture()
        model["margin_K"][:] = 10.
        action, info = self.controller(model).decide(8, before_fixture(), forecast_fixture())
        self.assertEqual(info["reason"], "no_margin_feasible_plan")
        self.assertEqual(info["mode"], "fallback")
        self.assertAlmostEqual(action["oveTSet_u"] - 273.15, 21.5)

    def test_outside_support_uses_rule_fallback(self):
        forecast = forecast_fixture()
        forecast["TDryBul"] = 350.
        _, info = self.controller().decide(8, before_fixture(), forecast)
        self.assertEqual(info["reason"], "outside_training_support")

    def test_invalid_energy_prediction_is_not_optimized(self):
        model = model_fixture()
        model["power"] = np.array([-1., 0., 0.])
        _, info = self.controller(model).decide(8, before_fixture(), forecast_fixture())
        self.assertEqual(info["reason"], "invalid_model_prediction")

    def test_both_controllers_share_temperature_guard_and_slew(self):
        for temperature, direction in [(20.9, 1), (24.1, -1)]:
            actions = []
            for kind in ["mpc", "guarded_rule"]:
                action, _ = self.controller(kind=kind).decide(8, before_fixture(temperature), forecast_fixture())
                actions.append(action)
                self.assertAlmostEqual(action["oveTSet_u"] - 295.15, direction * .5)
            self.assertEqual(actions[0], actions[1])

    def test_stop_band_aborts_and_initial_observer_uses_rule(self):
        with self.assertRaises(ValueError):
            self.controller().decide(8, before_fixture(28.1), forecast_fixture())
        _, info = self.controller().decide(0, before_fixture(), forecast_fixture())
        self.assertEqual(info["reason"], "observer_initialization")

    def test_intervention_branches_have_identical_common_prelude(self):
        actions = []
        for setting in [21.5, 22., 22.5]:
            policy = ResponseController(model_fixture(), CONFIG, setting)
            policy.reset(before_fixture(), INPUTS)
            action, info = policy.decide(0, before_fixture(), forecast_fixture())
            actions.append(action)
            self.assertEqual(info["mode"], "common_prelude")
            json.dumps(info, allow_nan=False)
        self.assertTrue(all(a == actions[0] for a in actions))

    def test_limit_includes_first_command(self):
        self.assertAlmostEqual(limited_command(30, 22., CONFIG), 22.5)
        self.assertAlmostEqual(limited_command(5, 22., CONFIG), 21.5)


if __name__ == "__main__":
    unittest.main()
