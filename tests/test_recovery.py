import copy
import json
import unittest

import numpy as np

from thermal_comfort.greybox import energy_prediction
from thermal_comfort.recovery import RecoveryController, configuration, weather_view
from test_mpc import INPUTS, before_fixture, forecast_fixture, model_fixture


CONFIG = configuration()


class RecoveryTests(unittest.TestCase):
    def policy(self, kind="mpc", model=None, weather="perfect"):
        policy = RecoveryController(model or model_fixture(), CONFIG, kind, weather)
        policy.reset(before_fixture(), INPUTS)
        return policy

    def test_forecast_errors_preserve_measurements_schedule_and_source(self):
        forecast = forecast_fixture()
        original = forecast.copy(deep=True)
        forecast["HGloHor"] = 400.
        for mode, sign in [("warm_bias",1),("cold_bias",-1)]:
            biased = weather_view(forecast,0,CONFIG,mode)
            self.assertEqual(biased.TDryBul.iloc[0],forecast.TDryBul.iloc[0])
            self.assertEqual(biased.HGloHor.iloc[0],400.)
            self.assertAlmostEqual(biased.TDryBul.iloc[8]-forecast.TDryBul.iloc[8],sign*2)
            self.assertAlmostEqual(biased.HGloHor.iloc[8],400*(1+sign*.25))
            np.testing.assert_array_equal(biased["LowerSetp[1]"],original["LowerSetp[1]"])
        np.testing.assert_array_equal(forecast.TDryBul,original.TDryBul)

    def test_idle_power_and_transition_are_nonnegative_continuous(self):
        model = model_fixture()
        model.update(idle_power_kW=.001,power_transition_kW=.5)
        q = np.array([0.,.25,.5,.500001,10.])
        energy = energy_prediction(model,q,np.full(5,5.))
        self.assertAlmostEqual(energy[0],.00025)
        self.assertTrue((np.diff(energy)>0).all())
        self.assertLess(energy[3]-energy[2],1e-5)
        self.assertTrue((energy>=0).all())

    def test_recovery_anticipates_occupied_schedule_and_obeys_slew(self):
        forecast = forecast_fixture()
        forecast.loc[:11,"LowerSetp[1]"] = 288.15
        forecast.loc[:11,"UpperSetp[1]"] = 303.15
        forecast.loc[:11,"Occupancy[1]"] = 0.
        for kind in ["mpc","guarded_rule"]:
            action,info = self.policy(kind).decide(8,before_fixture(20.7),forecast)
            self.assertEqual(info["reason"],"anticipatory_recovery")
            self.assertAlmostEqual(action["oveTSet_u"]-273.15,22.5)
            json.dumps(info,allow_nan=False)

    def test_recovery_hysteresis_requires_release_threshold(self):
        policy = self.policy("guarded_rule")
        _,info = policy.decide(8,before_fixture(21.2),forecast_fixture())
        self.assertEqual(info["reason"],"anticipatory_recovery")
        _,info = policy.decide(9,before_fixture(21.4),forecast_fixture())
        self.assertEqual(info["reason"],"anticipatory_recovery")
        _,info = policy.decide(10,before_fixture(21.61),forecast_fixture())
        self.assertEqual(info["reason"],"rule")

    def test_unoccupied_floor_limits_deep_setback(self):
        forecast = forecast_fixture()
        forecast["LowerSetp[1]"] = 288.15
        forecast["UpperSetp[1]"] = 303.15
        forecast["Occupancy[1]"] = 0.
        policy = self.policy("guarded_rule")
        policy.previous = 20.5
        action,_ = policy.decide(8,before_fixture(),forecast)
        self.assertAlmostEqual(action["oveTSet_u"]-273.15,20.5)

    def test_coast_overrides_recovery_latch(self):
        policy = self.policy()
        policy.recovering = True
        _,info = policy.decide(8,before_fixture(24.1),forecast_fixture())
        self.assertEqual(info["reason"],"coast")
        self.assertFalse(policy.recovering)

    def test_robust_plan_logs_all_scenarios_and_honors_bounds(self):
        action,info = self.policy().decide(8,before_fixture(),forecast_fixture())
        self.assertEqual(info["mode"],"mpc")
        self.assertTrue(np.all(np.array(info["worst_temperature_low_C"])>=21.1))
        self.assertTrue(np.all(np.array(info["worst_temperature_high_C"])<=23.9))
        self.assertAlmostEqual(action["oveTSet_u"]-273.15,info["planned_actions_C"][0])
        json.dumps(info,allow_nan=False)

    def test_infeasible_and_ood_predictions_retain_recovery_rule(self):
        for field,reason in [("margin_K","no_margin_feasible_plan"),("support_high","outside_training_support")]:
            model = copy.deepcopy(model_fixture())
            if field == "margin_K":
                model[field][:] = 10
            else:
                model[field][1] = -50
            action,info = self.policy(model=model).decide(8,before_fixture(),forecast_fixture())
            self.assertEqual(info["mode"],"fallback")
            self.assertEqual(info["reason"],reason)
            self.assertAlmostEqual(action["oveTSet_u"]-273.15,21.5)


if __name__ == "__main__":
    unittest.main()
