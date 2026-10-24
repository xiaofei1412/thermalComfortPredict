import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from thermal_comfort import greybox
from thermal_comfort.solar import SolarController, SolarResponse, configuration, discretize, episode_rollout, mass_state, observer_states, validate_split
from test_mpc import INPUTS, before_fixture, forecast_fixture as base_forecast_fixture, model_fixture


CONFIG = configuration()


def forecast_fixture():
    frame = base_forecast_fixture().reindex(range(64)).ffill()
    frame["time"] = np.arange(64)*900
    return frame


def solar_fixture():
    model = model_fixture()
    model["parameters"] = np.array([3.,.15,2.,12.,8.,12.,.1])
    model["A"],model["B"] = discretize(model["parameters"])
    model.update(mass_window_steps=16,mass_ridge=.001,margin_K=np.full(32,.1))
    return model


class SolarTests(unittest.TestCase):
    def policy(self, kind="solar_mpc"):
        policy = SolarController(solar_fixture(),CONFIG,kind)
        policy.reset(before_fixture(),INPUTS)
        return policy

    def test_positive_stored_solar_network_and_stable_equilibrium(self):
        model = solar_fixture()
        self.assertLess(max(abs(np.linalg.eigvals(model["A"]))),1)
        self.assertTrue((model["B"]>=-1e-12).all())
        self.assertGreater(model["B"][1,2],0)
        dark = np.array([[20.,0.,0.,0.,1.]])
        np.testing.assert_allclose(greybox.thermal_step(model,np.array([[20.,20.]]),np.array([0.]),dark),[[20.,20.]],atol=1e-12)
        lit = dark.copy(); lit[:,1] = 500
        state = greybox.thermal_step(model,np.array([[20.,20.]]),np.array([0.]),lit)
        self.assertGreater(state[0,1],20.)
        after_sunset = greybox.thermal_step(model,state,np.array([0.]),dark)
        self.assertGreater(after_sunset[0,0],20.)

    def test_past_temperature_window_recovers_hidden_mass(self):
        model = solar_fixture(); model["mass_ridge"] = 1e-12
        state = np.array([22.,25.])
        history=[]
        exog = np.array([10.,300.,5.,0.,1.])
        for _ in range(16):
            after = greybox.thermal_step(model,state[None],np.array([3.]),exog[None])[0]
            history.append((state[0],after[0],3.,exog))
            state = after
        np.testing.assert_allclose(mass_state(model,history,22.),state,atol=1e-7)

    def test_future_observed_temperature_and_heat_do_not_change_rollout(self):
        model = solar_fixture()
        e = {"states":np.tile([22.,0.,0.,.2],(65,1)),"heat":np.full(64,4.),"initial_heat":4.,
             "actions":np.full(64,22.),"exog":np.tile([5.,100.,5.,0.,1.],(64,1))}
        before = episode_rollout(model,e,np.array([20]),24)
        changed = copy.deepcopy(e)
        changed["states"][21:,0] += 100
        changed["heat"][20:] += 100
        np.testing.assert_array_equal(before,episode_rollout(model,changed,np.array([20]),24))

    def test_online_mass_update_matches_causal_batch_estimate(self):
        policy = self.policy()
        e = {"states":np.tile([22.,0.,0.,.2],(25,1)),"heat":np.full(24,4.),"initial_heat":4.,
             "actions":np.full(24,22.),"exog":np.zeros((24,5))}
        forecast = forecast_fixture()
        from thermal_comfort.mpc import exogenous
        e["exog"] = exogenous(forecast.iloc[:24])
        states = observer_states(policy.model,e)
        for k in range(24):
            before = {**before_fixture(),"time":k*900}
            after = {**before_fixture(),"time":(k+1)*900}
            interval = pd.DataFrame({"time":[k*900,(k+1)*900],"reaQFloHea_y":[4000.,4000.]})
            policy.update(before,after,interval,forecast.iloc[k])
            np.testing.assert_allclose(policy.state,states[k+1],rtol=1e-12,atol=1e-12)

    def test_early_stop_requires_enough_stored_heat_to_prevent_cold(self):
        policy = self.policy()
        prediction = np.zeros((1,32,4)); prediction[:,:,0] = 23.9
        with patch.object(policy,"predict",return_value=prediction):
            action,info = policy.decide(16,before_fixture(),forecast_fixture())
        self.assertEqual(info["mode"],"early_coast")
        self.assertAlmostEqual(action["oveTSet_u"]-273.15,21.5)
        json.dumps(info,allow_nan=False)
        prediction[:,:,0] = 20.
        with patch.object(policy,"predict",return_value=prediction):
            _,info = policy.decide(17,before_fixture(),forecast_fixture())
        self.assertNotEqual(info["mode"],"early_coast")

    def test_minimum_heating_is_reactive_and_respects_slew(self):
        policy = self.policy("minimum_heat")
        action,info = policy.decide(16,before_fixture(22.),forecast_fixture())
        self.assertEqual(info["reason"],"minimum_setpoint")
        self.assertAlmostEqual(action["oveTSet_u"]-273.15,21.5)
        action,info = policy.decide(17,before_fixture(21.1),forecast_fixture())
        self.assertEqual(info["reason"],"reactive_heat")
        self.assertAlmostEqual(action["oveTSet_u"]-273.15,22.5)

    def test_plan_cannot_add_predicted_heat_storage_above_rule(self):
        policy = self.policy()
        policy.q = 0.
        policy.state = np.array([22.,22.])
        policy.model["actuator"] = np.zeros(5)
        forecast = forecast_fixture(); forecast["TDryBul"] = 295.15
        action,info = policy.decide(16,before_fixture(),forecast)
        self.assertEqual(info["mode"],"mpc")
        self.assertLessEqual(action["oveTSet_u"]-273.15,21.5+1e-9)
        self.assertTrue((np.array(info["predicted_temperature_C"])<=np.array(info["reference_temperature_C"])+1e-9).all())
        self.assertTrue((np.array(info["predicted_mass_C"])<=np.array(info["reference_mass_C"])+1e-9).all())

    def test_common_startup_and_stop_band_are_shared(self):
        actions=[self.policy(k).decide(0,before_fixture(),forecast_fixture())[0] for k in CONFIG["controllers"]]
        self.assertTrue(all(a==actions[0] for a in actions))
        with self.assertRaises(ValueError):
            self.policy().decide(16,before_fixture(35.1),forecast_fixture())

    def test_intervention_prediction_uses_rate_limited_actions(self):
        policy = SolarResponse(solar_fixture(),CONFIG,18.)
        policy.reset(before_fixture(),INPUTS)
        action,info = policy.decide(16,before_fixture(),forecast_fixture())
        self.assertAlmostEqual(action["oveTSet_u"]-273.15,21.5)
        plan = np.array(info["planned_actions_C"])
        self.assertLessEqual(abs(np.diff(np.r_[22.,plan])).max(),.5)
        self.assertEqual(plan[-1],18.)
        json.dumps(info,allow_nan=False)

    def test_new_split_rejects_warmup_overlap_and_overlapping_tests(self):
        splits=[{"split":"train","end_s":154,"warmup_start_s":0},
                {"split":"validation","end_s":196,"warmup_start_s":161},
                {"split":"test","end_s":245,"warmup_start_s":231}]
        validate_split(splits)
        bad=copy.deepcopy(splits); bad[1]["warmup_start_s"]=153
        with self.assertRaises(ValueError): validate_split(bad)
        bad=copy.deepcopy(splits); bad.append({"split":"test","end_s":260,"warmup_start_s":240})
        with self.assertRaises(ValueError): validate_split(bad)


if __name__ == "__main__":
    unittest.main()
