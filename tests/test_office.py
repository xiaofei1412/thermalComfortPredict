import copy
import unittest

import numpy as np
import pandas as pd

from thermal_comfort import office_model as om
from thermal_comfort.office import OfficeController, configuration, mean_hvac, validate_windows
from thermal_comfort.solar import SolarController, configuration as policy_configuration
from test_mpc import INPUTS, before_fixture, model_fixture
from test_solar import forecast_fixture


def office_fixture():
    parameters=np.array([3.,.5,.3,5.,1.,6.,1.])
    a,b=om.discretize(parameters)
    return {"parameters":parameters,"A":a,"B":b,"observer_gain":.2,"residual_weight":0.,
        "terminal":np.array([1.,1.,0.,0.]),"fan":np.array([0.,.01,.01,0.,0.,0.,0.]),
        "thermal_cap_kW":20.,"fan_cap_kW":1.,"heat_efficiency":.9,"cooling_COP":3.,
        "margin_K":np.full(32,.2),"support_low":np.array([15.,-20.,0.]),"support_high":np.array([35.,40.,1500.])}


class OfficeTests(unittest.TestCase):
    def test_convective_heat_enters_air_and_cooling_has_opposite_sign(self):
        m=office_fixture(); s=np.array([[20.,20.]]); x=np.array([[20.,0.,0.,0.,1.,0.,0.]])
        base=om.thermal_step(m,s,np.array([0.]),x)
        high=om.thermal_step(m,s,np.array([1.]),x)
        low=om.thermal_step(m,s,np.array([-1.]),x)
        np.testing.assert_allclose(base,s,atol=1e-12)
        self.assertGreater(high[0,0]-20,high[0,1]-20)
        np.testing.assert_allclose(high-base,base-low,atol=1e-12)
        self.assertLess(max(abs(np.linalg.eigvals(m["A"]))),1)

    def test_heating_action_response_and_distinct_nonnegative_energy(self):
        m=office_fixture(); exog=np.tile([20.,0.,0.,0.,1.,0.,0.],(1,8,1))
        low=om.rollout(m,np.array([[20.,20.]]),np.zeros((1,3)),np.full((1,8),20.),exog,np.full((1,8),30.))
        high=om.rollout(m,np.array([[20.,20.]]),np.zeros((1,3)),np.full((1,8),22.),exog,np.full((1,8),30.))
        self.assertGreater(high[0,0,0],low[0,0,0])
        self.assertGreater(high[0,:,6].sum(),low[0,:,6].sum())
        np.testing.assert_allclose(high[:,:,6],high[:,:,2]/.9*.25)
        np.testing.assert_allclose(high[:,:,5],(high[:,:,3]/3+high[:,:,4])*.25)
        self.assertTrue((high[:,:,2:]>=0).all())

    def test_future_temperature_and_hvac_are_not_read_by_free_rollout(self):
        m=office_fixture(); e={"times":np.arange(80)*900,"temperature":np.full(81,22.),
            "hvac":np.tile([1.,0.,.01],(80,1)),"actions":np.full(80,22.),"exog":np.tile([10.,100.,2.,0.,1.,.2,.4],(80,1))}
        before=om.episode_rollout(m,e,np.array([20]),32)
        altered=copy.deepcopy(e);altered["temperature"][21:]+=100;altered["hvac"][20:]+=100
        np.testing.assert_array_equal(before,om.episode_rollout(m,altered,np.array([20]),32))

    def test_known_daily_cooling_schedule_does_not_use_future_measurements(self):
        np.testing.assert_array_equal(om.cooling_setpoint(np.array([0,8,17.75,18,32])*3600),[30,24,24,30,24])

    def test_measured_carriers_convert_to_heat_cool_and_fan_units(self):
        native=pd.DataFrame({"time":[0.,900.],"fcu_reaPHea_y":[2000.]*2,"fcu_reaPCoo_y":[1000.]*2,"fcu_reaPFan_y":[100.]*2})
        np.testing.assert_allclose(mean_hvac(native,configuration()),[1.8,3.,.1])

    def test_cooling_uses_electricity_without_gas_and_fan_stays_bounded(self):
        m=office_fixture();m["fan"]=np.full(7,100.)
        p=om.rollout(m,np.array([[30.,30.]]),np.zeros((1,3)),np.full((1,8),18.),
            np.tile([30.,0.,0.,0.,1.,0.,0.],(1,8,1)),np.full((1,8),24.))
        self.assertTrue((p[:,:,3]>0).any())
        self.assertTrue((p[:,:,6]==0).all())
        self.assertTrue((p[:,:,4]<=m["fan_cap_kW"]).all())
        np.testing.assert_allclose(p[:,:,5],.25*(p[:,:,3]/3+p[:,:,4]))

    def test_radiant_internal_gain_enters_mass_before_air(self):
        m=office_fixture();s=np.array([[20.,20.]]);x=np.array([[20.,0.,0.,0.,1.,0.,0.]])
        baseline=om.thermal_step(m,s,np.zeros(1),x)
        x[0,6]=1
        delta=om.thermal_step(m,s,np.zeros(1),x)-baseline
        self.assertGreater(delta[0,1],delta[0,0])

    def test_roles_and_new_test_windows_reject_calendar_leakage(self):
        c=configuration();validate_windows(c)
        c["episodes"][-1]["day"]=219
        with self.assertRaises(ValueError):validate_windows(c)
        c=configuration();c["test_days"][0]=210
        with self.assertRaises(ValueError):validate_windows(c)

    def test_fitting_rejects_selection_or_calibration_rows(self):
        with self.assertRaises(ValueError):om.fit([{"role":"calibration"}],configuration())

    def test_common_initial_rule_and_actuator_bounds_are_preserved(self):
        c=configuration();p=policy_configuration();b={**before_fixture(),"fcu_reaPHea_y":0.,"fcu_reaPCoo_y":0.,"fcu_reaPFan_y":0.}
        calibrated=OfficeController(office_fixture(),c,p);calibrated.reset(b,INPUTS)
        source=model_fixture();source["mass_window_steps"]=16
        rule=SolarController(source,p,"guarded_rule");rule.reset(b,INPUTS)
        self.assertEqual(calibrated.decide(0,b,forecast_fixture())[0],rule.decide(0,b,forecast_fixture())[0])
        self.assertLessEqual(abs(calibrated.ramp(18,32)[0]-22),.5)
        self.assertEqual(calibrated.ramp(18,32)[-1],18)


if __name__=="__main__":unittest.main()
