import copy
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from thermal_comfort import terminal as t, terminal_model as tm
from test_office import office_fixture
from test_mpc import INPUTS, before_fixture
from thermal_comfort.solar import configuration as source_policy


def fixture():
    m=office_fixture();m.update(family="dynamic",lag_parameters=[np.array([1.,0.,120.,.1])]*2,
        mean_fan=np.array([0,0,0,0,0,0,0,.001]),
        end_fan=np.array([0,0,0,0,0,0,0,.002]))
    return m


class TerminalTests(unittest.TestCase):
    def test_lag_outputs_are_convex_and_partial_data_cannot_train_or_calibrate(self):
        for tau in [10,120,1800]:
            mean,end=tm.lagged_flow([1.,0.,tau,0.],np.array([0.,3.]),np.array([3.,0.]),np.zeros(2),20.)
            self.assertTrue(((mean>=0)&(mean<=3)).all());self.assertTrue(((end>=0)&(end<=3)).all())
        for role in ["train","selection","calibration"]:
            with self.assertRaises(ValueError):t.read_episode(Path("unused"),{"sha256":{},"complete":False},role)

    def test_controller_reset_has_physical_carrier_parameters(self):
        b={**before_fixture(),"fcu_reaPHea_y":1000.,"fcu_reaPCoo_y":0.,"fcu_reaPFan_y":10.}
        pc={**source_policy(),"initial_rule_steps":16}
        controller=t.Controller(fixture(),t.configuration(),pc,{})
        controller.reset(b,INPUTS)
        np.testing.assert_allclose(controller.hvac,[.9,0,.01])

    def test_average_energy_is_distinct_from_endpoint_memory(self):
        m=fixture();p=tm.rollout(m,np.array([[20.,20.]]),np.array([[1.,0.,0.]]),np.full((1,4),22.),
            np.array([22.]),np.tile([20.,0.,0.,0.,1.,0.,0.],(1,4,1)),np.full((1,4),30.),np.array([30.]))
        self.assertTrue((p[:,:,2]!=p[:,:,7]).any())
        np.testing.assert_allclose(p[:,:,6],p[:,:,2]*.25/.9)
        np.testing.assert_allclose(p[:,:,5],(p[:,:,3]/3+p[:,:,4])*.25)
        self.assertTrue((p[:,:,2:]>=0).all())

    def test_idle_gate_prevents_head_intercepts_creating_gas_or_cooling(self):
        m=fixture()
        for a in m["lag_parameters"]:a[1]=2
        p=tm.rollout(m,np.array([[20.,20.]]),np.zeros((1,3)),np.full((1,8),18.),np.array([18.]),
            np.tile([20.,0.,0.,0.,1.,0.,0.],(1,8,1)),np.full((1,8),30.),np.array([30.]))
        self.assertTrue((p[:,:,2:4]==0).all());self.assertTrue((p[:,:,6]==0).all())
        self.assertTrue((p[:,:,5]>0).all())

    def test_current_action_increases_initial_heating_response(self):
        m=fixture();args=[m,np.array([[20.,20.]]),np.zeros((1,3))]
        x=np.tile([10.,0.,0.,0.,1.,0.,0.],(1,4,1));settings=np.full((1,4),30.)
        low=tm.rollout(*args,np.full((1,4),20.),np.array([20.]),x,settings,np.array([30.]))
        high=tm.rollout(*args,np.full((1,4),22.),np.array([20.]),x,settings,np.array([30.]))
        self.assertGreater(high[0,0,0],low[0,0,0]);self.assertGreater(high[0,0,6],low[0,0,6])

    def test_regimes_use_measured_endpoint_and_known_schedule(self):
        c=t.configuration()
        np.testing.assert_array_equal(tm.regime_indices([[1,0,0],[0,1,0],[0,0,0],[1,1,0]],0,4,c),[0,1,2,3])
        self.assertEqual(tm.regime_indices([0,0,0],7*3600,4,c)[0],3)

    def test_temporal_blocks_exclude_previous_office_data(self):
        c=t.configuration();t.validate_windows(c);c["episodes"][0]["day"]=224
        with self.assertRaises(ValueError):t.validate_windows(c)

    def test_final_margin_roles_and_sparse_pooling(self):
        c=t.configuration();e={"role":"calibration","actions":np.full(96,22.),"times":np.arange(96)*900,
            "temperature":np.full(97,22.),"end_hvac":np.zeros((96,3))}
        def predicted(model,episode,origins,horizon):
            result=np.zeros((len(origins),horizon,10));result[:,:,0]=23.;return result
        with patch.object(tm,"episode_rollout",predicted):
            margins=tm.calibrate({},[e],c)
        np.testing.assert_allclose(margins["lower_K"],1.1);np.testing.assert_allclose(margins["upper_K"],.1)
        self.assertTrue(np.array(margins["pooled_fallback"])[0].all())
        e["role"]="selection"
        with self.assertRaises(ValueError):tm.calibrate({},[e],c)

    def test_future_endpoint_labels_do_not_leak_into_prediction(self):
        m=fixture();e={"times":np.arange(80)*900,"temperature":np.full(81,22.),"hvac":np.tile([1.,0.,.01],(80,1)),
            "end_hvac":np.tile([.2,0.,.02],(80,1)),"actions":np.full(80,22.),"exog":np.tile([10.,100.,2.,0.,1.,.2,.4],(80,1))}
        before=tm.episode_rollout(m,e,np.array([20]),32);altered=copy.deepcopy(e)
        altered["temperature"][21:]+=100;altered["hvac"][20:]+=100;altered["end_hvac"][20:]+=100
        np.testing.assert_array_equal(before,tm.episode_rollout(m,altered,np.array([20]),32))


if __name__=="__main__":unittest.main()
