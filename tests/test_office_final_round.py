import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermal_comfort import native as n, source_terminal as sm, office_final_round as f
from test_native import PARAMETERS, episode


PARAMETERS11=np.r_[PARAMETERS[:7],PARAMETERS[10:14]]


class FinalOfficeTests(unittest.TestCase):
    def test_source_two_poles_have_the_exact_step_response(self):
        ih=ic=x1=x2=mode=0.
        for second in range(1,31):
            ih,ic,x1,x2,mode,_,_=sm.terminal_step(ih,ic,x1,x2,mode,20.,40.,30.)
            self.assertAlmostEqual(x2,1-(1+sm.RATE*second)*np.exp(-sm.RATE*second),places=13)

    def test_past_temperature_replay_preserves_unsaturated_integrator_memory(self):
        e=episode();h=sm.history(e)
        self.assertAlmostEqual(h[-1,1],.08,places=8)
        np.testing.assert_array_equal(h[-1,2:4],[0.,0.])

    def test_one_second_thermal_operator_has_correct_energy_units(self):
        p=PARAMETERS11.copy();p[:2]=1e10;a,b=sm.operators(p,1)
        state=np.array([20.,20.]);after=a@state+b@np.array([20.,1.,0.,0.,0.])
        self.assertAlmostEqual(np.sum((after-state)*p[2:4]),1/3600,places=9)

    def test_future_labels_and_oracle_fields_cannot_change_current_prediction(self):
        e=episode();start=np.array([480]);seconds=n.configuration()['horizons_seconds']
        before=sm.prediction(PARAMETERS11,e,start,seconds);changed=copy.deepcopy(e)
        changed['temperature'][481:]+=100;changed['power'][481:]+=100;changed['fan'][481:]=1
        changed['integrals']=np.full((1201,2),100.);changed['mode']=np.ones(1201)
        changed['enthalpy_correction']=np.full(1201,100000.)
        np.testing.assert_array_equal(before,sm.prediction(PARAMETERS11,changed,start,seconds))

    def test_current_action_affects_heat_and_metering_is_native_trapezoid(self):
        e=episode();start=np.array([480]);low=sm.prediction(PARAMETERS11,e,start,[30,300])
        e['actions'][480:]=23.25;high=sm.prediction(PARAMETERS11,e,start,[30,300])
        self.assertGreater(high[0,0,1],low[0,0,1])
        np.testing.assert_allclose(high[0,0,4:7],high[0,0,1:4]*30/3600/2)
        self.assertTrue(np.isfinite(high).all());self.assertTrue((high[:,:,1:]>=0).all())

    def test_fitting_rejects_nontraining_roles_before_optimization(self):
        e=episode();e['role']='development'
        with self.assertRaises(ValueError):sm.fit([e],PARAMETERS11,f.configuration(),n.configuration())

    def test_horizon_validation_and_prediction_prefix(self):
        e=episode();start=np.array([480]);long=sm.prediction(PARAMETERS11,e,start,[30,300,21600])
        np.testing.assert_array_equal(long[:,:2],sm.prediction(PARAMETERS11,e,start,[30,300]))
        with self.assertRaises(ValueError):sm.prediction(PARAMETERS11,e,start,[15])

    def test_gate_adapter_changes_names_without_changing_limits_or_sparse_decisions(self):
        coverage=[{'day':150,'trajectory':'rule','origins':63,'active_origins':[0,38,28]}]
        old=[g for g in n.capability_gates([],coverage,n.configuration()) if g['model']=='native_joint']
        adapted=f.existing_gates([],coverage)
        self.assertEqual(adapted,[{**g,'model':f.configuration()['candidate']} for g in old])
        self.assertFalse(next(g['passed'] for g in adapted if g['target']=='gas'))

    def test_closed_round_rejects_primary_refit_evaluation_and_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            closure=Path(tmp)/'closure.json';closure.write_text('{}')
            with patch.object(f,'CLOSURE',closure),patch.object(f.d,'check_baseline') as baseline:
                for call in [f.fit,lambda:f.fit(True),f.evaluate,f.prepare]:
                    with self.assertRaisesRegex(ValueError,'round is closed'):call()
                baseline.assert_not_called()

    def test_passing_assessment_cannot_be_mislabeled_as_closed_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            f.save_json(root/'assessment.json',{'eligible_for_calibration':True});f.save_json(root/'audit.json',{})
            with patch.object(f,'ROOT',root),patch.object(f,'check_protocol'):
                with self.assertRaisesRegex(ValueError,'passing candidate'):f.seal()


if __name__=='__main__':unittest.main()
