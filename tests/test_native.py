import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermal_comfort import native as n, native_model as nm


PARAMETERS=np.array([3,2,.0432,5,1,6,1,120,.9,30,1,1,.15,.001])


def episode():
    count=1200
    return {"role":"train","temperature":np.full(count+1,22.),"power":np.zeros((count+1,3)),
        "fan":np.zeros(count+1),"actions":np.full(count,22.),"initial_action":22.,"initial_setting":30.,
        "settings":np.full(count,30.),"exog":np.tile([10.,100.,0.,0.,1.,0.,0.],(count,1))}


class NativeTests(unittest.TestCase):
    def test_native_discretization_has_thirty_second_energy_input(self):
        p=PARAMETERS.copy();p[0:2]=1e10
        a,b=nm.operators(p);state=np.array([20.,20.]);x=np.zeros(7);x[0]=20
        after=nm.thermal(a,b,state,1.,x)
        energy=np.sum((after-state)*p[2:4])
        self.assertAlmostEqual(energy,30/3600,places=9)

    def test_native_free_run_never_inserts_future_measured_labels(self):
        e=episode();start=np.array([480]);before=nm.prediction(PARAMETERS,e,start,[30,300,21600])
        altered=copy.deepcopy(e);altered["temperature"][481:]+=50
        altered["power"][481:]+=10;altered["fan"][481:]=1
        np.testing.assert_array_equal(before,nm.prediction(PARAMETERS,altered,start,[30,300,21600]))

    def test_new_action_has_an_instantaneous_response_without_reinitializing_pi(self):
        e=episode();low=nm.prediction(PARAMETERS,e,np.array([480]),[30,300])
        e["actions"][480:]=23.25
        high=nm.prediction(PARAMETERS,e,np.array([480]),[30,300])
        self.assertGreater(high[0,0,1],low[0,0,1])
        self.assertGreater(high[0,1,4],low[0,1,4])

    def test_energy_is_integral_of_endpoints_and_rollout_prefix_is_consistent(self):
        e=episode();e["actions"][480:]=23.25
        p=nm.prediction(PARAMETERS,e,np.array([480]),[30,300,21600])
        np.testing.assert_allclose(p[0,0,4:7],p[0,0,1:4]*nm.DH/2)
        np.testing.assert_array_equal(p[:,:2],nm.prediction(PARAMETERS,e,np.array([480]),[30,300]))
        self.assertTrue(np.isfinite(p).all());self.assertTrue((p[:,:,1:]>=0).all())

    def test_native_targets_use_trapezoidal_units(self):
        e=episode();e["power"][:]=[1,2,.1]
        y=nm.truth(e,np.array([480]),[30,3600,21600])
        np.testing.assert_allclose(y[0,:,4:7],np.array([30/3600,1,6])[:,None]*[1,2,.1])

    def test_training_and_partial_trajectory_roles_are_rejected(self):
        e=episode();e["role"]="calibration"
        with self.assertRaises(ValueError):nm.fit([e],PARAMETERS,n.configuration(),True)
        with self.assertRaises(ValueError):n.read_native(Path("unused"),{"complete":False},"train")
        with self.assertRaises(ValueError):n.read_native(Path("unused"),{},"selection")
        for role in ["calibration","evaluation"]:
            with self.assertRaisesRegex(ValueError,"Historical source cannot supply"):n.source_episodes(role)

    def test_reserved_calibration_excludes_old_test_and_initialization(self):
        c=n.configuration();n.validate_roles(c);c["reserved_calibration_days"][0]=224
        with self.assertRaises(ValueError):n.validate_roles(c)
        c=n.configuration();c["training_days"][0]=224
        with self.assertRaises(ValueError):n.validate_roles(c)

    def test_no_development_success_cannot_open_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with patch.object(n,"MODEL",path),patch.object(n,"CALIBRATION",path/"calibration"):
                n.save_json(path/"development_assessment.json",{"code_sha256":n.hashes(),
                    "summary":{"native_joint":{"all_capability_gates_passed":False}}})
                with self.assertRaisesRegex(ValueError,"capability gate failed"):n.lock_development()
                self.assertFalse((path/"calibration").exists())

    def test_sparse_energy_evidence_is_not_an_accuracy_success(self):
        c=n.configuration();activity=[{"day":62,"trajectory":"rule","origins":50,"active_origins":[50,0,50]}]
        gates=n.capability_gates([],activity,c)
        cooling=[g for g in gates if g["target"]=="cooling_electricity"]
        self.assertTrue(cooling);self.assertTrue(all(not g["passed"] for g in cooling))


if __name__=="__main__":unittest.main()
