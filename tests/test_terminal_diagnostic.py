import copy
import unittest

import numpy as np

from thermal_comfort import native_model as nm, terminal_diagnostic as d
from test_native import PARAMETERS, episode


def fixture():
    e=episode();count=len(e['temperature'])
    e.update(integrals=np.zeros((count,2)),mode=np.zeros(count),enthalpy_correction=np.zeros(count))
    return e


class DiagnosticTests(unittest.TestCase):
    def test_baseline_reproduces_original_and_ignores_oracles(self):
        e=fixture();origins=np.array([480]);expected=nm.prediction(PARAMETERS,e,origins,d.SECONDS)
        np.testing.assert_array_equal(expected,d.predict(PARAMETERS,e,origins,'baseline'))
        e['integrals'][:]=100;e['mode'][:]=1;e['enthalpy_correction'][:]=100000
        np.testing.assert_array_equal(expected,d.predict(PARAMETERS,e,origins,'baseline'))

    def test_initial_pi_oracle_uses_only_origin_integrals(self):
        e=fixture();origins=np.array([480]);before=d.predict(PARAMETERS,e,origins,'initial_PI_oracle')
        e['integrals'][481:]=100
        np.testing.assert_array_equal(before,d.predict(PARAMETERS,e,origins,'initial_PI_oracle'))
        e['integrals'][480,0]=.5
        self.assertGreater(d.predict(PARAMETERS,e,origins,'initial_PI_oracle')[0,0,1],before[0,0,1])

    def test_oracle_free_runs_do_not_insert_future_temperature_or_power(self):
        e=fixture();origins=np.array([480])
        for variant in d.VARIANTS:
            before=d.predict(PARAMETERS,e,origins,variant);changed=copy.deepcopy(e)
            changed['temperature'][481:]+=50;changed['power'][481:]+=10
            np.testing.assert_array_equal(before,d.predict(PARAMETERS,changed,origins,variant))

    def test_flow_oracle_is_explicitly_noncausal_and_changes_only_its_channel(self):
        e=fixture();origins=np.array([480]);before=d.predict(PARAMETERS,e,origins,'flow_oracle')
        e['fan'][481:]=.8
        self.assertFalse(np.array_equal(before,d.predict(PARAMETERS,e,origins,'flow_oracle')))
        # Oracle use is intentional for diagnosis; it must never be described as a deployable predictor.
        self.assertIn('flow_oracle',d.VARIANTS)

    def test_stream_enthalpy_correction_has_joule_per_kg_units(self):
        p=PARAMETERS.copy();zero=d.corrected_power(p,20.,0.,0.,1.,0.,0.)
        warm=d.corrected_power(p,20.,0.,0.,1.,0.,1000.)
        self.assertAlmostEqual(warm[0]-zero[0],.55/.9)
        np.testing.assert_array_equal(zero,nm.powers(p,20.,0.,0.,1.,0.))

    def test_filter_two_poles_match_step_solution(self):
        rate=.259487839385516
        result=d.fan_replay(np.full(121,.7),np.zeros(2),274.,rate,True)
        seconds=np.arange(121)*30
        exact=.7*(1-(1+rate*seconds)*np.exp(-rate*seconds))
        np.testing.assert_allclose(result,exact,atol=2e-4,rtol=0)
        one=d.fan_replay(np.full(121,.7),np.zeros(2),274.,0.,False)
        np.testing.assert_allclose(one,.7*(1-np.exp(-seconds/274.)),atol=1e-14,rtol=0)

    def test_saturated_output_does_not_identify_integral(self):
        error=-8.
        self.assertEqual(nm.clipped(.1*error+.08),nm.clipped(.1*error+.5))
        replay=d.pi_replay(np.full(1001,22.),np.full(1000,22.),np.full(1000,30.),np.zeros(2),120.,.9)
        self.assertAlmostEqual(replay[-1,1],.08,places=8)

    def test_unregistered_variants_are_rejected(self):
        with self.assertRaises(ValueError):d.predict(PARAMETERS,fixture(),np.array([480]),'adaptive_oracle')


if __name__=='__main__':unittest.main()
