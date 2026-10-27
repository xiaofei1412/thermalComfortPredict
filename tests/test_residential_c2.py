import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermal_comfort import greybox
from thermal_comfort.residential_c2 import (EVENTS, PLAN, RUNS, EventScript, causal_check, collect,
    coverage_rows, group_masks, read, require_open, score_model, verify_data)


def synthetic_episode():
    script = EventScript(read(EVENTS)); records = []
    before = {"time": 287*86400, "reaTZon_y": 294.15, "oveTSet_u": 294.15}
    for k in range(192):
        item = script.tick(k, before)
        records.append({"before": copy.deepcopy(before), **item})
        before = {**before, "time": before["time"]+900, "oveTSet_u": item["action"]["oveTSet_u"]}
    times = np.array([r["before"]["time"] for r in records])
    states = np.zeros((193, 4)); states[:, 0] = 21.; states[:, 3] = .1
    return {"id": "synthetic_contract_fixture", "split": "development", "records": records, "times": times,
            "states": states, "actions": np.array([r["action"]["oveTSet_u"]-273.15 for r in records]),
            "heat": np.full(192, .2), "initial_heat": .2, "observed_heat": np.zeros(192), "exog": np.zeros((192, 5))}


def toy_model():
    return {"parameters": np.ones(6), "A": np.array([[.99, .01], [.01, .99]]), "B": np.zeros((2, 4)),
            "observer_gain": .2, "residual_weight": 0., "heat_max_kW": 15., "actuator": np.array([.5, 1., 0., 0., 0.]),
            "power": np.array([.1, .2, .1]), "idle_power_kW": .01, "power_transition_kW": .5,
            "support_low": np.array([-100., -100., -100.]), "support_high": np.array([100., 100., 100.])}


class ResidentialC2Contracts(unittest.TestCase):
    def test_immutable_run_budget_contains_no_calibration_or_test(self):
        self.assertEqual(sum(r[3] for r in RUNS), 464)
        self.assertEqual([r[2] for r in RUNS], [224, 287, 287, 287])
        self.assertTrue(all(r[1] in ("train", "development", "response") for r in RUNS))

    def test_event_script_is_reproducible_and_actions_stay_limited(self):
        a, b = synthetic_episode(), synthetic_episode()
        self.assertEqual(a["records"], b["records"])
        self.assertTrue(np.all((a["actions"] >= 18) & (a["actions"] <= 23.25)))
        self.assertTrue(np.all(abs(np.diff(a["actions"])) <= .5+1e-8))

    def test_expiry_does_not_renew_and_resolution_requires_explicit_rejection(self):
        r = synthetic_episode()["records"]
        self.assertTrue(r[4]["rejections"])
        self.assertFalse(r[4]["rejections"][0]["advance_called"])
        self.assertEqual(r[4]["decision"]["demand"]["status"], "infeasible")
        self.assertEqual(r[8]["decision"]["demand"]["status"], "no_active")
        self.assertIn("expire", r[8]["labels"])
        self.assertIn("return_to_base", r[8]["labels"])
        self.assertEqual(r[12]["decision"]["demand"]["status"], "conflict")
        self.assertIn("replace", r[16]["labels"])
        self.assertIn("withdraw", r[20]["labels"])

    def test_semantic_groups_are_not_repeated_as_new_confirmations(self):
        e = synthetic_episode()
        self.assertIn("conflict_resolved", e["records"][12]["labels"])
        self.assertNotIn("conflict_resolved", e["records"][13]["labels"])
        rows = coverage_rows(e, read(PLAN))
        semantic = [r for r in rows if r["group"] not in ("active", "idle", "transition")]
        self.assertTrue(all(r["origins"] >= 12 and r["distinct_days"] == 2 for r in semantic))

    def test_future_events_cannot_change_current_masks(self):
        e = synthetic_episode(); changed = copy.deepcopy(e)
        for r in changed["records"][40:]: r["labels"] = ["withdraw", "replace"]
        a, b = group_masks(e, read(PLAN)), group_masks(changed, read(PLAN))
        self.assertTrue(all(np.array_equal(a[k][:40], b[k][:40]) for k in a))

    def test_response_has_no_injected_person_requests_and_common_prelude(self):
        scripts = [EventScript(read(EVENTS)), EventScript(read(EVENTS))]
        before = {"time": 287*86400, "reaTZon_y": 294.15, "oveTSet_u": 294.15}
        for step in range(16):
            low, high = [s.tick(step, before, target) for s, target in zip(scripts, [18., 23.25])]
            self.assertEqual(low, high)
            self.assertFalse(low["decision"]["demand"]["active_requests"])
            before = {**before, "time": before["time"]+900, "oveTSet_u": low["action"]["oveTSet_u"]}

    def test_endpoint_energy_and_power_metrics_match_definitions(self):
        e, m = synthetic_episode(), toy_model(); config = read(PLAN)
        rows = score_model(m, e, config)
        row = next(r for r in rows if r["group"] == "all" and r["horizon_steps"] == 4)
        origins = np.arange(16, 192-4+1, 4)
        p = greybox.episode_rollout(m, e, origins, 4)
        self.assertAlmostEqual(row["energy_RMSE_kWh"], np.sqrt(np.mean((p[:, :, 3].sum(axis=1)-.4)**2)))
        self.assertAlmostEqual(row["power_MAE_kW"], np.mean(abs(p[:, :, 3]*4-.4)))
        self.assertTrue(causal_check(m, e))

    def test_missing_groups_remain_missing_instead_of_being_pooled(self):
        rows = score_model(toy_model(), synthetic_episode(), read(PLAN))
        active = next(r for r in rows if r["group"] == "active")
        self.assertEqual(active["origins"], 0)
        self.assertIsNone(active["temperature_RMSE_K"])

    def test_closed_phase_rejects_future_collection_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); (p/"closure.json").write_text('{"status":"closed_negative"}')
            with self.assertRaises(ValueError): require_open(p)

    def test_collection_cannot_retry_previous_attempt(self):
        with tempfile.TemporaryDirectory() as directory, patch('thermal_comfort.residential_c2.require_open'):
            p = Path(directory); (p/"operations.json").write_text('[]')
            with self.assertRaises(ValueError): collect(p, 'http://127.0.0.1:0')

    def test_operation_verifier_rejects_any_protected_initialization_before_data_read(self):
        with tempfile.TemporaryDirectory() as directory, patch('thermal_comfort.residential_c2.verify_lock'):
            p = Path(directory); (p/"operations.json").write_text(json.dumps([{'path':'initialize/example','data':{'start_time':300*86400}}]))
            with self.assertRaises(ValueError): verify_data(p)


if __name__ == '__main__':
    unittest.main()
