import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import gymnasium
except ImportError:
    gymnasium = None

if gymnasium is not None:
    from thermal_comfort.v2_baselines import (
        ScopedClient, check_action, command, comparison, coverage_gate, response_check,
        run_episode, weather_coverage, assert_same_values)
    from thermal_comfort.v2_protocol import ROOT, read


class FakeClient:
    def __init__(self):
        self.calls = []
        self.fail = None

    def call(self, method, path, data=None):
        self.calls.append(path)
        if path.startswith(self.fail or "never/"):
            raise TimeoutError("unknown result")
        if path.endswith("/select"):
            return {"testid": "fake"}
        return None


@unittest.skipIf(gymnasium is None, "V2 interface dependencies required")
class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(ROOT/"configs/v2_2.json")
        self.gates = read(ROOT/"configs/v2_0.json")["comparison_gates"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.raw = FakeClient()
        self.client = ScopedClient(self.raw, self.plan, [84, 168, 189], self.directory)

    def test_date_scope_rejects_train_and_holdout_before_io(self):
        for day in [14, 210, 259, 342, 238, 300, 329, 357]:
            with self.assertRaises(ValueError):
                self.client.call("PUT", "initialize/x", {"start_time": day*86400, "warmup_period": 604800})
        self.assertEqual(self.raw.calls, [])
        with self.assertRaises(ValueError):
            ScopedClient(self.raw, self.plan, [84, 168, 210], self.directory)

    def test_future_results_and_oversized_forecasts_rejected(self):
        self.client.call("PUT", "initialize/x", {"start_time": 84*86400, "warmup_period": 604800})
        for data in [{"start_time": 84*86400-30, "final_time": 84*86400},
                     {"start_time": 84*86400, "final_time": 84*86400+30}]:
            with self.assertRaises(ValueError):
                self.client.call("PUT", "results/x", data)
        with self.assertRaises(ValueError):
            self.client.call("PUT", "forecast/x", {"horizon": 86400})
        self.assertEqual(len(self.raw.calls), 1)

    def test_failed_advance_consumes_budget_without_retry(self):
        self.plan["live_budget"]["advance_attempts_max"] = 1
        self.client.call("PUT", "initialize/x", {"start_time": 84*86400, "warmup_period": 604800})
        self.raw.fail = "advance/"
        with self.assertRaises(TimeoutError):
            self.client.call("POST", "advance/x", {})
        with self.assertRaises(RuntimeError):
            self.client.call("POST", "advance/x", {})
        self.assertEqual(self.raw.calls.count("advance/x"), 1)

    def test_metadata_failure_releases_selected_worker(self):
        from unittest.mock import patch
        self.raw.fail = "inputs/"
        with patch("thermal_comfort.v2_baselines.SolarController"):
            with self.assertRaises(TimeoutError):
                run_episode(self.client, {"start_day": 84}, "guarded_rule", None, {}, self.directory/"run")
        self.assertEqual(self.raw.calls[-1], "stop/fake")
        self.assertTrue((self.directory/"run/failure.json").exists())

    def test_mechanical_and_native_temperature_checks(self):
        before = {"time": 0, "oveTSet_u": 294.15}
        action = {"oveTSet_u": 294.65, "oveTSet_activate": 1}
        native = pd.DataFrame({"time": np.arange(31)*30, "oveTSet_u": [294.15]+[294.65]*30,
            "oveTSet_activate": [0]+[1]*30, "reaTZon_y": [294.15]*31})
        check_action(action, before, native)
        for field, value in [("reaTZon_y", 310), ("oveTSet_u", 295)]:
            broken = native.copy()
            broken.loc[15, field] = value
            with self.assertRaises(ValueError):
                check_action(action, before, broken)
        with self.assertRaises(ValueError):
            check_action({**action, "oveTSet_u": 295.15}, before)

    def test_response_prefix_uses_frozen_rule_then_slew(self):
        from types import SimpleNamespace
        def decide(*args):
            return {"frozen": True}, {}
        policy = SimpleNamespace(previous=22, config={"action_min_C": 18, "action_max_C": 23.25,
            "slew_K_per_step": .5}, decide=decide)
        self.assertEqual(command(policy, 7, {}, None, 18)[0], {"frozen": True})
        self.assertAlmostEqual(command(policy, 8, {}, None, 18)[0]["oveTSet_u"]-273.15, 21.5)

    def test_weather_hours_and_solar_energy_units(self):
        schedule = pd.DataFrame({"time": np.arange(49)*1800, "TDryBul": [281.15]*49,
                                 "HGloHor": [200]*49})
        coverage = weather_coverage(schedule, self.plan["coverage"])
        self.assertEqual(coverage["cold_hours"], 24)
        self.assertEqual(coverage["warm_hours"], 0)
        self.assertEqual(coverage["sunny_hours"], 24)
        self.assertAlmostEqual(coverage["solar_total_kWh_m2"], 4.8)

    def test_csv_zero_prefix_types_are_compatible_but_value_drift_rejected(self):
        long_trace = pd.DataFrame({"heat": [0., 0., 1.25]})
        short_trace = pd.DataFrame({"heat": [0, 0]})
        assert_same_values(long_trace.iloc[:2], short_trace)
        with self.assertRaises(AssertionError):
            assert_same_values(long_trace.iloc[:2], pd.DataFrame({"heat": [0., 1e-12]}))

    def test_response_needs_real_heat_temperature_and_energy(self):
        low = {"energy_kWh": 1., "heat_kWh": 3., "operative_end_C": 21.}
        high = {"energy_kWh": 1.1, "heat_kWh": 3.2, "operative_end_C": 21.1}
        self.assertTrue(response_check(low, high, self.plan["coverage"])["responsive"])
        self.assertFalse(response_check(low, low, self.plan["coverage"])["responsive"])
        self.assertTrue(response_check(high, low, self.plan["coverage"])["reversed"])
        self.assertFalse(response_check(low, {**high, "operative_end_C": 21}, self.plan["coverage"])["responsive"])

    def metrics(self, energy):
        return {"energy_kWh": energy, **{f"{kind}_{m}": 0. for kind in ["cold", "hot"]
                                         for m in ["Kh", "duration_s", "max_K"]}}

    def test_energy_saving_cannot_hide_any_thermal_increase(self):
        rule, other = self.metrics(10), self.metrics(9)
        self.assertTrue(comparison(rule, other, self.gates)["joint_gate_passed"])
        for kind in ["cold", "hot"]:
            for metric in ["Kh", "duration_s", "max_K"]:
                self.assertFalse(comparison(rule, {**other, f"{kind}_{metric}": .1}, self.gates)["joint_gate_passed"])

    def test_low_load_has_no_relative_saving_and_small_absolute_tolerance(self):
        result = comparison(self.metrics(.1), self.metrics(.105), self.gates)
        self.assertIsNone(result["saving_fraction"])
        self.assertTrue(result["joint_gate_passed"])
        self.assertFalse(comparison(self.metrics(.1), self.metrics(.12), self.gates)["joint_gate_passed"])

    def test_coverage_cannot_pass_with_only_solar_or_one_active_date(self):
        coverage = {"a": {"cold_hours": 24, "warm_hours": 24, "sunny_hours": 12},
                    "b": {"cold_hours": 0, "warm_hours": 24, "sunny_hours": 12}}
        active = {"a": True, "b": True}
        responses = {"a": {"responsive": True, "reversed": False}, "b": {"responsive": False, "reversed": False}}
        result = coverage_gate(coverage, active, responses, self.plan["coverage"])
        self.assertTrue(result["passed"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(coverage_gate(coverage, {"a": True, "b": False}, responses, self.plan["coverage"])["passed"])
        broken = copy.deepcopy(coverage)
        for c in broken.values():
            c["warm_hours"] = 0
        self.assertFalse(coverage_gate(broken, active, responses, self.plan["coverage"])["passed"])


if __name__ == "__main__":
    unittest.main()
