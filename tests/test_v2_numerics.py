from decimal import Decimal
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
    from thermal_comfort.v2_numerics import DiagnosticClient, check_zero, decimal_thermal, difference, precision_check
    from thermal_comfort.simulation import POWER
    from thermal_comfort.v2_protocol import ROOT, read


class FakeClient:
    def __init__(self):
        self.calls = []
        self.fail = False

    def call(self, method, path, data=None):
        self.calls.append(path)
        if self.fail:
            raise TimeoutError("uncertain applied status")
        return {"testid": "fake"} if path.startswith("testcases/") else None


@unittest.skipIf(gymnasium is None, "Install the separate V2 interface environment")
class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(ROOT/"configs/v2_3_numerics.json")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.raw = FakeClient()
        self.client = DiagnosticClient(self.raw, self.plan, self.folder)

    def fixture(self):
        native = pd.DataFrame({"time": [0., 450., 900.], "reaTZon_y": [305.15]*3,
            "reaQFloHea_y": [0.]*3, "reaQHeaPumCon_y": [0.]*3, "reaQHeaPumEva_y": [0.]*3,
            "oveTSet_u": [291.15]*3, "oveTSet_activate": [1.]*3,
            "oveHeaPumY_activate": [0.]*3, "oveFan_activate": [0.]*3, "ovePum_activate": [0.]*3,
            **{p: [0.]*3 for p in POWER}})
        schedule = pd.DataFrame({"time": [0., 450., 900.], "LowerSetp[1]": [293.15]*3,
            "UpperSetp[1]": [303.15]*3, "PriceElectricPowerConstant": [.2]*3,
            "EmissionsElectricPower": [.1]*3})
        return native, schedule

    def test_only_development189_may_initialize(self):
        for day in [0, 14, 84, 168, 210, 259, 342, 300, 329, 357]:
            with self.assertRaises(ValueError):
                self.client.call("PUT", "initialize/fake", {"start_time": day*86400, "warmup_period": 604800})
        self.assertEqual(self.raw.calls, [])
        self.client.call("PUT", "initialize/fake", {"start_time": 189*86400, "warmup_period": 604800})
        self.assertEqual(len(self.raw.calls), 1)

    def test_select_budget_limits_instance_creation(self):
        path = "testcases/bestest_hydronic_heat_pump/select"
        for _ in range(4):
            self.client.call("POST", path)
        with self.assertRaises(RuntimeError):
            self.client.call("POST", path)
        with self.assertRaises(ValueError):
            self.client.call("POST", "testcases/other/select")
        self.assertEqual(len(self.raw.calls), 4)

    def test_failed_advance_consumes_budget_and_is_not_retried(self):
        self.plan["live_budget"]["advance_attempts_max"] = 1
        self.client.call("PUT", "initialize/fake", {"start_time": 189*86400, "warmup_period": 604800})
        self.raw.fail = True
        with self.assertRaises(TimeoutError):
            self.client.call("POST", "advance/fake", {})
        with self.assertRaises(RuntimeError):
            self.client.call("POST", "advance/fake", {})
        self.assertEqual(self.raw.calls.count("advance/fake"), 1)

    def test_bootstrap_results_cannot_be_read_before_explicit_initialization(self):
        self.client.call("POST", "testcases/bestest_hydronic_heat_pump/select")
        with self.assertRaises(ValueError):
            self.client.call("PUT", "results/fake", {"start_time": 0, "final_time": 0})
        self.assertEqual(len(self.raw.calls), 1)

    def test_forecast_and_future_result_permissions_stay_bounded(self):
        self.client.call("PUT", "initialize/fake", {"start_time": 189*86400, "warmup_period": 604800})
        with self.assertRaises(ValueError):
            self.client.call("PUT", "forecast/fake", {"horizon": 604800})
        with self.assertRaises(ValueError):
            self.client.call("PUT", "results/fake", {"start_time": 189*86400, "final_time": 190*86400})
        self.assertEqual(len(self.raw.calls), 1)

    def test_zero_diagnostic_checks_heat_pump_fluxes_and_overrides(self):
        native, _ = self.fixture()
        self.assertEqual(max(check_zero(native, self.plan).values()), 0)
        for field in self.plan["zero_channels"]:
            broken = native.copy()
            broken.loc[1, field] = -1e-5
            with self.assertRaises(ValueError):
                check_zero(broken, self.plan)
        native.loc[1, "oveFan_activate"] = 1
        with self.assertRaises(ValueError):
            check_zero(native, self.plan)

    def test_decimal_integral_preserves_small_exact_difference(self):
        native, schedule = self.fixture()
        self.assertEqual(Decimal(decimal_thermal(native, schedule)["hot_Kh"]), Decimal("0.5"))
        native.reaTZon_y += 1e-5
        self.assertAlmostEqual(float(decimal_thermal(native, schedule)["hot_Kh"]), .5000025, places=12)

    def test_cold_and_hot_decimal_integrals_are_separate(self):
        native, schedule = self.fixture()
        native.reaTZon_y = [292.15, 294.15, 304.15]
        result = decimal_thermal(native, schedule)
        self.assertEqual(float(result["cold_Kh"]), .0625)
        self.assertEqual(float(result["hot_Kh"]), .0625)

    def test_same_waveform_has_zero_differences(self):
        native, schedule = self.fixture()
        result = difference(native, native.copy(), schedule, 1e-6)
        self.assertEqual(result["temperature_max_abs_K"], 0)
        self.assertIsNone(result["first_temperature_difference_s"])
        self.assertIsNone(result["first_action_difference_s"])

    def test_tiny_temperature_drift_still_fails_original_integral_gate(self):
        native, schedule = self.fixture()
        other = native.copy()
        other.reaTZon_y += 1e-5
        result = difference(native, other, schedule, 1e-6)
        self.assertFalse(result["original_degree_hour_nonincrease_passed"])
        self.assertAlmostEqual(result["hot_integral_abs_difference_over_original_tolerance"], 2.5, places=6)

    def test_csv_round_trip_checked_against_raw_values(self):
        native, schedule = self.fixture()
        native.to_csv(self.folder/"native.csv.gz", index=False)
        result = precision_check(self.folder, native, schedule, self.plan)
        self.assertTrue(result["api_json_to_csv_round_trip_equal"])
        self.assertTrue(result["postprocessing_below_registered_diagnostic_limit"])
        native.loc[1, "reaTZon_y"] += 1e-7
        with self.assertRaises(AssertionError):
            precision_check(self.folder, native, schedule, self.plan)


if __name__ == "__main__":
    unittest.main()
