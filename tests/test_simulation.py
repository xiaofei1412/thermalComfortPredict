import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from thermal_comfort.simulation import (
    Client, POWER, integrate, rule_action, run_episode, schedule_at,
    transitions_frame, validate_kpis,
)


CONFIG = {"step_s": 900, "area_m2": 192, "days": 900 / 86400,
          "testcase": "bestest_hydronic_heat_pump",
          "occupied_offset_K": 0.5, "unoccupied_offset_K": 3.0}


def fixture():
    native = pd.DataFrame({"time": [0, 450, 900], "reaTZon_y": [290., 290., 290.],
                           "oveTSet_u": [291., 293.5, 293.5], "oveTSet_activate": [0, 1, 1],
                           **{p: [1000., 1000., 1000.] for p in POWER}})
    forecast = pd.DataFrame({"time": [0, 450, 900], "LowerSetp[1]": [293., 293., 293.],
                            "UpperSetp[1]": [303., 303., 303.], "Occupancy[1]": [1, 1, 1],
                            "TDryBul": [280., 280., 280.], "HGloHor": [0., 0., 0.],
                            "PriceElectricPowerConstant": [.2, .2, .2],
                            "EmissionsElectricPower": [.1, .1, .1]})
    records = [{"before": {"time": 0, "reaTZon_y": 290.},
                "after": {"time": 900, "reaTZon_y": 290.},
                "action": {"oveTSet_u": 293.5, "oveTSet_activate": 1}}]
    return native, forecast, records


class SimulationTests(unittest.TestCase):
    def test_api_envelope_and_inner_error(self):
        for body, expected in [({"status": 200, "payload": 900}, 900), ({"testid": "abc"}, {"testid": "abc"})]:
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
                self.assertEqual(Client("http://example").call("GET", "step/a"), expected)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"status":400,"payload":null}')):
            with self.assertRaises(RuntimeError):
                Client("http://example").call("GET", "step/a")

    def test_mutation_timeout_is_not_retried(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError) as request:
            with self.assertRaises(TimeoutError):
                Client("http://example").call("POST", "advance/a", {})
            self.assertEqual(request.call_count, 1)

    def test_empty_mutation_body_and_plain_stop_response(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'Stopped')) as request:
            self.assertIsNone(Client("http://example").call("PUT", "stop/a"))
            self.assertEqual(request.call_args.args[0].data, b'{}')

    def test_rule_uses_kelvin_schedule_and_clips(self):
        limits = {"Minimum": 278.15, "Maximum": 308.15}
        self.assertEqual(rule_action(293.15, 1, limits, CONFIG),
                         {"oveTSet_u": 293.65, "oveTSet_activate": 1})
        self.assertEqual(rule_action(289.15, 0, limits, CONFIG)["oveTSet_u"], 292.15)
        self.assertEqual(rule_action(308, 0, limits, CONFIG)["oveTSet_u"], 308.15)

    def test_schedule_is_zero_order_hold(self):
        schedule = pd.DataFrame({"time": [0, 900], "LowerSetp[1]": [289, 293]})
        self.assertEqual(schedule_at(schedule, np.array([0, 899, 900]))["LowerSetp[1]"].tolist(), [289, 289, 293])
        with self.assertRaises(ValueError):
            schedule_at(schedule, np.array([901]))

    def test_energy_integral_units_and_comfort(self):
        native, forecast, _ = fixture()
        metrics = integrate(native, forecast, 192)
        self.assertAlmostEqual(metrics["energy_kWh"], .75)
        self.assertAlmostEqual(metrics["discomfort_Kh"], .75)
        self.assertAlmostEqual(metrics["boptest_energy_kWh_m2"], .75 * 1.0000008 / 192)
        # Native power variation must be integrated, not replaced by the final power.
        native[POWER] = np.array([[0]*3, [1000]*3, [0]*3])
        self.assertAlmostEqual(integrate(native, forecast, 192)["energy_kWh"], .375)

    def test_invalid_native_data_is_rejected(self):
        native, forecast, _ = fixture()
        native.loc[1, "time"] = 1000
        with self.assertRaises(ValueError):
            integrate(native, forecast, 192)
        native.loc[1, "time"] = 450
        native.loc[1, POWER[0]] = np.nan
        with self.assertRaises(ValueError):
            integrate(native, forecast, 192)

    def test_kpis_must_match_independent_integration(self):
        native, forecast, _ = fixture()
        metrics = integrate(native, forecast, 192)
        kpis = {"ener_tot": metrics["boptest_energy_kWh_m2"], "tdis_tot": .75,
                "cost_tot": metrics["cost_EUR_m2"], "emis_tot": metrics["emissions_kg_m2"]}
        self.assertTrue(all(x == 0 for x in validate_kpis(metrics, kpis).values()))
        kpis["ener_tot"] *= 2
        with self.assertRaises(ValueError):
            validate_kpis(metrics, kpis)

    def test_transition_alignment_and_applied_action(self):
        native, forecast, records = fixture()
        frame = transitions_frame(records, native, forecast, CONFIG)
        self.assertAlmostEqual(frame.iloc[0].operative_start_C, 16.85)
        self.assertAlmostEqual(frame.iloc[0].interval_energy_kWh, .75)
        native.loc[1, "oveTSet_u"] = 300
        with self.assertRaises(ValueError):
            transitions_frame(records, native, forecast, CONFIG)
        records[0]["after"]["time"] = 899
        with self.assertRaises(ValueError):
            transitions_frame(records, native, forecast, CONFIG)

    def test_session_is_stopped_after_failure(self):
        class BrokenClient:
            def __init__(self):
                self.stopped = False

            def call(self, method, path, data=None):
                if path.endswith("/select"):
                    return {"testid": "abc"}
                if path == "stop/abc":
                    self.stopped = True
                    return None
                raise RuntimeError("bad metadata")

        client = BrokenClient()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                run_episode(client, CONFIG, "peak_heat_day", "rule", Path(folder) / "episode")
        self.assertTrue(client.stopped)


if __name__ == "__main__":
    unittest.main()
