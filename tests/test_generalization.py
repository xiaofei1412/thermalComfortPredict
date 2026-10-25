from pathlib import Path
import tempfile
import unittest

import pandas as pd

from thermal_comfort.generalization import canonical, check_freeze, integrate, interface
from thermal_comfort.simulation import digest, save_json
from scripts.verify_generalization import acceptance, csv


class GeneralizationTests(unittest.TestCase):
    def test_office_alias_preserves_air_temperature_and_gas_heat_semantics(self):
        io = interface({"interface":"office"})
        raw = {"zon_reaTRooAir_y":295.,"con_oveTSetHea_u":294.,"fcu_reaPHea_y":1000.}
        actual = canonical(raw,io)
        self.assertEqual(actual["reaQFloHea_y"],900.)
        self.assertEqual(actual["reaTZon_y"],295.)
        self.assertEqual(io["temperature_kind"],"air")
        self.assertNotIn("reaTZon_y",raw)

    def test_electricity_and_gas_are_separate_and_temperature_components_do_not_cancel(self):
        io = interface({"interface":"office"})
        native = pd.DataFrame({"time":[0.,1800.,3600.],"zon_reaTRooAir_y":[292.,296.,300.],
            "fcu_reaPHea_y":[2000.]*3,"fcu_reaPCoo_y":[500.]*3,"fcu_reaPFan_y":[100.]*3})
        forecast = pd.DataFrame({"time":[0.,3600.],"LowerSetp[1]":[293.]*2,"UpperSetp[1]":[299.]*2})
        metrics = integrate(native,forecast,io,48)
        self.assertAlmostEqual(metrics["electricity_kWh"],.6)
        self.assertAlmostEqual(metrics["gas_kWh"],2.)
        self.assertAlmostEqual(metrics["total_energy_kWh"],2.6)
        self.assertAlmostEqual(metrics["cold_Kh"],.25)
        self.assertAlmostEqual(metrics["hot_Kh"],.25)
        self.assertAlmostEqual(metrics["discomfort_Kh"],.5)

    def test_invalid_native_time_and_unknown_interface_rejected(self):
        with self.assertRaises(ValueError):
            interface({"interface":"unknown"})
        with self.assertRaises(ValueError):
            integrate(pd.DataFrame({"time":[0.,0.]}),None,{},1)

    def test_freeze_rejects_changed_model_config_source_and_fmu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["model.joblib","source.py","case.fmu"]:
                (root/name).write_text("unchanged")
            save_json(root/"policy_config.json",{"horizon":24})
            frozen = {"model_sha256":digest(root/"model.joblib"),"policy_config":{"horizon":24},
                "files_sha256":{str(root/"source.py"):digest(root/"source.py")},
                "fmus":{"case":{"path":str(root/"case.fmu"),"sha256":digest(root/"case.fmu")}}}
            save_json(root/"freeze.json",frozen)
            check_freeze(root)
            for name in ["model.joblib","source.py","case.fmu","policy_config.json"]:
                original = (root/name).read_bytes()
                (root/name).write_text('{}' if name.endswith("json") else "changed")
                with self.assertRaises(ValueError):
                    check_freeze(root)
                (root/name).write_bytes(original)

    def test_acceptance_rejects_comfort_tradeoff_and_gas_shifting(self):
        rule = {"electricity_kWh":10.,"gas_kWh":10.,"total_energy_kWh":20.,
            "discomfort_Kh":2.,"cold_Kh":0.,"hot_Kh":2.,"outside_hours":1.,
            "maximum_cold_K":0.,"maximum_hot_K":2.}
        good = {**rule,"electricity_kWh":9.,"total_energy_kWh":19.}
        self.assertTrue(acceptance(rule,good)["passed"])
        self.assertFalse(acceptance(rule,{**good,"cold_Kh":.1,"hot_Kh":1.9})["passed"])
        self.assertFalse(acceptance(rule,{**good,"gas_kWh":10.1})["passed"])
        self.assertFalse(acceptance(rule,good,complete=False)["passed"])

    def test_zero_electricity_baseline_is_not_a_percentage_saving(self):
        rule = {"electricity_kWh":0.,"gas_kWh":0.,"total_energy_kWh":0.,
            "discomfort_Kh":0.,"cold_Kh":0.,"hot_Kh":0.,"outside_hours":0.,
            "maximum_cold_K":0.,"maximum_hot_K":0.}
        result = acceptance(rule,rule)
        self.assertIsNone(result["energy_saved_percent"])
        self.assertFalse(result["passed"])

    def test_zero_only_night_prefix_uses_same_numeric_schema_as_full_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            full,short = Path(tmp)/"full.csv",Path(tmp)/"short.csv"
            full.write_text("time,solar\n0,0\n30,0\n60,1.25\n")
            short.write_text("time,solar\n0,0\n30,0\n")
            pd.testing.assert_frame_equal(csv(full).iloc[:2],csv(short),check_exact=True)


if __name__ == "__main__":
    unittest.main()
