import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from scripts.authorize_v2_safe import authorize, close, REPLICAS
    from scripts.audit_v2_safe_formal import pairs
    from thermal_comfort.rl_budget_safe import BudgetLedger
    from thermal_comfort.v2_protocol import ROOT, read
    from thermal_comfort.v2_protocol import verify_files


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class SafeFormalAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/"new"
        self.preflight = {"status": "passed", "replicas": REPLICAS, "FMU_select_or_initialize_calls": 0}
        self.asset_checks = []
        def verify(directory, files):
            # CI contains pinned source/reports, not private model/FMUs in outputs.
            if set(files) == {"outputs/solar-selected/model.joblib",
                    "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"}:
                self.asset_checks.append(files)
                return
            verify_files(directory, files)
        patcher = patch("scripts.authorize_v2_safe.verify_files", side_effect=verify)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_bad_preflight_does_not_create_directory(self):
        with self.assertRaises(ValueError):
            authorize(self.output, {**self.preflight, "FMU_select_or_initialize_calls": 1})
        self.assertFalse(self.output.exists())

    def test_fresh_authorization_keeps_original_false_and_zero_budget(self):
        authorize(self.output, self.preflight)
        self.assertEqual(len(self.asset_checks), 1)
        self.assertFalse(read(self.output/"registration-before-authorization.json")["formal_training_authorized"])
        registration = read(self.output/"registration.json")
        self.assertTrue(registration["formal_training_authorized"])
        self.assertFalse(registration["protocol"]["training_authorized"])
        self.assertEqual(len(registration["parent_locks_sha256"]), 9)
        self.assertEqual(read(self.output/"budget.json")["counts"], {"select": 0, "initialize": 0, "advance": 0})
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            authorize(self.output, self.preflight)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_close_requires_no_running_seed_and_cannot_reclose(self):
        authorize(self.output, self.preflight)
        ledger = BudgetLedger(self.output, read(self.output/"registration.json")["protocol"])
        key = ledger.start_seed("SAC", 11)
        with self.assertRaises(RuntimeError):
            close(self.output)
        ledger.finish(key, "failed")
        close(self.output)
        self.assertFalse(read(self.output/"registration.json")["formal_training_authorized"])
        self.assertTrue(read(self.output/"registration-authorized.json")["formal_training_authorized"])
        with self.assertRaises(PermissionError):
            close(self.output)

    def test_pairing_preserves_zero_load_no_percentage_and_strict_gate(self):
        baseline = read(ROOT/"reports/v2_2/summary.json")["runs"]["development189__guarded_rule"]
        trial = {"outcome": {"score": {"metrics": baseline["metrics"]}}, "end_state": baseline["end_state"],
            "fallback_fraction_excluding_prefix": 0.}
        result = pairs({"development": {"development189": trial}})["development189"]
        self.assertIsNone(result["comparisons"]["guarded_rule"]["saving_fraction"])
        self.assertTrue(result["comparisons"]["guarded_rule"]["thermal_nonincrease"])


if __name__ == "__main__":
    unittest.main()
