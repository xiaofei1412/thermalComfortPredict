import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from scripts.run_v2_formal import authorize, check_first, seed
    from scripts.dispatch_v2_formal import preflight
    from thermal_comfort.rl_budget import atomic_json
    from thermal_comfort.v2_protocol import ROOT, read


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class FormalLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = SimpleNamespace(spec={"id": "train014"}, role="train", step_seconds=[.1]*664)
        self.summary = {"transitions": 672, "actor_transitions": 664, "outcome": {"score": {
            "native_samples": 20161, "kpis": {"ener_tot": 1.}, "kpi_absolute_difference": {"ener_tot": 0.}}}}
        self.budget = {"seeds": {"SAC-11": {"elapsed_seconds": 70.,
            "counts": {"select": 1, "initialize": 1, "advance": 672}}}}
        self.ledger = SimpleNamespace(reserve=lambda key: self.budget)
        self.model = SimpleNamespace(num_timesteps=664)

    def test_first_gate_includes_prefix_KPI_and_original_cost_no_new_calls(self):
        result = check_first(self.env, self.summary, self.ledger, "SAC-11", self.model, 65.)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["extra_FMU_calls"], 0)
        self.assertEqual(result["original_seed_elapsed_seconds"], 70.)

    def test_duplicate_or_incomplete_attempts_fail(self):
        self.budget["seeds"]["SAC-11"]["counts"]["initialize"] = 2
        with self.assertRaises(ValueError):
            check_first(self.env, self.summary, self.ledger, "SAC-11", self.model, 65.)

    def test_bad_or_nonfinite_KPI_fails(self):
        for difference in [1e-5, float("nan"), -1.]:
            summary = copy.deepcopy(self.summary)
            summary["outcome"]["score"]["kpi_absolute_difference"]["ener_tot"] = difference
            with self.assertRaises(ValueError):
                check_first(self.env, summary, self.ledger, "SAC-11", self.model, 65.)

    def test_unauthorized_seed_never_constructs_client(self):
        atomic_json(self.root/"registration.json", {"formal_training_authorized": False})
        atomic_json(self.root/"authorization.json", {})
        with patch("scripts.run_v2_formal.Client", side_effect=AssertionError("No client authorized")):
            with self.assertRaises(PermissionError):
                seed(self.root, "SAC", 11, 8083)

    def test_failed_preflight_cannot_register_or_authorize(self):
        with patch("scripts.run_v2_formal.verify_files"), patch("scripts.run_v2_formal.register") as register:
            with self.assertRaises(ValueError):
                authorize(self.root/"new", {"status": "failed"})
            register.assert_not_called()

    def test_preflight_checks_target_object_not_exclusivity_of_registry(self):
        baseline = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
        class FakeClient:
            def __init__(self, url):
                pass
            def call(self, method, endpoint):
                return {"version": "0.9.0"} if endpoint == "version" else [
                    {"testcaseid": "bestest_air"}, {"testcaseid": "bestest_hydronic_heat_pump"}]
        def command(arguments, **kwargs):
            if "inspect" in arguments:
                return json.dumps([{"Config": {"Image": "ghcr.io/ibpsa/boptest-worker:v0.9.0@sha256:08ae9e6f49ff05c38847174fb776e547888c962f2d0b283ae45ebc6f972f0bb7"},
                    "Image": "fixture", "State": {"StartedAt": "fixture"}}]).encode()
            if "exec" in arguments:
                return json.dumps({"sha256": baseline["fmu_sha256"], "bytes": 100}).encode()
            return "fixture-head" if kwargs.get("text") else b"fixture-head"
        with patch("scripts.dispatch_v2_formal.Client", FakeClient), patch("scripts.dispatch_v2_formal.subprocess.check_output", command):
            result = preflight(self.root/"preflight.json")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["FMU_select_or_initialize_calls"], 0)

    def test_isolated_reader_probe_documents_platform_limit_without_trial_calls(self):
        from scripts.diagnose_v2_formal_io import diagnose
        result = diagnose(self.root/"io-probe")
        self.assertEqual(result["new_training_or_FMU_calls"], 0)
        self.assertEqual(result["after_close_and_serialization"], {"counter": 3})
        if os.name == "nt":
            self.assertTrue(result["matches_live_Windows_failure_pattern"])
            self.assertEqual(result["after_held_reader_attempt"], {"counter": 0})


if __name__ == "__main__":
    unittest.main()
