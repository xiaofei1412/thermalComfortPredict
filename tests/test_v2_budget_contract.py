import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from scripts.check_v2_4_budget import check
    from scripts.run_v2_residual import plan
    from thermal_comfort.rl_budget import BudgetTransport
    from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
    from thermal_comfort.rl_budget_contract import validate_budget, corrected_residual_plan


@unittest.skipIf(stable_baselines3 is None,"Install requirements/sac.txt")
class BudgetContractTests(unittest.TestCase):
    def test_actual_registered_fault_and_candidate_roundtrip(self):
        result = check()
        self.assertEqual(result["candidate_fixture_predebits"],{"select":1,"initialize":1,"advance":1})
        self.assertEqual(result["FMU_calls"],0)
        self.assertFalse(result["new_authorization_or_trial_created"])

    def test_failure_occurs_before_transport_client(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = BudgetLedger(Path(name),plan())
            ledger.create()
            key = ledger.start_seed("SAC",11)
            client = Mock()
            transport = BudgetTransport(client,ledger,key)
            with self.assertRaisesRegex(KeyError,"select_max"):
                transport.call("POST","testcases/bestest_hydronic_heat_pump/select")
            client.call.assert_not_called()
            self.assertEqual(ledger.read()["counts"]["select"],0)

    def test_candidate_keeps_limits_and_original_unchanged(self):
        original = plan()
        before = copy.deepcopy(original)
        candidate = corrected_residual_plan(original)
        self.assertEqual(original,before)
        self.assertEqual(candidate["budget_total_all_algorithms_seeds"],
            {"select_max":225,"initialize_max":225,"advance_max":151200,"stage_wall_seconds_max":18000})
        for key in original.keys()-{"budget_total_all_algorithms_seeds"}:
            self.assertEqual(candidate[key],original[key])

    def test_missing_ambiguous_and_wrong_total_rejected(self):
        for kind in ["select","initialize","advance"]:
            for mutation in ["missing","ambiguous","wrong"]:
                value = corrected_residual_plan(plan())
                total = value["budget_total_all_algorithms_seeds"]
                if mutation == "missing":
                    del total[kind+"_max"]
                elif mutation == "ambiguous":
                    total[kind] = total[kind+"_max"]
                else:
                    total[kind+"_max"] += 1
                with self.assertRaises(ValueError):
                    validate_budget(value)

    def test_candidate_enforces_seed_total_and_clock_caps_before_network(self):
        for level in ["seed","total","clock"]:
            with tempfile.TemporaryDirectory() as name:
                value = corrected_residual_plan(plan())
                ledger = BudgetLedger(Path(name),value)
                ledger.create()
                key = ledger.start_seed("SAC",11)
                client = Mock()
                transport = BudgetTransport(client,ledger,key)
                state = ledger.read()
                if level == "seed":
                    state["seeds"][key]["counts"]["select"] = value["budget_per_algorithm_seed"]["select_max"]
                elif level == "total":
                    state["counts"]["select"] = value["budget_total_all_algorithms_seeds"]["select_max"]
                write_snapshot(ledger.path,state)
                with patch.object(ledger,"elapsed",return_value=10801 if level == "clock" else 0):
                    with self.assertRaises(RuntimeError):
                        transport.call("POST","testcases/bestest_hydronic_heat_pump/select")
                client.call.assert_not_called()

    def test_candidate_transport_predebits_before_mock_service(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = BudgetLedger(Path(name),corrected_residual_plan(plan()))
            ledger.create()
            key = ledger.start_seed("SAC",11)
            def receipt(*args):
                self.assertEqual(ledger.read()["counts"]["select"],1)
                return {"testid":"fixture"}
            client = Mock()
            client.call.side_effect = receipt
            self.assertEqual(BudgetTransport(client,ledger,key).call("POST",
                "testcases/bestest_hydronic_heat_pump/select"),{"testid":"fixture"})
            client.call.assert_called_once()
