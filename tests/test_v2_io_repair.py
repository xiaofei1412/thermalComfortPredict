import ast
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from thermal_comfort.rl_budget import BudgetLedger as FrozenLedger, BudgetTransport
from thermal_comfort.rl_budget_safe import BudgetLedger, locked, read_snapshot, write_snapshot

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    import numpy as np
    from scripts.run_v2_safe import dispatch, register, seed, status
    from thermal_comfort.rl_failure_export import FirstGateRunner, export_fragment
    from thermal_comfort.rl_runner import ComparisonEnv
    from thermal_comfort.sac_env import CampaignClient
    from thermal_comfort.v2_protocol import ROOT, read
    from test_rl_interface import FakeClient, FakeRule


def plan_fixture():
    return {"algorithms": ["SAC"], "seeds": [11, 23, 37],
        "budget_per_algorithm_seed": {"select_max": 75, "initialize_max": 75, "advance_max": 50400, "wall_seconds_max": 10800},
        "budget_total_all_algorithms_seeds": {"select_max": 675, "initialize_max": 675, "advance_max": 453600, "stage_wall_seconds_max": 43200}}


class SafeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = plan_fixture()
        self.ledger = BudgetLedger(self.root, self.plan)
        self.ledger.create()
        self.key = self.ledger.start_seed("SAC", 11)

    def test_budget_mutation_AST_identical_except_private_read(self):
        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if node.attr == "_read_under_lock":
                    node.attr = "read"
                return self.generic_visit(node)
        for name in ["start_seed", "reserve", "finish"]:
            original = ast.parse(textwrap.dedent(inspect.getsource(getattr(FrozenLedger, name))))
            repaired = Normalize().visit(ast.parse(textwrap.dedent(inspect.getsource(getattr(BudgetLedger, name)))))
            self.assertEqual(ast.dump(original), ast.dump(repaired), name)

    def test_internal_mutations_never_reenter_public_read_lock(self):
        with patch.object(self.ledger, "read", side_effect=AssertionError("Nested public read")):
            self.ledger.start_seed("SAC", 23)
            self.ledger.reserve(self.key, "select")
            self.ledger.reserve(self.key, "advance")
            self.ledger.finish(self.key, "failed")

    def test_snapshot_has_no_handle_and_is_detached(self):
        snapshot = self.ledger.read()
        snapshot["counts"]["advance"] = -99
        self.ledger.reserve(self.key, "advance")
        self.assertEqual(self.ledger.read()["counts"]["advance"], 1)
        path = self.root/"progress.json"
        write_snapshot(path, {"counter": 1})
        value = read_snapshot(path)
        write_snapshot(path, {"counter": 2})  # Would fail on Windows if the prior read handle leaked.
        self.assertEqual(value, {"counter": 1})
        self.assertEqual(read_snapshot(path), {"counter": 2})

    def test_mutex_timeout_is_bounded_and_does_not_release_another_owner(self):
        path = self.root/"held.lock"
        with locked(path):
            with patch("thermal_comfort.rl_budget_safe.LOCK_WAIT_SECONDS", .01):
                with self.assertRaises(TimeoutError):
                    with locked(path):
                        self.fail("Bypassed existing owner")
        with locked(path):
            pass

    def test_parse_failure_closes_reader_before_releasing_mutex(self):
        path = self.root/"broken.json"
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            read_snapshot(path)
        write_snapshot(path, {"recovered_fixture": True})
        self.assertEqual(read_snapshot(path), {"recovered_fixture": True})

    def test_failed_transport_is_charged_once_and_stopped_seed_cannot_restart(self):
        class Broken:
            def call(self, *args):
                raise TimeoutError("unknown applied status")
        with self.assertRaises(TimeoutError):
            BudgetTransport(Broken(), self.ledger, self.key).call("POST", "advance/fixture", {})
        self.assertEqual(self.ledger.read()["counts"]["advance"], 1)
        self.ledger.finish(self.key, "failed")
        with self.assertRaises(RuntimeError):
            self.ledger.reserve(self.key, "advance")
        with self.assertRaises(ValueError):
            self.ledger.start_seed("SAC", 11)
        self.ledger.reserve(self.key, "stop")

    def test_original_clock_cap_and_protocol_drift_remain_enforced(self):
        state = self.ledger.read()
        state["seeds"][self.key]["counts"]["advance"] = 50400
        write_snapshot(self.ledger.path, state)
        with self.assertRaises(RuntimeError):
            self.ledger.reserve(self.key, "advance")
        self.plan["budget_per_algorithm_seed"]["advance_max"] += 1
        with self.assertRaises(ValueError):
            self.ledger.read()

    def test_original_clock_survives_wall_rollback_and_reconstruction(self):
        from types import SimpleNamespace
        original = self.ledger.read()["seeds"][self.key]["start"]
        clock = SimpleNamespace(time=lambda: original["wall"]-100,
            monotonic=lambda: original["monotonic"]+10801)
        fresh = BudgetLedger(self.root, self.plan, clock)
        with self.assertRaises(RuntimeError):
            fresh.reserve(self.key, "advance")
        state = fresh.read()["seeds"][self.key]
        self.assertEqual(state["start"], original)
        self.assertGreaterEqual(state["elapsed_seconds"], 10801)

    def test_persistence_failure_never_sends_or_retries_a_transport_call(self):
        from unittest.mock import Mock
        client = Mock()
        with patch("thermal_comfort.rl_budget_safe.atomic_json", side_effect=PermissionError("fixture denied")) as writer:
            with self.assertRaises(PermissionError):
                BudgetTransport(client, self.ledger, self.key).call("POST", "advance/fixture", {})
            self.assertEqual(writer.call_count, 1)
        client.call.assert_not_called()
        self.assertEqual(self.ledger.read()["counts"]["advance"], 0)


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class FragmentAndEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = read(ROOT/"configs/v2_comparison.json")

    def env(self):
        raw = FakeClient()
        adapter = {"train_ids": self.plan["train_ids"], "development_ids": self.plan["development_ids"],
            "budget_per_seed": self.plan["budget_per_algorithm_seed"],
            "fixed_observation_scaling": self.plan["shared"]["observation_scaling"],
            "reward": self.plan["shared"]["reward"], "guard": self.plan["shared"]["guard"]}
        spec = read(ROOT/"configs/v2_0.json")["episodes"][0]
        client = CampaignClient(raw, adapter, self.root)
        env = ComparisonEnv(FakeRule(), client, spec, "train", adapter, algorithm="SAC")
        env.reset(seed=11)
        env.step(np.array([0.], np.float32))
        self.addCleanup(env.close)
        return env, raw

    def test_partial_export_works_without_score_and_does_not_query_engine(self):
        env, raw = self.env()
        before = list(raw.log)
        summary = export_fragment(env, self.root/"fragment", 0.)
        self.assertFalse(summary["eligible_for_comparison_or_resume"])
        self.assertNotIn("score", summary["last_info"])
        self.assertEqual(summary["transitions"], 9)
        self.assertIsNone(summary["wall_seconds"])
        self.assertEqual(raw.log, before)

    def test_original_failure_preserved_without_reentering_first_gate(self):
        env, raw = self.env()
        class Ledger:
            def read(self):
                return {"seeds": {"SAC-11": {"status": "running"}}}
            def finish(self, key, state):
                self.finished = (key, state)
        runner = FirstGateRunner.__new__(FirstGateRunner)
        runner.root, runner.env, runner.ledger = self.root, env, Ledger()
        runner.key, runner.cycle, runner.folders, runner.model = "SAC-11", 0, [], None
        runner.first_checked = False
        runner.exporter = runner.export_with_gate
        before = len(raw.log)
        with patch("scripts.run_v2_formal.check_first", side_effect=AssertionError("No first gate on failure")):
            runner.fail(RuntimeError("original injected transport failure"))
        failure = read(self.root/"failure.json")
        self.assertEqual(failure["error"], "original injected transport failure")
        self.assertNotIn("partial_export_error", failure)
        self.assertFalse(runner.first_checked)
        self.assertEqual(raw.log[before:], [("stop", None)])

    def test_empty_fragment_needs_neither_cached_observation_nor_KPI(self):
        from types import SimpleNamespace
        env = SimpleNamespace(records=[], observations=[], spec={"id": "train014"}, role="train", final_info=None)
        summary = export_fragment(env, self.root/"empty", 0.)
        self.assertFalse(summary["last_observation_available"])
        self.assertEqual(summary["transitions"], 0)
        with np.load(self.root/"empty/actor_observations.npz") as values:
            self.assertEqual(values["observations"].shape, (0, 503))
            self.assertNotIn("terminal_observation", values)

    def test_complete_path_still_uses_the_frozen_first_gate(self):
        from types import SimpleNamespace
        runner = FirstGateRunner.__new__(FirstGateRunner)
        runner.root, runner.first_checked, runner.key = self.root, False, "SAC-11"
        runner.ledger, runner.model = object(), object()
        summary = {"outcome": {"score": {"metrics": {"energy_kWh": 1.}}}}
        with patch("scripts.run_v2_formal.export", return_value=summary), patch(
                "scripts.run_v2_formal.check_first", return_value={"status": "passed"}) as gate, patch(
                "scripts.run_v2_formal.pd.read_csv", return_value=None), patch(
                "scripts.run_v2_formal.native_score", return_value={"energy_kWh": 1.}):
            self.assertIs(runner.export_with_gate(SimpleNamespace(), self.root/"complete", 1.), summary)
        gate.assert_called_once()
        self.assertTrue(runner.first_checked)
        self.assertTrue((self.root/"first-gate.json").exists())

    def test_new_entry_registers_false_and_cannot_launch(self):
        path = self.root/"new"
        register(path)
        before = (path/"budget.json").read_bytes()
        with patch("scripts.run_v2_safe.Client", side_effect=AssertionError("No Client")), patch(
                "scripts.run_v2_safe.subprocess.Popen", side_effect=AssertionError("No process")):
            with self.assertRaises(PermissionError):
                seed(path, "SAC", 11, 8083)
            with self.assertRaises(PermissionError):
                dispatch(path)
        self.assertFalse(status(path)["formal_training_authorized"])
        self.assertEqual(before, (path/"budget.json").read_bytes())

    def test_old_closed_trial_rejected_without_creating_sidecars(self):
        path = self.root/"old"
        path.mkdir()
        (path/"registration.json").write_text(json.dumps({"formal_training_authorized": False}))
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        for method in [dispatch, status]:
            with self.assertRaises(PermissionError):
                method(path)
        with self.assertRaises(FileExistsError):
            register(path)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_dispatch_snapshot_wiring_with_fake_processes_only(self):
        path = self.root/"dispatch-fixture"
        register(path)
        registration = read(path/"registration.json")
        ledger, handles = BudgetLedger(path, self.plan), []
        class Process:
            def __init__(self, args, stdout, **kwargs):
                algorithm, number = args[args.index("--algorithm")+1], int(args[args.index("--seed")+1])
                key = ledger.start_seed(algorithm, number)
                ledger.finish(key, "failed")
                handles.append(stdout)
                self.pid, self.returncode = len(handles), 1
            def poll(self):
                return self.returncode
        with patch("scripts.run_v2_safe.registration_for", return_value=registration), patch(
                "scripts.run_v2_safe.subprocess.Popen", Process):
            dispatch(path)
        self.assertEqual(len(handles), 9)
        self.assertTrue(all(handle.closed for handle in handles))
        self.assertEqual(ledger.read()["counts"]["advance"], 0)
        self.assertEqual(read_snapshot(path/"dispatch-result.json")["status"], "finished_no_retries")
        self.assertFalse(read(path/"registration.json")["formal_training_authorized"])

    def test_dispatch_failure_requests_stop_and_closes_parent_log_handles(self):
        path = self.root/"dispatch-failure-fixture"
        register(path)
        registration = read(path/"registration.json")
        ledger, handles = BudgetLedger(path, self.plan), []
        class Process:
            pid = 123
            def __init__(self, args, stdout, **kwargs):
                ledger.start_seed("SAC", 11)
                handles.append(stdout)
            def poll(self):
                return None
        def broken_progress(*args):
            raise OSError("injected progress store failure")
        with patch("scripts.run_v2_safe.registration_for", return_value=registration), patch(
                "scripts.run_v2_safe.subprocess.Popen", Process), patch("scripts.run_v2_safe.write_snapshot", broken_progress):
            with self.assertRaises(OSError):
                dispatch(path)
        self.assertTrue(all(handle.closed for handle in handles))
        self.assertEqual(ledger.read()["seeds"]["SAC-11"]["status"], "failed")
        self.assertEqual(ledger.read()["counts"]["advance"], 0)


if __name__ == "__main__":
    unittest.main()
