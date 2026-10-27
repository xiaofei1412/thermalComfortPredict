import copy
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

from thermal_comfort.rl_budget import BudgetLedger, BudgetTransport
from thermal_comfort.v2_protocol import ROOT, read

if stable_baselines3 is not None:
    from stable_baselines3 import SAC as StockSAC, TD3 as StockTD3
    import torch
    from scripts.rl_runner_fixture import FixtureEnv
    from scripts.run_v2_comparison import register, run_seed
    from thermal_comfort.rl_runner import ActionJournal, build_model, seed_local, train_episode


class Clock:
    def __init__(self):
        self.wall, self.mono = 1000., 50.
    def time(self):
        return self.wall
    def monotonic(self):
        return self.mono


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = read(ROOT/"configs/v2_comparison.json")
        self.clock = Clock()
        self.ledger = BudgetLedger(self.root, self.plan, self.clock)
        self.ledger.create()
        self.key = self.ledger.start_seed("SAC", 11)

    def test_failed_request_counts_once_and_cannot_refund(self):
        class Failure:
            def call(self, *args):
                raise TimeoutError("unknown applied state")
        client = BudgetTransport(Failure(), self.ledger, self.key)
        with self.assertRaises(TimeoutError):
            client.call("POST", "advance/owned", {})
        self.assertEqual(self.ledger.read()["counts"]["advance"], 1)
        self.ledger.finish(self.key, "failed")
        with self.assertRaises(RuntimeError):
            client.call("POST", "advance/owned", {})
        self.assertEqual(self.ledger.read()["counts"]["advance"], 1)

    def test_original_clock_survives_new_ledger_and_wall_rollback(self):
        self.clock.mono += 10801
        self.clock.wall -= 100
        other = BudgetLedger(self.root, self.plan, self.clock)
        with self.assertRaises(RuntimeError):
            other.reserve(self.key, "select")
        self.assertEqual(other.read()["counts"]["select"], 0)
        other.reserve(self.key, "stop")
        with self.assertRaises(ValueError):
            other.start_seed("SAC", 11)
        with self.assertRaises(FileExistsError):
            other.create()

    def test_stage_clock_starts_on_first_select_and_is_not_reset(self):
        self.assertIsNone(self.ledger.read()["stage_start"])
        self.ledger.reserve(self.key, "select")
        original = self.ledger.read()["stage_start"]
        self.clock.wall += 3
        self.clock.mono += 3
        self.ledger.reserve(self.key, "select")
        self.assertEqual(original, self.ledger.read()["stage_start"])

    def test_seed_and_global_cap_boundaries(self):
        state = self.ledger.read()
        state["seeds"][self.key]["counts"]["advance"] = 50399
        state["counts"]["advance"] = 453599
        from thermal_comfort.rl_budget import atomic_json
        atomic_json(self.ledger.path, state)
        self.ledger.reserve(self.key, "advance")
        with self.assertRaises(RuntimeError):
            self.ledger.reserve(self.key, "advance")
        self.assertEqual(self.ledger.read()["counts"]["advance"], 453600)

    def test_parallel_reservations_and_algorithm_order(self):
        keys = [self.key, self.ledger.start_seed("SAC", 23), self.ledger.start_seed("SAC", 37)]
        with self.assertRaises(ValueError):
            self.ledger.start_seed("PPO", 11)
        def reserve(key):
            for _ in range(20):
                self.ledger.reserve(key, "advance")
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(reserve, keys))
        self.assertEqual(self.ledger.read()["counts"]["advance"], 60)
        for key in keys:
            self.assertEqual(self.ledger.read()["seeds"][key]["counts"]["advance"], 20)

    def test_protocol_drift_and_monotonic_reset_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed["budget_per_algorithm_seed"]["advance_max"] += 1
        with self.assertRaises(ValueError):
            BudgetLedger(self.root, changed, self.clock).reserve(self.key)
        self.clock.mono -= 1
        with self.assertRaises(RuntimeError):
            self.ledger.reserve(self.key)

    def test_in_place_protocol_mutation_is_rejected(self):
        self.plan["budget_per_algorithm_seed"]["advance_max"] += 1
        with self.assertRaises(ValueError):
            self.ledger.reserve(self.key)

    def test_completion_cannot_hide_elapsed_result_export_cost(self):
        self.clock.mono += 10801
        with self.assertRaises(RuntimeError):
            self.ledger.finish(self.key, "completed")
        self.assertEqual(self.ledger.read()["seeds"][self.key]["status"], "failed")


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = read(ROOT/"configs/v2_comparison.json")
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def fixture(self, algorithm):
        plan = copy.deepcopy(self.plan)
        plan["algorithms"], plan["seeds"] = [algorithm], [11]  # owner registry fixture only
        ledger = BudgetLedger(self.root, plan)
        ledger.create()
        key = ledger.start_seed(algorithm, 11)
        spec = read(ROOT/"configs/v2_0.json")["episodes"][0]
        return FixtureEnv(spec, "train", ledger, key, self.plan["shared"]), ledger, key

    def test_formal_seed_rejected_before_client_or_budget_mutation(self):
        output = self.root/"registered"
        register(output)
        before = (output/"budget.json").read_bytes()
        with patch("scripts.run_v2_comparison.Client", side_effect=AssertionError("No client authorized")):
            with self.assertRaises(PermissionError):
                run_seed(output, "SAC", 11, "http://forbidden")
        self.assertEqual(before, (output/"budget.json").read_bytes())

    def test_retention_keeps_two_and_all_manifests_and_never_deletes_partial(self):
        from thermal_comfort.rl_checkpoint import retain_latest
        from thermal_comfort.rl_budget import atomic_json
        from thermal_comfort.simulation import digest
        snapshots, archive = self.root/"checkpoints", self.root/"checkpoint-manifests"
        snapshots.mkdir(); archive.mkdir()
        for cycle in range(1, 4):
            directory = snapshots/f"cycle-{cycle:02}"
            directory.mkdir()
            atomic_json(directory/"manifest.json", {"test_cycle": cycle})
            atomic_json(archive/(directory.name+".json"), {"manifest_sha256": digest(directory/"manifest.json")})
        (snapshots/"cycle-04.partial").mkdir()
        retain_latest(self.root)
        self.assertEqual(sorted(p.name for p in snapshots.iterdir()), ["cycle-02", "cycle-03", "cycle-04.partial"])
        self.assertEqual(len(list(archive.iterdir())), 3)
        (snapshots/"cycle-04").mkdir()
        (snapshots/"cycle-02/manifest.json").write_text("corrupted")
        with self.assertRaises(ValueError):
            retain_latest(self.root)
        self.assertTrue((snapshots/"cycle-02").exists())

    def test_two_PPO_664_rollouts_no_extra_reset_raw_action_and_log_prob(self):
        env, ledger, key = self.fixture("PPO")
        model = build_model("PPO", env, 11, self.plan)
        with torch.no_grad():
            model.policy.action_net.bias.fill_(4.)
        journal = ActionJournal(self.root, "PPO", lambda: ledger.reserve(key))
        for episode in range(2):
            if episode:
                env = FixtureEnv(env.spec, "train", ledger, key, self.plan["shared"])
            seed_local(model, env, 11+episode)
            state = train_episode(model, env, journal, episode+1, lambda: ledger.reserve(key))
            self.assertEqual(state["optimizer_steps"]["policy.optimizer"], [80*(episode+1)])
        self.assertEqual(model.num_timesteps, 1328)
        self.assertEqual(ledger.read()["counts"], {"select": 2, "initialize": 2, "advance": 1344})
        rows = [json.loads(line) for line in (self.root/"actions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1328)
        self.assertTrue(any(abs(r["algorithm_raw"][0]) > 1 for r in rows))
        for row in rows:
            self.assertEqual(row["buffer_action"], row["algorithm_raw"])
            self.assertLessEqual(abs(row["library_action"][0]), 1)
            self.assertTrue(np.isfinite(row["log_probability"]))
        self.assertTrue(rows[0]["control"]["fallback"])
        self.assertNotEqual(rows[0]["control"]["actor_target_C"], rows[0]["control"]["actual_command_C"])
        np.testing.assert_array_equal(model.rollout_buffer.actions[:, 0], [r["buffer_action"][0] for r in rows[-664:]])

    def test_off_policy_sampler_matches_stock_draws_and_rng_order(self):
        from stable_baselines3.common.noise import NormalActionNoise
        from scripts.verify_rl_random_state import NoInteractionEnv
        for name, cls in [("SAC", StockSAC), ("TD3", StockTD3)]:
            env = NoInteractionEnv()
            traced = build_model(name, env, 11, self.plan)
            stock = cls("MlpPolicy", env, seed=11, device="cpu", policy_kwargs={"net_arch": [64, 64]}, buffer_size=32)
            stock.set_parameters(traced.get_parameters())
            if name == "SAC":
                stock.log_ent_coef.data.copy_(traced.log_ent_coef.data)
            for steps in [0, 300]:
                outputs, states = [], []
                for model in [stock, traced]:
                    model.num_timesteps = steps
                    model._last_obs = np.zeros((1, 503), np.float32)
                    model.action_space.seed(77)
                    np.random.seed(78); torch.manual_seed(79)
                    noise = NormalActionNoise(np.zeros(1), .1*np.ones(1)) if name == "TD3" else None
                    outputs.append(model._sample_action(256, noise))
                    states.append((np.random.get_state()[1].copy(), torch.get_rng_state().clone()))
                for left, right in zip(outputs[0], outputs[1]):
                    np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(states[0][0], states[1][0])
                self.assertTrue(torch.equal(states[0][1], states[1][1]))

    def test_ppo_drift_rejected_before_sampling_and_incomplete_eval_forbidden(self):
        env, ledger, key = self.fixture("PPO")
        model = build_model("PPO", env, 11, self.plan)
        model.n_steps = 2048
        journal = ActionJournal(self.root, "PPO", lambda: ledger.reserve(key))
        with self.assertRaises(ValueError):
            train_episode(model, env, journal, 1, lambda: ledger.reserve(key))
        self.assertEqual(ledger.read()["counts"]["advance"], 0)

    def test_failure_stops_runner_and_preserves_attempts_no_retry(self):
        from thermal_comfort.rl_campaign import TrainingRunner
        plan = copy.deepcopy(self.plan)
        plan["algorithms"], plan["seeds"] = ["SAC"], [11]
        plan["budget_per_algorithm_seed"]["advance_max"] = 11
        ledger = BudgetLedger(self.root, plan)
        ledger.create()
        def factory(spec, role):
            return FixtureEnv(spec, role, ledger, "SAC-11", plan["shared"])
        runner = TrainingRunner(self.root/"SAC-11", "SAC", 11, plan, ledger, factory, {})
        with self.assertRaises(RuntimeError):
            runner.run_cycle()
        self.assertTrue((runner.root/"failure.json").exists())
        self.assertEqual(ledger.read()["seeds"]["SAC-11"]["status"], "failed")
        before = ledger.read()["counts"]
        with self.assertRaises(RuntimeError):
            runner.run_cycle()
        with self.assertRaises(RuntimeError):
            runner.evaluate_final()
        self.assertEqual(before, ledger.read()["counts"])

    def test_real_control_adapter_keeps_actor_projection_and_rule_takeover(self):
        from test_rl_interface import FakeClient, FakeRule
        from thermal_comfort.rl_runner import ComparisonEnv
        from thermal_comfort.sac_env import CampaignClient
        spec = read(ROOT/"configs/v2_0.json")["episodes"][0]
        adapter = {"train_ids": self.plan["train_ids"], "development_ids": self.plan["development_ids"],
            "budget_per_seed": self.plan["budget_per_algorithm_seed"],
            "fixed_observation_scaling": self.plan["shared"]["observation_scaling"],
            "reward": self.plan["shared"]["reward"], "guard": self.plan["shared"]["guard"]}
        for algorithm in ["SAC", "PPO", "TD3"]:
            raw = FakeClient()
            client = CampaignClient(raw, adapter, self.root)
            env = ComparisonEnv(FakeRule(), client, spec, "train", adapter, algorithm=algorithm)
            env.reset(seed=11)
            raw.temp, env.before["reaTZon_y"] = 291.85, 291.85
            _, _, _, _, info = env.step(np.array([1.], np.float32))
            self.assertTrue(info["control"]["fallback"])
            self.assertEqual(info["control"]["actor_projection"]["target_C"], 21.5)
            self.assertEqual(info["control"]["actual_command_C"], 21.)
            self.assertEqual(info["control"]["measured_setpoint_C"], 21.)
            env.close()


if __name__ == "__main__":
    unittest.main()
