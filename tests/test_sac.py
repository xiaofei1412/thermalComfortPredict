import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env
    import torch
    from test_rl_interface import FakeClient, FakeRule
    from thermal_comfort.sac_env import (
        CampaignClient, SACEnv, replay_hash, reward_value, scaled_observation, weights_hash)
    from thermal_comfort.v2_protocol import ROOT, read


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt in the separate SAC environment")
class SACTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(ROOT/"configs/v2_sac.json")
        self.specs = {e["id"]: e for e in read(ROOT/"configs/v2_0.json")["episodes"]}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.raw = FakeClient()
        self.client = CampaignClient(self.raw, self.plan, self.folder)

    def env(self):
        env = SACEnv(FakeRule(), self.client, self.specs["train014"], "train", self.plan)
        self.addCleanup(env.close)
        return env

    def test_six_training_dates_allowed_without_development(self):
        for name in self.plan["train_ids"]:
            self.client.set_episode(self.specs[name], "train")
        self.assertEqual(self.raw.log, [])
        for name in self.plan["development_ids"]:
            with self.assertRaises(ValueError):
                self.client.set_episode(self.specs[name], "train")

    def test_eval_roles_and_all_holdouts_rejected_for_training(self):
        for name in self.plan["development_ids"]:
            self.client.set_episode(self.specs[name], "development")
        for name in ["holdout210", "holdout259", "holdout342", "historical238", "train014"]:
            with self.assertRaises(ValueError):
                self.client.set_episode(self.specs[name], "development")
        with self.assertRaises(ValueError):
            self.client.set_episode({"id": "train014", "role": "train", "start_day": 329}, "train")
        self.assertEqual(self.raw.log, [])

    def test_owned_worker_and_exact_date_and_forecast_permissions(self):
        self.client.set_episode(self.specs["train014"], "train")
        with self.assertRaises(ValueError):
            self.client.call("GET", "kpi/foreign")
        testid = self.client.call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
        with self.assertRaises(ValueError):
            self.client.call("PUT", f"initialize/{testid}", {"start_time": 35*86400, "warmup_period": 604800})
        with self.assertRaises(ValueError):
            self.client.call("POST", "testcases/bestest_hydronic_heat_pump/select")
        with self.assertRaises(ValueError):
            self.client.call("PUT", f"results/{testid}", {"start_time": 0, "final_time": 0})

    def test_failed_attempt_counted_no_retry(self):
        env = self.env()
        env.reset()
        self.plan["budget_per_seed"]["advance_max"] = 9
        self.raw.fail = "advance"
        with self.assertRaises(TimeoutError):
            env.step([0])
        self.assertEqual(self.client.counts["advance"], 9)
        self.assertEqual(len(env.records), 8)
        self.assertEqual(sum(k == "advance" for k, _ in self.raw.log), 9)

    def test_fixed_scaling_padding_and_no_running_stats(self):
        env = self.env()
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (503,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        np.testing.assert_equal(obs[343:343+7*9], 0)
        self.assertEqual(obs[-16:].sum(), 9)
        np.testing.assert_array_equal(env.observe(), env.observe())
        self.assertEqual(len(env.observations), 0)
        self.assertEqual(len(env.records), 8)

    def test_scaling_does_not_mutate_raw_and_masks_unobservable_padding(self):
        raw = {"history": np.full((16, 9), 100., np.float32), "history_mask": np.zeros(16, np.int8),
               "forecast": np.zeros((49, 7), np.float32)}
        before = copy.deepcopy(raw)
        obs = scaled_observation(raw, self.plan)
        np.testing.assert_equal(obs[343:487], 0)
        for key in raw:
            np.testing.assert_array_equal(raw[key], before[key])

    def test_reward_and_failure_are_explicit_not_score_substitution(self):
        metrics = {"energy_kWh": .1, "cold_Kh": .2, "hot_Kh": .3}
        value, costs = reward_value(metrics, .5, True, False, 0, self.plan)
        self.assertAlmostEqual(value, -1.01525)
        failed, extra = reward_value(metrics, .5, True, True, 10, self.plan)
        self.assertEqual(extra["failure"], 11000)
        self.assertLess(failed, value-1000)

    def test_guard_raw_action_and_actual_action_logged_with_hysteresis(self):
        env = self.env()
        env.reset()
        for temperature, expected in [(18.9, True), (19.3, True), (19.5, False)]:
            env.before["reaTZon_y"] = self.raw.temp = temperature+273.15
            _, _, _, _, info = env.step(np.array([-1], np.float32))
            self.assertEqual(info["control"]["fallback"], expected)
            self.assertEqual(env.records[-1]["control"]["fallback"], expected)
            self.assertEqual(env.records[-1]["control"]["raw_actor_normalized"], -1)
        self.assertEqual(len(env.observations), 3)

    def test_invalid_actor_input_has_no_physical_call(self):
        env = self.env()
        env.reset()
        before = len(self.raw.log)
        with self.assertRaises(ValueError):
            env.step([np.nan])
        self.assertEqual(len(self.raw.log), before)

    def test_terminal_autoreset_is_noop_and_future_steps_rejected(self):
        with patch("thermal_comfort.rl_interface.STEPS", 10):
            env = self.env()
            env.reset()
            env.step([0])
            terminal_obs, _, terminal, truncated, _ = env.step([0])
            self.assertTrue(terminal)
            self.assertFalse(truncated)
            before = dict(self.client.counts)
            obs, info = env.reset()
            np.testing.assert_array_equal(obs, terminal_obs)
            self.assertTrue(info["budget_terminal_placeholder"])
            self.assertEqual(self.client.counts, before)
            with self.assertRaises(RuntimeError):
                env.step([0])
            self.assertEqual(self.client.counts, {"select": 1, "initialize": 1, "advance": 10})

    def test_sb3_checker_on_fake_engine(self):
        check_env(self.env(), warn=True)

    def test_sac_fake_training_buffer_terminal_and_save_reload(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with patch("thermal_comfort.rl_interface.STEPS", 10):
            env = self.env()
            model = SAC("MlpPolicy", env, learning_starts=0, buffer_size=20, batch_size=2,
                        policy_kwargs={"net_arch": [8, 8]}, seed=11, device="cpu", verbose=0)
            model.learn(total_timesteps=2)
            state = replay_hash(model)
            self.assertEqual(state["size"], 2)
            self.assertEqual(state["terminal_transitions"], 1)
            self.assertEqual(state["timeouts"], 0)
            self.assertEqual(model.replay_buffer.next_observations[1, 0, 486], 0)
            self.assertGreater(model._n_updates, 0)
            self.assertEqual(self.client.counts["initialize"], 1)
            model.save(self.folder/"model.zip")
            model.save_replay_buffer(self.folder/"replay.pkl")
            loaded = SAC.load(self.folder/"model.zip", device="cpu")
            loaded.load_replay_buffer(self.folder/"replay.pkl")
            self.assertEqual(weights_hash(loaded), weights_hash(model))
            self.assertEqual(replay_hash(loaded), state)
            np.testing.assert_array_equal(loaded.predict(env.cached_obs, deterministic=True)[0],
                                          model.predict(env.cached_obs, deterministic=True)[0])

    def test_multiple_learn_chunks_keep_buffer_and_do_not_extra_initialize(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with patch("thermal_comfort.rl_interface.STEPS", 10):
            model = SAC("MlpPolicy", self.env(), learning_starts=0, buffer_size=20, batch_size=2,
                        policy_kwargs={"net_arch": [8]}, seed=23, device="cpu", verbose=0)
            model.learn(total_timesteps=2, reset_num_timesteps=False)
            model.set_env(self.env())
            model.learn(total_timesteps=2, reset_num_timesteps=False)
            self.assertEqual(model.num_timesteps, 4)
            self.assertEqual(replay_hash(model)["size"], 4)
            self.assertEqual(replay_hash(model)["terminal_transitions"], 2)
            self.assertEqual(self.client.counts, {"select": 2, "initialize": 2, "advance": 20})

    def test_export_after_sb3_mutates_terminal_info_is_json_safe(self):
        from scripts.run_sac import export
        with patch("thermal_comfort.rl_interface.STEPS", 10):
            env = self.env()
            env.reset()
            env.step([0])
            env.step([0])
            env.final_info["terminal_observation"] = env.cached_obs
            result = export(env, self.folder/"export", .1)
            json.dumps(result, allow_nan=False)
            self.assertNotIn("terminal_observation", result["outcome"])
            with np.load(self.folder/"export/actor_observations.npz") as observations:
                np.testing.assert_array_equal(observations["terminal_observation"], env.cached_obs)


if __name__ == "__main__":
    unittest.main()
