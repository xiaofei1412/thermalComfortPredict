import copy
import hashlib
import pickle
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    import torch
    from scripts.verify_rl_random_state import NoInteractionEnv, probes
    from thermal_comfort.rl_random_state import restore_rng, roots, save_rng


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class RandomStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"rng.pkl"
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.addCleanup(random.setstate, python)
        self.addCleanup(np.random.set_state, numpy)
        self.addCleanup(torch.set_rng_state, cpu)
        self.env = NoInteractionEnv()
        self.model = SAC("MlpPolicy", self.env, seed=11, buffer_size=32,
            policy_kwargs={"net_arch": [8]}, device="cpu")
        for i in range(8):
            self.model.replay_buffer.add(np.full((1, 503), i), np.full((1, 503), i+1),
                np.zeros((1, 1)), np.zeros(1), np.zeros(1), [{}])
        self.binding = {"model_sha256": "a"*64, "next_episode": "train035", "completed": 1}

    def save(self):
        return save_rng(self.path, self.model, self.env, self.binding)

    def test_all_global_local_policy_and_replay_streams_restored(self):
        random.gauss(0, 1); np.random.randn()
        saved = self.save()
        expected = probes(self.model, self.env)
        random.seed(999); np.random.seed(999); torch.manual_seed(999)
        self.env._np_random = np.random.default_rng(999)
        restore_rng(self.path, self.model, self.env, self.binding, saved)
        self.assertEqual(expected, probes(self.model, self.env))

    def test_original_model_to_loaded_model_preserves_shared_generator_aliases(self):
        model_path, replay_path = self.path.parent/"model.zip", self.path.parent/"replay.pkl"
        self.model.save(model_path)
        self.model.save_replay_buffer(replay_path)
        sha = self.save()
        expected = probes(self.model, self.env)
        env = NoInteractionEnv()
        loaded = SAC.load(model_path, env=env, device="cpu")
        loaded.load_replay_buffer(replay_path)
        self.assertIsNot(loaded.action_space, env.action_space)
        restore_rng(self.path, loaded, env, self.binding, sha)
        self.assertIs(loaded.action_space._np_random, env.action_space._np_random)
        self.assertEqual(expected, probes(loaded, env))

    def test_hash_rejection_does_not_change_random_streams(self):
        self.save()
        python, numpy, cpu = random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            restore_rng(self.path, self.model, self.env, self.binding, "0"*64)
        self.assertEqual(python, random.getstate())
        np.testing.assert_array_equal(numpy[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    def test_model_or_next_episode_binding_mismatch_rejected(self):
        sha = self.save()
        for key, value in [("model_sha256", "b"*64), ("next_episode", "holdout210")]:
            binding = {**self.binding, key: value}
            with self.assertRaisesRegex(ValueError, "bindings mismatch"):
                restore_rng(self.path, self.model, self.env, binding, sha)

    def test_runtime_version_mismatch_rejected(self):
        self.save()
        state = pickle.loads(self.path.read_bytes())
        state["runtime"]["numpy"] = "other"
        self.path.write_bytes(pickle.dumps(state))
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "runtime bindings mismatch"):
            restore_rng(self.path, self.model, self.env, self.binding, sha)

    def test_missing_file_and_overwrite_rejected(self):
        with self.assertRaises(FileNotFoundError):
            restore_rng(self.path, self.model, self.env, self.binding, "0"*64)
        self.save()
        with self.assertRaises(FileExistsError):
            self.save()

    def test_empty_binding_rejected(self):
        with self.assertRaises(ValueError):
            save_rng(self.path, self.model, self.env, {})

    def test_uninitialized_generators_rejected_without_silent_entropy(self):
        for obj in roots(self.model, self.env).values():
            obj._np_random = None
        self.env._np_random_seed = None
        with self.assertRaisesRegex(ValueError, "Seed all local"):
            self.save()
        self.assertTrue(all(obj._np_random is None for obj in roots(self.model, self.env).values()))
        self.assertIsNone(self.env._np_random_seed)
        self.assertFalse(self.path.exists())

    def test_unregistered_sde_and_stateful_noise_rejected(self):
        self.model.use_sde = True
        with self.assertRaises(ValueError):
            self.save()
        self.model.use_sde = False
        self.model.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(1), np.ones(1))
        with self.assertRaises(ValueError):
            self.save()


if __name__ == "__main__":
    unittest.main()
