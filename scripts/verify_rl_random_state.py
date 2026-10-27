"""Verify RNG save/load across fresh processes, without learn, network or env steps."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.noise import NormalActionNoise
import torch

from thermal_comfort.rl_random_state import restore_rng, runtime, save_rng
from thermal_comfort.simulation import digest, save_json


class NoInteractionEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (503,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self._np_random = np.random.default_rng(91)
        self._np_random_seed = 91
        self.action_space.seed(92)
        self.observation_space.seed(93)

    def reset(self, **kwargs):
        raise AssertionError("No environment reset authorized")

    def step(self, action):
        raise AssertionError("No environment step authorized")


def probes(model, env):
    obs = np.linspace(-.2, .2, 503, dtype=np.float32)
    result = {"python": [random.random(), random.gauss(0, 1)],
        "numpy": np.random.randn(8).tolist(), "torch": torch.randn(8).tolist(),
        "environment": env.np_random.normal(size=8).tolist(),
        "env_action": [env.action_space.sample().tolist() for _ in range(8)],
        "model_action": [model.action_space.sample().tolist() for _ in range(8)],
        "env_observation": env.observation_space.sample().tolist(),
        "model_observation": model.observation_space.sample().tolist(),
        "policy": [model.predict(obs, deterministic=False)[0].tolist() for _ in range(8)],
        "target_noise": torch.randn((8, 1)).clamp(-.5, .5).tolist()}
    if getattr(model, "action_noise", None) is not None:
        result["exploration_noise"] = [model.action_noise().tolist() for _ in range(8)]
    if getattr(model, "replay_buffer", None) is not None:
        sample = model.replay_buffer.sample(16)
        result["replay_sample"] = sample.observations[:, :2].cpu().numpy().tolist()
    # Exercise the actual PPO buffer's permutation, without rollout collection/update.
    buffer = RolloutBuffer(8, env.observation_space, env.action_space, device="cpu")
    for i in range(8):
        buffer.add(np.full((1, 503), i, np.float32), np.zeros((1, 1)), np.zeros(1),
                   np.zeros(1), torch.zeros(1), torch.zeros(1))
    buffer.advantages[:] = 0
    buffer.returns[:] = 0
    result["ppo_minibatch_order"] = [batch.observations[:, 0].tolist() for batch in buffer.get(4)]
    return result


def worker(folder, restore):
    torch.set_num_threads(1)
    metadata = json.loads((folder/"case.json").read_text())
    env = NoInteractionEnv()
    model = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[metadata["algorithm"]].load(
        metadata["model"], env=env, device="cpu")
    if metadata["replay"]:
        model.load_replay_buffer(metadata["replay"])
    binding = metadata["bindings"]
    if digest(Path(metadata["model"])) != binding["model_sha256"]:
        raise ValueError("Model hash differs from RNG binding")
    if metadata["replay"] and digest(Path(metadata["replay"])) != binding["replay_sha256"]:
        raise ValueError("Replay hash differs from RNG binding")
    if restore:
        # Loading and arbitrary setup consume/reseed streams; restoration must be LAST.
        random.seed(999); np.random.seed(999); torch.manual_seed(999)
        random.random(); np.random.randn(7); torch.randn(7)
        env.action_space.sample(); model.action_space.sample()
        restore_rng(folder/"rng.pkl", model, env, binding, metadata["rng_sha256"])
    else:
        random.seed(71); np.random.seed(72); torch.manual_seed(73)
        model.observation_space.seed(94)
        model.action_space.seed(95)
        random.gauss(0, 1); np.random.randn()  # Include cached Gaussian values.
        model.action_space.sample(); env.np_random.normal()
        metadata["rng_sha256"] = save_rng(folder/"rng.pkl", model, env, binding)
        save_json(folder/"case.json", metadata)
    save_json(folder/("restored.json" if restore else "expected.json"), probes(model, env))


def verify(output, archive):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cases = []
    for seed in [11, 23, 37]:
        for model, replay in [("partial_checkpoint.zip", "partial_replay.pkl"), ("checkpoint.zip", "replay.pkl")]:
            cases.append((f"SAC-{seed}-"+model.split('.')[0], "SAC", archive/f"seed{seed}"/model, archive/f"seed{seed}"/replay))
    # Initialized PPO/TD3 verify library RNG paths without any training/optimizer step.
    for algorithm, cls in [("PPO", PPO), ("TD3", TD3)]:
        env = NoInteractionEnv()
        params = {"n_steps": 8, "batch_size": 4} if algorithm == "PPO" else {
            "buffer_size": 32, "action_noise": NormalActionNoise(np.zeros(1), .1*np.ones(1))}
        model = cls("MlpPolicy", env, seed=11, device="cpu", policy_kwargs={"net_arch": [8, 8]}, **params)
        model_path, replay_path = output/(algorithm+".zip"), None
        model.save(model_path)
        if algorithm == "TD3":
            for i in range(16):
                model.replay_buffer.add(np.full((1, 503), i, np.float32), np.full((1, 503), i+1, np.float32),
                    np.zeros((1, 1)), np.zeros(1), np.zeros(1), [{}])
            replay_path = output/"TD3.pkl"
            model.save_replay_buffer(replay_path)
        cases.append((algorithm+"-initialized-no-training", algorithm, model_path, replay_path))
    results = {}
    for name, algorithm, model, replay in cases:
        folder = output/name
        folder.mkdir()
        bindings = {"model_sha256": digest(model), "replay_sha256": digest(replay) if replay else None,
            "case": name, "purpose": "offline_random_stream_acceptance_not_campaign_resume"}
        save_json(folder/"case.json", {"algorithm": algorithm, "model": str(model.resolve()),
            "replay": str(replay.resolve()) if replay else None, "bindings": bindings})
        for mode in ["capture", "restore"]:
            subprocess.run([sys.executable, "-m", "scripts.verify_rl_random_state", mode, "--output", str(folder)], check=True)
        expected = json.loads((folder/"expected.json").read_text())
        restored = json.loads((folder/"restored.json").read_text())
        if expected != restored:
            raise ValueError("Fresh-process RNG mismatch: "+name)
        results[name] = {"all_probes_exact": True, "probes": list(expected), "model_sha256": bindings["model_sha256"],
            "replay_sha256": bindings["replay_sha256"], "probe_sha256": hashlib.sha256(
                json.dumps(expected, sort_keys=True).encode()).hexdigest()}
    report = {"status": "passed", "runtime": runtime(), "cases": results,
        "fresh_worker_processes": 2*len(cases), "research_training_steps": 0, "optimizer_updates": 0,
        "FMU_calls": 0, "heldout_reads_or_initializations": 0,
        "scope": "RNG streams and checkpoint loading only; not a proof of uninterrupted learning or FMU state recovery"}
    save_json(output/"audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "capture", "restore"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("outputs/v2-sac/live-registered"))
    args = parser.parse_args()
    with patch("socket.socket.connect", side_effect=AssertionError("Network prohibited in RNG acceptance")):
        if args.mode == "verify":
            print(json.dumps(verify(args.output, args.archive), indent=2))
        else:
            worker(args.output, args.mode == "restore")
