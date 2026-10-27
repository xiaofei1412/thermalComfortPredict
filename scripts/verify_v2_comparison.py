"""Validate budget and construct all algorithms without training or env calls."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch

from scripts.verify_rl_random_state import NoInteractionEnv
from thermal_comfort.rl_comparison_protocol import parameters, validate
from thermal_comfort.simulation import save_json
from thermal_comfort.v2_protocol import ROOT, read


def verify():
    torch.set_num_threads(1)
    plan = read(ROOT/"configs/v2_comparison.json")
    report = validate(plan)
    report["algorithm_construction"] = {}
    for name, cls in [("SAC", SAC), ("PPO", PPO), ("TD3", TD3)]:
        params, policy = parameters(plan, name)
        policy["activation_fn"] = torch.nn.ReLU
        if name == "TD3":
            noise = params.pop("action_noise")
            params["action_noise"] = NormalActionNoise(np.array([noise["mean"]]), np.array([noise["sigma"]]))
        model = cls("MlpPolicy", NoInteractionEnv(), seed=11, device="cpu", policy_kwargs=policy, **params)
        if model.num_timesteps != 0 or model._n_updates != 0:
            raise ValueError("Protocol validation must not train")
        report["algorithm_construction"][name] = {"steps": 0, "updates": 0,
            "observation_shape": list(model.observation_space.shape), "action_shape": list(model.action_space.shape),
            "buffer_capacity": model.replay_buffer.buffer_size if name != "PPO" else model.rollout_buffer.buffer_size,
            "policy": str(model.policy)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch("socket.socket.connect", side_effect=AssertionError("No network authorized")):
        save_json(args.output, verify())
