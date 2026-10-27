"""Shared SB3 loop and action provenance; formal launch remains disabled."""

from collections import deque
import json
import time

import numpy as np
from stable_baselines3 import SAC as SB3SAC, PPO as SB3PPO, TD3 as SB3TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import torch

from .rl_comparison_protocol import parameters
from .sac_env import SACEnv
from .rl_interface import project_action


def append_json(path, record):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, allow_nan=False)+"\n")


class BoxSampling:
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # SB3 2.9.2's Box/no-SDE path, with the same calls and random draw order.
        if n_envs != 1 or self.use_sde:
            raise ValueError("Only the frozen one-env/no-SDE interface is supported")
        warmup = self.num_timesteps < learning_starts
        raw = np.array([self.action_space.sample()]) if warmup else self.predict(self._last_obs, deterministic=False)[0]
        scaled = self.policy.scale_action(raw)
        noise = action_noise() if action_noise is not None else np.zeros_like(scaled)
        proposal = scaled+noise
        buffer = np.clip(proposal, -1, 1) if action_noise is not None else scaled
        action = self.policy.unscale_action(buffer)
        self.sample_trace = {"source": "random_warmup" if warmup else "policy",
            "algorithm_raw": raw[0].tolist(), "exploration_noise": np.asarray(noise).reshape(-1).tolist(),
            "normalized_proposal": proposal[0].tolist(), "library_action": action[0].tolist(),
            "buffer_action": buffer[0].tolist()}
        return action, buffer


class TimedUpdate:
    def train(self, *args, **kwargs):
        self.update_in_progress = True
        started = time.perf_counter()
        try:
            result = super().train(*args, **kwargs)
        except Exception:
            self.update_failed = True
            raise
        finally:
            self.update_seconds = getattr(self, "update_seconds", 0.)+time.perf_counter()-started
            self.update_in_progress = False
        return result


class SAC(TimedUpdate, BoxSampling, SB3SAC):
    pass


class TD3(TimedUpdate, BoxSampling, SB3TD3):
    pass


class PPO(TimedUpdate, SB3PPO):
    pass


ALGORITHMS = {"SAC": SAC, "PPO": PPO, "TD3": TD3}


def build_model(name, env, seed, plan):
    params, policy = parameters(plan, name)
    policy["activation_fn"] = torch.nn.ReLU
    if name == "TD3":
        noise = params.pop("action_noise")
        params["action_noise"] = NormalActionNoise(np.array([noise["mean"]]), np.array([noise["sigma"]]))
    return ALGORITHMS[name]("MlpPolicy", env, seed=seed, device="cpu", policy_kwargs=policy, **params)


def seed_local(model, env, seed):
    env._np_random = np.random.default_rng(seed)
    env._np_random_seed = seed
    # Explicit local seeds only; never reseed global training streams per episode.
    for offset, space in enumerate([env.action_space, env.observation_space, model.action_space, model.observation_space]):
        space.seed(seed+offset+1)


class ComparisonEnv(SACEnv):
    def __init__(self, *args, algorithm, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm

    def step(self, action):
        _, projection = project_action(action, self.before["oveTSet_u"]-273.15)
        obs, reward, terminated, truncated, info = super().step(action)
        if info["control"]["mode"] == "sac":
            info["control"]["mode"] = self.algorithm.lower()
        info["control"]["actual_command_C"] = self.records[-1]["action"]["oveTSet_u"]-273.15
        info["control"]["actor_projection"] = projection
        info["control"]["measured_setpoint_C"] = self.records[-1]["after"]["oveTSet_u"]-273.15
        self.records[-1]["control"].update(info["control"])
        return obs, reward, terminated, truncated, info


class ActionJournal(BaseCallback):
    def __init__(self, directory, algorithm, budget_check):
        super().__init__()
        self.directory, self.algorithm, self.budget_check = directory, algorithm, budget_check
        self.rewards = deque(maxlen=96)
        self.rows, self.episode = 0, None

    def _on_step(self):
        self.budget_check()
        info = self.locals["infos"][0]
        if self.algorithm == "PPO":
            raw, applied = self.locals["actions"][0], self.locals["clipped_actions"][0]
            trace = {"source": "policy", "algorithm_raw": raw.tolist(),
                "library_action": applied.tolist(), "buffer_action": raw.tolist(),
                "log_probability": float(self.locals["log_probs"][0].item())}
        else:
            trace = dict(self.model.sample_trace)
            if not np.array_equal(self.locals["actions"][0], trace["library_action"]):
                raise ValueError("Sampling trace and library command differ")
        record = {"actor_step": self.num_timesteps, "episode": self.episode,
            "algorithm": self.algorithm, **trace, "control": info["control"],
            "terminated": bool(info["terminated"]), "truncated": bool(info["truncated"])}
        if not np.isfinite(np.asarray(record["algorithm_raw"])).all():
            raise ValueError("Nonfinite actor proposal")
        append_json(self.directory/"actions.jsonl", record)
        self.rows += 1
        self.rewards.append(float(self.locals["rewards"][0]))
        if self.num_timesteps % 96 == 0:
            append_json(self.directory/"curve.jsonl", {"actor_steps": self.num_timesteps,
                "mean_reward_last96": float(np.mean(self.rewards))})
        return True

    def state(self):
        return {"rows": self.rows, "rewards": list(self.rewards),
            "actions_bytes": (self.directory/"actions.jsonl").stat().st_size,
            "curve_bytes": (self.directory/"curve.jsonl").stat().st_size if (self.directory/"curve.jsonl").exists() else 0}


def optimizer_steps(model):
    return {key: sorted({int(value["step"].item()) for value in state["state"].values()})
        for key, state in model.get_parameters().items() if key.endswith("optimizer")}


def assert_boundary(model, algorithm, episodes):
    steps = episodes*664
    updates = 10*episodes if algorithm == "PPO" else max(0, steps-256)
    expected = ({"policy.optimizer": [80*episodes]} if algorithm == "PPO" else
        {"actor.optimizer": [updates if algorithm == "SAC" else updates//2], "critic.optimizer": [updates]})
    if algorithm == "SAC":
        expected["ent_coef_optimizer"] = [updates]
    if (model.num_timesteps != steps or model._n_updates != updates
            or getattr(model, "update_in_progress", False) or getattr(model, "update_failed", False)
            or optimizer_steps(model) != expected):
        raise ValueError("Actor/update/optimizer boundary is incomplete")
    if algorithm == "PPO" and (model.n_steps != 664 or not model.rollout_buffer.full
            or not model.rollout_buffer.generator_ready):
        raise ValueError("PPO rollout is incomplete or not consumed")
    return {"actor_steps": steps, "updates": updates, "optimizer_steps": expected, "episodes": episodes}


def train_episode(model, env, journal, completed_episodes, check_budget):
    if journal.algorithm == "PPO" and (model.n_steps != 664 or model.batch_size != 83 or model.n_epochs != 10):
        raise ValueError("PPO must use the frozen 664/83/10 rollout before any reset")
    before = model.num_timesteps
    model.set_env(env)
    journal.episode = env.spec["id"]
    started = time.perf_counter()
    model.learn(total_timesteps=664, reset_num_timesteps=False, callback=journal)
    if model.num_timesteps-before != 664 or not env.final_info["episode_complete"]:
        raise ValueError("Incomplete/overshot finite training episode")
    check_budget()  # Include the final PPO epochs / TD3/SAC update, not just env time.
    state = assert_boundary(model, journal.algorithm, completed_episodes)
    state.update(wall_seconds=time.perf_counter()-started, update_seconds=model.update_seconds)
    return state


def deterministic_action(model, algorithm, obs):
    # Record the Gaussian mean BEFORE PPO predict()'s Box clipping, without sampling.
    with torch.no_grad():
        model.policy.set_training_mode(False)
        if algorithm == "PPO":
            raw = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.mean.cpu().numpy()[0]
            return np.clip(raw, -1, 1), raw
        action, _ = model.predict(obs, deterministic=True)
        return action, action.copy()
