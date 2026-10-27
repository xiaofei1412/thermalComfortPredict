"""CPU, single-environment RNG sidecar for trusted episode-boundary checkpoints."""

import copy
import hashlib
import pickle
import platform
import random
import sys

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.noise import NormalActionNoise
import torch


def runtime():
    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "gymnasium": gym.__version__, "sb3": stable_baselines3.__version__,
        "platform": platform.system(), "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def roots(model, env):
    if (model.__class__.__name__ not in ["SAC", "PPO", "TD3"] or model.device.type != "cpu"
            or model.n_envs != 1 or model.use_sde or torch.cuda.is_initialized()
            or torch.get_num_threads() != 1 or isinstance(env, gym.Wrapper)
            or not isinstance(env.action_space, gym.spaces.Box)
            or not isinstance(env.observation_space, gym.spaces.Box)
            or model.get_vec_normalize_env() is not None
            or type(getattr(model, "action_noise", None)) not in [type(None), NormalActionNoise]):
        raise ValueError("RNG sidecar only supports the registered CPU/one-env/Box/no-SDE configuration")
    return {"env": env, "env_action": env.action_space, "env_observation": env.observation_space,
        "model_action": model.action_space, "model_observation": model.observation_space}


def save_rng(path, model, env, bindings):
    """Call after model/replay export and before any next-episode sampling.

    Bindings identify the model/replay hashes, protocol, run and next episode.
    The returned sidecar hash belongs in the committed checkpoint manifest.
    """
    objects = roots(model, env)
    if not bindings:
        raise ValueError("Checkpoint bindings must be nonempty")
    if any(obj._np_random is None for obj in objects.values()):
        raise ValueError("Seed all local generators explicitly before checkpointing")
    state = {"schema": 1, "runtime": runtime(), "bindings": copy.deepcopy(bindings),
        "python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
        "generators": copy.deepcopy({name: obj._np_random for name, obj in objects.items()}),
        "env_seed": env._np_random_seed}
    # Exclusive creation: an incomplete sidecar is ineligible, never overwritten.
    with path.open("xb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_rng(path, model, env, bindings, expected_sha256):
    """Restore LAST, after model/replay loading and env construction/seeding.

    This restores random streams, not an FMU, rollout, callback or budget.
    The pickle and its separately pinned SHA must come from the trusted archive.
    """
    objects = roots(model, env)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("RNG sidecar hash mismatch")
    state = pickle.loads(raw)
    if (state["schema"] != 1 or state["runtime"] != runtime() or not bindings
            or state["bindings"] != bindings or state["generators"].keys() != objects.keys()):
        raise ValueError("RNG checkpoint/runtime bindings mismatch")
    generators = copy.deepcopy(state["generators"])
    for name, obj in objects.items():
        obj._np_random = generators[name]
    env._np_random_seed = state["env_seed"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
