"""Compare restored SAC optimizer states with the exact serialized dictionaries."""

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.save_util import load_from_zip_file
import torch

from thermal_comfort.simulation import save_json


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def check(directory):
    torch.set_num_threads(1)
    reports = {}
    for seed in [11, 23, 37]:
        for name in ["partial_checkpoint.zip", "checkpoint.zip"]:
            path = directory/f"seed{seed}"/name
            _, serialized, _ = load_from_zip_file(path, device="cpu")
            model = SAC.load(path, device="cpu")
            restored = model.get_parameters()
            steps = {}
            for key in ["actor.optimizer", "critic.optimizer", "ent_coef_optimizer"]:
                if not equal(serialized[key], restored[key]):
                    raise ValueError(f"Optimizer restore mismatch: {seed}/{name}/{key}")
                steps[key] = sorted({int(v["step"].item()) for v in restored[key]["state"].values()})
            reports[f"seed{seed}/{name}"] = {"steps": model.num_timesteps, "updates": model._n_updates,
                "optimizer_steps": steps, "serialized_and_restored_states_equal": True}
    return {"status": "passed", "checkpoints": reports, "additional_training_or_FMU_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, check(args.directory))
