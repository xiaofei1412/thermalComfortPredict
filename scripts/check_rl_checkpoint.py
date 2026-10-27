"""Read-only fresh-process probe of complete runner bundles; no learn/reset/step."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

from scripts.verify_rl_random_state import NoInteractionEnv, probes
from thermal_comfort.rl_budget import atomic_json
from thermal_comfort.rl_checkpoint import inspect_bundle
from thermal_comfort.v2_protocol import read


def probe(bundle, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pin = read(bundle.parent.parent/"checkpoint-manifests"/(bundle.name+".json"))["manifest_sha256"]
    env = NoInteractionEnv()
    model, _, _ = inspect_bundle(bundle, env, pin)
    atomic_json(output, probes(model, env))


def verify(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for algorithm in ["SAC", "PPO", "TD3"]:
        bundle = archive/algorithm/(algorithm+"-11")/"checkpoints/cycle-01"
        for index in range(2):
            subprocess.run([sys.executable, "-m", "scripts.check_rl_checkpoint", "probe",
                "--archive", str(bundle), "--output", str(output/f"{algorithm}-{index}.json")], check=True)
        expected, restored = [read(output/f"{algorithm}-{i}.json") for i in range(2)]
        if expected != restored:
            raise ValueError("Complete-bundle RNG differs across fresh processes")
        results[algorithm] = {"exact": True, "probes": list(expected)}
    report = {"status": "passed", "cases": results, "fresh_processes": 6,
        "optimizer_updates": 0, "FMU_calls": 0, "scope": "loaded complete bundles and RNG only; no training continuation"}
    atomic_json(output/"audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "probe"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch("socket.socket.connect", side_effect=AssertionError("No network authorized")):
        if args.mode == "verify":
            print(json.dumps(verify(args.archive, args.output), indent=2))
        else:
            probe(args.archive, args.output)
