"""Bounded, seeded excitation with a reactive temperature supervisor."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .simulation import Client, digest, run_episode, save_json, schedule_at


def validate_plan(config):
    episodes = config["episodes"]
    if len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Episode IDs must be unique")
    spans = sorted((e["start_day"] - config["warmup_days"],
                    e["start_day"] + config["days"], e["split"]) for e in episodes)
    if spans[0][0] < 0 or any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("Episode evaluation/warmup windows must be separated")
    if [s[2] for s in spans] != sorted([s[2] for s in spans], key={"train": 0, "validation": 1, "test": 2}.get):
        raise ValueError("Training must precede validation and test in model time")
    if set(e["split"] for e in episodes) != {"train", "validation", "test"}:
        raise ValueError("All three splits are required")


class SafeExcitation:
    def __init__(self, config, limits, initial):
        self.config = config["excitation"]
        self.limits = limits
        self.previous_C = initial["oveTSet_u"] - 273.15
        rng = np.random.default_rng(config["episode"]["seed"])
        n = int(config["days"] * 86400 / config["step_s"])
        self.levels = rng.choice(self.config["levels_K"], size=(n + self.config["hold_steps"] - 1) // self.config["hold_steps"])

    def check_temperature(self, temperature_C):
        c = self.config
        if not np.isfinite(temperature_C) or not c["abort_min_C"] <= temperature_C <= c["abort_max_C"]:
            raise ValueError(f"Excitation aborted: operative temperature {temperature_C} °C outside stop band")

    def action(self, index, before, current):
        c = self.config
        temperature = before["reaTZon_y"] - 273.15
        self.check_temperature(temperature)
        nominal = c["nominal_C"] + float(self.levels[index // c["hold_steps"]])
        lower, upper = current["LowerSetp[1]"] - 273.15, current["UpperSetp[1]"] - 273.15
        if temperature < max(lower + c["soft_margin_K"], c["abort_min_C"] + 1):
            mode, target = "recover_heat", c["action_max_C"]
        elif temperature > min(upper - c["soft_margin_K"], c["abort_max_C"] - 1):
            mode, target = "coast", c["action_min_C"]
        else:
            mode, target = "excite", nominal
        low = max(c["action_min_C"], self.limits["Minimum"] - 273.15)
        high = min(c["action_max_C"], self.limits["Maximum"] - 273.15)
        bounded = float(np.clip(target, low, high))
        requested = float(np.clip(bounded, self.previous_C - c["slew_K_per_step"],
                                  self.previous_C + c["slew_K_per_step"]))
        # Also covers the first step if the embedded initialization is outside our action band.
        if not low <= requested <= high:
            raise ValueError("Initial embedded setpoint is incompatible with excitation action/slew limits")
        info = {"mode": mode, "nominal_C": nominal, "target_C": target,
                "previous_C": self.previous_C, "requested_C": requested,
                "slew_limited": abs(requested - bounded) > 1e-10}
        self.previous_C = requested
        return {"oveTSet_u": requested + 273.15, "oveTSet_activate": 1}, info


def verify_excitation(directory, summary):
    config = summary["config"]
    records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    metadata = json.loads((directory / "api_metadata.json").read_text())
    forecast = pd.read_csv(directory / "exogenous.csv.gz")
    policy = SafeExcitation(config, metadata["inputs"]["oveTSet_u"], records[0]["before"])
    modes, slew_count = {}, 0
    for i, record in enumerate(records):
        current = schedule_at(forecast, np.array([record["before"]["time"]])).iloc[0]
        action, info = policy.action(i, record["before"], current)
        if action != record["action"] or info != record["safety"]:
            raise ValueError("Saved actions/supervisor decisions do not reproduce")
        modes[info["mode"]] = modes.get(info["mode"], 0) + 1
        slew_count += int(info["slew_limited"])
    native = pd.read_csv(directory / "native.csv.gz")
    temperature = native.reaTZon_y - 273.15
    for value in [temperature.min(), temperature.max()]:
        policy.check_temperature(value)
    return {"modes": modes, "slew_limited_steps": slew_count,
            "operative_min_C": float(temperature.min()), "operative_max_C": float(temperature.max()),
            "action_min_C": min(r["safety"]["requested_C"] for r in records),
            "action_max_C": max(r["safety"]["requested_C"] for r in records),
            "distinct_actions": len({round(r["safety"]["requested_C"], 6) for r in records}),
            "stop_band_violations": 0, "exact_policy_replay": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dynamics.json"))
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, default=Path("outputs/excitation"))
    parser.add_argument("--days", type=float, help="Smoke run only; effective plan is saved")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.days is not None:
        config["days"] = args.days
    validate_plan(config)
    client = Client(args.url)
    version = client.call("GET", "version")
    if version["version"] != config["version"]:
        raise ValueError("BOPTEST version mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", {"config": config, "version": version, "url": args.url,
              "code_sha256": {name: digest(Path(__file__).parent / name) for name in ["excitation.py", "simulation.py"]},
              "config_sha256": digest(args.config)})
    summaries, checks = [], []
    for episode in config["episodes"]:
        effective = {**config, "episode": episode}
        directory = args.output / (episode["id"] + "__excitation")
        summary = run_episode(client, effective, episode["id"], "excitation", directory)
        summaries.append(summary)
        checks.append({"episode": episode["id"], **verify_excitation(directory, summary)})
        save_json(args.output / "summary.json", summaries)
        save_json(args.output / "safety.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
