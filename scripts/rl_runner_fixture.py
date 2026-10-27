"""Synthetic, socket-free fixture. Never use these rows as building evidence."""

import gzip
import json

import gymnasium as gym
import numpy as np
import pandas as pd

from scripts.run_sac import outcome_for_json

from thermal_comfort.rl_interface import native_score, project_action
from thermal_comfort.sac_env import reward_value
from thermal_comfort.simulation import FORECAST, POWER, digest, validate_kpis
from thermal_comfort.simulation import save_json


class FixtureEnv(gym.Env):
    def __init__(self, spec, role, ledger, key, shared):
        self.spec, self.role, self.ledger, self.key, self.shared = spec, role, ledger, key, shared
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (503,), np.float32)
        self.index, self.command, self.completed, self.testid = 0, 21., False, None
        self.records, self.observations = [], []

    def observation(self):
        obs = np.zeros(503, np.float32)
        obs[:3] = [self.command/30, (672-self.index)/672, self.index/672]
        return obs

    def reset(self, seed=None, options=None):
        if self.completed:
            return self.cached_obs.copy(), {"budget_terminal_placeholder": True}
        if self.testid is not None:
            raise AssertionError("Extra physical reset")
        super().reset(seed=seed)
        self.ledger.reserve(self.key, "select")
        self.ledger.reserve(self.key, "initialize")
        self.testid = "synthetic"
        for index in range(8):
            self.ledger.reserve(self.key, "advance")
            self.records.append({"index": index, "command_C": self.command, "prefix": True})
        self.index = 8
        return self.observation(), {"prefix_steps": 8}

    def step(self, action):
        if self.completed or self.testid is None:
            raise AssertionError("Overstep after finite horizon")
        self.observations.append(self.observation())
        proposal, detail = project_action(action, self.command)
        fallback = self.index == 8
        # Deliberate fixture takeover to test separation of proposed/actual action.
        actual, control = project_action(np.array([-.5]) if fallback else np.asarray(action), self.command)
        previous = self.command
        self.command = actual["oveTSet_u"]-273.15
        self.ledger.reserve(self.key, "advance")
        self.records.append({"index": self.index, "command_C": self.command, "prefix": False})
        self.index += 1
        self.completed = self.index == 672
        if self.completed:
            self.testid = None
        control.update(fallback=fallback, actual_command_C=self.command, actor_target_C=proposal["oveTSet_u"]-273.15,
            actor_projection=detail, reason="synthetic_takeover" if fallback else "actor")
        metrics = {"energy_kWh": .075, "cold_Kh": 0., "hot_Kh": 0.}
        reward, _ = reward_value(metrics, self.command-previous, fallback, False, 672-self.index, self.shared)
        info = {"control": control, "terminated": self.completed, "truncated": False,
            "aborted": False, "episode_complete": self.completed}
        self.cached_obs = self.observation()
        self.final_info = info
        return self.cached_obs.copy(), reward, self.completed, False, info

    def close(self):
        self.ledger.reserve(self.key, "stop")
        self.testid = None


def export_fixture(env, folder, wall_seconds):
    folder.mkdir(exist_ok=False)
    start = env.spec["start_day"]*86400
    times = start+np.arange(20161)*30
    commands = np.array([r["command_C"]+273.15 for r in env.records])
    actual = np.r_[commands[0], np.repeat(commands, 30)]
    native = pd.DataFrame({"time": times, "reaTZon_y": 294.15, "oveTSet_u": actual,
        "oveTSet_activate": 1, "reaQFloHea_y": 2000., "reaTRet_y": 300., "reaTSup_y": 305.,
        "reaTSetHea_y": 293.15, "reaTSetCoo_y": 303.15, **{p: 100. for p in POWER}})
    schedule = pd.DataFrame({"time": times, **{key: value for key, value in zip(FORECAST, [293.15, 303.15, 1., 280., 50., .2, .1])}})
    metrics = native_score(native, schedule)
    kpis = {"ener_tot": metrics["boptest_energy_kWh_m2"], "tdis_tot": 0.,
        "cost_tot": metrics["cost_EUR_m2"], "emis_tot": metrics["emissions_kg_m2"]}
    for name, frame in [("native", native), ("schedule", schedule)]:
        frame.to_csv(folder/(name+".csv.gz"), index=False)
    with gzip.open(folder/"steps.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in env.records:
            stream.write(json.dumps(record)+"\n")
    constants = {key: [value]*1441 for key, value in zip(FORECAST, [293.15, 303.15, 1., 280., 50., .2, .1])}
    suffix = json.dumps(constants)[1:]
    with gzip.open(folder/"forecasts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(672):
            stream.write('{"time": '+json.dumps((start+index*900+np.arange(1441)*30).tolist())+", "+suffix+"\n")
    with gzip.open(folder/"terminal_forecast.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"time": (start+604800+np.arange(1441)*30).tolist(), **constants}, stream)
    np.savez_compressed(folder/"actor_observations.npz", observations=np.asarray(env.observations), terminal_observation=env.cached_obs)
    outcome = {**outcome_for_json(env.final_info), "score": {"metrics": metrics, "kpis": kpis,
        "kpi_absolute_difference": validate_kpis(metrics, kpis), "transitions": 672, "native_samples": 20161}}
    summary = {"synthetic_fixture_not_research": True, "transitions": 672, "actor_transitions": 664,
        "outcome": outcome, "wall_seconds": wall_seconds,
        "sha256": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}}
    save_json(folder/"summary.json", summary)
    return summary
