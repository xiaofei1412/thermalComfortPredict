"""Explicit V2 SAC permissions, fixed unit scaling and accountable rule fallback."""

from collections import deque
import hashlib
import json
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .rl_interface import ResidentialEnv, PREFIX, project_action
from .simulation import FORECAST, POWER
from .v2_baselines import assert_same_values, stitch
from .v2_protocol import ROOT, read, verify_files


def load_plan():
    plan = read(ROOT/"configs/v2_sac.json")
    parent = read(ROOT/"configs/v2_0.json")
    if (not plan["training_authorized"] or plan["algorithms"] != ["SAC"]
            or plan["seeds"] != parent["learning"]["seed_ids"]
            or plan["train_ids"] != [e["id"] for e in parent["episodes"] if e["role"] == "train"]
            or plan["development_ids"] != [e["id"] for e in parent["episodes"] if e["role"] == "development"]):
        raise ValueError("SAC learning scope differs from its registered child protocol")
    for filename in plan["parent_locks"]:
        verify_files(ROOT, read(ROOT/filename)["files_sha256"])
    return plan


class CampaignClient:
    def __init__(self, client, plan, output, deadline=None):
        self.client, self.plan, self.output = client, plan, output
        self.counts = {"select": 0, "initialize": 0, "advance": 0}
        self.started = time.perf_counter()
        self.spec = None
        self.deadline = deadline
        self.owned, self.now = set(), {}

    def set_episode(self, spec, role):
        allowed = self.plan["train_ids"] if role == "train" else self.plan["development_ids"] if role == "development" else []
        expected = next((e for e in read(ROOT/"configs/v2_0.json")["episodes"] if e["id"] == spec["id"]), None)
        if spec["id"] not in allowed or expected != spec or spec["role"] != role:
            raise ValueError("Training/development/held-out permissions violated")
        if self.owned:
            raise ValueError("Previous worker must be released before another episode")
        self.spec = dict(spec)

    def call(self, method, path, data=None):
        kind, _, testid = path.partition("/")
        budget = self.plan["budget_per_seed"]
        if kind != "stop" and time.perf_counter()-self.started > budget["wall_seconds_max"]:
            raise RuntimeError("SAC seed wall budget exhausted")
        if kind != "stop" and self.deadline is not None and time.time() > self.deadline:
            raise RuntimeError("SAC campaign wall budget exhausted")
        if kind != "version" and self.spec is None:
            raise ValueError("An authorized episode must be selected first")
        if kind not in ["version", "testcases"] and testid not in self.owned:
            raise ValueError("Cannot access an unowned worker")
        if kind == "testcases" and (path != "testcases/bestest_hydronic_heat_pump/select" or method != "POST"):
            raise ValueError("Only the original residential case may be selected")
        if kind == "testcases" and self.owned:
            raise ValueError("One worker per environment must remain exclusive")
        start = self.spec["start_day"]*86400 if self.spec else None
        if kind == "initialize" and data != {"start_time": start, "warmup_period": 604800}:
            raise ValueError("Unexpected initialization/warmup")
        if kind == "forecast" and (testid not in self.now or data != {
                "point_names": FORECAST, "horizon": 43200, "interval": 30}):
            raise ValueError("Forecast permissions violated")
        if kind == "results" and (testid not in self.now or data["start_time"] < start
                or data["final_time"] > self.now[testid] or data["start_time"] > data["final_time"]):
            raise ValueError("Only completed observations may be queried")
        if kind == "advance" and (testid not in self.now or self.now[testid] >= start+604800):
            raise ValueError("Cannot advance an uninitialized/completed episode")
        counter = "select" if kind == "testcases" else kind
        if counter in self.counts:
            if self.counts[counter] >= budget[counter+"_max"]:
                raise RuntimeError("SAC interaction budget exhausted")
            self.counts[counter] += 1
        entry = {"path": path, "method": method, "request": data, "episode": self.spec,
                 "counts": dict(self.counts), "ok": False}
        with (self.output/"attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry)+"\n")
        started = time.perf_counter()
        try:
            result = self.client.call(method, path, data)
            entry["ok"] = True
            if kind == "testcases":
                self.owned.add(result["testid"])
            elif kind == "initialize":
                self.now[testid] = start
            elif kind == "advance":
                self.now[testid] = result["time"]
            elif kind == "stop":
                self.owned.remove(testid)
                self.now.pop(testid, None)
            return result
        finally:
            entry["seconds"] = time.perf_counter()-started
            with (self.output/"calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, allow_nan=False)+"\n")


def scaled_observation(raw, plan):
    c = plan["fixed_observation_scaling"]
    mask = raw["history_mask"]
    history = (raw["history"].astype(float)-c["history_offset"])/c["history_scale"]
    history[mask == 0] = 0
    parts = {"history": history, "history_mask": mask,
             "forecast": (raw["forecast"].astype(float)-c["forecast_offset"])/c["forecast_scale"]}
    result = np.concatenate([parts[key].ravel() for key in c["flatten_order"]]).astype(np.float32)
    if result.shape != (503,) or not np.isfinite(result).all():
        raise ValueError("Invalid scaled observation")
    return result


def reward_value(metrics, move, fallback, aborted, remaining, plan):
    c = plan["reward"]
    costs = {"energy": c["energy"]*metrics["energy_kWh"], "cold": c["cold_Kh"]*metrics["cold_Kh"],
        "hot": c["hot_Kh"]*metrics["hot_Kh"], "move": c["move_squared_K"]*move**2,
        "fallback": c["fallback"]*bool(fallback),
        "failure": c["abort_fixed"]+c["abort_per_remaining_step"]*remaining if aborted else 0.}
    return -c["scale"]*sum(costs.values()), costs


class SACEnv(ResidentialEnv):
    def __init__(self, policy, client, spec, role, plan, reference=None):
        # The old engineering environment remains frozen and restricted to train014.
        super().__init__(policy, client=client)
        client.set_episode(spec, role)
        self.spec, self.role, self.plan, self.reference = spec, role, plan, reference
        self.observation_space = spaces.Box(-np.inf, np.inf, (503,), np.float32)
        self.completed = False
        self.guard_mode = None
        self.final_info = None
        self.observations, self.step_seconds = [], []

    def observe(self):
        return scaled_observation(super().observe(), self.plan)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Episode dates/roles cannot be overridden")
        if self.completed:
            return self.cached_obs.copy(), {"budget_terminal_placeholder": True}
        gym.Env.reset(self, seed=seed)
        self.close()
        self.client.set_episode(self.spec, self.role)
        self.records, self.calls, self.history = [], [], deque(maxlen=16)
        self.observations, self.step_seconds = [], []
        self.index, self.done, self.guard_mode = 0, False, None
        try:
            if self.call("GET", "version")["version"] != "0.9.0":
                raise ValueError("BOPTEST version mismatch")
            self.testid = self.call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
            inputs = self.call("GET", f"inputs/{self.testid}")
            measurements = self.call("GET", f"measurements/{self.testid}")
            if inputs["oveTSet_u"]["Unit"] != "K" or measurements["reaTZon_y"]["Unit"] != "K":
                raise ValueError("Temperature units mismatch")
            if any(measurements[p]["Unit"] != "W" for p in [*POWER, "reaQFloHea_y"]):
                raise ValueError("Power units mismatch")
            self.call("PUT", f"scenario/{self.testid}", {"electricity_price": "constant"})
            self.before = self.call("PUT", f"initialize/{self.testid}", {
                "start_time": self.spec["start_day"]*86400, "warmup_period": 604800})
            if self.before["time"] != self.spec["start_day"]*86400:
                raise ValueError("Initialization time mismatch")
            self.policy.reset(self.before, inputs)
            self.call("PUT", f"step/{self.testid}", {"step": 900})
            if self.call("GET", f"step/{self.testid}") != 900:
                raise ValueError("Control step mismatch")
            self.append_history(self.before["oveTSet_u"]-273.15, 0)
            self.view = self.forecast()
            for _ in range(PREFIX):
                action, control = self.policy.decide(self.index, self.before, self.view)
                self.advance(action, {"common_prefix": True, "actor_transition": False, **control})
                if self.done:
                    raise ValueError("Temperature abort in common prefix")
            if self.reference is not None:
                previous = pd.read_csv(self.reference/"native.csv.gz", float_precision="round_trip").iloc[:241]
                current = stitch([r["native"] for r in self.records])
                assert_same_values(previous, current[previous.columns])
            self.cached_obs = self.observe()
            return self.cached_obs.copy(), {"episode": self.spec["id"], "role": self.role, "prefix_steps": PREFIX}
        except Exception:
            self.invalidate()
            raise

    def step(self, action):
        if self.completed or self.done:
            raise RuntimeError("Finite episode exhausted; no additional physical step is authorized")
        started = time.perf_counter()
        proposed, _ = project_action(action, self.before["oveTSet_u"]-273.15)
        previous = self.before["oveTSet_u"]
        temperature, g = self.before["reaTZon_y"]-273.15, self.plan["guard"]
        if self.guard_mode == "low" and temperature >= g["low_release_C"]:
            self.guard_mode = None
        if self.guard_mode == "high" and temperature <= g["high_release_C"]:
            self.guard_mode = None
        if temperature <= g["low_enter_C"]:
            self.guard_mode = "low"
        elif temperature >= g["high_enter_C"]:
            self.guard_mode = "high"
        shadow, shadow_info = self.policy.decide(self.index, self.before, self.view)
        selected = shadow if self.guard_mode else proposed
        applied_normalized = np.clip(np.array([(selected["oveTSet_u"]-previous)/.5], np.float64), -1, 1)
        observation_before = self.observe()
        obs, _, terminal, truncated, info = super().step(applied_normalized)
        move = self.records[-1]["action"]["oveTSet_u"]-previous
        reward, costs = reward_value(info["metrics"], move, self.guard_mode is not None,
                                    info["aborted"], 672-self.index, self.plan)
        info["control"].update(raw_actor_normalized=float(np.asarray(action)[0]),
            actor_target_C=proposed["oveTSet_u"]-273.15, fallback=self.guard_mode is not None,
            mode="fallback" if self.guard_mode else "sac",
            reason="engineering_"+self.guard_mode if self.guard_mode else "actor",
            shadow_reason=shadow_info.get("reason"))
        info["reward_costs"] = costs
        info.update(terminated=terminal, truncated=truncated)
        self.records[-1]["control"].update(info["control"])
        self.records[-1].update(reward=reward, reward_costs=costs)
        self.observations.append(observation_before)
        self.step_seconds.append(time.perf_counter()-started)
        self.cached_obs = obs.copy()
        self.completed = bool(terminal or truncated)
        self.final_info = info
        if (self.index-PREFIX) % 96 == 0 or self.completed:
            print(f"{self.spec['id']}: actor {self.index-PREFIX}/664, done={self.completed}", flush=True)
        return obs, float(reward), terminal, truncated, info


def weights_hash(model):
    digestor = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digestor.update(name.encode())
        digestor.update(tensor.detach().cpu().numpy().tobytes())
    if model.log_ent_coef is not None:
        digestor.update(model.log_ent_coef.detach().cpu().numpy().tobytes())
    return digestor.hexdigest()


def replay_hash(model):
    buffer = model.replay_buffer
    count = buffer.buffer_size if buffer.full else buffer.pos
    digestor = hashlib.sha256()
    for name in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
        digestor.update(name.encode())
        digestor.update(getattr(buffer, name)[:count].tobytes())
    return {"size": count, "sha256": digestor.hexdigest(),
        "terminal_transitions": int(buffer.dones[:count].sum()), "timeouts": int(buffer.timeouts[:count].sum())}
