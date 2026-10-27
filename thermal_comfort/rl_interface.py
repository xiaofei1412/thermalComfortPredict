"""V2-1 engineering-only Gymnasium interface, restricted to train014."""

from collections import deque
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .simulation import Client, FORECAST, POINTS, POWER, integrate, validate_kpis


START = 14 * 86400
STEPS = 672
PREFIX = 8


def project_action(action, previous):
    value = np.asarray(action)
    if value.shape != (1,) or not np.isfinite(value).all() or abs(value[0]) > 1:
        raise ValueError("Expected one finite normalized action in [-1, 1]")
    target = previous + float(value[0]) * 0.5
    applied = float(np.clip(target, 18, 23.25))
    if abs(applied - previous) > 0.5 + 1e-7:
        raise ValueError("Previous actual setpoint is outside mechanical bounds")
    return {"oveTSet_u": applied + 273.15, "oveTSet_activate": 1}, {
        "raw_normalized": float(value[0]), "target_C": target,
        "applied_C": applied, "projected": abs(target - applied) > 1e-7,
        "fallback": False, "reason": "mechanical_projection" if target != applied else "actor"}


def bounded_forecast(payload, now):
    frame = pd.DataFrame(payload)
    if set(frame) != {"time", *FORECAST} or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Invalid forecast fields/values")
    expected = now + np.arange(1441) * 30
    if len(frame) != len(expected) or not np.array_equal(frame.time, expected):
        raise ValueError("Forecast must cover exactly current..+12h at 30s")
    return frame[["time", *FORECAST]]


def native_score(native, schedule):
    metrics = integrate(native, schedule, 192)
    cold = np.maximum(schedule["LowerSetp[1]"].to_numpy() - native.reaTZon_y, 0)
    hot = np.maximum(native.reaTZon_y - schedule["UpperSetp[1]"].to_numpy(), 0)
    for name, deviation in [("cold", cold), ("hot", hot)]:
        metrics[name + "_Kh"] = float(np.trapezoid(deviation, native.time) / 3600)
        metrics[name + "_duration_s"] = float(np.trapezoid((deviation > 0).astype(float), native.time))
        metrics[name + "_max_K"] = float(np.max(deviation))
    return metrics


class ResidentialEnv(gym.Env):
    """No training authorization; one worker per instance, no hidden-state observations."""

    metadata = {"render_modes": []}

    def __init__(self, policy, *, client=None, episode="train014", max_agent_steps=None):
        if episode != "train014":
            raise ValueError("V2-1 only authorizes train014; no other dates may be initialized")
        if max_agent_steps is not None and not 1 <= max_agent_steps <= STEPS - PREFIX:
            raise ValueError("Invalid engineering step cap")
        self.policy = policy
        self.client = client if client is not None else Client("http://127.0.0.1:8082")
        self.max_agent_steps = max_agent_steps
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = spaces.Dict({
            "history": spaces.Box(-np.inf, np.inf, (16, 9), np.float32),
            "history_mask": spaces.MultiBinary(16),
            "forecast": spaces.Box(-np.inf, np.inf, (49, 7), np.float32)})
        self.testid = None
        self.done = True
        self.records = []
        self.calls = []

    def call(self, method, path, data=None):
        started = time.perf_counter()
        try:
            result = self.client.call(method, path, data)
        except Exception:
            self.calls.append({"path": path.split("/")[0], "seconds": time.perf_counter()-started,
                               "ok": False})
            raise
        self.calls.append({"path": path.split("/")[0], "seconds": time.perf_counter()-started,
                           "ok": True})
        return result

    def forecast(self):
        return bounded_forecast(self.call("PUT", f"forecast/{self.testid}", {
            "point_names": FORECAST, "horizon": 43200, "interval": 30}), self.before["time"])

    def observe(self):
        history = np.zeros((16, 9), dtype=np.float32)
        mask = np.zeros(16, dtype=np.int8)
        history[-len(self.history):] = np.asarray(self.history, dtype=np.float32)
        mask[-len(self.history):] = 1
        return {"history": history, "history_mask": mask,
                "forecast": self.view.iloc[::30][FORECAST].to_numpy(dtype=np.float32).copy()}

    def append_history(self, previous, mean_heat):
        b = self.before
        angle = 2 * np.pi * (b["time"] % 86400) / 86400
        row = [b["reaTZon_y"]-273.15, b["oveTSet_u"]-273.15, previous,
               b["reaQFloHea_y"]/1000, sum(b[p] for p in POWER)/1000,
               mean_heat, np.sin(angle), np.cos(angle), STEPS-self.index]
        if not np.isfinite(row).all():
            raise ValueError("Invalid observable measurement")
        self.history.append(row)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("V2-1 has no reset date/scenario overrides")
        super().reset(seed=seed)
        self.close()
        self.records, self.calls, self.history = [], [], deque(maxlen=16)
        self.index = 0
        self.done = False
        started = time.perf_counter()
        try:
            if self.call("GET", "version")["version"] != "0.9.0":
                raise ValueError("Expected BOPTEST 0.9.0")
            self.testid = self.call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
            inputs = self.call("GET", f"inputs/{self.testid}")
            measurements = self.call("GET", f"measurements/{self.testid}")
            if inputs["oveTSet_u"]["Unit"] != "K" or measurements["reaTZon_y"]["Unit"] != "K":
                raise ValueError("Unexpected temperature units")
            if any(measurements[p]["Unit"] != "W" for p in [*POWER, "reaQFloHea_y"]):
                raise ValueError("Unexpected power units")
            self.call("PUT", f"scenario/{self.testid}", {"electricity_price": "constant"})
            init_started = time.perf_counter()
            self.before = self.call("PUT", f"initialize/{self.testid}", {
                "start_time": START, "warmup_period": 7*86400})
            initialize_s = time.perf_counter() - init_started
            if self.before["time"] != START:
                raise ValueError("Initialization timestamp differs")
            self.policy.reset(self.before, inputs)
            self.call("PUT", f"step/{self.testid}", {"step": 900})
            if self.call("GET", f"step/{self.testid}") != 900:
                raise ValueError("Unexpected control step")
            self.append_history(self.before["oveTSet_u"]-273.15, 0)
            self.view = self.forecast()
            for _ in range(PREFIX):
                action, control = self.policy.decide(self.index, self.before, self.view)
                self.advance(action, {"common_prefix": True, "actor_transition": False, **control})
                if self.done:
                    raise ValueError("Temperature abort during common prefix")
            self.reset_seconds = time.perf_counter() - started
            return self.observe(), {"episode": "train014", "common_prefix_steps": PREFIX,
                "initialize_seconds": initialize_s, "reset_seconds": self.reset_seconds,
                "prefix_energy_kWh": sum(r["metrics"]["energy_kWh"] for r in self.records)}
        except Exception:
            self.invalidate()
            raise

    def advance(self, action, control):
        started = time.perf_counter()
        before, view = self.before, self.view
        previous = before["oveTSet_u"] - 273.15
        applied = action["oveTSet_u"] - 273.15
        if (not 18-1e-7 <= applied <= 23.25+1e-7 or abs(applied-previous) > .5+1e-7
                or action["oveTSet_activate"] != 1):
            raise ValueError("Mechanical constraint violated before advance")
        after = self.call("POST", f"advance/{self.testid}", action)
        if after["time"] != before["time"] + 900:
            raise ValueError("Unexpected advance timestamp")
        native = pd.DataFrame(self.call("PUT", f"results/{self.testid}", {
            "point_names": POINTS, "start_time": before["time"], "final_time": after["time"]}))
        if not np.array_equal(native.time, before["time"]+np.arange(31)*30):
            raise ValueError("Missing or irregular 30s native interval")
        if (not np.allclose(native.oveTSet_u.iloc[1:], action["oveTSet_u"], atol=1e-7, rtol=0)
                or not (native.oveTSet_activate.iloc[1:] == 1).all()):
            raise ValueError("FMU did not apply the requested action")
        schedule = view.iloc[:31].reset_index(drop=True)
        metrics = native_score(native, schedule)
        abort = bool(((native.reaTZon_y-273.15 < 18) | (native.reaTZon_y-273.15 > 35)).any())
        if not abort:
            self.policy.update(before, after, native, view.iloc[0])
        self.index += 1
        self.before = after
        self.append_history(previous, float(np.trapezoid(native.reaQFloHea_y, native.time)/900000))
        self.done = abort or self.index == STEPS
        self.records.append({"index": self.index-1, "before": before, "action": action,
            "after": after, "control": control, "metrics": metrics, "aborted": abort,
            "native": native, "schedule": schedule, "forecast": view,
            "seconds": time.perf_counter()-started})
        self.view = self.forecast()
        return metrics, abort

    def step(self, action):
        if self.done or self.testid is None:
            raise RuntimeError("Call reset before stepping an ended/inactive episode")
        # Invalid actor input has no side effects and does not invalidate the worker.
        command, control = project_action(action, self.before["oveTSet_u"]-273.15)
        try:
            metrics, aborted = self.advance(command, {
                "common_prefix": False, "actor_transition": True, **control})
            terminated = aborted or self.index == STEPS
            truncated = bool(not terminated and self.max_agent_steps is not None
                             and self.index-PREFIX >= self.max_agent_steps)
            self.done = terminated or truncated
            obs = self.observe()
            info = {"metrics": metrics, "control": control, "aborted": aborted,
                    "episode_complete": self.index == STEPS and not aborted,
                    "end_reason": "temperature_abort" if aborted else (
                        "finite_horizon" if terminated else "external_cap" if truncated else None)}
            if self.done:
                info["score"] = self.score()
                self.close()
            return obs, -metrics["energy_kWh"], bool(terminated), truncated, info
        except Exception:
            self.invalidate()
            raise

    def score(self):
        native = pd.concat([r["native"].iloc[:-1] for r in self.records] +
                           [self.records[-1]["native"].iloc[-1:]], ignore_index=True)
        schedule = pd.concat([r["schedule"].iloc[:-1] for r in self.records] +
                             [self.records[-1]["schedule"].iloc[-1:]], ignore_index=True)
        metrics = native_score(native, schedule)
        kpis = self.call("GET", f"kpi/{self.testid}")
        return {"metrics": metrics, "kpis": kpis,
                "kpi_absolute_difference": validate_kpis(metrics, kpis),
                "transitions": len(self.records), "native_samples": len(native)}

    def invalidate(self):
        self.done = True
        self.close()

    def close(self):
        self.done = True
        if self.testid is not None:
            testid, self.testid = self.testid, None
            self.call("PUT", f"stop/{testid}")
