"""Deterministic short-horizon shooting MPC with explicit guarded-rule fallback."""

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .greybox import initial_state, observe, rollout
from .simulation import Client, digest, run_episode, save_json, schedule_at


def exogenous(frame):
    angle = 2 * np.pi * frame["time"].to_numpy() / 86400
    return np.column_stack([frame["TDryBul"] - 273.15, frame["HGloHor"], frame["Occupancy[1]"],
                            np.sin(angle), np.cos(angle)])


def candidate_actions(previous, config):
    if config["horizon_steps"] % config["move_block_steps"] or max(abs(np.array(config["moves_K"]))) > config["slew_K_per_step"]:
        raise ValueError("Invalid move blocking or slew constraints")
    blocks = config["horizon_steps"] // config["move_block_steps"]
    moves = np.array(list(itertools.product(config["moves_K"], repeat=blocks)))
    values, current = [], np.full(len(moves), previous)
    for block in range(blocks):
        current = np.clip(current + moves[:, block], config["action_min_C"], config["action_max_C"])
        values.append(current.copy())
    return np.unique(np.repeat(np.column_stack(values), config["move_block_steps"], axis=1), axis=0)


def limited_command(target, previous, config):
    bounded = np.clip(target, config["action_min_C"], config["action_max_C"])
    value = float(np.clip(bounded, previous - config["slew_K_per_step"], previous + config["slew_K_per_step"]))
    if not config["action_min_C"] <= value <= config["action_max_C"]:
        raise ValueError("Previous actuator setting incompatible with command/slew limits")
    return value


class Controller:
    def __init__(self, model, config, kind):
        self.model, self.config, self.kind = model, config, kind
        self.horizon_steps = config["horizon_steps"]

    def reset(self, initial, inputs):
        self.previous = initial["oveTSet_u"] - 273.15
        self.q = initial["reaQFloHea_y"] / 1000
        temperature = initial["reaTZon_y"] - 273.15
        self.state = initial_state(self.model, temperature, self.q)
        self.error = self.previous - temperature
        if (self.config["action_min_C"] + 273.15 < inputs["oveTSet_u"]["Minimum"] or
                self.config["action_max_C"] + 273.15 > inputs["oveTSet_u"]["Maximum"]):
            raise ValueError("Controller bounds exceed model input bounds")

    def check_temperature(self, temperature):
        if not np.isfinite(temperature) or not self.config["abort_min_C"] <= temperature <= self.config["abort_max_C"]:
            raise ValueError(f"Control aborted: operative temperature {temperature} °C outside stop band")

    def guarded_rule(self, before, current):
        c = self.config
        temperature = before["reaTZon_y"] - 273.15
        self.check_temperature(temperature)
        lower, upper = current["LowerSetp[1]"] - 273.15, current["UpperSetp[1]"] - 273.15
        if temperature < max(lower + c["guard_margin_K"], c["abort_min_C"] + 1):
            return c["action_max_C"], "recover_heat"
        if temperature > min(upper - c["guard_margin_K"], c["abort_max_C"] - 1):
            return c["action_min_C"], "coast"
        offset = c["rule_occupied_offset_K"] if current["Occupancy[1]"] > 0 else c["rule_unoccupied_offset_K"]
        return lower + offset, "rule"

    def decide(self, index, before, forecast):
        current = schedule_at(forecast, np.array([before["time"]])).iloc[0]
        target, guard = self.guarded_rule(before, current)
        info = {"mode": guard if guard != "rule" else "guarded_rule", "reason": guard}
        if self.kind == "mpc" and guard == "rule":
            if index < self.config["initial_rule_steps"]:
                info["reason"] = "observer_initialization"
            else:
                try:
                    target, extra = self.plan(before["time"], forecast, target)
                    info.update(extra)
                except (ValueError, FloatingPointError) as error:
                    info.update({"mode": "fallback", "reason": "invalid_model_prediction", "detail": str(error)})
        elif self.kind == "mpc":
            info["mode"] = "fallback"
        value = limited_command(target, self.previous, self.config)
        info.update({"requested_C": value, "previous_C": self.previous,
                     "slew_limited": bool(abs(value - np.clip(target, self.config["action_min_C"], self.config["action_max_C"])) > 1e-10)})
        return {"oveTSet_u": value + 273.15, "oveTSet_activate": 1}, info

    def plan(self, start, forecast, fallback):
        h = self.horizon_steps
        c = self.config
        starts = schedule_at(forecast, start + np.arange(h) * 900)
        ends = schedule_at(forecast, start + np.arange(1, h+1) * 900)
        future = exogenous(starts)
        support = np.column_stack([np.full(h, self.state[0]), future[:, :2]])
        tolerance = np.array(c["support_tolerance"])
        if ((support < self.model["support_low"] - tolerance) | (support > self.model["support_high"] + tolerance)).any():
            return fallback, {"mode": "fallback", "reason": "outside_training_support"}
        actions = candidate_actions(self.previous, c)
        prediction = rollout(self.model, np.tile(self.state, (len(actions), 1)),
                             np.full(len(actions), self.q), np.full(len(actions), self.error),
                             actions, np.tile(future, (len(actions), 1, 1)))
        lower = ends["LowerSetp[1]"].to_numpy() - 273.15 + self.model["margin_K"]
        upper = ends["UpperSetp[1]"].to_numpy() - 273.15 - self.model["margin_K"]
        feasible = ((prediction[:, :, 0] >= lower) & (prediction[:, :, 0] <= upper)).all(axis=1)
        feasible &= ((prediction[:, :, 0] >= self.model["support_low"][0] - tolerance[0]) &
                     (prediction[:, :, 0] <= self.model["support_high"][0] + tolerance[0])).all(axis=1)
        if not feasible.any():
            return fallback, {"mode": "fallback", "reason": "no_margin_feasible_plan", "candidates": len(actions)}
        changes = np.diff(np.column_stack([np.full(len(actions), self.previous), actions]), axis=1)
        costs = (prediction[:, :, 3] * starts["PriceElectricPowerConstant"].to_numpy()).sum(axis=1)
        costs += c["move_weight"] * (changes**2).sum(axis=1)
        costs[~feasible] = np.inf
        selected = int(np.argmin(costs))
        return actions[selected, 0], {"mode": "mpc", "reason": "feasible_plan", "candidates": len(actions),
                    "feasible_candidates": int(feasible.sum()), "objective_EUR": float(costs[selected]),
                    "planned_actions_C": actions[selected].tolist(),
                    "predicted_temperature_C": prediction[selected, :, 0].tolist(),
                    "predicted_energy_kWh": prediction[selected, :, 3].tolist()}

    def update(self, before, after, interval, current):
        temperature = after["reaTZon_y"] - 273.15
        self.check_temperature(temperature)
        heat = float(np.trapezoid(interval.reaQFloHea_y, interval.time) / 900000)
        self.state = observe(self.model, self.state, temperature, heat, exogenous(pd.DataFrame([current]))[0])
        self.q = heat
        self.previous = after["oveTSet_u"] - 273.15
        self.error = self.previous - (before["reaTZon_y"] - 273.15)


class ResponseController(Controller):
    def __init__(self, model, config, setpoint):
        super().__init__(model, config, "response")
        self.setpoint = setpoint
        self.horizon_steps = config["response"]["steps"]

    def decide(self, index, before, forecast):
        self.check_temperature(before["reaTZon_y"] - 273.15)
        prelude = self.config["response"]["prelude_steps"]
        target = 22.0 if index < prelude else self.setpoint
        value = limited_command(target, self.previous, self.config)
        info = {"mode": "common_prelude" if index < prelude else "intervention", "requested_C": value,
                "previous_C": self.previous, "slew_limited": value != target}
        if index == prelude:
            times = before["time"] + np.arange(self.horizon_steps) * 900
            future = exogenous(schedule_at(forecast, times))
            actions = np.full((1, self.horizon_steps), self.setpoint)
            prediction = rollout(self.model, self.state[None], np.array([self.q]), np.array([self.error]), actions, future[None])[0]
            info["predicted_temperature_C"] = prediction[:, 0].tolist()
            info["predicted_energy_kWh"] = prediction[:, 3].tolist()
        return {"oveTSet_u": value + 273.15, "oveTSet_activate": 1}, info


def response_checks(output, config, model_path, config_path):
    rows, gates = [], []
    prelude = config["response"]["prelude_steps"]
    for day in config["response"]["start_days"]:
        curves = {}
        for setting in config["response"]["setpoints_C"]:
            directory = output / f"day{day:g}__response_{setting:g}"
            records = [json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()]
            frame = pd.read_csv(directory / "transitions.csv.gz").iloc[prelude:].reset_index(drop=True)
            predicted = records[prelude]["control"]
            curves[setting] = (records[prelude]["before"], frame, predicted)
        reference = next(iter(curves.values()))[0]
        if any(before != reference for before, _, _ in curves.values()):
            raise ValueError("Paired FMU interventions have different initial conditions")
        low, high = min(curves), max(curves)
        for h in [1,4,8,16,32]:
            actual_t = curves[high][1].operative_end_C.iloc[h-1] - curves[low][1].operative_end_C.iloc[h-1]
            predicted_t = curves[high][2]["predicted_temperature_C"][h-1] - curves[low][2]["predicted_temperature_C"][h-1]
            actual_e = curves[high][1].interval_energy_kWh.iloc[:h].sum() - curves[low][1].interval_energy_kWh.iloc[:h].sum()
            predicted_e = sum(curves[high][2]["predicted_energy_kWh"][:h]) - sum(curves[low][2]["predicted_energy_kWh"][:h])
            rows.append({"start_day": day, "horizon_steps": h, "actual_delta_temperature_C": actual_t,
                         "predicted_delta_temperature_C": predicted_t, "actual_delta_energy_kWh": actual_e,
                         "predicted_delta_energy_kWh": predicted_e})
            if h == config["horizon_steps"]:
                gates.append(actual_t >= -.05 and predicted_t >= -1e-8 and actual_e >= -.1 and predicted_e >= -1e-8)
        for setting, (_, frame, predicted) in curves.items():
            for h in [8,32]:
                rows.append({"start_day": day, "setpoint_C": setting, "horizon_steps": h,
                             "temperature_error_C": predicted["predicted_temperature_C"][h-1] - frame.operative_end_C.iloc[h-1],
                             "energy_error_kWh": sum(predicted["predicted_energy_kWh"][:h]) - frame.interval_energy_kWh.iloc[:h].sum()})
    informative = any(r.get("actual_delta_energy_kWh", 0) > .5 for r in rows if r["horizon_steps"] == config["horizon_steps"])
    pd.DataFrame(rows).to_csv(output / "response_metrics.csv", index=False, lineterminator="\n")
    save_json(output / "response_gate.json", {"passed": bool(all(gates) and informative),
              "direction_checks": [bool(g) for g in gates], "nonzero_response_observed": bool(informative),
              "model_sha256": digest(model_path), "config_sha256": digest(config_path),
              "scope": "Two-hour monotonic direction check on two calibration contexts; not an error bound or arbitrary-action certificate"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["response", "run"])
    parser.add_argument("--config", type=Path, default=Path("configs/mpc.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/greybox-mpc/model.joblib"))
    parser.add_argument("--response", type=Path, default=Path("outputs/action-response-mpc"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--days", type=float, help="Short closed-loop smoke run only")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base = json.loads(Path(config["dynamics_config"]).read_text())
    model = joblib.load(args.model)
    client = Client("http://127.0.0.1:8080")
    if client.call("GET", "version")["version"] != base["version"]:
        raise ValueError("BOPTEST version mismatch")
    output = args.output or Path("outputs/action-response-mpc" if args.mode == "response" else "outputs/mpc")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", {"config": config, "model_sha256": digest(args.model),
              "code_sha256": digest(Path(__file__)), "config_sha256": digest(args.config)})
    summaries = []
    if args.mode == "response":
        for day in config["response"]["start_days"]:
            for setting in config["response"]["setpoints_C"]:
                episode = {**base, "episode": {"start_day": day},
                           "days": (config["response"]["prelude_steps"] + config["response"]["steps"]) / 96}
                name, kind = f"day{day:g}", f"response_{setting:g}"
                summaries.append(run_episode(client, episode, name, kind, output / f"{name}__{kind}", ResponseController(model, config, setting)))
                save_json(output / "summary.json", summaries)
        response_checks(output, config, args.model, args.config)
    else:
        gate = json.loads((args.response / "response_gate.json").read_text())
        if not gate["passed"] or gate["model_sha256"] != digest(args.model) or gate["config_sha256"] != digest(args.config):
            raise ValueError("A matching, passed FMU action-response check is required")
        for day in config["closed_loop"]["start_days"]:
            for kind in config["closed_loop"]["controllers"]:
                episode = {**base, "episode": {"start_day": day}, "days": args.days or config["closed_loop"]["days"]}
                name = f"day{day:g}"
                summaries.append(run_episode(client, episode, name, kind, output / f"{name}__{kind}", Controller(model, config, kind)))
                save_json(output / "summary.json", summaries)
    print(json.dumps([{k: s[k] for k in ["scenario", "controller", "metrics"]} for s in summaries], indent=2))


if __name__ == "__main__":
    main()
