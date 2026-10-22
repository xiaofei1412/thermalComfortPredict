"""Chronological calibration extension and weather-stressed recovery MPC."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import nnls

from .excitation import SafeExcitation
from .greybox import energy_prediction, episode_rollout, fit, load_heat_episodes, metrics, rollout
from .mpc import Controller, ResponseController, candidate_actions, exogenous, limited_command, response_checks
from .simulation import Client, digest, run_episode, save_json, schedule_at


def configuration(path=Path("configs/recovery_mpc.json")):
    extra = json.loads(path.read_text())
    return {**json.loads(Path(extra["base_config"]).read_text()), **extra}


def weather_view(forecast, start, config, mode):
    """Synthetic forecast bias ramps from measured lead zero to full bias at 2 h."""
    result = forecast.copy()
    sign = {"perfect": 0, "warm_bias": 1, "cold_bias": -1}[mode]
    ramp = np.clip((result.time.to_numpy() - start) / (config["horizon_steps"] * 900), 0, 1)
    result["TDryBul"] += sign * ramp * config["forecast_temperature_error_K"]
    result["HGloHor"] *= 1 + sign * ramp * config["forecast_solar_error_fraction"]
    return result


class CollectionPolicy:
    horizon_steps = 0

    def __init__(self, config):
        self.config = config

    def reset(self, initial, inputs):
        self.policy = SafeExcitation(self.config, inputs["oveTSet_u"], initial)

    def decide(self, index, before, forecast):
        current = schedule_at(forecast, [before["time"]]).iloc[0]
        return self.policy.action(index, before, current)

    def update(self, before, after, interval, current):
        self.policy.check_temperature(after["reaTZon_y"] - 273.15)


def load_data(extra_root, config):
    base = json.loads(Path("configs/dynamics.json").read_text())
    episodes, splits = load_heat_episodes(Path("outputs/excitation"), base)
    summaries = json.loads((extra_root / "summary.json").read_text())
    if len(summaries) != len(config["extra_episodes"]):
        raise ValueError("Incomplete calibration collection")
    for expected_plan, summary in zip(config["extra_episodes"], summaries):
        plan = summary["config"]["episode"]
        if plan != expected_plan or summary["config"]["days"] != 7 or summary["config"]["warmup_days"] != 7:
            raise ValueError("Calibration collection plan mismatch")
        directory = extra_root / f"{summary['scenario']}__{summary['controller']}"
        for name, expected in summary["sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError("Changed extra calibration artifact")
        frame = pd.read_csv(directory / "transitions.csv.gz")
        native = pd.read_csv(directory / "native.csv.gz")
        records = [json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()]
        states = np.zeros((len(frame)+1, 4))
        states[0, 0] = records[0]["before"]["reaTZon_y"] - 273.15
        states[0, 3] = np.nan
        states[1:, 0] = frame.operative_end_C
        states[1:, 3] = frame.interval_energy_kWh
        times = frame.time_start_s.to_numpy()
        if not np.array_equal(times, plan["start_day"]*86400+np.arange(672)*900):
            raise ValueError("Incomplete or misaligned calibration episode")
        boundaries = np.append(times, times[-1]+900)
        cumulative = cumulative_trapezoid(native.reaQFloHea_y, native.time, initial=0)
        angle = 2*np.pi*times/86400
        episodes[plan["id"]] = {"id": plan["id"], "split": plan["split"], "states": states,
            "actions": frame.setpoint_requested_C.to_numpy(), "times": times,
            "heat": np.diff(np.interp(boundaries, native.time, cumulative))/900000,
            "initial_heat": float(native.reaQFloHea_y.iloc[0]/1000),
            "exog": np.column_stack([frame.outdoor_start_C, frame.solar_start_W_m2,
                                      frame.occupancy_start, np.sin(angle), np.cos(angle)])}
        splits.append({**plan, "warmup_start_s": (plan["start_day"]-7)*86400,
                       "start_s": float(times[0]), "end_s": float(boundaries[-1]),
                       "transitions": len(frame), "source_hashes": summary["sha256"]})
    # Shared warmup/evaluation dates within a role are correlated, not independent data.
    # Across roles use half-open intervals: training ends exactly when validation warmup starts.
    for left, right in [("train", "validation"), ("validation", "test")]:
        if max(s["end_s"] for s in splits if s["split"] == left) > min(s["warmup_start_s"] for s in splits if s["split"] == right):
            raise ValueError("Calibration roles overlap in model time")
    return episodes, splits


def fit_model(config, source, output):
    episodes, splits = load_data(source,config)
    output.mkdir(parents=True, exist_ok=False)
    model = fit(episodes, config)
    train = [e for e in episodes.values() if e["split"] == "train"]
    heat = np.concatenate([e["heat"] for e in train])
    outdoor = np.concatenate([e["exog"][:, 0] for e in train])
    power = np.concatenate([e["states"][1:, 3]*4 for e in train])
    idle = heat < config["idle_heat_threshold_kW"]
    if not idle.any():
        raise ValueError("No training idle observations")
    model["idle_power_kW"] = float(np.mean(power[idle]))
    model["power_transition_kW"] = config["power_transition_kW"]
    x = np.column_stack([np.minimum(heat/model["power_transition_kW"], 1), heat,
                         heat*np.maximum(20-outdoor, 0)/20])
    model["power"] = nnls(x, np.maximum(power-model["idle_power_kW"], 0))[0]
    validation = [e for e in episodes.values() if e["split"] == "validation"]
    selection_episode = validation[0]
    selection_origins = np.arange(16, config["validation_selection_days"]*96-16+1, 4)
    scores = []
    for weight in config["residual_weights"]:
        predicted = episode_rollout({**model, "residual_weight": weight}, selection_episode, selection_origins, 16)
        scores.append({"weight": weight, "score_C": float(np.mean([r["temperature_RMSE_C"] for r in metrics(predicted, selection_episode, selection_origins, [1,4,8])]))})
    model["residual_weight"] = min(scores, key=lambda r: r["score_C"])["weight"]
    errors, calibration = [], []
    for e in validation:
        start = config["validation_calibration_start_day"]*96 if e["id"] == selection_episode["id"] else 16
        origins = np.arange(start, len(e["actions"])-config["horizon_steps"]+1, 4)
        predicted = episode_rollout(model, e, origins, config["horizon_steps"])
        errors.append(predicted[:, :, 0] - e["states"][origins[:, None]+np.arange(1,config["horizon_steps"]+1), 0])
        calibration.append({"episode": e["id"], "origins": origins.tolist()})
    model["margin_K"] = np.quantile(abs(np.concatenate(errors)), config["margin_quantile"], axis=0)+config["sensor_margin_K"]
    joblib.dump(model, output / "model.joblib")
    save_json(output / "selection.json", {"scores": scores, "selected_weight": model["residual_weight"],
        "selection_episode": selection_episode["id"], "selection_origins": selection_origins.tolist(),
        "calibration": calibration, "margin_K": model["margin_K"].tolist(),
        "idle_training_rows": int(idle.sum()), "idle_power_kW": model["idle_power_kW"],
        "power": model["power"].tolist(), "support_low": model["support_low"].tolist(),
        "support_high": model["support_high"].tolist(), "fit_diagnostics": model["fit_diagnostics"]})
    old = joblib.load("outputs/greybox-mpc/model.joblib")
    rows, energy_rows = [], []
    for e in episodes.values():
        if e["split"] == "train":
            continue
        origins = np.arange(16, len(e["actions"])-16+1, 4)
        for name, m in [("previous", old), ("expanded_gated", model)]:
            prediction = episode_rollout(m, e, origins, 16)
            rows.extend([{"episode": e["id"], "split": e["split"], "model": name, **r} for r in metrics(prediction,e,origins,[1,4,8,16])])
            # Isolate electrical mapping error from actuator and thermal rollout error.
            predicted = energy_prediction(m, e["heat"], e["exog"][:,0])
            for regime, mask in [("all", np.ones(len(predicted),dtype=bool)), ("idle", e["heat"] < config["idle_heat_threshold_kW"])]:
                error = predicted[mask]-e["states"][1:,3][mask]
                energy_rows.append({"episode": e["id"], "split": e["split"], "model": name,
                                    "regime": regime, "rows": int(mask.sum()), "bias_kWh": float(error.mean()),
                                    "RMSE_kWh": float(np.sqrt(np.mean(error**2)))})
    pd.DataFrame(rows).to_csv(output/"metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(energy_rows).to_csv(output/"power_metrics.csv", index=False, lineterminator="\n")
    save_json(output/"split_manifest.json", splits)
    save_json(output/"manifest.json", {"config": config,
        "sources_sha256": {str(p): digest(p) for p in [source/"summary.json",Path("outputs/excitation/summary.json"),Path("outputs/greybox-mpc/model.joblib")]},
        "code_sha256": {n: digest(Path(__file__).with_name(n)) for n in ["recovery.py","greybox.py","mpc.py","simulation.py"]},
        "artifacts_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(json.loads((output/"selection.json").read_text()),indent=2), flush=True)


class RecoveryController(Controller):
    def __init__(self, model, config, kind, weather="perfect"):
        super().__init__(model, config, kind)
        self.weather = weather
        self.horizon_steps = max(config["horizon_steps"],config["recovery_lookahead_steps"])

    def reset(self, initial, inputs):
        super().reset(initial, inputs)
        self.recovering = False

    def decide(self, index, before, forecast):
        self.view = weather_view(forecast, before["time"], self.config, self.weather)
        action, info = super().decide(index,before,self.view)
        info["weather_mode"] = self.weather
        return action, info

    def guarded_rule(self, before, current):
        c = self.config
        target, reason = super().guarded_rule(before,current)
        temperature = before["reaTZon_y"]-273.15
        future = schedule_at(self.view, before["time"]+np.arange(c["recovery_lookahead_steps"]+1)*900)
        required = float(future["LowerSetp[1]"].max()-273.15)
        upper = current["UpperSetp[1]"]-273.15
        if reason == "coast":
            self.recovering = False
            return target, reason
        if self.recovering and temperature >= required+c["recovery_release_K"]:
            self.recovering = False
        if temperature < required+c["recovery_trigger_K"]:
            self.recovering = True
        if self.recovering:
            return c["action_max_C"], "anticipatory_recovery"
        target = max(target,c["unoccupied_floor_C"])
        if required > current["LowerSetp[1]"]-273.15:
            target = max(target,required+c["rule_occupied_offset_K"])
        return min(target,upper-c["guard_margin_K"]), "rule"

    def plan(self, start, forecast, fallback):
        c, h = self.config, self.config["horizon_steps"]
        starts = schedule_at(forecast,start+np.arange(h)*900)
        ends = schedule_at(forecast,start+np.arange(1,h+1)*900)
        future = exogenous(starts)
        tolerance = np.array(c["support_tolerance"])
        support = np.column_stack([np.full(h,self.state[0]),future[:,:2]])
        if ((support < self.model["support_low"]-tolerance)|(support > self.model["support_high"]+tolerance)).any():
            return fallback,{"mode":"fallback","reason":"outside_training_support"}
        actions = candidate_actions(self.previous,c)
        predictions = []
        # Robust envelope around the supplied (possibly biased) forecast, lead zero measured.
        for mode in c["weather_modes"]:
            scenario = weather_view(starts,start,c,mode)
            predictions.append(rollout(self.model,np.tile(self.state,(len(actions),1)),
                np.full(len(actions),self.q),np.full(len(actions),self.error),actions,
                np.tile(exogenous(scenario),(len(actions),1,1))))
        prediction = np.stack(predictions)
        lower = ends["LowerSetp[1]"].to_numpy()-273.15+self.model["margin_K"]
        upper = ends["UpperSetp[1]"].to_numpy()-273.15-self.model["margin_K"]
        feasible = ((prediction[:,:,:,0]>=lower)&(prediction[:,:,:,0]<=upper)).all(axis=(0,2))
        feasible &= ((prediction[:,:,:,0]>=self.model["support_low"][0]-tolerance[0])&
                     (prediction[:,:,:,0]<=self.model["support_high"][0]+tolerance[0])).all(axis=(0,2))
        if not feasible.any():
            return fallback,{"mode":"fallback","reason":"no_margin_feasible_plan","candidates":len(actions)}
        changes = np.diff(np.column_stack([np.full(len(actions),self.previous),actions]),axis=1)
        costs = (prediction[:,:,:,3]*starts["PriceElectricPowerConstant"].to_numpy()).sum(axis=2).max(axis=0)
        costs += c["move_weight"]*(changes**2).sum(axis=1)
        costs[~feasible] = np.inf
        selected = int(np.argmin(costs))
        return actions[selected,0],{"mode":"mpc","reason":"robust_feasible_plan","candidates":len(actions),
            "feasible_candidates":int(feasible.sum()),"objective_EUR":float(costs[selected]),
            "planned_actions_C":actions[selected].tolist(),"predicted_temperature_C":prediction[0,selected,:,0].tolist(),
            "predicted_energy_kWh":prediction[0,selected,:,3].tolist(),
            "worst_temperature_low_C":prediction[:,selected,:,0].min(axis=0).tolist(),
            "worst_temperature_high_C":prediction[:,selected,:,0].max(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["collect","fit","response","run"])
    parser.add_argument("--config",type=Path,default=Path("configs/recovery_mpc.json"))
    parser.add_argument("--source",type=Path,default=Path("outputs/recovery-calibration"))
    parser.add_argument("--model",type=Path,default=Path("outputs/recovery-model/model.joblib"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--days",type=float)
    parser.add_argument("--start-days",type=int,nargs="+")
    parser.add_argument("--weather",nargs="+",choices=["perfect","warm_bias","cold_bias"])
    args = parser.parse_args()
    config = configuration(args.config)
    if args.mode == "fit":
        fit_model(config,args.source,args.output)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    client = Client("http://127.0.0.1:8080")
    base = json.loads(Path(config["dynamics_config"]).read_text())
    if client.call("GET","version")["version"] != base["version"]:
        raise ValueError("BOPTEST version mismatch")
    model = None if args.mode == "collect" else joblib.load(args.model)
    save_json(args.output/"manifest.json",{"config":config,"config_sha256":digest(args.config),
        "code_sha256":{n:digest(Path(__file__).with_name(n)) for n in ["recovery.py","greybox.py","mpc.py","simulation.py"]},
        "model_sha256":None if model is None else digest(args.model),
        "effective_days":args.days,"effective_start_days":args.start_days,"effective_weather":args.weather})
    summaries = []
    if args.mode == "collect":
        for plan in config["extra_episodes"]:
            episode = {**base,"episode":plan,"days":args.days or 7,
                "excitation":{**base["excitation"],"nominal_C":config["extra_nominal_C"],
                    "levels_K":config["extra_levels_K"],"hold_steps":config["extra_hold_steps"]}}
            kind = "warm_excitation"
            summaries.append(run_episode(client,episode,plan["id"],kind,args.output/f"{plan['id']}__{kind}",CollectionPolicy(episode)))
            save_json(args.output/"summary.json",summaries)
    elif args.mode == "response":
        for day in config["response_start_days"]:
            for setting in config["response"]["setpoints_C"]:
                episode = {**base,"episode":{"start_day":day},
                    "days":(config["response"]["prelude_steps"]+config["response"]["steps"])/96}
                name,kind = f"day{day}",f"response_{setting:g}"
                summaries.append(run_episode(client,episode,name,kind,args.output/f"{name}__{kind}",ResponseController(model,config,setting)))
                save_json(args.output/"summary.json",summaries)
        response_checks(args.output,config,args.model,args.config)
    else:
        gate = json.loads(Path("outputs/recovery-response/response_gate.json").read_text())
        if not gate["passed"] or gate["model_sha256"]!=digest(args.model) or gate["config_sha256"]!=digest(args.config):
            raise ValueError("Matching action response gate required")
        for day in args.start_days or config["test_start_days"]:
            for weather in args.weather or config["weather_modes"]:
                for kind in ["guarded_rule","mpc"]:
                    episode = {**base,"episode":{"start_day":day},"days":args.days or config["test_days"]}
                    name = f"day{day}_{weather}"
                    summaries.append(run_episode(client,episode,name,kind,args.output/f"{name}__{kind}",RecoveryController(model,config,kind,weather)))
                    save_json(args.output/"summary.json",summaries)
    print(json.dumps([{k:s[k] for k in ["scenario","controller","metrics"]} for s in summaries],indent=2))


if __name__ == "__main__":
    main()
