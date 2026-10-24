"""Warm-season split, stored solar gain, causal mass estimation and early coasting."""

import argparse
from collections import deque
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import expm
from scipy.optimize import least_squares, nnls
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import greybox
from .mpc import candidate_actions, exogenous, limited_command
from .recovery import CollectionPolicy, RecoveryController, configuration as recovery_configuration, weather_view
from .simulation import Client, digest, run_episode, save_json, schedule_at


def configuration():
    extra = json.loads(Path("configs/solar_mpc.json").read_text())
    return {**recovery_configuration(Path(extra["base_config"])), **extra}


def validate_split(splits):
    if {s["split"] for s in splits} != {"train", "validation", "test"}:
        raise ValueError("All roles required")
    for left, right in [("train", "validation"), ("validation", "test")]:
        if max(s["end_s"] for s in splits if s["split"] == left) > min(s["warmup_start_s"] for s in splits if s["split"] == right):
            raise ValueError("Role evaluation/warmup windows overlap")
    held = sorted((s["warmup_start_s"], s["end_s"]) for s in splits if s["split"] == "test")
    if any(a[1] > b[0] for a, b in zip(held, held[1:])):
        raise ValueError("Test episodes overlap")


def read_episode(directory, summary, role):
    for name, expected in summary["sha256"].items():
        if digest(directory/name) != expected:
            raise ValueError("Source hash mismatch")
    frame = pd.read_csv(directory/"transitions.csv.gz")
    native = pd.read_csv(directory/"native.csv.gz")
    records = [json.loads(s) for s in (directory/"steps.jsonl").read_text().splitlines()]
    times = frame.time_start_s.to_numpy()
    if len(frame) != 672 or not np.array_equal(times, times[0]+np.arange(672)*900):
        raise ValueError("Expected complete 7-day episode")
    states = np.zeros((673, 4))
    states[0, 0] = records[0]["before"]["reaTZon_y"]-273.15
    states[0, 3] = np.nan
    states[1:, 0] = frame.operative_end_C
    states[1:, 3] = frame.interval_energy_kWh
    angle = 2*np.pi*times/86400
    boundaries = np.append(times, times[-1]+900)
    cumulative = cumulative_trapezoid(native.reaQFloHea_y, native.time, initial=0)
    episode = {"id": summary["scenario"], "split": role, "states": states,
        "actions": frame.setpoint_requested_C.to_numpy(), "times": times,
        "heat": np.diff(np.interp(boundaries, native.time, cumulative))/900000,
        "initial_heat": float(native.reaQFloHea_y.iloc[0]/1000),
        "exog": np.column_stack([frame.outdoor_start_C, frame.solar_start_W_m2, frame.occupancy_start, np.sin(angle), np.cos(angle)])}
    split = {"id": episode["id"], "split": role, "original_role": summary["config"]["episode"]["split"],
        "warmup_start_s": float(times[0]-7*86400), "start_s": float(times[0]), "end_s": float(boundaries[-1]),
        "source": str(directory), "source_summary_sha256": digest(directory/"summary.json")}
    return episode, split


def load_data(config, source=Path("outputs/solar-data")):
    episodes, splits = {}, []
    collected = json.loads((source/"manifest.json").read_text())["config"]
    for key in ["new_episodes","collection_nominal_C","collection_levels_K","abort_max_C"]:
        if collected[key] != config[key]:
            raise ValueError("New collection configuration differs")
    for root, selected in [(Path("outputs/excitation"), [14,35,56,119]),
                           (Path("outputs/recovery-calibration"), [63]), (source, [147,168,189])]:
        summaries = json.loads((root/"summary.json").read_text())
        chosen = [s for s in summaries if s["config"]["episode"]["start_day"] in selected]
        if len(chosen) != len(selected):
            raise ValueError("Calibration source incomplete")
        for summary in chosen:
            day = summary["config"]["episode"]["start_day"]
            if root == source:
                expected = next(p for p in config["new_episodes"] if p["start_day"] == day)
                if summary["config"]["episode"] != expected:
                    raise ValueError("New episode identity differs")
            role = "train" if day <= 147 else "validation"
            episode, split = read_episode(root/f"{summary['scenario']}__{summary['controller']}",summary,role)
            episodes[episode["id"]] = episode
            splits.append(split)
    for day in config["test_start_days"]:
        splits.append({"id": f"day{day}", "split": "test", "warmup_start_s": (day-7)*86400,
                       "start_s": day*86400, "end_s": (day+7)*86400})
    validate_split(splits)
    return episodes, splits


def discretize(parameters):
    rz, rm, cz, cm, fast, stored, person = parameters
    a = np.array([[-(1/rz+1/rm)/cz, 1/rm/cz], [1/rm/cm, -1/rm/cm]])
    b = np.array([[1/rz/cz, 0, fast/cz, person/cz], [0, 1/cm, stored/cm, 0]])
    ad = expm(a*.25)
    return ad, np.linalg.solve(a, (ad-np.eye(2))@b)


def mass_state(model, history, prior):
    """Fit initial latent temperature to past zone observations, then propagate to now."""
    state = np.array([history[0][0], 0.])
    sensitivity = np.array([0., 1.])
    numerator, denominator = 0., model["mass_ridge"]
    for _, actual_after, heat, exog in history:
        state = greybox.thermal_step(model,state[None],np.array([heat]),exog[None])[0]
        sensitivity = model["A"]@sensitivity
        numerator += sensitivity[0]*(actual_after-state[0])
        denominator += sensitivity[0]**2
    # Weak prior is for the initial mass in this window, not a future observation.
    initial_mass = (numerator+model["mass_ridge"]*prior)/denominator
    result = state+sensitivity*initial_mass
    result[0] = history[-1][1]
    result[1] = np.clip(result[1], result[0]-10, result[0]+15)
    return result


def observer_states(model, episode):
    if model.get("observer_kind") == "recursive":
        return greybox.observer_states(model,episode)
    state = greybox.initial_state(model,episode["states"][0,0],episode["initial_heat"])
    states = [state]
    history = deque(maxlen=model["mass_window_steps"])
    for k, heat in enumerate(episode["heat"]):
        history.append((episode["states"][k,0],episode["states"][k+1,0],heat,episode["exog"][k]))
        state = greybox.observe(model,state,episode["states"][k+1,0],heat,episode["exog"][k])
        if len(history) == model["mass_window_steps"]:
            state = mass_state(model,history,history[0][0])
        states.append(state)
    return np.array(states)


def episode_rollout(model, episode, origins, horizon):
    states = observer_states(model,episode)
    return greybox.rollout(model,states[origins],episode["heat"][origins-1],
        episode["actions"][origins-1]-episode["states"][origins-1,0],
        np.stack([episode["actions"][k:k+horizon] for k in origins]),
        np.stack([episode["exog"][k:k+horizon] for k in origins]))


def fit_model(config, output):
    episodes, splits = load_data(config)
    output.mkdir(parents=True,exist_ok=False)
    train = {key:e for key,e in episodes.items() if e["split"] == "train"}
    model = greybox.fit(train,config)
    expanded = copy.deepcopy(model)
    def objective(log_parameters):
        parameters = np.exp(log_parameters)
        a,b = discretize(parameters)
        candidate = {**model,"parameters":parameters,"A":a,"B":b,"residual_weight":0}
        errors = []
        for e in train.values():
            estimated = greybox.observer_states(candidate,e)
            origins = np.arange(16,len(e["actions"])-32,16)
            state = estimated[origins]
            for h in range(32):
                state = greybox.thermal_step(candidate,state,e["heat"][origins+h],e["exog"][origins+h])
                if h in [3,7,15,31]:
                    errors.extend(state[:,0]-e["states"][origins+h+1,0])
        return np.array(errors)
    result = least_squares(objective,np.log([3.,.15,2.,12.,10.,10.,.1]),
        bounds=(np.log([.1,.01,.5,1.,.01,.01,.001]),np.log([20.,5.,100.,500.,100.,100.,.5])),
        max_nfev=config["fit_max_nfev"])
    if not result.success:
        raise RuntimeError("Stored-solar RC fit did not converge")
    model["parameters"] = np.exp(result.x)
    model["A"], model["B"] = discretize(model["parameters"])
    model["mass_window_steps"], model["mass_ridge"] = config["mass_window_steps"],config["mass_ridge"]
    # Train a bounded residual after replacing the physical solar network.
    xs,ys = [],[]
    for e in train.values():
        estimated = observer_states(model,e)
        pred = greybox.thermal_step(model,estimated[:-1],e["heat"],e["exog"])
        xs.append(e["exog"][16:]); ys.append(e["states"][17:,0]-pred[16:,0])
    x,y = np.concatenate(xs),np.concatenate(ys)
    model["residual"] = make_pipeline(StandardScaler(),Ridge(alpha=config["residual_alpha"])).fit(x,y)
    model["residual_cap_K"] = float(np.quantile(abs(y),.95))
    heat = np.concatenate([e["heat"] for e in train.values()])
    outdoor = np.concatenate([e["exog"][:,0] for e in train.values()])
    power = np.concatenate([e["states"][1:,3]*4 for e in train.values()])
    model["idle_power_kW"] = float(power[heat<config["idle_heat_threshold_kW"]].mean())
    model["power_transition_kW"] = config["power_transition_kW"]
    px = np.column_stack([np.minimum(heat/model["power_transition_kW"],1),heat,heat*np.maximum(20-outdoor,0)/20])
    model["power"] = nnls(px,np.maximum(power-model["idle_power_kW"],0))[0]
    expanded.update({key:model[key] for key in ["idle_power_kW","power_transition_kW","power"]})
    validation = [e for e in episodes.values() if e["split"] == "validation"]
    first = validation[0]
    origins = np.arange(16,3*96-32+1,4)
    expanded.update(mass_window_steps=config["mass_window_steps"],mass_ridge=config["mass_ridge"],observer_kind="recursive")
    expanded_mhe = {**expanded,"observer_kind":"window"}
    grid = [("expanded_baseline",expanded),("expanded_mhe",expanded_mhe),("stored_solar_mhe",model)]
    scores = []
    for name,m in grid:
        weight_scores = []
        for weight in config["residual_weights"]:
            p = episode_rollout({**m,"residual_weight":weight},first,origins,32)
            weight_scores.append({"model":name,"weight":weight,"RMSE_C":float(np.mean([r["temperature_RMSE_C"] for r in greybox.metrics(p,first,origins,[4,8,24,32])]))})
        m["residual_weight"] = min(weight_scores,key=lambda r:r["RMSE_C"])["weight"]
        scores.extend(weight_scores)
    models = {"expanded_baseline":expanded,"expanded_mhe":expanded_mhe,"stored_solar_mhe":model}
    selected = min(scores,key=lambda r:r["RMSE_C"])["model"]
    model = models[selected]
    errors, rows, calibration = [],[],[]
    previous = joblib.load("outputs/recovery-model/model.joblib")
    for e in validation:
        origins = np.arange(16,len(e["actions"])-32+1,4)
        for name,m in [("previous",previous),*models.items()]:
            p = (greybox.episode_rollout if name == "previous" else episode_rollout)(m,e,origins,32)
            rows.extend([{"episode":e["id"],"model":name,**r} for r in greybox.metrics(p,e,origins,[1,4,8,24,32])])
        start = 4*96 if e is first else 16
        origins = np.arange(start,len(e["actions"])-32+1,4)
        p = episode_rollout(model,e,origins,32)
        errors.append(p[:,:,0]-e["states"][origins[:,None]+np.arange(1,33),0])
        calibration.append({"episode":e["id"],"origins":origins.tolist()})
    model["margin_K"] = np.quantile(abs(np.concatenate(errors)),config["margin_quantile"],axis=0)+config["sensor_margin_K"]
    joblib.dump(model,output/"model.joblib")
    for name,m in models.items():
        joblib.dump(m,output/(name+".joblib"))
    pd.DataFrame(rows).to_csv(output/"metrics.csv",index=False,lineterminator="\n")
    save_json(output/"selection.json",{"scores":scores,"selected_model":selected,"selected_weight":model["residual_weight"],
        "selection_episode":first["id"],"selection_origins":np.arange(16,3*96-32+1,4).tolist(),
        "calibration":calibration,"margin_K":model["margin_K"].tolist(),"parameters":model["parameters"].tolist(),
        "support_low":model["support_low"].tolist(),"support_high":model["support_high"].tolist(),
        "fit_cost":float(result.cost),"fit_nfev":result.nfev})
    save_json(output/"split_manifest.json",splits)
    save_json(output/"manifest.json",{"config":config,"code_sha256":digest(Path(__file__)),
        "source_summaries":{s["source"]:s["source_summary_sha256"] for s in splits if "source" in s},
        "artifacts_sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}})


class SolarController(RecoveryController):
    def __init__(self, model, config, kind, weather="perfect"):
        super().__init__(model,config,kind,weather)
        self.horizon_steps = 48  # 8h projection plus the rule's 4h schedule lookahead.

    def reset(self, initial, inputs):
        super().reset(initial,inputs)
        self.history = deque(maxlen=self.model["mass_window_steps"])
        self.min_recovering = False

    def update(self, before, after, interval, current):
        super().update(before,after,interval,current)
        self.history.append((before["reaTZon_y"]-273.15,after["reaTZon_y"]-273.15,self.q,exogenous(pd.DataFrame([current]))[0]))
        if self.model.get("observer_kind") != "recursive" and len(self.history) == self.model["mass_window_steps"]:
            self.state = mass_state(self.model,self.history,self.history[0][0])

    def predict(self, actions, forecast, start):
        future = exogenous(schedule_at(forecast,start+np.arange(actions.shape[1])*900))
        return greybox.rollout(self.model,np.tile(self.state,(len(actions),1)),np.full(len(actions),self.q),
            np.full(len(actions),self.error),actions,np.tile(future,(len(actions),1,1)))

    def ramp(self, target, length):
        values, previous = [],self.previous
        for _ in range(length):
            previous = limited_command(target,previous,self.config)
            values.append(previous)
        return np.array(values)

    def reference_trajectory(self, start, forecast):
        shadow = RecoveryController(self.model,self.config,"guarded_rule")
        shadow.state,shadow.q,shadow.error = self.state.copy(),self.q,self.error
        shadow.previous,shadow.recovering = self.previous,self.recovering
        shadow.view = forecast
        actions,states = [],[]
        for k in range(32):
            time = start+k*900
            current = schedule_at(forecast,[time]).iloc[0]
            before = {"time":time,"reaTZon_y":shadow.state[0]+273.15}
            target,_ = shadow.guarded_rule(before,current)
            value = limited_command(target,shadow.previous,self.config)
            prediction = greybox.rollout(self.model,shadow.state[None],np.array([shadow.q]),np.array([shadow.error]),
                np.array([[value]]),exogenous(pd.DataFrame([current]))[None])[0,0]
            shadow.error = value-shadow.state[0]
            shadow.state,shadow.q,shadow.previous = prediction[:2],prediction[2],value
            actions.append(value); states.append(prediction)
        return np.array(actions),np.array(states)

    def decide(self, index, before, forecast):
        if self.kind == "guarded_rule" or index < self.config["initial_rule_steps"]:
            return super().decide(index,before,forecast)
        self.check_temperature(before["reaTZon_y"]-273.15)
        self.view = weather_view(forecast,before["time"],self.config,self.weather)
        current = schedule_at(self.view,[before["time"]]).iloc[0]
        t = before["reaTZon_y"]-273.15
        lower = current["LowerSetp[1]"]-273.15
        if self.kind == "minimum_heat":
            if t < max(lower+self.config["minimum_trigger_K"],19):
                self.min_recovering = True
            if t >= max(lower+self.config["minimum_release_K"],19.3):
                self.min_recovering = False
            target = self.config["action_max_C"] if self.min_recovering else self.config["action_min_C"]
            info = {"mode":"minimum_heat","reason":"reactive_heat" if self.min_recovering else "minimum_setpoint"}
        else:
            target, reason = self.guarded_rule(before,current)
            info = {"mode":"fallback","reason":reason}
            try:
                target,extra = self.plan_solar(index,before["time"],self.view,target)
                info.update(extra)
            except (ValueError,FloatingPointError) as error:
                info.update(reason="invalid_model_prediction",detail=str(error))
        value = limited_command(target,self.previous,self.config)
        info.update(requested_C=value,previous_C=self.previous,weather_mode=self.weather)
        return {"oveTSet_u":value+273.15,"oveTSet_activate":1},info

    def plan_solar(self, index, start, forecast, fallback):
        c,h = self.config,self.config["horizon_steps"]
        starts = schedule_at(forecast,start+np.arange(32)*900)
        ends = schedule_at(forecast,start+np.arange(1,33)*900)
        support = np.column_stack([np.full(32,self.state[0]),exogenous(starts)[:,:2]])
        if ((support<self.model["support_low"]-c["support_tolerance"])|(support>self.model["support_high"]+c["support_tolerance"])).any():
            return fallback,{"mode":"fallback","reason":"outside_training_support"}
        coast = self.ramp(c["action_min_C"],32)
        passive = self.predict(coast[None],forecast,start)[0,:,0]
        lower = ends["LowerSetp[1]"].to_numpy()-273.15
        upper = ends["UpperSetp[1]"].to_numpy()-273.15
        margin = self.model["margin_K"]
        # Stop heating early only when stored/solar heat predicts no cold violation.
        if (passive>=lower+margin).all() and (passive>=upper-c["solar_coast_buffer_K"]).any():
            self.recovering = False
            return coast[0],{"mode":"early_coast","reason":"stored_solar_heat","passive_temperature_C":passive.tolist()}
        reference_actions,reference = self.reference_trajectory(start,forecast)
        actions = candidate_actions(self.previous,c)
        actions = np.unique(np.vstack([actions,coast[:h],reference_actions[:h]]),axis=0)
        prediction = self.predict(actions,forecast,start)
        lo,hi = lower[:h]+margin[:h],upper[:h]-margin[:h]
        feasible = ((prediction[:,:,0]>=lo)&(prediction[:,:,0]<=hi)).all(axis=1)
        feasible &= ((prediction[:,:,0]>=self.model["support_low"][0]-c["support_tolerance"][0])&
                     (prediction[:,:,0]<=self.model["support_high"][0]+c["support_tolerance"][0])).all(axis=1)
        feasible &= (prediction[:,:,:2]<=reference[None,:h,:2]+1e-9).all(axis=(1,2))
        feasible &= actions[:,0]<=reference_actions[0]+1e-9
        if not feasible.any():
            return fallback,{"mode":"fallback","reason":"no_margin_feasible_plan"}
        changes = np.diff(np.column_stack([np.full(len(actions),self.previous),actions]),axis=1)
        costs = (prediction[:,:,3]*starts["PriceElectricPowerConstant"].to_numpy()[:h]).sum(axis=1)+c["move_weight"]*(changes**2).sum(axis=1)
        costs[~feasible] = np.inf
        selected = int(np.argmin(costs))
        return actions[selected,0],{"mode":"mpc","reason":"solar_feasible_plan","feasible_candidates":int(feasible.sum()),
            "predicted_temperature_C":prediction[selected,:,0].tolist(),"predicted_energy_kWh":prediction[selected,:,3].tolist(),
            "planned_actions_C":actions[selected].tolist(),"predicted_mass_C":prediction[selected,:,1].tolist(),
            "reference_temperature_C":reference[:h,0].tolist(),"reference_mass_C":reference[:h,1].tolist()}


class SolarResponse(SolarController):
    def __init__(self, model, config, setting):
        super().__init__(model,config,"response")
        self.setting = setting

    def decide(self, index, before, forecast):
        self.check_temperature(before["reaTZon_y"]-273.15)
        prelude = self.config["response"]["prelude_steps"]
        target = 22. if index < prelude else self.setting
        value = limited_command(target,self.previous,self.config)
        info = {"mode":"common_prelude" if index < prelude else "intervention","requested_C":value}
        if index == prelude:
            actions = self.ramp(self.setting,32)
            prediction = self.predict(actions[None],forecast,before["time"])[0]
            info.update(predicted_temperature_C=prediction[:,0].tolist(),predicted_energy_kWh=prediction[:,3].tolist(),
                        planned_actions_C=actions.tolist())
        return {"oveTSet_u":value+273.15,"oveTSet_activate":1},info


def response_gate(root, model_path, config):
    rows=[]
    n=config["response"]["prelude_steps"]
    for day in config["response"]["start_days"]:
        branches={}
        for setting in config["response"]["setpoints_C"]:
            d=root/f"day{day}__response_{setting:g}"
            records=[json.loads(s) for s in (d/"steps.jsonl").read_text().splitlines()]
            branches[setting]=(records[n],pd.read_csv(d/"transitions.csv.gz").iloc[n:])
        low,high=min(branches),max(branches)
        if branches[low][0]["before"]!=branches[high][0]["before"]:
            raise ValueError("Intervention initial states differ")
        for h in [8,24,32]:
            a,b=branches[low],branches[high]
            rows.append({"start_day":day,"horizon_steps":h,
                "actual_delta_temperature_C":float(b[1].operative_end_C.iloc[h-1]-a[1].operative_end_C.iloc[h-1]),
                "predicted_delta_temperature_C":b[0]["control"]["predicted_temperature_C"][h-1]-a[0]["control"]["predicted_temperature_C"][h-1],
                "actual_delta_energy_kWh":float(b[1].interval_energy_kWh.iloc[:h].sum()-a[1].interval_energy_kWh.iloc[:h].sum()),
                "predicted_delta_energy_kWh":sum(b[0]["control"]["predicted_energy_kWh"][:h])-sum(a[0]["control"]["predicted_energy_kWh"][:h])})
    pd.DataFrame(rows).to_csv(root/"response_metrics.csv",index=False,lineterminator="\n")
    checked=[r for r in rows if r["horizon_steps"]==config["horizon_steps"]]
    passed=all(r["actual_delta_temperature_C"]>=-.05 and r["predicted_delta_temperature_C"]>=-1e-8 and
               r["actual_delta_energy_kWh"]>=-.1 and r["predicted_delta_energy_kWh"]>=-1e-8 for r in checked)
    passed &= any(r["actual_delta_energy_kWh"]>.2 for r in checked)
    save_json(root/"response_gate.json",{"passed":bool(passed),"model_sha256":digest(model_path),
        "config_sha256":digest(Path("configs/solar_mpc.json")),"scope":"6h paired direction, not arbitrary-action accuracy or safety"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["collect","fit","response","validation","test"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",type=Path,default=Path("outputs/solar-selected/model.joblib"))
    parser.add_argument("--days",type=float)
    args = parser.parse_args()
    config = configuration()
    if args.mode == "fit":
        fit_model(config,args.output)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    client = Client("http://127.0.0.1:8080")
    base = json.loads(Path(config["dynamics_config"]).read_text())
    if client.call("GET","version")["version"] != base["version"]:
        raise ValueError("BOPTEST version mismatch")
    model = None if args.mode == "collect" else joblib.load(args.model)
    save_json(args.output/"manifest.json",{"config":config,"mode":args.mode,"effective_days":args.days,
        "model_sha256":None if model is None else digest(args.model),
        "code_sha256":{n:digest(Path(__file__).with_name(n)) for n in ["solar.py","greybox.py","mpc.py","recovery.py","simulation.py"]}})
    summaries = []
    if args.mode == "collect":
        for plan in config["new_episodes"]:
            episode = {**base,"episode":plan,"days":args.days or 7,
                "excitation":{**base["excitation"],"nominal_C":config["collection_nominal_C"],
                "levels_K":config["collection_levels_K"],"hold_steps":16,"abort_max_C":config["abort_max_C"]}}
            # Custom name avoids the legacy collection branch and retains native heat observations.
            kind = "solar_excitation"
            summaries.append(run_episode(client,episode,plan["id"],kind,args.output/f"{plan['id']}__{kind}",CollectionPolicy(episode)))
            save_json(args.output/"summary.json",summaries)
    elif args.mode == "response":
        for day in config["response"]["start_days"]:
            for setting in config["response"]["setpoints_C"]:
                name,kind=f"day{day}",f"response_{setting:g}"
                episode={**base,"episode":{"start_day":day},"days":.5}
                summaries.append(run_episode(client,episode,name,kind,args.output/f"{name}__{kind}",SolarResponse(model,config,setting)))
                save_json(args.output/"summary.json",summaries)
        response_gate(args.output,args.model,config)
    else:
        gate=json.loads(Path("outputs/solar-response/response_gate.json").read_text())
        if not gate["passed"] or gate["model_sha256"]!=digest(args.model) or gate["config_sha256"]!=digest(Path("configs/solar_mpc.json")):
            raise ValueError("Matching FMU direction gate required")
        days = config["validation_start_days"] if args.mode == "validation" else config["test_start_days"]
        modes = ["perfect"] if args.mode == "validation" else config["weather_modes"]
        for day in days:
            for weather in modes:
                for kind in config["controllers"]:
                    name = f"day{day}_{weather}"
                    episode = {**base,"episode":{"start_day":day},"days":args.days or 7}
                    summaries.append(run_episode(client,episode,name,kind,args.output/f"{name}__{kind}",SolarController(model,config,kind,weather)))
                    save_json(args.output/"summary.json",summaries)


if __name__ == "__main__":
    main()
