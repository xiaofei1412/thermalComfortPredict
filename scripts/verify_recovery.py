"""Rebuild calibration, replay every policy and audit weather-stressed comparisons."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort.greybox import episode_rollout, fit, metrics, observer_states, rollout
from thermal_comfort.recovery import CollectionPolicy, RecoveryController, configuration, load_data
from thermal_comfort.mpc import response_checks
from thermal_comfort.simulation import digest, integrate, save_json, schedule_at
from verify_mpc import assert_same_info, verify_policies
from verify_simulation import compare_repeat, verify


def verify_model(root, repeat, config):
    manifest = json.loads((root/"manifest.json").read_text())
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("Model artifact hash mismatch")
    for name, expected in manifest["sources_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Changed model source")
    if manifest["config"] != config:
        raise ValueError("Model/config mismatch")
    for name in ["model.joblib","selection.json","metrics.csv","power_metrics.csv","split_manifest.json"]:
        if (root/name).read_bytes() != (repeat/name).read_bytes():
            raise ValueError("Independent refit differs: "+name)
    episodes,splits = load_data(Path("outputs/recovery-calibration"),config)
    if splits != json.loads((root/"split_manifest.json").read_text()):
        raise ValueError("Chronological split changed")
    model = joblib.load(root/"model.joblib")
    if not ((model["parameters"]>0).all() and (model["power"]>=0).all() and model["idle_power_kW"]>=0
            and (model["actuator"][:4]>=0).all() and model["actuator"][0]<=1
            and max(abs(np.linalg.eigvals(model["A"])))<1):
        raise ValueError("Physical constraints failed")
    train = [e for e in episodes.values() if e["split"]=="train"]
    # Refit with training dictionaries only: held-out states/heat cannot enter identification.
    training_only = fit({e["id"]:e for e in train},config)
    for key in ["parameters","A","B","actuator","support_low","support_high"]:
        np.testing.assert_array_equal(model[key],training_only[key])
    heat = np.concatenate([e["heat"] for e in train])
    outdoor = np.concatenate([e["exog"][:,0] for e in train])
    power = np.concatenate([e["states"][1:,3]*4 for e in train])
    idle = heat<config["idle_heat_threshold_kW"]
    np.testing.assert_equal(model["idle_power_kW"],power[idle].mean())
    x = np.column_stack([np.minimum(heat/config["power_transition_kW"],1),heat,heat*np.maximum(20-outdoor,0)/20])
    np.testing.assert_array_equal(model["power"],nnls(x,np.maximum(power-power[idle].mean(),0))[0])
    selection = json.loads((root/"selection.json").read_text())
    e = episodes[selection["selection_episode"]]
    origins = np.array(selection["selection_origins"])
    scores = []
    for weight in config["residual_weights"]:
        p = episode_rollout({**model,"residual_weight":weight},e,origins,16)
        scores.append(float(np.mean([m["temperature_RMSE_C"] for m in metrics(p,e,origins,[1,4,8])])))
    np.testing.assert_allclose(scores,[s["score_C"] for s in selection["scores"]],rtol=1e-12)
    if model["residual_weight"]!=config["residual_weights"][int(np.argmin(scores))]:
        raise ValueError("Incorrect residual selection")
    selection_end = e["times"][origins.max()]+16*900
    errors=[]
    for block in selection["calibration"]:
        e=episodes[block["episode"]]
        origins=np.array(block["origins"])
        if e["split"]!="validation" or e["times"][origins.min()]<=selection_end:
            raise ValueError("Selection/calibration overlap or test contamination")
        p=episode_rollout(model,e,origins,config["horizon_steps"])
        errors.append(p[:,:,0]-e["states"][origins[:,None]+np.arange(1,config["horizon_steps"]+1),0])
    np.testing.assert_allclose(model["margin_K"],np.quantile(abs(np.concatenate(errors)),config["margin_quantile"],axis=0)+config["sensor_margin_K"],rtol=1e-12)
    rows,coverage,weather_rows=[],[],[]
    previous=joblib.load("outputs/greybox-mpc/model.joblib")
    for e in episodes.values():
        if e["split"]=="train":
            continue
        origins=np.arange(16,len(e["actions"])-16+1,4)
        p=episode_rollout(model,e,origins,16)
        rows.extend([{"episode":e["id"],"split":e["split"],"model":"expanded_gated",**m} for m in metrics(p,e,origins,[1,4,8,16])])
        for h in [1,4,8]:
            error=p[:,h-1,0]-e["states"][origins+h,0]
            support=np.stack([np.column_stack([e["states"][origins,0],e["exog"][origins+k,:2]]) for k in range(h)],axis=1)
            in_range=((support>=model["support_low"]-config["support_tolerance"])&(support<=model["support_high"]+config["support_tolerance"])).all(axis=(1,2))
            old_range=((support>=previous["support_low"]-config["support_tolerance"])&(support<=previous["support_high"]+config["support_tolerance"])).all(axis=(1,2))
            coverage.append({"episode":e["id"],"horizon_steps":h,"margin_K":model["margin_K"][h-1],
                "margin_coverage":float(np.mean(abs(error)<=model["margin_K"][h-1])),"input_support_fraction":float(in_range.mean()),
                "previous_input_support_fraction":float(old_range.mean()),
                "temperature_support_fraction":float(((support[:,0,0]>=model["support_low"][0]-config["support_tolerance"][0])&
                    (support[:,0,0]<=model["support_high"][0]+config["support_tolerance"][0])).mean()),
                "previous_temperature_support_fraction":float(((support[:,0,0]>=previous["support_low"][0]-config["support_tolerance"][0])&
                    (support[:,0,0]<=previous["support_high"][0]+config["support_tolerance"][0])).mean())})
        estimated=observer_states(model,e)
        actions=np.stack([e["actions"][k:k+16] for k in origins])
        base_future=np.stack([e["exog"][k:k+16] for k in origins])
        ramp=np.clip(np.arange(16)/config["horizon_steps"],0,1)
        for mode,sign in [("perfect",0),("warm_bias",1),("cold_bias",-1)]:
            future=base_future.copy()
            future[:,:,0]+=sign*ramp*config["forecast_temperature_error_K"]
            future[:,:,1]*=1+sign*ramp*config["forecast_solar_error_fraction"]
            p=rollout(model,estimated[origins],e["heat"][origins-1],
                e["actions"][origins-1]-e["states"][origins-1,0],actions,future)
            weather_rows.extend([{"episode":e["id"],"split":e["split"],"weather_mode":mode,**m} for m in metrics(p,e,origins,[1,4,8,16])])
    saved=pd.read_csv(root/"metrics.csv")
    pd.testing.assert_frame_equal(pd.DataFrame(rows),saved[saved.model=="expanded_gated"].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-12)
    pd.DataFrame(coverage).to_csv(root/"coverage.csv",index=False,lineterminator="\n")
    pd.DataFrame(weather_rows).to_csv(root/"weather_prediction_metrics.csv",index=False,lineterminator="\n")
    return {"independent_refit_byte_identical":True,"train_only_refit_matches":True,
            "selection_margin_and_power_rebuilt":True,"chronological_role_isolation":True,
            "training_rows":sum(len(e["actions"]) for e in train),"idle_training_rows":int(idle.sum()),
            "calibration_origins":sum(len(b["origins"]) for b in selection["calibration"])}


def verify_collection(root):
    summaries=json.loads((root/"summary.json").read_text())
    results=[]
    for s in summaries:
        d=root/f"{s['scenario']}__{s['controller']}"
        records=[json.loads(line) for line in (d/"steps.jsonl").read_text().splitlines()]
        forecast=pd.read_csv(d/"exogenous.csv.gz")
        native=pd.read_csv(d/"native.csv.gz")
        policy=CollectionPolicy(s["config"])
        policy.reset(records[0]["before"],json.loads((d/"api_metadata.json").read_text())["inputs"])
        for k,r in enumerate(records):
            action,info=policy.decide(k,r["before"],forecast)
            if action!=r["action"]:
                raise ValueError("Calibration policy replay failed")
            assert_same_info(info,r["control"])
        t=native.reaTZon_y-273.15
        if t.min()<18 or t.max()>28:
            raise ValueError("Calibration native stop band failed")
        results.append({"episode":s["scenario"],"replayed_steps":len(records),"temperature_min_C":float(t.min()),"temperature_max_C":float(t.max())})
    return results


def comfort_breakdown(native,forecast):
    schedule=schedule_at(forecast,native.time.to_numpy())
    low=np.maximum(schedule["LowerSetp[1]"].to_numpy()-native.reaTZon_y.to_numpy(),0)
    high=np.maximum(native.reaTZon_y.to_numpy()-schedule["UpperSetp[1]"].to_numpy(),0)
    return {"cold_Kh":float(np.trapezoid(low,native.time)/3600),"hot_Kh":float(np.trapezoid(high,native.time)/3600),
            "outside_hours":float(np.trapezoid(((low+high)>1e-8).astype(float),native.time)/3600),
            "maximum_cold_K":float(low.max()),"maximum_hot_K":float(high.max())}


def comparison(root):
    summaries=json.loads((root/"summary.json").read_text())
    rows,daily=[],[]
    for s in summaries:
        d=root/f"{s['scenario']}__{s['controller']}"
        native=pd.read_csv(d/"native.csv.gz")
        forecast=pd.read_csv(d/"exogenous.csv.gz")
        rows.append({"scenario":s["scenario"],"controller":s["controller"],**s["metrics"],**comfort_breakdown(native,forecast)})
        for day in range(int(s["config"]["days"])):
            start=s["start_time_s"]+day*86400
            part=native.loc[native.time.between(start,start+86400)]
            daily.append({"scenario":s["scenario"],"controller":s["controller"],"evaluation_day":day,
                          **integrate(part,forecast,192),**comfort_breakdown(part,forecast)})
    frame=pd.DataFrame(rows)
    frame.to_csv(root/"comparison.csv",index=False,lineterminator="\n")
    pd.DataFrame(daily).to_csv(root/"daily_comparison.csv",index=False,lineterminator="\n")
    gates=[]
    for name,group in frame.groupby("scenario",sort=False):
        rule=group[group.controller=="guarded_rule"].iloc[0]
        mpc=group[group.controller=="mpc"].iloc[0]
        gates.append({"scenario":name,"energy_saved_percent":float((rule.energy_kWh-mpc.energy_kWh)/rule.energy_kWh*100),
            "discomfort_delta_Kh":float(mpc.discomfort_Kh-rule.discomfort_Kh),
            "cold_delta_Kh":float(mpc.cold_Kh-rule.cold_Kh),"hot_delta_Kh":float(mpc.hot_Kh-rule.hot_Kh),
            "energy_lower":bool(mpc.energy_kWh<rule.energy_kWh-1e-6),
            "discomfort_not_increased":bool(mpc.discomfort_Kh<=rule.discomfort_Kh+1e-6),
            "both_components_not_increased":bool(mpc.cold_Kh<=rule.cold_Kh+1e-6 and mpc.hot_Kh<=rule.hot_Kh+1e-6),
            "outside_hours_not_increased":bool(mpc.outside_hours<=rule.outside_hours+1e-6),
            "maximum_deviations_not_increased":bool(mpc.maximum_cold_K<=rule.maximum_cold_K+1e-6 and mpc.maximum_hot_K<=rule.maximum_hot_K+1e-6)})
    save_json(root/"acceptance.json",{"scenarios":gates,
        "all_energy_and_integral_pass":all(g["energy_lower"] and g["discomfort_not_increased"] for g in gates),
        "all_comfort_measures_pass":all(g["both_components_not_increased"] and g["outside_hours_not_increased"] and g["maximum_deviations_not_increased"] for g in gates)})
    return gates


def verify_plan(root, config, model_path, days, start_days):
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["config"]!=config or manifest["model_sha256"]!=digest(model_path):
        raise ValueError("Frozen control model/config mismatch")
    if manifest["config_sha256"]!=digest(Path("configs/recovery_mpc.json")):
        raise ValueError("Frozen config file changed")
    for name,expected in manifest["code_sha256"].items():
        if digest(Path("thermal_comfort")/name)!=expected:
            raise ValueError("Controller implementation changed after run began")
    summaries=json.loads((root/"summary.json").read_text())
    expected={(f"day{day}_{mode}",kind) for day in start_days for mode in config["weather_modes"] for kind in ["guarded_rule","mpc"]}
    actual={(s["scenario"],s["controller"]) for s in summaries}
    if actual!=expected or len(actual)!=len(summaries) or any(s["transitions"]!=days*96 for s in summaries):
        raise ValueError("Incomplete control comparison grid")


if __name__=="__main__":
    config=configuration()
    model_root=Path("outputs/recovery-model")
    model=joblib.load(model_root/"model.joblib")
    root=Path("outputs/recovery-control")
    response=Path("outputs/recovery-response")
    verify_plan(root,config,model_root/"model.joblib",7,config["test_start_days"])
    verify_plan(Path("outputs/recovery-repeat"),config,model_root/"model.joblib",.25,config["test_start_days"])
    result={"model":verify_model(model_root,Path("outputs/recovery-model-repeat"),config),
        "collection_trajectories":verify(Path("outputs/recovery-calibration")),
        "collection_policy_replay":verify_collection(Path("outputs/recovery-calibration")),
        "response_trajectories":verify(response),"validation_trajectories":verify(Path("outputs/recovery-validation")),
        "control_trajectories":verify(root),"repeat_trajectories":verify(Path("outputs/recovery-repeat")),
        "response_policy_replay":verify_policies(response,model,config,True)}
    factory=lambda s:RecoveryController(model,config,s["controller"],s["scenario"].split("_",1)[1])
    result["validation_policy_replay"]=verify_policies(Path("outputs/recovery-validation"),model,config,policy_factory=factory)
    result["control_policy_replay"]=verify_policies(root,model,config,policy_factory=factory)
    gate=json.loads((response/"response_gate.json").read_text())
    original=(response/"response_metrics.csv").read_bytes()
    response_checks(response,config,model_root/"model.joblib",Path("configs/recovery_mpc.json"))
    if gate!=json.loads((response/"response_gate.json").read_text()) or not gate["passed"] or original!=(response/"response_metrics.csv").read_bytes():
        raise ValueError("Response gate changed")
    result["paired_action_response_gate_rebuilt"]=True
    result["short_repeat_exact"]=compare_repeat(root,Path("outputs/recovery-repeat"))
    result["acceptance"]=comparison(root)
    # Same measured FMU weather for all error modes; weather-independent rule is invariant.
    for day in config["test_start_days"]:
        reference=root/f"day{day}_perfect__guarded_rule"
        for mode in config["weather_modes"][1:]:
            other=root/f"day{day}_{mode}__guarded_rule"
            for name in ["native.csv.gz","transitions.csv.gz","exogenous.csv.gz"]:
                pd.testing.assert_frame_equal(pd.read_csv(reference/name),pd.read_csv(other/name),check_exact=True)
    result["weather_independent_rule_invariant"]=True
    result["frozen_controller_and_complete_grid"]=True
    save_json(root/"verification.json",result)
    print(json.dumps(result["acceptance"],indent=2))
