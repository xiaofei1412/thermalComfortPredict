"""Audit the seasonal split, candidate selection, frozen policies and FMU repeats."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import greybox
from thermal_comfort.recovery import CollectionPolicy
from thermal_comfort.solar import SolarController, SolarResponse, configuration, episode_rollout, load_data, observer_states, read_episode, response_gate
from thermal_comfort.simulation import digest, integrate, save_json, schedule_at
from verify_mpc import assert_same_info, verify_policies
from verify_recovery import comfort_breakdown
from verify_simulation import compare_repeat, verify


def model_check(config):
    root,repeat=Path("outputs/solar-selected"),Path("outputs/solar-refit")
    manifest=json.loads((root/"manifest.json").read_text())
    refit_manifest=json.loads((repeat/"manifest.json").read_text())
    if refit_manifest["config"]!=config or refit_manifest["code_sha256"]!=digest(Path("thermal_comfort/solar.py")):
        raise ValueError("Independent refit does not match the final implementation")
    for name,expected in manifest["artifacts_sha256"].items():
        if digest(root/name)!=expected or (root/name).read_bytes()!=(repeat/name).read_bytes():
            raise ValueError("Changed artifact or nonidentical independent fit: "+name)
    episodes,splits=load_data(config)
    if splits!=json.loads((root/"split_manifest.json").read_text()):
        raise ValueError("Split changed")
    train={key:e for key,e in episodes.items() if e["split"]=="train"}
    raw=greybox.fit(train,config)
    expanded=joblib.load(root/"expanded_baseline.joblib")
    for name in ["A","B","parameters","actuator","support_low","support_high"]:
        np.testing.assert_array_equal(raw[name],expanded[name])
    heat=np.concatenate([e["heat"] for e in train.values()])
    outdoor=np.concatenate([e["exog"][:,0] for e in train.values()])
    power=np.concatenate([e["states"][1:,3]*4 for e in train.values()])
    idle=power[heat<config["idle_heat_threshold_kW"]].mean()
    x=np.column_stack([np.minimum(heat/config["power_transition_kW"],1),heat,heat*np.maximum(20-outdoor,0)/20])
    np.testing.assert_equal(expanded["idle_power_kW"],idle)
    np.testing.assert_array_equal(expanded["power"],nnls(x,np.maximum(power-idle,0))[0])
    selection=json.loads((root/"selection.json").read_text())
    first=episodes[selection["selection_episode"]]
    origins=np.array(selection["selection_origins"])
    scores=[]
    for name in ["expanded_baseline","expanded_mhe","stored_solar_mhe"]:
        model=joblib.load(root/(name+".joblib"))
        if not ((model["parameters"]>0).all() and (model["power"]>=0).all() and max(abs(np.linalg.eigvals(model["A"])))<1):
            raise ValueError("Physical constraints failed")
        for weight in config["residual_weights"]:
            p=episode_rollout({**model,"residual_weight":weight},first,origins,32)
            scores.append({"model":name,"weight":weight,"RMSE_C":float(np.mean([r["temperature_RMSE_C"] for r in greybox.metrics(p,first,origins,[4,8,24,32])]))})
    pd.testing.assert_frame_equal(pd.DataFrame(scores),pd.DataFrame(selection["scores"]),rtol=1e-12,atol=1e-12)
    chosen=min(scores,key=lambda r:r["RMSE_C"])
    if chosen["model"]!=selection["selected_model"] or chosen["weight"]!=selection["selected_weight"]:
        raise ValueError("Candidate selection failed")
    model=joblib.load(root/"model.joblib")
    errors=[]
    for block in selection["calibration"]:
        e=episodes[block["episode"]]; starts=np.array(block["origins"])
        if e["split"]!="validation" or e["times"][starts.min()]<=first["times"][origins.max()]+32*900:
            raise ValueError("Selection/calibration overlap")
        p=episode_rollout(model,e,starts,32)
        errors.append(p[:,:,0]-e["states"][starts[:,None]+np.arange(1,33),0])
    np.testing.assert_allclose(model["margin_K"],np.quantile(abs(np.concatenate(errors)),config["margin_quantile"],axis=0)+config["sensor_margin_K"],rtol=1e-12)
    rows=[]
    previous=joblib.load("outputs/recovery-model/model.joblib")
    for e in episodes.values():
        if e["split"]!="validation": continue
        origins=np.arange(16,641,4)
        for name in ["previous","expanded_baseline","expanded_mhe","stored_solar_mhe"]:
            m=previous if name=="previous" else joblib.load(root/(name+".joblib"))
            p=(greybox.episode_rollout if name=="previous" else episode_rollout)(m,e,origins,32)
            rows.extend([{"episode":e["id"],"model":name,**r} for r in greybox.metrics(p,e,origins,[1,4,8,24,32])])
    pd.testing.assert_frame_equal(pd.DataFrame(rows),pd.read_csv(root/"metrics.csv"),check_dtype=False,rtol=1e-12,atol=1e-12)
    return {"independent_refit_byte_identical":True,"train_only_RC_actuator_and_power_rebuilt":True,
            "selection_and_margins_rebuilt":True,"chronological_split_pass":True,
            "retired_test_day119_used_as_training":True,"selected_model":chosen["model"],
            "training_rows":sum(len(e["actions"]) for e in train.values()),"calibration_origins":sum(len(b["origins"]) for b in selection["calibration"])}


def check_grid(root,config,days,dates,modes):
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["config"]!=config or manifest["model_sha256"]!=digest(Path("outputs/solar-selected/model.joblib")):
        raise ValueError("Frozen model/config mismatch")
    for name,expected in manifest["code_sha256"].items():
        if digest(Path("thermal_comfort")/name)!=expected:
            raise ValueError("Controller source changed after run began")
    summaries=json.loads((root/"summary.json").read_text())
    expected={(f"day{d}_{w}",k) for d in dates for w in modes for k in config["controllers"]}
    actual={(s["scenario"],s["controller"]) for s in summaries}
    if actual!=expected or len(actual)!=len(summaries) or any(s["transitions"]!=days*96 for s in summaries):
        raise ValueError("Comparison grid incomplete")


def collect_replay(config):
    root=Path("outputs/solar-data")
    result=[]
    for s in json.loads((root/"summary.json").read_text()):
        d=root/f"{s['scenario']}__{s['controller']}"
        records=[json.loads(x) for x in (d/"steps.jsonl").read_text().splitlines()]
        forecast=pd.read_csv(d/"exogenous.csv.gz")
        policy=CollectionPolicy(s["config"])
        policy.reset(records[0]["before"],json.loads((d/"api_metadata.json").read_text())["inputs"])
        for k,r in enumerate(records):
            action,info=policy.decide(k,r["before"],forecast)
            if action!=r["action"]: raise ValueError("Collection action replay failed")
            assert_same_info(info,r["control"])
        native=pd.read_csv(d/"native.csv.gz")
        t=native.reaTZon_y-273.15
        if t.min()<config["abort_min_C"] or t.max()>config["abort_max_C"]: raise ValueError("Collection stop band violated")
        result.append({"episode":s["scenario"],"replayed":len(records),"temperature_min_C":float(t.min()),
                       "temperature_max_C":float(t.max()),"native_points_above_28C":int((t>28).sum())})
    return result


def summarize(root,config):
    rows,daily=[],[]
    for s in json.loads((root/"summary.json").read_text()):
        d=root/f"{s['scenario']}__{s['controller']}"
        native=pd.read_csv(d/"native.csv.gz"); forecast=pd.read_csv(d/"exogenous.csv.gz")
        rows.append({"scenario":s["scenario"],"controller":s["controller"],**s["metrics"],**comfort_breakdown(native,forecast),
                     "native_points_above_28C":int((native.reaTZon_y>301.15).sum())})
        for day in range(int(s["config"]["days"])):
            start=s["start_time_s"]+day*86400
            part=native.loc[native.time.between(start,start+86400)]
            daily.append({"scenario":s["scenario"],"controller":s["controller"],"evaluation_day":day,
                          **integrate(part,forecast,192),**comfort_breakdown(part,forecast)})
    frame=pd.DataFrame(rows)
    frame.to_csv(root/"comparison.csv",index=False,lineterminator="\n")
    pd.DataFrame(daily).to_csv(root/"daily_comparison.csv",index=False,lineterminator="\n")
    gates=[]
    for scenario,f in frame.groupby("scenario",sort=False):
        rule=f[f.controller=="guarded_rule"].iloc[0]; mpc=f[f.controller=="solar_mpc"].iloc[0]
        minimum=f[f.controller=="minimum_heat"].iloc[0]
        gates.append({"scenario":scenario,"energy_saved_percent":float((rule.energy_kWh-mpc.energy_kWh)/rule.energy_kWh*100) if rule.energy_kWh>1e-9 else None,
            "energy_delta_kWh":float(mpc.energy_kWh-rule.energy_kWh),"discomfort_delta_Kh":float(mpc.discomfort_Kh-rule.discomfort_Kh),
            "energy_lower":bool(mpc.energy_kWh<rule.energy_kWh-1e-6),"discomfort_not_increased":bool(mpc.discomfort_Kh<=rule.discomfort_Kh+1e-6),
            "all_comfort_measures_not_increased":bool(all(mpc[key]<=rule[key]+1e-6 for key in ["cold_Kh","hot_Kh","outside_hours","maximum_cold_K","maximum_hot_K"])),
            "energy_delta_vs_minimum_kWh":float(mpc.energy_kWh-minimum.energy_kWh),
            "discomfort_delta_vs_minimum_Kh":float(mpc.discomfort_Kh-minimum.discomfort_Kh)})
    save_json(root/"acceptance.json",{"scenarios":gates,"all_energy_and_integral_pass":all(g["energy_lower"] and g["discomfort_not_increased"] for g in gates)})
    return gates


def test_predictions(root,config):
    rows=[]
    models={name:joblib.load(Path("outputs/solar-selected")/(name+".joblib")) for name in ["expanded_baseline","expanded_mhe","stored_solar_mhe"]}
    models["previous"]=joblib.load("outputs/recovery-model/model.joblib")
    for s in json.loads((root/"summary.json").read_text()):
        d=root/f"{s['scenario']}__{s['controller']}"
        adapted=copy.deepcopy(s); adapted["config"]["episode"]["split"]="test"
        e,_=read_episode(d,adapted,"test")
        origins=np.arange(16,641,4)
        mode=s["scenario"].split("_",1)[1]
        sign={"perfect":0,"warm_bias":1,"cold_bias":-1}[mode]
        future=np.stack([e["exog"][k:k+32] for k in origins])
        ramp=np.clip(np.arange(32)/config["horizon_steps"],0,1)
        future[:,:,0]+=sign*ramp*config["forecast_temperature_error_K"]
        future[:,:,1]*=1+sign*ramp*config["forecast_solar_error_fraction"]
        for name,m in models.items():
            estimated=(greybox.observer_states if name=="previous" else observer_states)(m,e)
            p=greybox.rollout(m,estimated[origins],e["heat"][origins-1],e["actions"][origins-1]-e["states"][origins-1,0],
                np.stack([e["actions"][k:k+32] for k in origins]),future)
            for r in greybox.metrics(p,e,origins,[1,4,8,24,32]):
                h=r["horizon_steps"]
                error=p[:,h-1,0]-e["states"][origins+h,0]
                margin=m.get("margin_K")
                support=np.dstack([np.tile(e["states"][origins,0,None],(1,h)),future[:,:h,0],future[:,:h,1]])
                in_support=((support>=m["support_low"]-config["support_tolerance"])&
                            (support<=m["support_high"]+config["support_tolerance"])).all(axis=(1,2))
                rows.append({"scenario":s["scenario"],"controller":s["controller"],"model":name,**r,
                    "margin_coverage":float(np.mean(abs(error)<=margin[h-1])) if margin is not None and h<=len(margin) else None,
                    "input_support_fraction":float(in_support.mean())})
    pd.DataFrame(rows).to_csv(root/"prediction_metrics.csv",index=False,lineterminator="\n")


if __name__=="__main__":
    config=configuration(); model=joblib.load("outputs/solar-selected/model.joblib")
    root=Path("outputs/solar-control"); repeat=Path("outputs/solar-repeat"); validation=Path("outputs/solar-validation-v2"); response=Path("outputs/solar-response")
    check_grid(root,config,7,config["test_start_days"],config["weather_modes"])
    check_grid(repeat,config,.25,config["test_start_days"],config["weather_modes"])
    check_grid(validation,config,7,config["validation_start_days"],["perfect"])
    audit_config={**config,"closed_loop":{"controllers":["guarded_rule","solar_mpc"]}}
    factory=lambda s:SolarController(model,config,s["controller"],s["scenario"].split("_",1)[1])
    result={"model":model_check(config),"collection_trajectories":verify(Path("outputs/solar-data")),"collection_replay":collect_replay(config),
        "response_trajectories":verify(response),"validation_trajectories":verify(validation),"control_trajectories":verify(root),"repeat_trajectories":verify(repeat),
        "response_replay":verify_policies(response,model,audit_config,True,lambda s:SolarResponse(model,config,float(s["controller"].split("_")[-1]))),
        "validation_replay":verify_policies(validation,model,audit_config,policy_factory=factory),
        "control_replay":verify_policies(root,model,audit_config,policy_factory=factory)}
    v1=Path("outputs/solar-validation")
    source=v1/"source_solar_v1.py"
    expected=json.loads((v1/"manifest.json").read_text())["code_sha256"]["solar.py"]
    if digest(source)!=expected:
        raise ValueError("Historical validation source does not match its frozen hash")
    spec=importlib.util.spec_from_file_location("thermal_comfort._solar_v1_evidence",source)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    result["validation_v1_trajectories"]=verify(v1)
    result["validation_v1_replay"]=verify_policies(v1,model,audit_config,policy_factory=lambda s:old.SolarController(model,config,s["controller"],"perfect"))
    result["validation_v1_source_hash_matches"]=True
    original_gate=json.loads((response/"response_gate.json").read_text()); original_metrics=(response/"response_metrics.csv").read_bytes()
    response_gate(response,Path("outputs/solar-selected/model.joblib"),config)
    if original_gate!=json.loads((response/"response_gate.json").read_text()) or not original_gate["passed"] or original_metrics!=(response/"response_metrics.csv").read_bytes():
        raise ValueError("Intervention gate failed to reproduce")
    for day in config["response"]["start_days"]:
        branches=[]
        for setting in config["response"]["setpoints_C"]:
            d=response/f"day{day}__response_{setting:g}"
            branches.append([json.loads(x) for x in (d/"steps.jsonl").read_text().splitlines()])
        n=config["response"]["prelude_steps"]
        if any(branch[n]["before"]!=branches[0][n]["before"] for branch in branches[1:]):
            raise ValueError("All three intervention branches must share the initial state")
    for directory in [root,validation]:
        summaries=json.loads((directory/"summary.json").read_text())
        for scenario in dict.fromkeys(s["scenario"] for s in summaries):
            paths=[directory/f"{scenario}__{k}" for k in config["controllers"]]
            records=[[json.loads(x) for x in (p/"steps.jsonl").read_text().splitlines()] for p in paths]
            for branch in records[1:]:
                if branch[0]["before"]!=records[0][0]["before"] or any(a["action"]!=b["action"] or a["after"]!=b["after"] for a,b in zip(records[0][:8],branch[:8])):
                    raise ValueError("All-three-controller common initialization failed")
            for path in paths[1:]:
                pd.testing.assert_frame_equal(pd.read_csv(paths[0]/"exogenous.csv.gz"),pd.read_csv(path/"exogenous.csv.gz"),check_exact=True)
    for day in config["test_start_days"]:
        for kind in ["guarded_rule","minimum_heat"]:
            for mode in config["weather_modes"][1:]:
                for name in ["native.csv.gz","transitions.csv.gz","exogenous.csv.gz"]:
                    pd.testing.assert_frame_equal(pd.read_csv(root/f"day{day}_perfect__{kind}"/name),pd.read_csv(root/f"day{day}_{mode}__{kind}"/name),check_exact=True)
    result["repeat_exact"]=compare_repeat(root,repeat)
    result["paired_initialization_true_weather_and_reference_invariance"]=True
    result["acceptance"]=summarize(root,config)
    result["validation_acceptance"]=summarize(validation,config)
    test_predictions(root,config)
    save_json(root/"verification.json",result)
    print(json.dumps(result["acceptance"],indent=2))
