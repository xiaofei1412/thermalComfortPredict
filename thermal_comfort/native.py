"""Native-time-scale development; calibration remains closed unless all gates pass."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import native_model as nm, office as o, office_model as om, terminal as t, terminal_model as tm
from .solar import SolarController
from .simulation import digest, save_json, schedule_at

CONFIG=Path("configs/native_terminal.json")
MODEL=Path("outputs/native-model")
RULE=Path("outputs/native-rule-development")
CALIBRATION=Path("outputs/native-calibration")
LOCK=Path("outputs/native-lock/development_lock.json")
CORE=["thermal_comfort/native.py","thermal_comfort/native_model.py",str(CONFIG),"docs/experiments/native_terminal.md"]
CARRIERS=["gas","cooling_electricity","fan_electricity"]


def configuration():return t.load(CONFIG)
def hashes():return {n:digest(Path(n)) for n in CORE}


def validate_roles(config):
    old=t.configuration();t.validate_windows(old)
    if config["training_days"]!=[e["day"] for e in old["episodes"] if e["role"]=="train"]:
        raise ValueError("Native train role differs from registered development source")
    if [e["day"] for e in config["development"]]!=[e["day"] for e in old["episodes"] if e["role"]=="selection"]:
        raise ValueError("Native development role differs from source")
    previous=o.configuration();g=t.load(Path("configs/generalization.json"))
    used=[(e["day"]-7,e["day"]+3) for e in old["episodes"]+previous["episodes"]]
    used += [(d-7,d+7) for d in previous["test_days"]+g["start_days"]]
    used += [(d-7,d+.5) for d in previous["response_days"]]
    windows=[(d-7,d+config["reserved_calibration_length_days"]) for d in config["reserved_calibration_days"]]
    if any(a<d and b>c for a,b in windows for c,d in used):raise ValueError("Reserved calibration overlaps historical office data")
    if any(a<d and b>c for i,(a,b) in enumerate(windows) for c,d in windows[i+1:]):raise ValueError("Calibration windows overlap")


def read_native(root,summary,role):
    if role not in ["train","development","calibration","evaluation"]:raise ValueError("Unknown data role")
    if not summary["complete"]:raise ValueError("Native identification requires complete trajectories")
    for name,h in summary["sha256"].items():
        if digest(root/name)!=h:raise ValueError("Native source trajectory changed")
    frame=pd.read_csv(root/"native.csv.gz",dtype=float,float_precision="round_trip")
    forecast=pd.read_csv(root/"exogenous.csv.gz",dtype=float,float_precision="round_trip")
    records=[json.loads(line) for line in (root/"steps.jsonl").read_text().splitlines()]
    times=frame.time.to_numpy();n=len(times)-1
    if not np.all(np.diff(times)==30) or n!=30*len(records):raise ValueError("Native grid/decision length mismatch")
    power=frame[["fcu_reaPHea_y","fcu_reaPCoo_y","fcu_reaPFan_y"]].to_numpy()/1000
    flow=frame.fcu_reaFloSup_y.to_numpy()/.55
    if (power < -1e-9).any() or (flow < -1e-6).any() or (flow>1+1e-6).any():
        raise ValueError("Unexpected native energy/flow outside numerical tolerance")
    actions=np.repeat([r["action"]["oveTSet_u"]-273.15 for r in records],30)
    return {"id":root.name,"day":summary["day"],"role":role,"times":times[:-1],
        "temperature":frame[o.IO["temperature"]].to_numpy()-273.15,"power":np.maximum(power,0),
        "fan":np.clip(flow,0,1),"actions":actions,"initial_action":records[0]["before"]["oveTSet_u"]-273.15,
        "settings":om.cooling_setpoint(times[:-1]),"initial_setting":float(om.cooling_setpoint(times[0]-30)),
        "exog":o.office_exogenous(schedule_at(forecast,times[:-1])),"root":root}


def source_episodes(role):
    if role not in ["train","development"]:raise ValueError("Historical source cannot supply calibration or evaluation roles")
    config=configuration();manifest=t.load(t.DEVELOPMENT/"manifest.json");summaries=t.load(t.DEVELOPMENT/"summary.json")
    plans=manifest["plans"]
    for name,h in manifest["code_sha256"].items():
        if digest(t.DEVELOPMENT/"source"/Path(name).name)!=h:raise ValueError("Historical collection source snapshot changed")
    if any(manifest["config"][key]!=t.configuration()[key] for key in ["episodes","collection_days","dwell_steps"]):
        raise ValueError("Historical collection plan changed")
    wanted=config["training_days"] if role=="train" else [e["day"] for e in config["development"]]
    result=[]
    for p,s in zip(plans,summaries):
        if s["day"] in wanted:
            if p["role"]!=("train" if role=="train" else "selection"):raise ValueError("Source role mismatch")
            result.append(read_native(t.DEVELOPMENT/f"day{s['day']}__excitation",s,role))
    if len(result)!=len(wanted):raise ValueError("Missing native development data")
    return result


def collect_rule():
    if LOCK.exists() or CALIBRATION.exists():raise ValueError("Development is closed")
    t.check_final();config=configuration();validate_roles(config)
    RULE.mkdir(parents=True,exist_ok=False)
    save_json(RULE/"manifest.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"role":"development",
        "config":config,"collection_protocol_sha256":digest(Path(CORE[-1])),"prior_freeze_sha256":digest(t.FINAL/"freeze.json")})
    results=[]
    for plan in config["development"]:
        day=plan["day"];policy=SolarController(joblib.load(t.FREEZE/"model.joblib"),t.policy_config(),"guarded_rule")
        results.append(t.run_episode(config,"bestest_air",day,3,"guarded_rule","perfect",policy,RULE/f"day{day}__guarded_rule"))
        save_json(RULE/"summary.json",results)


def fit(output):
    if LOCK.exists() or CALIBRATION.exists():raise ValueError("Native fitting is closed")
    t.check_final();config=configuration();validate_roles(config)
    episodes=source_episodes("train")
    base=joblib.load(t.MODEL/"model.joblib")
    start=np.r_[base["parameters"],120.,.9,30.,1.,1.,.15,.001]
    output.mkdir(parents=True,exist_ok=False)
    first=nm.fit(episodes,start,config,False)
    joblib.dump(first,output/"native_one_step.joblib");save_json(output/"native_one_step.json",first["diagnostics"])
    print("Native one-step identification: "+json.dumps(first["diagnostics"]),flush=True)
    joint=nm.fit(episodes,first["parameters"],config,True)
    joblib.dump(joint,output/"native_joint.joblib");save_json(output/"native_joint.json",joint["diagnostics"])
    save_json(output/"manifest.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"code_sha256":hashes(),
        "source_summary_sha256":digest(t.DEVELOPMENT/"summary.json"),"training_days":config["training_days"],
        "artifacts_sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print("Native joint identification: "+json.dumps(joint["diagnostics"]),flush=True)


def metric_rows(name,episode,p,y,seconds):
    rows=[]
    for j,h in enumerate(seconds):
        for k,label in enumerate(["temperature",*CARRIERS,*[c+"_energy" for c in CARRIERS]]):
            rows.append({"model":name,"day":episode["day"],"trajectory":episode["id"],"seconds":h,"target":label,
                "origins":len(p),"RMSE":float(np.sqrt(np.mean((p[:,j,k]-y[:,j,k])**2))),
                "bias":float(np.mean(p[:,j,k]-y[:,j,k])),"truth_RMS":float(np.sqrt(np.mean(y[:,j,k]**2)))})
    return rows


def baseline_prediction(episode,origins):
    root=episode["root"];summary=t.load(root/"summary.json");coarse=t.read_episode(root,summary,"selection")
    p=tm.episode_rollout(joblib.load(t.MODEL/"model.joblib"),coarse,origins//30,24)
    result=np.empty((len(origins),2,7))
    for j,index in enumerate([4,24]):
        result[:,j,0]=p[:,index-1,0]
        result[:,j,1:4]=p[:,index-1,2:5]/[.9,3,1]  # mean powers only, never used as endpoint gates
        result[:,j,4]=p[:,:index,6].sum(axis=1)
        result[:,j,5]=p[:,:index,3].sum(axis=1)*.25/3
        result[:,j,6]=p[:,:index,4].sum(axis=1)*.25
    return result


def capability_gates(rows,activity,config):
    gates=[];limits=config["gates"]
    def add(model,day,trajectory,target,seconds,criterion,value,limit,passed):
        gates.append(dict(model=model,day=day,trajectory=trajectory,target=target,seconds=seconds,
            criterion=criterion,value=float(value),limit=float(limit),passed=bool(passed)))
    lookup={(r["model"],r["day"],r["trajectory"],r["target"],r["seconds"]):r for r in rows}
    for r in rows:
        model,day,trajectory,target,h=[r[k] for k in ["model","day","trajectory","target","seconds"]]
        if model=="frozen_static":continue
        if target=="temperature" and str(h) in limits["temperature_RMSE_K"]:
            base=lookup[("frozen_static",day,trajectory,target,h)]
            limit=min(limits["temperature_RMSE_K"][str(h)],base["RMSE"]+limits["temperature_regression_tolerance_K"])
            add(model,day,trajectory,target,h,"temperature_accuracy",r["RMSE"],limit,r["RMSE"]<=limit)
        for j,c in enumerate(CARRIERS):
            if target==c+"_energy" and h in [3600,21600]:
                base=lookup[("frozen_static",day,trajectory,target,h)]
                limit=min(max(limits["energy_absolute_floor_kWh"][j],limits["energy_relative_RMSE"]*r["truth_RMS"]),
                    base["RMSE"]+limits["energy_regression_tolerance_kWh"][j])
                add(model,day,trajectory,target,h,"energy_accuracy",r["RMSE"],limit,r["RMSE"]<=limit)
            if target==c and h in [300,900]:
                limit=max(limits["power_absolute_floor_kW"][j],limits["power_relative_RMSE"]*r["truth_RMS"])
                add(model,day,trajectory,target,h,"native_endpoint_accuracy",r["RMSE"],limit,r["RMSE"]<=limit)
    for model in ["native_one_step","native_joint"]:
        for row in activity:
            for j,c in enumerate(CARRIERS):
                count=row["active_origins"][j]
                add(model,row["day"],row["trajectory"],c,21600,"active_support",count,limits["minimum_active_origins"],count>=limits["minimum_active_origins"])
            add(model,row["day"],row["trajectory"],"all",21600,"origin_count",row["origins"],limits["minimum_origins"],row["origins"]>=limits["minimum_origins"])
    return gates


def evaluate():
    if LOCK.exists() or CALIBRATION.exists():raise ValueError("Native development assessment is closed")
    t.check_final();config=configuration();validate_roles(config)
    manifest=t.load(MODEL/"manifest.json")
    if manifest["code_sha256"]!=hashes():raise ValueError("Native source differs from fitted manifest")
    for name,h in manifest["artifacts_sha256"].items():
        if digest(MODEL/name)!=h:raise ValueError("Native model artifact changed")
    episodes=source_episodes("development")
    summaries=t.load(RULE/"summary.json")
    expected={(p["day"],config["development_days"],"bestest_air","guarded_rule","perfect") for p in config["development"]}
    observed={(s["day"],s["days"],s["case"],s["controller"],s["weather"]) for s in summaries}
    if observed!=expected or len(summaries)!=len(expected):raise ValueError("Incomplete rule development coverage")
    for s in summaries:
        episodes.append(read_native(RULE/f"day{s['day']}__guarded_rule",s,"development"))
    rows=[];activity=[];seconds=config["horizons_seconds"]
    for e in episodes:
        starts=nm.origin_indices(e,config["validation_stride_seconds"]);y=nm.truth(e,starts,seconds)
        activity.append({"day":e["day"],"trajectory":e["id"],"origins":len(starts),
            "active_origins":(y[:,-1,4:7]/6>config["gates"]["active_mean_kW"]).sum(axis=0).tolist()})
        for name in ["native_one_step","native_joint"]:
            model=joblib.load(MODEL/(name+".joblib"));p=nm.prediction(model["parameters"],e,starts,seconds)
            if not np.isfinite(p).all() or (p[:,:,1:]<0).any():raise ValueError("Invalid native free run")
            rows.extend(metric_rows(name,e,p,y,seconds))
        base=baseline_prediction(e,starts);yy=nm.truth(e,starts,[3600,21600])
        # The old model has interval-average powers, so report comparable temperature/energy only.
        rows.extend(r for r in metric_rows("frozen_static",e,base,yy,[3600,21600]) if r["target"]=="temperature" or r["target"].endswith("_energy"))
    gates=capability_gates(rows,activity,config)
    response=[]
    for day in [p["day"] for p in config["development"]]:
        pair=[]
        for setting in [20.5,23.25]:
            root=Path(f"outputs/terminal-response/day{day}_perfect__response_{setting:g}")
            pair.append(read_native(root,t.load(root/"summary.json"),"development"))
        origin=np.array([16*30]);observed=[nm.truth(e,origin,[3600,21600]) for e in pair];delta=observed[1]-observed[0]
        for name in ["native_one_step","native_joint"]:
            parameters=joblib.load(MODEL/(name+".joblib"))["parameters"]
            predicted=[nm.prediction(parameters,e,origin,[3600,21600]) for e in pair];difference=predicted[1]-predicted[0]
            active=abs(delta[0,0,4])>.001
            direction=active and difference[0,0,0]*delta[0,0,0]>0 and difference[0,0,4]*delta[0,0,4]>0
            limit=max(config["gates"]["response_absolute_floor_kWh"],config["gates"]["response_relative_error"]*abs(delta[0,1,4]))
            error=abs(difference[0,1,4]-delta[0,1,4]);passed=direction and error<=limit
            response.append({"model":name,"day":day,"active":bool(active),"direction_passed":bool(direction),
                "observed_gas_delta_6h_kWh":float(delta[0,1,4]),"predicted_gas_delta_6h_kWh":float(difference[0,1,4]),
                "absolute_error_kWh":float(error),"limit_kWh":float(limit),"passed":bool(passed)})
    summary={name:{"all_capability_gates_passed":all(g["passed"] for g in gates if g["model"]==name) and all(r["passed"] for r in response if r["model"]==name)
        and joblib.load(MODEL/(name+".joblib"))["diagnostics"]["success"],
        "gates_passed":sum(g["passed"] for g in gates if g["model"]==name),"gates_total":sum(g["model"]==name for g in gates)} for name in ["native_one_step","native_joint"]}
    save_json(MODEL/"development_metrics.json",rows);save_json(MODEL/"capability_gates.json",gates)
    save_json(MODEL/"development_assessment.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"code_sha256":hashes(),
        "summary":summary,"activity":activity,"response":response,"calibration_exists":CALIBRATION.exists()})
    print(json.dumps(summary,indent=2),flush=True)


def lock_development():
    assessment=t.load(MODEL/"development_assessment.json")
    if CALIBRATION.exists():raise ValueError("Calibration must not exist before development lock")
    if assessment["code_sha256"]!=hashes():raise ValueError("Assessed code changed")
    if not any(s["all_capability_gates_passed"] for s in assessment["summary"].values()):
        raise ValueError("Development capability gate failed; calibration and new-scenario evaluation remain closed")
    # Prediction eligibility alone cannot authorize a new closed-loop experiment.
    raise ValueError("Before locking a passing candidate, bind the independent audit, controller, calibration algorithm and unused scenario")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["collect-rule","fit","evaluate","lock"])
    parser.add_argument("--output",type=Path,default=MODEL);args=parser.parse_args()
    if args.mode=="collect-rule":collect_rule()
    elif args.mode=="fit":fit(args.output)
    elif args.mode=="evaluate":evaluate()
    else:lock_development()


if __name__=="__main__":main()
