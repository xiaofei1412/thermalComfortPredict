"""Independent office development data, few-shot fitting and frozen time holdouts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from . import office_model as om
from .generalization import FREEZE, canonical, check_freeze, integrate, interface
from .mpc import candidate_actions, exogenous, limited_command
from .recovery import RecoveryController, weather_view
from .simulation import Client, FORECAST, digest, save_json, schedule_at
from .solar import SolarController


CONFIG = Path("configs/office_calibration.json")
DATA = Path("outputs/office-data")
MODEL = Path("outputs/office-model")
DOMAIN = {"id":"office","testcase":"bestest_air","interface":"office","area_m2":48}
IO = interface(DOMAIN)
POINTS = [IO["temperature"],IO["setpoint"],IO["active"],IO["heat"],*IO["electric"],
    "con_oveTSetCoo_activate","fcu_oveFan_activate","fcu_oveTSup_activate","fcu_reaFloSup_y",
    *[IO["weather_prefix"]+n+"_y" for n in ["TDryBul","HGloHor","Lat","Lon"]]]
SOURCE_FILES = ["thermal_comfort/office.py","thermal_comfort/office_model.py",str(CONFIG),"docs/experiments/office_calibration.md"]


def configuration():
    return json.loads(CONFIG.read_text())


def validate_windows(config):
    groups = {r:[(e["day"]-7,e["day"]+3) for e in config["episodes"] if e["role"]==r]
              for r in ["train","selection","calibration"]}
    groups["test"] = [(d-7,d+7) for d in config["test_days"]]
    roles = list(groups)
    for left,right in zip(roles,roles[1:]):
        if max(b for a,b in groups[left])>min(a for a,b in groups[right]):
            raise ValueError("Office role/warmup windows overlap")
    all_windows = [w for group in groups.values() for w in group]
    ordered = sorted(all_windows)
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError("Office episodes overlap")
    previous_tests = json.loads(Path("configs/generalization.json").read_text())["start_days"]
    previous_tests += json.loads(Path("configs/solar_mpc.json").read_text())["test_start_days"]
    if any(a<d+7 and b>d-7 for a,b in groups["test"] for d in previous_tests):
        raise ValueError("New tests overlap a previously used test or warmup")
    return groups


def current_hvac(raw,config):
    return np.array([raw["fcu_reaPHea_y"]*config["heat_efficiency"],
                     raw["fcu_reaPCoo_y"]*config["cooling_COP"],raw["fcu_reaPFan_y"]])/1000


def mean_hvac(native,config):
    seconds = native.time.iloc[-1]-native.time.iloc[0]
    return np.trapezoid(native[["fcu_reaPHea_y","fcu_reaPCoo_y","fcu_reaPFan_y"]].to_numpy(),native.time,axis=0)/seconds/1000*np.array([config["heat_efficiency"],config["cooling_COP"],1])


def office_exogenous(frame):
    return np.column_stack([exogenous(frame),frame["InternalGainsCon[1]"].to_numpy()/1000,
                            frame["InternalGainsRad[1]"].to_numpy()/1000])


class Excitation:
    def __init__(self,config,policy_config,seed):
        self.config,self.policy_config = config,policy_config
        self.rng = np.random.default_rng(seed)
        self.target = 22.

    def reset(self,initial,inputs):
        self.previous = initial["oveTSet_u"]-273.15

    def decide(self,index,before,forecast):
        c = self.policy_config
        t = before["reaTZon_y"]-273.15
        if not c["abort_min_C"]<=t<=c["abort_max_C"]: raise ValueError("Excitation stop band")
        if index%8==0: self.target = float(self.rng.choice([18.,20.5,22.,23.25]))
        current = schedule_at(forecast,[before["time"]]).iloc[0]
        target = self.target; reason = "seeded_excitation"
        if t<max(current["LowerSetp[1]"]-273.15+.3,19.): target,reason = 23.25,"heat_protection"
        if t>current["UpperSetp[1]"]-273.15-.2: target,reason = 18.,"coast_protection"
        value = limited_command(target,self.previous,c)
        return {"oveTSet_u":value+273.15,"oveTSet_activate":1},{"mode":"excitation","reason":reason,"requested_C":value}

    def update(self,before,after,interval,current):
        self.previous = after["oveTSet_u"]-273.15
        t = after["reaTZon_y"]-273.15
        if not self.policy_config["abort_min_C"]<=t<=self.policy_config["abort_max_C"]: raise ValueError("Excitation stop band")


class OfficeController(RecoveryController):
    def __init__(self,model,config,policy_config,weather="perfect",setting=None):
        super().__init__(model,policy_config,"office_mpc",weather)
        self.office_config = config
        self.horizon_steps = 48
        self.setting = setting

    def reset(self,initial,inputs):
        super().reset(initial,inputs)
        t = initial["reaTZon_y"]-273.15
        self.state = np.array([t,t]); self.hvac = current_hvac(initial,self.office_config)

    def update(self,before,after,interval,current):
        t = after["reaTZon_y"]-273.15
        self.check_temperature(t)
        self.hvac = mean_hvac(interval,self.office_config)
        self.state = om.observe(self.model,self.state,t,self.hvac,office_exogenous(pd.DataFrame([current]))[0])
        self.previous = after["oveTSet_u"]-273.15; self.q = self.hvac[0]

    def ramp(self,target,length):
        values=[]; previous=self.previous
        for _ in range(length):
            previous=limited_command(target,previous,self.config); values.append(previous)
        return np.array(values)

    def predict(self,actions,forecast,start):
        times = start+np.arange(actions.shape[1])*900
        future = office_exogenous(schedule_at(forecast,times))
        return om.rollout(self.model,np.tile(self.state,(len(actions),1)),np.tile(self.hvac,(len(actions),1)),
            actions,np.tile(future,(len(actions),1,1)),np.tile(om.cooling_setpoint(times),(len(actions),1)))

    def reference(self,start,forecast):
        shadow = RecoveryController(self.model,self.config,"guarded_rule")
        shadow.previous,shadow.recovering,shadow.view = self.previous,self.recovering,forecast
        state,hvac = self.state.copy(),self.hvac.copy()
        actions=[]; predictions=[]
        for k in range(32):
            t = start+k*900; current = schedule_at(forecast,[t]).iloc[0]
            target,_ = shadow.guarded_rule({"time":t,"reaTZon_y":state[0]+273.15},current)
            action = limited_command(target,shadow.previous,self.config)
            p = om.rollout(self.model,state[None],hvac[None],np.array([[action]]),
                office_exogenous(pd.DataFrame([current]))[None],np.array([[om.cooling_setpoint(t)]]))[0,0]
            state,hvac,shadow.previous = p[:2],p[2:5],action
            actions.append(action); predictions.append(p)
        return np.array(actions),np.array(predictions)

    def plan_office(self,start,forecast,fallback):
        c = self.config; h = c["horizon_steps"]
        times = start+np.arange(32)*900
        starts = schedule_at(forecast,times)
        support = np.column_stack([np.full(32,self.state[0]),exogenous(starts)[:,:2]])
        if ((support<self.model["support_low"]-c["support_tolerance"])|(support>self.model["support_high"]+c["support_tolerance"])).any():
            return fallback,{"mode":"fallback","reason":"outside_training_support"}
        reference_actions,reference = self.reference(start,forecast)
        ramps = np.array([self.ramp(v,h) for v in [18.,19.,20.,21.,22.,23.25]])
        actions = np.unique(np.vstack([candidate_actions(self.previous,c),ramps,reference_actions[:h]]),axis=0)
        p = self.predict(actions,forecast,start)
        ends = schedule_at(forecast,start+np.arange(1,h+1)*900)
        lower = ends["LowerSetp[1]"].to_numpy()-273.15+self.model["margin_K"][:h]
        upper = ends["UpperSetp[1]"].to_numpy()-273.15-self.model["margin_K"][:h]
        feasible = ((p[:,:,0]>=lower)&(p[:,:,0]<=upper)).all(axis=1)
        feasible &= (p[:,:,:2]<=reference[None,:h,:2]+1e-9).all(axis=(1,2))
        feasible &= (p[:,:,5:7].sum(axis=1)<=reference[None,:h,5:7].sum(axis=1)+1e-9).all(axis=1)
        feasible &= actions[:,0]<=reference_actions[0]+1e-9
        if not feasible.any(): return fallback,{"mode":"fallback","reason":"no_margin_feasible_plan"}
        changes = np.diff(np.column_stack([np.full(len(actions),self.previous),actions]),axis=1)
        costs = p[:,:,5:7].sum(axis=(1,2))+.05*(changes**2).sum(axis=1)
        costs[~feasible]=np.inf; j=int(np.argmin(costs))
        mode = "reference_selected" if abs(actions[j,0]-fallback)<1e-10 else "mpc"
        return actions[j,0],{"mode":mode,"reason":"office_feasible_plan","feasible_candidates":int(feasible.sum()),
            "planned_actions_C":actions[j].tolist(),"predicted_temperature_C":p[j,:,0].tolist(),
            "predicted_electricity_kWh":p[j,:,5].tolist(),"predicted_gas_kWh":p[j,:,6].tolist()}

    def decide(self,index,before,forecast):
        self.check_temperature(before["reaTZon_y"]-273.15)
        self.view = weather_view(forecast,before["time"],self.config,self.weather)
        current = schedule_at(self.view,[before["time"]]).iloc[0]
        target,reason = self.guarded_rule(before,current)
        info = {"mode":"fallback","reason":reason}
        if self.setting is not None:
            target = 22. if index<self.office_config["response_prelude_steps"] else self.setting
            info = {"mode":"intervention","reason":"common_prelude" if index<16 else "fixed_setting"}
            if index==16:
                actions = self.ramp(self.setting,32)
                p = self.predict(actions[None],forecast,before["time"])[0]
                info.update(planned_actions_C=actions.tolist(),predicted_temperature_C=p[:,0].tolist(),
                    predicted_electricity_kWh=p[:,5].tolist(),predicted_gas_kWh=p[:,6].tolist())
        elif index<self.config["initial_rule_steps"]:
            info = {"mode":"initial_rule","reason":"common_initialization"}
        else:
            try: target,extra = self.plan_office(before["time"],self.view,target); info.update(extra)
            except (ValueError,FloatingPointError) as error: info.update(reason="invalid_model_prediction",detail=str(error))
        value = limited_command(target,self.previous,self.config)
        info.update(requested_C=value,weather_mode=self.weather)
        return {"oveTSet_u":value+273.15,"oveTSet_activate":1},info


def run_episode(config,day,days,kind,weather,policy,output):
    output.mkdir(parents=True,exist_ok=False)
    client = Client("http://127.0.0.1:8080")
    if client.call("GET","version")["version"]!="0.9.0": raise ValueError("BOPTEST version mismatch")
    testid = client.call("POST","testcases/bestest_air/select")["testid"]
    records=[]; failure=None; started=time.perf_counter()
    try:
        inputs=client.call("GET",f"inputs/{testid}"); measurements=client.call("GET",f"measurements/{testid}")
        if set(POINTS)-inputs.keys()-measurements.keys(): raise ValueError("Requested native points are not exposed by this FMU API")
        if inputs[IO["setpoint"]]["Unit"]!="K" or measurements[IO["temperature"]]["Unit"]!="K": raise ValueError("Unexpected temperature units")
        if any(measurements[p]["Unit"]!="W" for p in ["fcu_reaPHea_y",*IO["electric"]]): raise ValueError("Unexpected energy units")
        save_json(output/"api_metadata.json",{"inputs":inputs,"measurements":measurements})
        client.call("PUT",f"scenario/{testid}",{"electricity_price":"constant"})
        before=client.call("PUT",f"initialize/{testid}",{"start_time":(day-1)*86400,"warmup_period":6*86400})
        client.call("PUT",f"step/{testid}",{"step":900})
        forecast=pd.DataFrame(client.call("PUT",f"forecast/{testid}",{"point_names":FORECAST+["InternalGainsCon[1]","InternalGainsRad[1]"],"horizon":(days+1)*86400+48*900,"interval":30}))
        forecast.to_csv(output/"exogenous.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        prelude=[]
        for _ in range(96):
            action={IO["setpoint"]:295.15,IO["active"]:1}
            after=client.call("POST",f"advance/{testid}",action)
            prelude.append({"before":before,"action":action,"after":after});before=after
        save_json(output/"prelude.json",prelude)
        if before["time"]!=day*86400: raise ValueError("Misaligned evaluation start")
        policy.reset(canonical(before,IO),{"oveTSet_u":inputs[IO["setpoint"]]})
        with (output/"steps.jsonl").open("w",encoding="utf-8") as stream:
            for index in range(int(days*96)):
                raw_before=canonical(before,IO)
                try:
                    tick=time.perf_counter();action,info=policy.decide(index,raw_before,forecast)
                    info["decision_seconds"]=time.perf_counter()-tick
                except ValueError as error: failure=str(error);break
                actual={IO["setpoint"]:action["oveTSet_u"],IO["active"]:1}
                after=client.call("POST",f"advance/{testid}",actual)
                record={"before":raw_before,"action":action,"actual_action":actual,"after":canonical(after,IO),"control":info}
                stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush();records.append(record)
                interval=pd.DataFrame(client.call("PUT",f"results/{testid}",{"point_names":["fcu_reaPHea_y",*IO["electric"]],"start_time":before["time"],"final_time":after["time"]}))
                interval["reaQFloHea_y"]=interval["fcu_reaPHea_y"]*.9
                try: policy.update(raw_before,canonical(after,IO),interval,schedule_at(forecast,[before["time"]]).iloc[0])
                except ValueError as error: failure=str(error);before=after;break
                before=after
                if (index+1)%96==0: print(f"day{day}/{weather}/{kind}: {(index+1)//96} days",flush=True)
        all_native=pd.DataFrame(client.call("PUT",f"results/{testid}",{"point_names":POINTS,"start_time":(day-1)*86400,"final_time":before["time"]}))
        native=all_native[all_native.time>=day*86400].reset_index(drop=True)
        for name,frame in [("native.csv.gz",native),("including_prelude.csv.gz",all_native)]:
            frame.to_csv(output/name,index=False,compression={"method":"gzip","mtime":0})
        kpis=client.call("GET",f"kpi/{testid}");save_json(output/"api_kpis.json",kpis)
        whole=integrate(all_native,forecast,IO,48)
        differences={k:abs(whole[m]-kpis[k]) for m,k in [("official_energy_kWh_m2","ener_tot"),("discomfort_Kh","tdis_tot")]}
        if max(differences.values())>1e-6: raise ValueError("Office native/API KPI mismatch")
        summary={"day":day,"days":days,"controller":kind,"weather":weather,"complete":failure is None and len(records)==int(days*96),
            "failure":failure,"transitions":len(records),"native_samples":len(native),"wall_seconds":time.perf_counter()-started,
            "metrics":integrate(native,forecast,IO,48) if len(native)>1 else None,"api_difference":differences,
            "sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
        save_json(output/"summary.json",summary);return summary
    finally: client.call("PUT",f"stop/{testid}")


def read_episode(root,summary,role):
    for name,expected in summary["sha256"].items():
        if digest(root/name)!=expected: raise ValueError("Office source hash changed")
    records=[json.loads(v) for v in (root/"steps.jsonl").read_text().splitlines()]
    native=pd.read_csv(root/"native.csv.gz",dtype=float,float_precision="round_trip")
    forecast=pd.read_csv(root/"exogenous.csv.gz",dtype=float,float_precision="round_trip")
    if not summary["complete"]: raise ValueError("Incomplete office development episode")
    hvac=[];energy=[]
    times=np.array([r["before"]["time"] for r in records])
    for t in times:
        segment=native[native.time.between(t,t+900)]
        hvac.append(mean_hvac(segment,configuration()))
        energy.append([np.trapezoid(segment[IO["electric"]].sum(axis=1),segment.time)/3600000,
                       np.trapezoid(segment["fcu_reaPHea_y"],segment.time)/3600000])
    return {"id":root.name,"day":summary["day"],"role":role,"times":times,"exog":office_exogenous(schedule_at(forecast,times)),
        "temperature":np.array([records[0]["before"]["reaTZon_y"]]+[r["after"]["reaTZon_y"] for r in records])-273.15,
        "actions":np.array([r["action"]["oveTSet_u"]-273.15 for r in records]),"hvac":np.array(hvac),"energy":np.array(energy),
        "initial_hvac":current_hvac(records[0]["before"],configuration())}


def fit_models(config,output):
    manifest=json.loads((DATA/"manifest.json").read_text())
    collection_keys=["baseline_commit","domain","episodes","collection_days","heat_efficiency","cooling_COP"]
    if any(manifest["config"][k]!=config[k] for k in collection_keys): raise ValueError("Collection config changed")
    summaries=json.loads((DATA/"summary.json").read_text())
    episodes=[]
    for plan,s in zip(config["episodes"],summaries):
        if s["day"]!=plan["day"] or s["days"]!=3: raise ValueError("Development identity/length mismatch")
        episodes.append(read_episode(DATA/f"day{s['day']}__excitation",s,plan["role"]))
    if len(episodes)!=len(config["episodes"]):raise ValueError("Development collection incomplete")
    output.mkdir(parents=True,exist_ok=False)
    selection=next(e for e in episodes if e["role"]=="selection")
    origins=np.arange(16,len(selection["actions"])-32+1,config["origin_stride"])
    scores=[]
    for days in config["training_days_per_episode"]:
        train=[]
        for e in episodes:
            if e["role"]!="train":continue
            n=days*96
            train.append({**e,**{k:e[k][:n] for k in ["times","exog","actions","hvac","energy"]},"temperature":e["temperature"][:n+1]})
        model=om.fit(train,config)
        for weight in config["residual_weights"]:
            trial={**model,"residual_weight":weight}
            p=om.episode_rollout(trial,selection,origins,32)
            score=float(np.mean([np.sqrt(np.mean((p[:,h-1,0]-selection["temperature"][origins+h])**2)) for h in [4,24,32]]))
            scores.append({"days_per_episode":days,"training_rows":days*96*3,"residual_weight":weight,"score_C":score})
        best=min((s for s in scores if s["days_per_episode"]==days),key=lambda s:s["score_C"])
        model["residual_weight"]=best["residual_weight"]
        errors=[];calibration=[]
        for e in episodes:
            if e["role"]!="calibration":continue
            starts=np.arange(16,len(e["actions"])-32+1,config["origin_stride"])
            p=om.episode_rollout(model,e,starts,32)
            errors.append(p[:,:,0]-e["temperature"][starts[:,None]+np.arange(1,33)])
            calibration.append({"id":e["id"],"origins":starts.tolist()})
        model["margin_K"]=np.quantile(abs(np.concatenate(errors)),config["margin_quantile"],axis=0)+config["sensor_margin_K"]
        model["training_days_per_episode"]=days
        joblib.dump(model,output/f"budget_{days}.joblib")
    chosen=min(scores,key=lambda s:s["score_C"])
    shutil.copyfile(output/f"budget_{chosen['days_per_episode']}.joblib",output/"model.joblib")
    save_json(output/"selection.json",{"scores":scores,"selected":chosen,"selection_origins":origins.tolist(),"calibration":calibration})
    save_json(output/"split_manifest.json",{"windows":validate_windows(config),"development":config["episodes"],"budgets_rows":[288,864],
        "selection_rows":288,"calibration_rows":864,"target_development_total_rows":2016})
    save_json(output/"manifest.json",{"config":config,"source_summary_sha256":digest(DATA/"summary.json"),
        "code_sha256":{n:digest(Path(n)) for n in SOURCE_FILES},"artifacts_sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"selected":chosen,"scores":scores},indent=2),flush=True)


def freeze(config):
    check_freeze()
    verified=json.loads(Path("outputs/office-validation/development_verification.json").read_text())
    gate=json.loads(Path("outputs/office-response/gate.json").read_text())
    hashes={n:digest(Path(n)) for n in SOURCE_FILES}
    manifest=json.loads((MODEL/"manifest.json").read_text())
    if manifest["code_sha256"]!=hashes or manifest["config"]!=config:
        raise ValueError("Fit must use current source and configuration")
    if not gate["passed"] or not verified["passed"] or verified["code_sha256"]!=hashes:
        raise ValueError("Current development audit and action response gate required")
    if gate["model_sha256"]!=digest(MODEL/"model.joblib") or verified["model_sha256"]!=gate["model_sha256"]:
        raise ValueError("Development evidence belongs to another model")
    root=Path("outputs/office-freeze");root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(MODEL/"model.joblib",root/"model.joblib")
    save_json(root/"freeze.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"config":config,
        "model_sha256":digest(MODEL/"model.joblib"),"source_freeze_sha256":digest(FREEZE/"freeze.json"),
        "code_sha256":{n:digest(Path(n)) for n in SOURCE_FILES},"model_manifest_sha256":digest(MODEL/"manifest.json"),
        "validation_summary_sha256":digest(Path("outputs/office-validation/summary.json")),
        "response_gate_sha256":digest(Path("outputs/office-response/gate.json")),
        "development_verification_sha256":digest(Path("outputs/office-validation/development_verification.json")),
        "audit_script_sha256":digest(Path("scripts/verify_office.py"))})


def check_office_freeze():
    check_freeze();root=Path("outputs/office-freeze");f=json.loads((root/"freeze.json").read_text())
    for name,expected in f["code_sha256"].items():
        if digest(Path(name))!=expected:raise ValueError("Frozen office source changed")
    if digest(root/"model.joblib")!=f["model_sha256"] or digest(MODEL/"model.joblib")!=f["model_sha256"]:raise ValueError("Office model changed")
    for path,key in [(MODEL/"manifest.json","model_manifest_sha256"),
            (Path("outputs/office-response/gate.json"),"response_gate_sha256"),
            (Path("outputs/office-validation/development_verification.json"),"development_verification_sha256"),
            (Path("scripts/verify_office.py"),"audit_script_sha256")]:
        if digest(path)!=f[key]:raise ValueError("Office freeze evidence changed")
    return f


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["collect","fit","validation","response","freeze","test"])
    parser.add_argument("--output",type=Path);parser.add_argument("--days",type=float)
    args=parser.parse_args();config=configuration();validate_windows(config);frozen=check_freeze()
    if args.mode=="fit":fit_models(config,args.output);return
    if args.mode=="freeze":freeze(config);return
    if args.mode=="test":check_office_freeze()
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/"manifest.json",{"mode":args.mode,"config":config,"days_override":args.days,
        "created_UTC":datetime.now(timezone.utc).isoformat(),"code_sha256":{n:digest(Path(n)) for n in SOURCE_FILES},
        "source_model_sha256":frozen["model_sha256"],"office_model_sha256":None if args.mode=="collect" else digest(MODEL/"model.joblib"),
        "source_freeze_sha256":digest(FREEZE/"freeze.json")})
    source_model=joblib.load(FREEZE/"model.joblib")
    target_model=None if args.mode=="collect" else joblib.load(MODEL/"model.joblib")
    summaries=[]
    if args.mode=="collect":
        tasks=[(e["day"],3,"excitation","perfect",Excitation(config,frozen["policy_config"],e["seed"])) for e in config["episodes"]]
    elif args.mode=="response":
        tasks=[(d,.5,f"response_{setting:g}","perfect",OfficeController(target_model,config,frozen["policy_config"],setting=setting))
               for d in config["response_days"] for setting in config["response_settings_C"]]
    else:
        tasks=[]
        dates=config["validation_days"] if args.mode=="validation" else config["test_days"]
        modes=["perfect"] if args.mode=="validation" else config["weather_modes"]
        for d in dates:
            for w in modes:
                for k in config["controllers"]:
                    policy=OfficeController(target_model,config,frozen["policy_config"],w) if k=="office_mpc" else SolarController(source_model,frozen["policy_config"],k,w)
                    tasks.append((d,args.days or (3 if args.mode=="validation" else 7),k,w,policy))
    for day,days,kind,weather,policy in tasks:
        if args.mode=="test":check_office_freeze()
        directory=args.output/f"day{day}_{weather}__{kind}" if args.mode not in ["collect","response"] else args.output/f"day{day}__{kind}"
        summaries.append(run_episode(config,day,days,kind,weather,policy,directory))
        save_json(args.output/"summary.json",summaries)


if __name__=="__main__":main()
