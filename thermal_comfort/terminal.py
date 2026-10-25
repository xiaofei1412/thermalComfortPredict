"""New seasonal development, locked calibration, and unseen office-climate tests."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from . import office as o, office_model as om, terminal_model as tm
from .generalization import FREEZE, SOURCE, canonical, check_freeze, integrate
from .mpc import candidate_actions, exogenous, limited_command
from .recovery import RecoveryController
from .simulation import Client, FORECAST, digest, save_json, schedule_at
from .solar import SolarController


CONFIG=Path("configs/terminal_dynamics.json")
DEVELOPMENT=Path("outputs/terminal-development")
CALIBRATION=Path("outputs/terminal-calibration")
MODEL=Path("outputs/terminal-model")
LOCK=Path("outputs/terminal-lock")
FINAL=Path("outputs/terminal-freeze")
CORE=["thermal_comfort/terminal.py","thermal_comfort/terminal_model.py",str(CONFIG),"docs/experiments/terminal_dynamics.md"]
IO=o.IO;POINTS=o.POINTS


def load(path):return json.loads(path.read_text(encoding="utf-8"))
def configuration():return load(CONFIG)
def hashes():return {n:digest(Path(n)) for n in CORE}


def policy_config():
    return {**check_freeze()["policy_config"],"initial_rule_steps":configuration()["initial_rule_steps"]}


def validate_windows(config):
    plans=config["episodes"];windows=[(p["day"]-7,p["day"]+3) for p in plans]
    ordered=sorted(windows)
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):raise ValueError("Seasonal role/warmup windows overlap")
    old=o.configuration()
    old_windows=[(p["day"]-7,p["day"]+3) for p in old["episodes"]]
    old_windows += [(d-7,d+7) for d in old["test_days"]+load(Path("configs/generalization.json"))["start_days"]]
    old_windows += [(d-7,d+.5) for d in old["response_days"]]
    if any(a<d and b>c for a,b in windows for c,d in old_windows):raise ValueError("New development reuses old office windows")
    return [{**p,"warmup_start_day":p["day"]-7,"end_day":p["day"]+3} for p in plans]


def prepare():
    o.check_office_freeze();config=configuration();validate_windows(config)
    root=Path("outputs/terminal-scenario");target=root/"testcases"/config["test_case"]/"models/wrapped.fmu"
    target.parent.mkdir(parents=True,exist_ok=False)
    source=SOURCE/"testcases/bestest_air/models/wrapped.fmu";donor=SOURCE/"testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"
    with ZipFile(source) as old,ZipFile(donor) as weather,ZipFile(target,"w") as new:
        for info in old.infolist():
            content=old.read(info.filename)
            if info.filename=="resources/0/DRYCOLD.mos":content=weather.read("resources/0/Brussels.mos")
            elif info.filename=="resources/weather.csv":content=weather.read(info.filename)
            elif info.filename=="resources/config.json":
                embedded=json.loads(content);embedded["name"]=config["test_case"]
                content=(json.dumps(embedded,indent=2)+"\n").encode()
            new.writestr(info,content)
    with ZipFile(source) as old,ZipFile(target) as new,ZipFile(donor) as weather:
        changed=[n for n in old.namelist() if old.read(n)!=new.read(n)]
        assert set(changed)=={"resources/0/DRYCOLD.mos","resources/weather.csv","resources/config.json"}
        assert new.read("resources/weather.csv")==weather.read("resources/weather.csv")
        assert new.read("resources/0/DRYCOLD.mos")==weather.read("resources/0/Brussels.mos")
    save_json(root/"scenario.json",{"source":str(source),"source_sha256":digest(source),"weather_donor":str(donor),
        "donor_sha256":digest(donor),"target":str(target),"target_sha256":digest(target),"only_changed_entries":changed,
        "old_office_freeze_sha256":digest(Path("outputs/office-freeze/freeze.json")),"config":config})


class Excitation(o.Excitation):
    def __init__(self,config,pc,seed):
        super().__init__(config,pc,seed);self.remaining=0

    def decide(self,index,before,forecast):
        c=self.policy_config;t=before["reaTZon_y"]-273.15
        if not c["abort_min_C"]<=t<=c["abort_max_C"]:raise ValueError("Excitation stop band")
        if self.remaining==0:
            self.target=float(self.rng.choice([18.,20.5,22.,23.25]));self.remaining=int(self.rng.choice(self.config["dwell_steps"]))
        self.remaining-=1;target=self.target;reason="variable_dwell_excitation"
        current=schedule_at(forecast,[before["time"]]).iloc[0]
        if t<max(current["LowerSetp[1]"]-273.15+.3,20.):target,reason=23.25,"heat_protection"
        if t>current["UpperSetp[1]"]-273.15-.2:target,reason=18.,"coast_protection"
        value=limited_command(target,self.previous,c)
        return {"oveTSet_u":value+273.15,"oveTSet_activate":1},{"mode":"excitation","reason":reason,"requested_C":value}


class Controller(o.OfficeController):
    def __init__(self,model,config,pc,margins,weather="perfect"):
        super().__init__(model,{**o.configuration(),**config},pc,weather);self.margins=margins;self.kind="terminal_mpc"

    def update(self,before,after,interval,current):
        self.check_temperature(after["reaTZon_y"]-273.15)
        mean=o.mean_hvac(interval,o.configuration())
        self.state=om.observe(self.model,self.state,after["reaTZon_y"]-273.15,mean,o.office_exogenous(pd.DataFrame([current]))[0])
        self.hvac=o.current_hvac(after,o.configuration());self.q=mean[0];self.previous=after["oveTSet_u"]-273.15

    def predict(self,actions,forecast,start):
        times=start+np.arange(actions.shape[1])*900;n=len(actions)
        return tm.rollout(self.model,np.tile(self.state,(n,1)),np.tile(self.hvac,(n,1)),actions,np.full(n,self.previous),
            np.tile(o.office_exogenous(schedule_at(forecast,times)),(n,1,1)),np.tile(om.cooling_setpoint(times),(n,1)),
            np.full(n,om.cooling_setpoint(start-900)))

    def reference(self,start,forecast):
        shadow=RecoveryController(None,self.config,"guarded_rule")
        shadow.previous,shadow.recovering,shadow.view=self.previous,self.recovering,forecast
        state,hvac=self.state.copy(),self.hvac.copy();actions=[];predictions=[]
        for k in range(32):
            t=start+k*900;current=schedule_at(forecast,[t]).iloc[0]
            target,_=shadow.guarded_rule({"time":t,"reaTZon_y":state[0]+273.15},current)
            action=limited_command(target,shadow.previous,self.config)
            p=tm.rollout(self.model,state[None],hvac[None],np.array([[action]]),np.array([shadow.previous]),
                o.office_exogenous(pd.DataFrame([current]))[None],np.array([[om.cooling_setpoint(t)]]),np.array([om.cooling_setpoint(t-900)]))[0,0]
            state,hvac,shadow.previous=p[:2],p[7:10],action;actions.append(action);predictions.append(p)
        return np.array(actions),np.array(predictions)

    def plan_office(self,start,forecast,fallback):
        c=self.config;h=c["horizon_steps"];times=start+np.arange(32)*900
        starts=schedule_at(forecast,times)
        support=np.column_stack([np.full(32,self.state[0]),exogenous(starts)[:,:2]])
        if ((support<self.model["support_low"]-c["support_tolerance"])|(support>self.model["support_high"]+c["support_tolerance"])).any():
            return fallback,{"mode":"fallback","reason":"outside_training_support"}
        lower_margin,upper_margin=tm.margins_for(self.margins,self.hvac,start,h,self.office_config)
        ends=schedule_at(forecast,start+np.arange(1,h+1)*900)
        lower=ends["LowerSetp[1]"].to_numpy()-273.15+lower_margin
        upper=ends["UpperSetp[1]"].to_numpy()-273.15-upper_margin
        if (lower>upper).any():return fallback,{"mode":"fallback","reason":"empty_tightened_band"}
        reference_actions,reference=self.reference(start,forecast)
        ramps=np.array([self.ramp(v,h) for v in [18.,19.,20.,21.,22.,23.25]])
        actions=np.unique(np.vstack([candidate_actions(self.previous,c),ramps,reference_actions[:h]]),axis=0)
        p=self.predict(actions,forecast,start)
        feasible=((p[:,:,0]>=lower)&(p[:,:,0]<=upper)).all(axis=1)
        feasible &= (p[:,:,:2]<=reference[None,:h,:2]+1e-9).all(axis=(1,2))
        feasible &= (p[:,:,5:7].sum(axis=1)<=reference[None,:h,5:7].sum(axis=1)+1e-9).all(axis=1)
        feasible &= actions[:,0]<=reference_actions[0]+1e-9
        if not feasible.any():return fallback,{"mode":"fallback","reason":"no_margin_feasible_plan"}
        changes=np.diff(np.column_stack([np.full(len(actions),self.previous),actions]),axis=1)
        costs=p[:,:,5:7].sum(axis=(1,2))+.05*(changes**2).sum(axis=1);costs[~feasible]=np.inf;j=int(np.argmin(costs))
        mode="reference_selected" if abs(actions[j,0]-limited_command(fallback,self.previous,c))<1e-10 else "mpc"
        return actions[j,0],{"mode":mode,"reason":"terminal_feasible_plan","feasible_candidates":int(feasible.sum()),
            "planned_actions_C":actions[j].tolist(),"predicted_temperature_C":p[j,:,0].tolist(),
            "predicted_electricity_kWh":p[j,:,5].tolist(),"predicted_gas_kWh":p[j,:,6].tolist()}


def read_episode(root,summary,role):
    for name,h in summary["sha256"].items():
        if digest(root/name)!=h:raise ValueError("Trajectory artifact changed")
    if not summary["complete"] and role!="evaluation":raise ValueError("Incomplete development/calibration data")
    records=[json.loads(v) for v in (root/"steps.jsonl").read_text().splitlines()]
    if not records:raise ValueError("No observed transitions")
    native=pd.read_csv(root/"native.csv.gz",dtype=float,float_precision="round_trip")
    forecast=pd.read_csv(root/"exogenous.csv.gz",dtype=float,float_precision="round_trip")
    times=np.array([r["before"]["time"] for r in records]);hvac=[];energy=[]
    for time in times:
        segment=native[native.time.between(time,time+900)]
        hvac.append(o.mean_hvac(segment,o.configuration()))
        energy.append([np.trapezoid(segment[o.IO["electric"]].sum(axis=1),segment.time)/3600000,
            np.trapezoid(segment["fcu_reaPHea_y"],segment.time)/3600000])
    end=native.set_index("time").loc[times+900]
    return {"id":root.name,"day":summary["day"],"role":role,"times":times,"exog":o.office_exogenous(schedule_at(forecast,times)),
        "temperature":np.array([records[0]["before"]["reaTZon_y"]]+[r["after"]["reaTZon_y"] for r in records])-273.15,
        "actions":np.array([r["action"]["oveTSet_u"]-273.15 for r in records]),"hvac":np.array(hvac),"energy":np.array(energy),
        "end_hvac":end[["fcu_reaPHea_y","fcu_reaPCoo_y","fcu_reaPFan_y"]].to_numpy()/1000*np.array([.9,3,1]),
        "initial_hvac":o.current_hvac(records[0]["before"],o.configuration()),"initial_action":records[0]["before"]["oveTSet_u"]-273.15}


def dataset(root,role=None):
    manifest=load(root/"manifest.json");plans=manifest["plans"];summaries=load(root/"summary.json")
    assert len(plans)==len(summaries)
    current=configuration()
    for key in ["episodes","collection_days","dwell_steps"]:
        if manifest["config"][key]!=current[key]:raise ValueError("Data collection plan changed")
    if any((p["day"],3)!=(s["day"],s["days"]) for p,s in zip(plans,summaries)):raise ValueError("Episode role/identity differs")
    return [read_episode(root/f"day{s['day']}__excitation",s,p["role"]) for p,s in zip(plans,summaries) if role is None or p["role"]==role]


def fit(output):
    if CALIBRATION.exists() or LOCK.exists():raise ValueError("Fitting forbidden after development lock or calibration")
    config=configuration();episodes=dataset(DEVELOPMENT);train=[e for e in episodes if e["role"]=="train"];selection=[e for e in episodes if e["role"]=="selection"]
    base=om.fit(train,o.configuration());base["family"]="static";dynamic=tm.fit_dynamic(base,train)
    output.mkdir(parents=True,exist_ok=False);scores=[]
    for family,model in [("static",base),("dynamic",dynamic)]:
        for weight in config["residual_weights"]:
            trial={**model,"residual_weight":weight};name=f"{family}_{weight:g}";parts=[]
            for e in selection:
                starts=np.arange(16,len(e["actions"])-32+1,config["origin_stride"]);p=tm.episode_rollout(trial,e,starts,32)
                temperature=np.mean([np.sqrt(np.mean((p[:,h-1,0]-e["temperature"][starts+h])**2)) for h in [4,24]])
                gas_truth=np.array([e["energy"][k:k+24,1].sum() for k in starts])
                cool_truth=np.array([.25*e["hvac"][k:k+24,1].sum()/3 for k in starts])
                gas=float(np.sqrt(np.mean((p[:,:24,6].sum(axis=1)-gas_truth)**2)))
                cool=float(np.sqrt(np.mean((.25*p[:,:24,3].sum(axis=1)/3-cool_truth)**2)))
                parts.append({"day":e["day"],"temperature_score_C":float(temperature),"gas_6h_RMSE_kWh":gas,"cooling_6h_RMSE_kWh":cool,"joint_score":float(temperature+gas+cool)})
            score={"candidate":name,"score":float(np.mean([p["joint_score"] for p in parts])),"seasons":parts};scores.append(score)
            joblib.dump(trial,output/(name+".joblib"))
    selected=min(scores,key=lambda s:s["score"])
    shutil.copyfile(output/(selected["candidate"]+".joblib"),output/"model.joblib")
    save_json(output/"selection.json",{"scores":scores,"selected":selected})
    save_json(output/"development_margins.json",tm.calibrate(joblib.load(output/"model.joblib"),selection,config,"selection"))
    save_json(output/"manifest.json",{"config":config,"code_sha256":hashes(),"data_sha256":digest(DEVELOPMENT/"summary.json"),
        "artifacts_sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(scores,indent=2),flush=True)


def check_lock():
    o.check_office_freeze();lock=load(LOCK/"development_lock.json")
    if lock["code_sha256"]!=hashes() or lock["model_sha256"]!=digest(MODEL/"model.joblib"):raise ValueError("Development lock changed")
    for name,h in lock["artifacts_sha256"].items():
        if digest(Path(name))!=h:raise ValueError("Locked development evidence changed")
    scenario=load(Path("outputs/terminal-scenario/scenario.json"))
    if digest(Path(scenario["target"]))!=scenario["target_sha256"]:raise ValueError("New scenario FMU changed")
    return lock


def lock_development():
    if CALIBRATION.exists():raise ValueError("Calibration must not exist before lock")
    manifest=load(MODEL/"manifest.json")
    assert manifest["code_sha256"]==hashes()
    for name,h in manifest["artifacts_sha256"].items():assert digest(MODEL/name)==h
    for name in [*manifest["artifacts_sha256"]]:assert digest(MODEL/name)==digest(Path("outputs/terminal-refit")/name)
    audit=load(Path("outputs/terminal-validation/development_audit.json"))
    assert audit["passed"] and audit["calibration_not_generated"] and audit["code_sha256"]==hashes()
    assert audit["model_sha256"]==digest(MODEL/"model.joblib")
    LOCK.mkdir(parents=True,exist_ok=False)
    evidence=[MODEL/"manifest.json",MODEL/"selection.json",DEVELOPMENT/"summary.json",Path("outputs/terminal-validation/development_audit.json"),
        Path("outputs/terminal-response/summary.json"),Path("outputs/terminal-scenario/scenario.json")]
    save_json(LOCK/"development_lock.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"code_sha256":hashes(),
        "model_sha256":digest(MODEL/"model.joblib"),"config":configuration(),"artifacts_sha256":{str(p):digest(p) for p in evidence}})


def calibrate():
    check_lock();config=configuration();episodes=dataset(CALIBRATION)
    if (CALIBRATION/"margins.json").exists():raise ValueError("Final calibration is write-once")
    margins=tm.calibrate(joblib.load(MODEL/"model.joblib"),episodes,config)
    save_json(CALIBRATION/"margins.json",margins)
    save_json(CALIBRATION/"calibration_manifest.json",{"development_lock_sha256":digest(LOCK/"development_lock.json"),
        "model_sha256":digest(MODEL/"model.joblib"),"calibration_summary_sha256":digest(CALIBRATION/"summary.json"),
        "margins_sha256":digest(CALIBRATION/"margins.json"),"code_sha256":hashes()})


def freeze():
    check_lock();meta=load(CALIBRATION/"calibration_manifest.json")
    assert meta["model_sha256"]==digest(MODEL/"model.joblib") and meta["code_sha256"]==hashes()
    assert meta["margins_sha256"]==digest(CALIBRATION/"margins.json")
    assert meta["calibration_summary_sha256"]==digest(CALIBRATION/"summary.json")
    assert load(CALIBRATION/"audit.json")["passed"]
    FINAL.mkdir(parents=True,exist_ok=False)
    save_json(FINAL/"freeze.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"code_sha256":hashes(),
        "model_sha256":digest(MODEL/"model.joblib"),"files_sha256":{str(p):digest(p) for p in [LOCK/"development_lock.json",CALIBRATION/"margins.json",CALIBRATION/"calibration_manifest.json",CALIBRATION/"audit.json",CALIBRATION/"summary.json",CALIBRATION/"manifest.json"]}})


def check_final():
    check_lock();final=load(FINAL/"freeze.json")
    assert final["code_sha256"]==hashes() and final["model_sha256"]==digest(MODEL/"model.joblib")
    for n,h in final["files_sha256"].items():assert digest(Path(n))==h
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","collect","fit","validation","response","lock","collect-calibration","calibrate","freeze","test"])
    parser.add_argument("--output",type=Path);parser.add_argument("--days",type=float)
    args=parser.parse_args();config=configuration();validate_windows(config);o.check_office_freeze();pc=policy_config()
    if args.mode=="prepare":prepare();return
    if args.mode=="fit":fit(args.output);return
    if args.mode=="lock":lock_development();return
    if args.mode=="calibrate":calibrate();return
    if args.mode=="freeze":freeze();return
    if args.mode=="collect-calibration":check_lock()
    if args.mode=="test":check_final()
    if args.mode in ["validation","response"] and LOCK.exists():raise ValueError("Control development is closed after lock")
    if args.mode=="collect" and (LOCK.exists() or CALIBRATION.exists()):raise ValueError("Development collection is closed")
    plans=[e for e in config["episodes"] if (e["role"]=="calibration")== (args.mode=="collect-calibration")]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"source").mkdir()
    for n in CORE:shutil.copyfile(n,args.output/"source"/Path(n).name)
    save_json(args.output/"manifest.json",{"mode":args.mode,"config":config,"plans":plans,"code_sha256":hashes(),
        "created_UTC":datetime.now(timezone.utc).isoformat(),"model_sha256":digest(MODEL/"model.joblib") if MODEL.exists() else None,
        "lock_sha256":digest(LOCK/"development_lock.json") if LOCK.exists() else None,"days_override":args.days})
    summaries=[]
    if args.mode in ["collect","collect-calibration"]:
        tasks=[(config["development_case"],p["day"],3,"excitation","perfect",Excitation(config,pc,p["seed"])) for p in plans]
    elif args.mode=="response":
        tasks=[]
        for d in config["response_days"]:
            for setting in config["response_settings_C"]:
                policy=Controller(joblib.load(MODEL/"model.joblib"),config,pc,load(MODEL/"development_margins.json"));policy.setting=setting
                tasks.append((config["development_case"],d,.5,f"response_{setting:g}","perfect",policy))
    else:
        dates=[p["day"] for p in plans if p["role"]=="selection"] if args.mode=="validation" else config["test_days"]
        case=config["development_case"] if args.mode=="validation" else config["test_case"]
        modes=["perfect"] if args.mode=="validation" else config["weather_modes"]
        margins=load(MODEL/"development_margins.json") if args.mode=="validation" else load(CALIBRATION/"margins.json")
        tasks=[]
        for d in dates:
            for w in modes:
                for kind in config["controllers"]:
                    if kind=="terminal_mpc":policy=Controller(joblib.load(MODEL/"model.joblib"),config,pc,margins,w)
                    elif kind=="office_prior":policy=o.OfficeController(joblib.load(o.MODEL/"model.joblib"),o.configuration(),pc,w)
                    else:policy=SolarController(joblib.load(FREEZE/"model.joblib"),pc,"guarded_rule",w)
                    tasks.append((case,d,args.days or (3 if args.mode=="validation" else 7),kind,w,policy))
    for case,day,days,kind,weather,policy in tasks:
        if args.mode=="test":check_final()
        if args.mode=="collect-calibration":check_lock()
        name=f"day{day}__excitation" if kind=="excitation" else f"day{day}_{weather}__{kind}"
        summaries.append(run_episode(config,case,day,days,kind,weather,policy,args.output/name))
        save_json(args.output/"summary.json",summaries)


# Frozen office acquisition contract, extended only with the explicit testcase.
def run_episode(config,case,day,days,kind,weather,policy,output):
    output.mkdir(parents=True,exist_ok=False)
    client = Client("http://127.0.0.1:8080")
    if client.call("GET","version")["version"]!="0.9.0": raise ValueError("BOPTEST version mismatch")
    testid = client.call("POST",f"testcases/{case}/select")["testid"]
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
        summary={"case":case,"day":day,"days":days,"controller":kind,"weather":weather,"complete":failure is None and len(records)==int(days*96),
            "failure":failure,"transitions":len(records),"native_samples":len(native),"wall_seconds":time.perf_counter()-started,
            "metrics":integrate(native,forecast,IO,48) if len(native)>1 else None,"api_difference":differences,
            "sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
        save_json(output/"summary.json",summary);return summary
    finally: client.call("PUT",f"stop/{testid}")


if __name__=="__main__":main()
