"""Reconstruct office development evidence and audit untouched time holdouts."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_generalization import acceptance, csv, load, same_info
from thermal_comfort import greybox, office as o, office_model as om
from thermal_comfort.generalization import FREEZE, check_freeze, integrate
from thermal_comfort.mpc import exogenous
from thermal_comfort.recovery import weather_view
from thermal_comfort.simulation import digest, save_json, schedule_at
from thermal_comfort.solar import SolarController


def records(directory):
    return [json.loads(s) for s in (directory/"steps.jsonl").read_text().splitlines()]


def directory(root,s):
    middle="" if s["controller"]=="excitation" or s["controller"].startswith("response_") else s["weather"]
    return root/f"day{s['day']}_{middle}__{s['controller']}" if middle else root/f"day{s['day']}__{s['controller']}"


def audit_episode(root,s,config,frozen):
    p=directory(root,s); e=o.read_episode(p,s,"evaluation")
    if load(p/"summary.json")!=s:raise ValueError("Summary differs")
    rows=records(p); native=csv(p/"native.csv.gz"); forecast=csv(p/"exogenous.csv.gz")
    pc=frozen["policy_config"]; kind=s["controller"]
    if kind=="excitation":
        seed=next(v["seed"] for v in config["episodes"] if v["day"]==s["day"])
        policy=o.Excitation(config,pc,seed)
    elif kind=="office_mpc" or kind.startswith("response_"):
        setting=float(kind.split("_")[-1]) if kind.startswith("response_") else None
        policy=o.OfficeController(joblib.load(o.MODEL/"model.joblib"),config,pc,s["weather"],setting)
    else:policy=SolarController(joblib.load(FREEZE/"model.joblib"),pc,kind,s["weather"])
    inputs=load(p/"api_metadata.json")["inputs"]
    policy.reset(rows[0]["before"],{"oveTSet_u":inputs[o.IO["setpoint"]]})
    prelude=load(p/"prelude.json")
    assert len(prelude)==96 and prelude[-1]["after"]["time"]==s["day"]*86400
    assert all(r["action"]=={o.IO["setpoint"]:295.15,o.IO["active"]:1} for r in prelude)
    assert s["transitions"]==int(s["days"]*96)==len(rows)
    assert np.all(np.diff(native.time)==30.)
    modes,reasons=Counter(),Counter()
    for k,r in enumerate(rows):
        t=s["day"]*86400+k*900
        assert r["before"]["time"]==t and r["after"]["time"]==t+900
        action,info=policy.decide(k,r["before"],forecast)
        assert action==r["action"]
        same_info(info,r["control"])
        assert r["actual_action"]=={o.IO["setpoint"]:action["oveTSet_u"],o.IO["active"]:1}
        assert pc["action_min_C"]<=action["oveTSet_u"]-273.15<=pc["action_max_C"]
        assert abs(action["oveTSet_u"]-r["before"]["oveTSet_u"])<=pc["slew_K_per_step"]+1e-9
        part=native[native.time.between(t,t+900)].copy()
        assert len(part)==31
        np.testing.assert_allclose(part[o.IO["temperature"]].iloc[[0,-1]],
            [r["before"]["reaTZon_y"],r["after"]["reaTZon_y"]],rtol=0,atol=1e-10)
        applied=part[part.time>t]
        np.testing.assert_allclose(applied[o.IO["setpoint"]],action["oveTSet_u"],rtol=0,atol=1e-8)
        assert (applied[o.IO["active"]]==1).all()
        part["reaQFloHea_y"]=.9*part["fcu_reaPHea_y"]
        policy.update(r["before"],r["after"],part,schedule_at(forecast,[t]).iloc[0])
        modes[info["mode"]]+=1; reasons[info["reason"]]+=1
    for key in ["con_oveTSetCoo_activate","fcu_oveFan_activate","fcu_oveTSup_activate"]:
        assert (native[key]==0).all(),"Non-heating override applied"
    assert (native[["fcu_reaPHea_y",*o.IO["electric"]]].to_numpy()>=-1e-8).all()
    schedule=schedule_at(forecast,e["times"])
    # The forecast table retains the left limit at 18:00; the controller switches then.
    away_from_evening_switch=e["times"]%86400!=18*3600
    np.testing.assert_allclose(om.cooling_setpoint(e["times"])[away_from_evening_switch],
        (schedule["UpperSetp[1]"]-273.15)[away_from_evening_switch],atol=1e-10)
    metrics=integrate(native,forecast,o.IO,48)
    for key,v in metrics.items():np.testing.assert_allclose(v,s["metrics"][key],rtol=1e-10,atol=1e-9)
    full=integrate(csv(p/"including_prelude.csv.gz"),forecast,o.IO,48); official=load(p/"api_kpis.json")
    delta=max(abs(full[m]-official[k]) for m,k in [("official_energy_kWh_m2","ener_tot"),("discomfort_Kh","tdis_tot")])
    assert delta<=1e-6
    assert pc["abort_min_C"]<=metrics["temperature_min_C"]<=metrics["temperature_max_C"]<=pc["abort_max_C"]
    n=max(0,len(rows)-pc["initial_rule_steps"])
    audit={"episode":p.name,"decisions_replayed":len(rows),"modes":dict(modes),"reasons":dict(reasons),
        "fallback_after_initialization":modes["fallback"]/n if n else None,
        "reference_selected_after_initialization":modes["reference_selected"]/n if n else None,
        "mpc_after_initialization":modes["mpc"]/n if n else None,"official_kpi_max_difference":delta}
    return e,audit


def prediction_metrics(root,s,config,frozen,e=None):
    p=directory(root,s)
    if e is None:e=o.read_episode(p,s,"evaluation")
    origins=np.arange(16,len(e["actions"])-32+1,config["origin_stride"])
    if not len(origins):return []
    forecast=csv(p/"exogenous.csv.gz"); saved=records(p); pc=frozen["policy_config"]
    source=joblib.load(FREEZE/"model.joblib"); observer=SolarController(source,pc,"guarded_rule")
    observer.reset(saved[0]["before"],{"oveTSet_u":load(p/"api_metadata.json")["inputs"][o.IO["setpoint"]]})
    snapshots=[];heat=[];error=[]
    for k,r in enumerate(saved):
        if k in origins:snapshots.append(observer.state.copy());heat.append(observer.q);error.append(observer.error)
        interval=pd.DataFrame({"time":[0.,900.],"reaQFloHea_y":[e["hvac"][k,0]*1000]*2})
        observer.update(r["before"],r["after"],interval,schedule_at(forecast,[e["times"][k]]).iloc[0])
    actions=np.array([e["actions"][k:k+32] for k in origins]); output=[]
    for view in ["perfect","supplied"]:
        future=np.array([o.office_exogenous(weather_view(schedule_at(forecast,e["times"][k]+np.arange(32)*900),
            e["times"][k],pc,"perfect" if view=="perfect" else s["weather"])) for k in origins])
        predictions={"frozen":greybox.rollout(source,np.array(snapshots),np.array(heat),np.array(error),actions,future[:,:,:5])}
        models={"frozen":source}
        for days in config["training_days_per_episode"]:
            name=f"calibrated_{days*3}d"; m=joblib.load(o.MODEL/f"budget_{days}.joblib");models[name]=m
            states=om.observer_states(m,e)
            settings=np.array([om.cooling_setpoint(e["times"][k:k+32]) for k in origins])
            predictions[name]=om.rollout(m,states[origins],e["hvac"][origins-1],actions,future,settings)
        for name,pred in predictions.items():
            for h in config["horizons"]:
                err=pred[:,h-1,0]-e["temperature"][origins+h]
                truth=np.array([e["energy"][k:k+h].sum(axis=0) for k in origins])
                er=pred[:,:h,3 if name=="frozen" else 5].sum(axis=1)-truth[:,0]
                gr=None if name=="frozen" else pred[:,:h,6].sum(axis=1)-truth[:,1]
                output.append({"episode":p.name,"day":s["day"],"controller":s["controller"],"weather":s["weather"],
                    "view":view,"model":name,"horizon_steps":h,"origins":len(origins),
                    "temperature_RMSE_C":float(np.sqrt(np.mean(err**2))),"temperature_bias_C":float(err.mean()),
                    "margin_K":float(models[name]["margin_K"][h-1]),"margin_coverage":float(np.mean(abs(err)<=models[name]["margin_K"][h-1])),
                    "electricity_RMSE_kWh":float(np.sqrt(np.mean(er**2))),"electricity_bias_kWh":float(er.mean()),
                    "gas_RMSE_kWh":None if gr is None else float(np.sqrt(np.mean(gr**2))),
                    "gas_bias_kWh":None if gr is None else float(gr.mean())})
    return output


def verify_development(config,frozen):
    audits=[];episodes=[]; regimes=[]
    summaries=load(o.DATA/"summary.json")
    collected=load(o.DATA/"manifest.json")["config"]
    assert all(collected[k]==config[k] for k in ["baseline_commit","domain","episodes","collection_days","heat_efficiency","cooling_COP"])
    assert len(summaries)==len(config["episodes"])
    for plan,s in zip(config["episodes"],summaries):
        assert (s["day"],s["days"],s["controller"])==(plan["day"],3,"excitation")
        e,a=audit_episode(o.DATA,s,config,frozen);e["role"]=plan["role"];episodes.append(e);audits.append(a)
        regimes.append({"day":s["day"],"role":plan["role"],"rows":len(e["actions"]),
            "heat_rows":int((e["hvac"][:,0]>.01).sum()),"cool_rows":int((e["hvac"][:,1]>.01).sum()),
            "idle_rows":int((e["hvac"][:,:2].max(axis=1)<=.01).sum())})
        old=Path("outputs/office-data-before-radiant")/directory(o.DATA,s).name
        if old.exists():
            pd.testing.assert_frame_equal(csv(old/"native.csv.gz"),csv(directory(o.DATA,s)/"native.csv.gz"),check_exact=True)
            for x,y in zip(records(old),records(directory(o.DATA,s))):
                assert all(x[k]==y[k] for k in ["before","action","after","actual_action"])
    selected=load(o.MODEL/"selection.json")
    assert selected["selected"]==min(selected["scores"],key=lambda s:s["score_C"])
    np.testing.assert_array_equal(joblib.load(o.MODEL/"model.joblib")["parameters"],
        joblib.load(o.MODEL/f"budget_{selected['selected']['days_per_episode']}.joblib")["parameters"])
    calibration_origins=[]
    for days in config["training_days_per_episode"]:
        m=joblib.load(o.MODEL/f"budget_{days}.joblib");assert m["fit_diagnostics"]["training_rows"]==days*288
        errors=[]
        for e in episodes:
            if e["role"]!="calibration":continue
            origins=np.arange(16,len(e["actions"])-32+1,config["origin_stride"])
            prediction=om.episode_rollout(m,e,origins,32)
            errors.append(prediction[:,:,0]-e["temperature"][origins[:,None]+np.arange(1,33)])
            if days==1:calibration_origins.append({"id":e["id"],"origins":origins.tolist()})
        np.testing.assert_array_equal(m["margin_K"],np.quantile(abs(np.concatenate(errors)),config["margin_quantile"],axis=0)+config["sensor_margin_K"])
        assert max(abs(np.linalg.eigvals(m["A"])))<1
    assert selected["calibration"]==calibration_origins
    for file,expected in load(o.MODEL/"manifest.json")["artifacts_sha256"].items():assert digest(o.MODEL/file)==expected
    for name in ["budget_1.joblib","budget_3.joblib","model.joblib","selection.json","split_manifest.json"]:
        assert digest(o.MODEL/name)==digest(Path("outputs/office-refit")/name),"Independent refit differs"
    validation=Path("outputs/office-validation"); response=Path("outputs/office-response")
    predictions=[]
    for s in load(validation/"summary.json"):
        e,a=audit_episode(validation,s,config,frozen);audits.append(a)
        predictions.extend(prediction_metrics(validation,s,config,frozen,e))
    response_rows=[];heating_gates=[];carrier_gates=[];mixed_coverage=[];long_direction=[]
    model=joblib.load(o.MODEL/"model.joblib")
    for s in load(response/"summary.json"):
        e,a=audit_episode(response,s,config,frozen);audits.append(a)
    for day in config["response_days"]:
        curves={}
        for setting in config["response_settings_C"]:
            p=response/f"day{day}__response_{setting:g}";r=records(p);s=load(p/"summary.json")
            e=o.read_episode(p,s,"response");curves[setting]=(r,e)
        low,high=curves[min(curves)],curves[max(curves)]
        for pair in curves.values():
            for x,y in zip(pair[0][:16],low[0][:16]):assert all(x[k]==y[k] for k in ["before","action","after"])
            assert pair[0][16]["before"]==low[0][16]["before"]
        for h in [1,4,24,32]:
            actual=high[1]["temperature"][16+h]-low[1]["temperature"][16+h]
            predicted=high[0][16]["control"]["predicted_temperature_C"][h-1]-low[0][16]["control"]["predicted_temperature_C"][h-1]
            gas=high[1]["energy"][16:16+h,1].sum()-low[1]["energy"][16:16+h,1].sum()
            pg=sum(high[0][16]["control"]["predicted_gas_kWh"][:h])-sum(low[0][16]["control"]["predicted_gas_kWh"][:h])
            row={"day":day,"horizon_steps":h,"actual_temperature_high_minus_low_C":float(actual),
                "predicted_temperature_high_minus_low_C":float(predicted),"actual_gas_delta_kWh":float(gas),"predicted_gas_delta_kWh":float(pg),
                "paired_temperature_margin_K":float(2*model["margin_K"][h-1])}
            response_rows.append(row)
            if h in [1,4]:heating_gates.append(actual>=-1e-6 and predicted>=-1e-6 and gas>0 and pg>0)
            if h in [24,32]:
                carrier_gates.append(gas>0 and pg>0)
                mixed_coverage.append(abs(actual-predicted)<=2*model["margin_K"][h-1])
                long_direction.append(actual>=-1e-6 and predicted>=-1e-6)
    gate={"passed":bool(all(heating_gates+carrier_gates+mixed_coverage)),
        "heating_first_hour_direction_pass":bool(all(heating_gates)),
        "mixed_six_eight_hour_gas_direction_pass":bool(all(carrier_gates)),
        "mixed_paired_error_within_two_empirical_margins":bool(all(mixed_coverage)),
        "original_long_temperature_monotonicity_pass":bool(all(long_direction)),
        "model_sha256":digest(o.MODEL/"model.joblib"),"limited_development_gate_not_safety_certificate":True,"responses":response_rows}
    save_json(response/"gate.json",gate)
    pd.DataFrame(predictions).to_csv(validation/"prediction_metrics.csv",index=False,lineterminator="\n")
    verification={"passed":gate["passed"],"model_sha256":digest(o.MODEL/"model.joblib"),
        "code_sha256":{n:digest(Path(n)) for n in o.SOURCE_FILES},"windows":o.validate_windows(config),
        "independent_refit_byte_identical":True,"margins_reconstructed":True,"recollected_six_native_trajectories_identical":True,
        "development_transitions":sum(len(e["actions"]) for e in episodes),"regimes":regimes,"audits":audits}
    save_json(validation/"development_verification.json",verification)
    assert gate["passed"],"Action response direction check failed"
    print(json.dumps({"development_passed":True,"responses":response_rows},indent=2))


def verify_pairs(root,config):
    for day in config["test_days"]:
        for weather in config["weather_modes"]:
            paths=[root/f"day{day}_{weather}__{k}" for k in config["controllers"]]
            baseline=paths[0]
            for p in paths[1:]:
                assert load(p/"prelude.json")==load(baseline/"prelude.json")
                pd.testing.assert_frame_equal(csv(p/"exogenous.csv.gz"),csv(baseline/"exogenous.csv.gz"),check_exact=True)
                for x,y in zip(records(p)[:8],records(baseline)[:8]):assert all(x[k]==y[k] for k in ["before","action","after"])
        for weather in config["weather_modes"][1:]:
            pd.testing.assert_frame_equal(csv(root/f"day{day}_{weather}__guarded_rule/native.csv.gz"),
                csv(root/f"day{day}_perfect__guarded_rule/native.csv.gz"),check_exact=True)


def verify_test(config,frozen):
    office_frozen=o.check_office_freeze();root=Path("outputs/office-control");repeat=Path("outputs/office-repeat")
    summaries=load(root/"summary.json");expected={(d,w,k) for d in config["test_days"] for w in config["weather_modes"] for k in config["controllers"]}
    assert len(summaries)==len(expected) and {(s["day"],s["weather"],s["controller"]) for s in summaries}==expected
    rows=[];audits=[];predictions=[];daily=[]
    for s in summaries:
        assert s["days"]==7
        e,a=audit_episode(root,s,config,frozen);audits.append(a)
        rows.append({k:s[k] for k in ["day","weather","controller","complete"]}|s["metrics"])
        predictions.extend(prediction_metrics(root,s,config,frozen,e))
        p=directory(root,s);native=csv(p/"native.csv.gz");forecast=csv(p/"exogenous.csv.gz")
        for d in range(7):
            t=(s["day"]+d)*86400
            daily.append({"episode":p.name,"evaluation_day":d,**integrate(native[native.time.between(t,t+86400)],forecast,o.IO,48)})
        print("Verified",p.name,flush=True)
    verify_pairs(root,config)
    gates=[]
    for d in config["test_days"]:
        for w in config["weather_modes"]:
            group={s["controller"]:s for s in summaries if s["day"]==d and s["weather"]==w}
            for baseline in ["guarded_rule","solar_mpc"]:
                gates.append({"day":d,"weather":w,"baseline":baseline,
                    **acceptance(group[baseline]["metrics"],group["office_mpc"]["metrics"],all(s["complete"] for s in group.values()))})
    repeated=load(repeat/"summary.json");assert len(repeated)==len(summaries)
    assert {(s["day"],s["weather"],s["controller"]) for s in repeated}==expected
    prefix=[]
    for s in repeated:
        assert s["days"]==.25 and s["complete"] and s["transitions"]==24
        a,b=directory(root,s),directory(repeat,s)
        for name,value in s["sha256"].items():assert digest(b/name)==value
        full,short=csv(a/"native.csv.gz"),csv(b/"native.csv.gz")
        pd.testing.assert_frame_equal(full[full.time<=short.time.iloc[-1]].reset_index(drop=True),short,check_exact=True)
        assert load(a/"prelude.json")==load(b/"prelude.json")
        for x,y in zip(records(a),records(b)):
            assert all(x[k]==y[k] for k in ["before","after","action","actual_action"])
            same_info({k:v for k,v in x["control"].items() if k!="decision_seconds"},y["control"])
        prefix.append({"episode":a.name,"steps":24,"native_samples":len(short),"exact_match":True})
    for p in [root,repeat]:
        manifest=load(p/"manifest.json")
        assert manifest["config"]==config and manifest["code_sha256"]==office_frozen["code_sha256"]
        assert manifest["office_model_sha256"]==office_frozen["model_sha256"]
        assert manifest["source_freeze_sha256"]==digest(FREEZE/"freeze.json")
    pd.DataFrame(rows).to_csv(root/"comparison.csv",index=False,lineterminator="\n")
    pd.DataFrame(predictions).to_csv(root/"prediction_metrics.csv",index=False,lineterminator="\n")
    pd.DataFrame(daily).to_csv(root/"daily_metrics.csv",index=False,lineterminator="\n")
    save_json(root/"decision_audit.json",audits);save_json(root/"acceptance.json",gates)
    save_json(root/"verification.json",{"frozen_sources_pass":True,"formal_episodes":len(summaries),
        "complete_episodes":sum(s["complete"] for s in summaries),"formal_transitions":sum(s["transitions"] for s in summaries),
        "native_samples":sum(s["native_samples"] for s in summaries),"decisions_replayed":sum(a["decisions_replayed"] for a in audits),
        "official_kpi_max_difference":max(a["official_kpi_max_difference"] for a in audits),
        "paired_initialization_rule_invariance":True,"repeat_prefixes":prefix,"full_seven_day_repeat_verified":False})
    o.check_office_freeze()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["development","test"])
    args=parser.parse_args();config=o.configuration();frozen=check_freeze()
    if args.mode=="development":verify_development(config,frozen)
    else:verify_test(config,frozen)


if __name__=="__main__":main()
