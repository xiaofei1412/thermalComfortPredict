"""Audit locked calibration, causal regime margins, and unseen climate experiments."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_generalization import acceptance, csv, same_info
from thermal_comfort import terminal as t, terminal_model as tm, office as o, office_model as om
from thermal_comfort.generalization import FREEZE, integrate
from thermal_comfort.recovery import weather_view
from thermal_comfort.simulation import digest, save_json, schedule_at
from thermal_comfort.solar import SolarController


def records(p):return [json.loads(v) for v in (p/"steps.jsonl").read_text().splitlines()]
def directory(root,s):return root/(f"day{s['day']}__excitation" if s["controller"]=="excitation" else f"day{s['day']}_{s['weather']}__{s['controller']}")


def policy_for(s,margins):
    c=t.configuration();pc=t.policy_config();kind=s["controller"]
    if kind=="excitation":return t.Excitation(c,pc,next(p["seed"] for p in c["episodes"] if p["day"]==s["day"]))
    if kind=="office_prior":return o.OfficeController(joblib.load(o.MODEL/"model.joblib"),o.configuration(),pc,s["weather"])
    if kind=="guarded_rule":return SolarController(joblib.load(FREEZE/"model.joblib"),pc,kind,s["weather"])
    p=t.Controller(joblib.load(t.MODEL/"model.joblib"),c,pc,margins,s["weather"])
    if kind.startswith("response_"):p.setting=float(kind.split("_")[-1])
    return p


def audit_one(root,s,margins):
    p=directory(root,s);e=t.read_episode(p,s,"evaluation");native=csv(p/"native.csv.gz");forecast=csv(p/"exogenous.csv.gz");r=records(p)
    assert t.load(p/"summary.json")==s and s["transitions"]==len(r)
    if s["complete"]:assert len(r)==int(s["days"]*96)
    else:assert s["failure"] and len(r)<int(s["days"]*96)
    assert np.all(np.diff(native.time)==30.)
    pc=t.policy_config();policy=policy_for(s,margins)
    policy.reset(r[0]["before"],{"oveTSet_u":t.load(p/"api_metadata.json")["inputs"][o.IO["setpoint"]]})
    prelude=t.load(p/"prelude.json");assert len(prelude)==96 and prelude[-1]["after"]["time"]==s["day"]*86400
    assert all(v["action"]=={o.IO["setpoint"]:295.15,o.IO["active"]:1} for v in prelude)
    modes=Counter();reasons=Counter()
    for k,v in enumerate(r):
        start=s["day"]*86400+k*900
        assert v["before"]["time"]==start and v["after"]["time"]==start+900
        action,info=policy.decide(k,v["before"],forecast);assert action==v["action"];same_info(info,v["control"])
        assert v["actual_action"]=={o.IO["setpoint"]:action["oveTSet_u"],o.IO["active"]:1}
        assert pc["action_min_C"]<=action["oveTSet_u"]-273.15<=pc["action_max_C"]
        assert abs(action["oveTSet_u"]-v["before"]["oveTSet_u"])<=pc["slew_K_per_step"]+1e-9
        part=native[native.time.between(start,start+900)].copy();assert len(part)==31
        np.testing.assert_allclose(part[o.IO["temperature"]].iloc[[0,-1]],[v["before"]["reaTZon_y"],v["after"]["reaTZon_y"]],atol=1e-10,rtol=0)
        applied=part[part.time>start]
        np.testing.assert_allclose(applied[o.IO["setpoint"]],action["oveTSet_u"],atol=1e-8,rtol=0)
        assert (applied[o.IO["active"]]==1).all()
        part["reaQFloHea_y"]=.9*part["fcu_reaPHea_y"]
        try:policy.update(v["before"],v["after"],part,schedule_at(forecast,[start]).iloc[0])
        except ValueError as error:
            if s["complete"] or k!=len(r)-1 or str(error)!=s["failure"]:raise
        modes[info["mode"]]+=1;reasons[info["reason"]]+=1
    assert (native[["con_oveTSetCoo_activate","fcu_oveFan_activate","fcu_oveTSup_activate"]]==0).all().all()
    assert (native[["fcu_reaPHea_y",*o.IO["electric"]]]>=-1e-8).all().all()
    metrics=integrate(native,forecast,o.IO,48)
    for key,value in metrics.items():np.testing.assert_allclose(value,s["metrics"][key],atol=1e-9,rtol=1e-10)
    full=integrate(csv(p/"including_prelude.csv.gz"),forecast,o.IO,48);official=t.load(p/"api_kpis.json")
    delta=max(abs(full[m]-official[k]) for m,k in [("official_energy_kWh_m2","ener_tot"),("discomfort_Kh","tdis_tot")]);assert delta<=1e-6
    native_stop_band=pc["abort_min_C"]<=metrics["temperature_min_C"]<=metrics["temperature_max_C"]<=pc["abort_max_C"]
    weather_error={}
    if s["case"]==t.configuration()["test_case"]:
        scenario=t.load(Path("outputs/terminal-scenario/scenario.json"))
        with ZipFile(scenario["weather_donor"]) as donor:w=pd.read_csv(donor.open("resources/weather.csv"))
        for name,column in [("Lat","lat"),("Lon","lon")]:
            np.testing.assert_allclose(native[o.IO["weather_prefix"]+name+"_y"],w[column].iloc[0],atol=1e-10,rtol=0)
        schedule=schedule_at(forecast,native.time.to_numpy())
        weather_error={key:float(np.max(abs(native[o.IO["weather_prefix"]+key+"_y"]-schedule[key]))) for key in ["TDryBul","HGloHor"]}
        for api,column in [("SolZen","solZen"),("SolTim","solTim")]:
            actual=np.array([v["before"][o.IO["weather_prefix"]+api+"_y"] for v in r])
            expected=np.interp(e["times"],w.time,w[column])
            weather_error[api+"_vs_donor_csv"]=float(np.max(abs(actual-expected)))
    n=max(0,len(r)-16)
    return e,{"episode":p.name,"complete":s["complete"],"failure":s["failure"],"native_stop_band_pass":native_stop_band,
        "decisions_replayed":len(r),"modes":dict(modes),"reasons":dict(reasons),
        "fallback_after_initialization":modes["fallback"]/n if n else None,"mpc_after_initialization":modes["mpc"]/n if n else None,
        "reference_selected_after_initialization":modes["reference_selected"]/n if n else None,
        "official_kpi_max_difference":delta,"actual_forecast_weather_max_error":weather_error}


def predictions(root,s,margins,e=None):
    p=directory(root,s);e=e or t.read_episode(p,s,"evaluation");c=t.configuration();pc=t.policy_config()
    origins=np.arange(16,len(e["actions"])-32+1,c["origin_stride"])
    if not len(origins):return []
    forecast=csv(p/"exogenous.csv.gz");prior=joblib.load(o.MODEL/"model.joblib");model=joblib.load(t.MODEL/"model.joblib")
    rows=[]
    for view in ["perfect","supplied"]:
        future=np.array([o.office_exogenous(weather_view(schedule_at(forecast,e["times"][k]+np.arange(32)*900),e["times"][k],pc,
            "perfect" if view=="perfect" else s["weather"])) for k in origins])
        old_states=om.observer_states(prior,e)
        actions=np.array([e["actions"][k:k+32] for k in origins]);settings=np.array([om.cooling_setpoint(e["times"][k:k+32]) for k in origins])
        pred={"office_prior":om.rollout(prior,old_states[origins],e["hvac"][origins-1],actions,future,settings),
            "selected":tm.episode_rollout(model,e,origins,32,future)}
        for h in c["horizons"]:
            regime=tm.regime_indices(e["end_hvac"][origins-1],e["times"][origins],h,c)
            truth=np.array([e["energy"][k:k+h].sum(axis=0) for k in origins])
            cool_truth=np.array([.25*e["hvac"][k:k+h,1].sum()/3 for k in origins])
            fan_truth=np.array([.25*e["hvac"][k:k+h,2].sum() for k in origins])
            for name,values in pred.items():
                error=values[:,h-1,0]-e["temperature"][origins+h]
                if name=="office_prior":lower=upper=np.full(len(origins),prior["margin_K"][h-1])
                else:lower=np.array(margins["lower_K"])[regime,h-1];upper=np.array(margins["upper_K"])[regime,h-1]
                for group,mask in [("all",np.ones(len(origins),bool)),*[(label,regime==j) for j,label in enumerate(c["regimes"])]]:
                    if not mask.any():continue
                    row={"episode":p.name,"day":s["day"],"controller":s["controller"],"weather":s["weather"],"view":view,"model":name,
                        "complete_trajectory":s["complete"],"observed_transitions":s["transitions"],
                        "horizon_steps":h,"regime":group,"origins":int(mask.sum()),"temperature_RMSE_C":float(np.sqrt(np.mean(error[mask]**2))),
                        "temperature_bias_C":float(error[mask].mean()),"lower_margin_mean_K":float(lower[mask].mean()),"upper_margin_mean_K":float(upper[mask].mean()),
                        "lower_coverage":float(np.mean(error[mask]<=lower[mask])),"upper_coverage":float(np.mean(-error[mask]<=upper[mask])),
                        "two_sided_coverage":float(np.mean((error[mask]<=lower[mask])&(-error[mask]<=upper[mask])))}
                    for label,er in [("electricity",values[:,:h,5].sum(axis=1)-truth[:,0]),("gas",values[:,:h,6].sum(axis=1)-truth[:,1]),
                        ("cooling",.25*values[:,:h,3].sum(axis=1)/3-cool_truth),("fan",.25*values[:,:h,4].sum(axis=1)-fan_truth)]:
                        row[label+"_RMSE_kWh"]=float(np.sqrt(np.mean(er[mask]**2)));row[label+"_bias_kWh"]=float(er[mask].mean())
                    rows.append(row)
    return rows


def paired(root,summaries):
    c=t.configuration()
    for day in dict.fromkeys(s["day"] for s in summaries):
        for weather in dict.fromkeys(s["weather"] for s in summaries):
            paths=[root/f"day{day}_{weather}__{k}" for k in c["controllers"]]
            baseline=paths[0]
            for p in paths[1:]:
                assert t.load(p/"prelude.json")==t.load(baseline/"prelude.json")
                pd.testing.assert_frame_equal(csv(p/"exogenous.csv.gz"),csv(baseline/"exogenous.csv.gz"),check_exact=True)
                for x,y in zip(records(p)[:16],records(baseline)[:16]):assert all(x[k]==y[k] for k in ["before","action","after"])
        if len({s["weather"] for s in summaries})>1:
            for weather in c["weather_modes"][1:]:
                pd.testing.assert_frame_equal(csv(root/f"day{day}_{weather}__guarded_rule/native.csv.gz"),csv(root/f"day{day}_perfect__guarded_rule/native.csv.gz"),check_exact=True)


def development():
    assert not t.CALIBRATION.exists() and not t.LOCK.exists()
    c=t.configuration();audits=[];ps=[]
    assert t.load(t.MODEL/"manifest.json")["code_sha256"]==t.hashes()
    margins=t.load(t.MODEL/"development_margins.json")
    for name,h in t.load(t.MODEL/"manifest.json")["artifacts_sha256"].items():
        assert digest(t.MODEL/name)==h==digest(Path("outputs/terminal-refit")/name)
    for root in [t.DEVELOPMENT,Path("outputs/terminal-validation"),Path("outputs/terminal-response")]:
        for name,h in t.load(root/"manifest.json")["code_sha256"].items():assert digest(root/"source"/Path(name).name)==h
        for s in t.load(root/"summary.json"):
            e,a=audit_one(root,s,margins);audits.append(a)
            if root.name=="terminal-validation":ps.extend(predictions(root,s,margins,e))
    paired(Path("outputs/terminal-validation"),t.load(Path("outputs/terminal-validation/summary.json")))
    assert all(a["complete"] and a["native_stop_band_pass"] for a in audits if not a["episode"].endswith("__office_prior"))
    responses=[];active=[]
    for day in c["response_days"]:
        pairs=[]
        for setting in c["response_settings_C"]:
            p=Path("outputs/terminal-response")/f"day{day}_perfect__response_{setting:g}";s=t.load(p/"summary.json")
            pairs.append((records(p),t.read_episode(p,s,"response")))
        low,high=pairs
        for x,y in zip(low[0][:16],high[0][:16]):assert all(x[k]==y[k] for k in ["before","action","after"])
        predicted_pair=[tm.episode_rollout(joblib.load(t.MODEL/"model.joblib"),v[1],np.array([16]),32)[0] for v in pairs]
        for v,prediction in zip(pairs,predicted_pair):
            np.testing.assert_allclose(prediction[:,0],v[0][16]["control"]["predicted_temperature_C"],atol=1e-10,rtol=1e-10)
        for h in c["horizons"]:
            actual_t=high[1]["temperature"][16+h]-low[1]["temperature"][16+h]
            predicted_t=high[0][16]["control"]["predicted_temperature_C"][h-1]-low[0][16]["control"]["predicted_temperature_C"][h-1]
            actual_gas=high[1]["energy"][16:16+h,1].sum()-low[1]["energy"][16:16+h,1].sum()
            predicted_gas=sum(high[0][16]["control"]["predicted_gas_kWh"][:h])-sum(low[0][16]["control"]["predicted_gas_kWh"][:h])
            actual_cool=.25*(high[1]["hvac"][16:16+h,1].sum()-low[1]["hvac"][16:16+h,1].sum())/3
            predicted_cool=.25*(predicted_pair[1][:h,3].sum()-predicted_pair[0][:h,3].sum())/3
            responses.append({"day":day,"horizon_steps":h,"actual_temperature_delta_C":float(actual_t),"predicted_temperature_delta_C":float(predicted_t),
                "actual_gas_delta_kWh":float(actual_gas),"predicted_gas_delta_kWh":float(predicted_gas),
                "actual_cooling_delta_kWh":float(actual_cool),"predicted_cooling_delta_kWh":float(predicted_cool)})
            if h==4 and actual_gas>.001:active.append(actual_t>=-1e-6 and predicted_t>=-1e-6 and predicted_gas>0)
    assert active and all(active),"Excited first-hour response direction failed"
    output=Path("outputs/terminal-validation")
    pd.DataFrame(ps).to_csv(output/"prediction_metrics.csv",index=False,lineterminator="\n")
    save_json(output/"development_audit.json",{"passed":True,"calibration_not_generated":True,"independent_refit_byte_identical":True,
        "code_sha256":t.hashes(),"model_sha256":digest(t.MODEL/"model.joblib"),"windows":t.validate_windows(c),"responses":responses,"audits":audits})
    print("Development audit passed",flush=True)


def calibration():
    lock=t.check_lock();c=t.configuration();meta=t.load(t.CALIBRATION/"manifest.json")
    assert meta["lock_sha256"]==digest(t.LOCK/"development_lock.json") and meta["code_sha256"]==t.hashes()
    assert datetime.fromisoformat(meta["created_UTC"])>datetime.fromisoformat(lock["created_UTC"])
    episodes=t.dataset(t.CALIBRATION);margins=t.load(t.CALIBRATION/"margins.json");model=joblib.load(t.MODEL/"model.joblib")
    assert tm.calibrate(model,episodes,c)==margins
    ps=[];audits=[];feasibility=[]
    for s in t.load(t.CALIBRATION/"summary.json"):
        e,a=audit_one(t.CALIBRATION,s,margins);audits.append(a);ps.extend(predictions(t.CALIBRATION,s,margins,e))
        p=directory(t.CALIBRATION,s);rs=records(p);native=csv(p/"native.csv.gz");forecast=csv(p/"exogenous.csv.gz")
        policy=t.Controller(model,c,t.policy_config(),margins);policy.reset(rs[0]["before"],{"oveTSet_u":t.load(p/"api_metadata.json")["inputs"][o.IO["setpoint"]]})
        reasons=Counter();modes=Counter()
        for k,r in enumerate(rs):
            action,info=policy.decide(k,r["before"],forecast)
            if k>=16:reasons[info["reason"]]+=1;modes[info["mode"]]+=1
            part=native[native.time.between(r["before"]["time"],r["after"]["time"])].copy();part["reaQFloHea_y"]=.9*part["fcu_reaPHea_y"]
            policy.update(r["before"],r["after"],part,schedule_at(forecast,[r["before"]["time"]]).iloc[0])
        feasibility.append({"day":s["day"],"operational_steps":len(rs)-16,"modes":dict(modes),"reasons":dict(reasons)})
    pd.DataFrame(ps).to_csv(t.CALIBRATION/"prediction_metrics.csv",index=False,lineterminator="\n")
    save_json(t.CALIBRATION/"audit.json",{"passed":True,"development_locked_before_collection":True,"margins_exactly_reconstructed":True,
        "model_sha256":digest(t.MODEL/"model.joblib"),"untouched_calibration_days":[e["day"] for e in episodes],"audits":audits,
        "shadow_feasibility_on_excitation_states_not_closed_loop":feasibility})
    t.check_lock();print("Untouched calibration and feasibility audit passed",flush=True)


def test():
    frozen=t.check_final();c=t.configuration();root=Path("outputs/terminal-control");repeat=Path("outputs/terminal-repeat")
    summaries=t.load(root/"summary.json");grid={(d,w,k) for d in c["test_days"] for w in c["weather_modes"] for k in c["controllers"]}
    assert len(summaries)==len(grid) and {(s["day"],s["weather"],s["controller"]) for s in summaries}==grid
    margins=t.load(t.CALIBRATION/"margins.json");audits=[];ps=[];rows=[];daily=[]
    for s in summaries:
        assert s["days"]==7 and s["case"]==c["test_case"]
        e,a=audit_one(root,s,margins);audits.append(a);ps.extend(predictions(root,s,margins,e));rows.append({k:s[k] for k in ["day","weather","controller","complete"]}|s["metrics"])
        p=directory(root,s);native=csv(p/"native.csv.gz");forecast=csv(p/"exogenous.csv.gz")
        for day in range(7):
            start=(s["day"]+day)*86400
            part=native[native.time.between(start,start+86400)]
            if len(part)>1:
                hours=float((part.time.iloc[-1]-part.time.iloc[0])/3600)
                daily.append({"episode":p.name,"day_index":day,"observed_hours":hours,"full_day":hours==24.,
                    **integrate(part,forecast,o.IO,48)})
        print("Verified",p.name,flush=True)
    paired(root,summaries);gates=[]
    for day in c["test_days"]:
        for weather in c["weather_modes"]:
            group={s["controller"]:s for s in summaries if s["day"]==day and s["weather"]==weather}
            for baseline in ["guarded_rule","office_prior"]:
                complete=group[baseline]["complete"] and group["terminal_mpc"]["complete"]
                gate=acceptance(group[baseline]["metrics"],group["terminal_mpc"]["metrics"],complete)
                gate["passed"] &= all(a["native_stop_band_pass"] for a in audits if a["episode"] in [f"day{day}_{weather}__{baseline}",f"day{day}_{weather}__terminal_mpc"])
                gates.append({"day":day,"weather":weather,"baseline":baseline,**gate})
    prefixes=[];repeated=t.load(repeat/"summary.json");assert len(repeated)==len(grid)
    assert {(s["day"],s["weather"],s["controller"]) for s in repeated}==grid
    for s in repeated:
        original=next(v for v in summaries if (v["day"],v["weather"],v["controller"])==(s["day"],s["weather"],s["controller"]))
        assert s["transitions"]==min(24,original["transitions"]) and s["days"]==.25 and s["case"]==c["test_case"]
        if s["transitions"]<24:assert not s["complete"] and s["failure"]==original["failure"]
        else:assert s["complete"]
        full,short=directory(root,s),directory(repeat,s)
        for name,h in s["sha256"].items():assert digest(short/name)==h
        a,b=csv(full/"native.csv.gz"),csv(short/"native.csv.gz")
        pd.testing.assert_frame_equal(a[a.time<=b.time.iloc[-1]].reset_index(drop=True),b,check_exact=True)
        assert t.load(full/"prelude.json")==t.load(short/"prelude.json")
        for x,y in zip(records(full),records(short)):
            assert all(x[k]==y[k] for k in ["before","action","after","actual_action"])
            same_info({k:v for k,v in x["control"].items() if k!="decision_seconds"},y["control"])
        prefixes.append({"episode":full.name,"steps":s["transitions"],"native_samples":len(b),"exact_match":True})
    for p in [root,repeat]:
        meta=t.load(p/"manifest.json");assert meta["code_sha256"]==frozen["code_sha256"] and meta["model_sha256"]==frozen["model_sha256"]
    pd.DataFrame(rows).to_csv(root/"comparison.csv",index=False,lineterminator="\n")
    pd.DataFrame(ps).to_csv(root/"prediction_metrics.csv",index=False,lineterminator="\n")
    pd.DataFrame(daily).to_csv(root/"daily_metrics.csv",index=False,lineterminator="\n")
    save_json(root/"acceptance.json",gates);save_json(root/"decision_audit.json",audits)
    save_json(root/"verification.json",{"frozen_source_model_margins_pass":True,"planned_episodes":len(summaries),"complete_episodes":sum(s["complete"] for s in summaries),
        "formal_transitions":sum(s["transitions"] for s in summaries),"native_samples":sum(s["native_samples"] for s in summaries),
        "decisions_replayed":sum(a["decisions_replayed"] for a in audits),"official_kpi_max_difference":max(a["official_kpi_max_difference"] for a in audits),
        "paired_and_rule_weather_invariance":True,"prefixes":prefixes,"full_seven_day_repeat_verified":False})
    t.check_final()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["development","calibration","test"])
    args=parser.parse_args();{"development":development,"calibration":calibration,"test":test}[args.mode]()
