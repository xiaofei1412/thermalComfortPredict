"""Audit frozen transfer experiments, including failures and both energy carriers."""

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import greybox
from thermal_comfort.generalization import FREEZE, canonical, check_freeze, integrate, interface
from thermal_comfort.mpc import exogenous
from thermal_comfort.recovery import weather_view
from thermal_comfort.simulation import digest, save_json, schedule_at
from thermal_comfort.solar import SolarController


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv(path):
    # Numeric API signals have one physical schema even when a night-only column is all zeros.
    return pd.read_csv(path,float_precision="round_trip",dtype=float)


def acceptance(rule, mpc, complete=True, tolerance=1e-6):
    comfort_keys = ["discomfort_Kh","cold_Kh","hot_Kh","outside_hours","maximum_cold_K","maximum_hot_K"]
    electric = bool(complete and mpc["electricity_kWh"] < rule["electricity_kWh"]-tolerance)
    comfort = bool(complete and all(mpc[k] <= rule[k]+tolerance for k in comfort_keys))
    carriers = bool(complete and all(mpc[k] <= rule[k]+tolerance for k in ["gas_kWh","total_energy_kWh"]))
    return {"energy_saved_percent":float(100*(rule["electricity_kWh"]-mpc["electricity_kWh"])/rule["electricity_kWh"])
                if complete and rule["electricity_kWh"]>tolerance else None,
            "electricity_lower":electric,"comfort_not_increased":comfort,"energy_carriers_not_increased":carriers,
            "passed":electric and comfort and carriers,
            **{k+"_delta":float(mpc[k]-rule[k]) if complete else None
               for k in comfort_keys+["electricity_kWh","gas_kWh","total_energy_kWh"]}}


def same_info(expected,saved):
    actual = {k:v for k,v in saved.items() if k!="decision_seconds"}
    if expected.keys()!=actual.keys():
        raise ValueError("Decision fields changed")
    for key,value in expected.items():
        if isinstance(value,(int,float,list)) and not isinstance(value,bool):
            np.testing.assert_allclose(value,actual[key],rtol=1e-10,atol=1e-10)
        elif value!=actual[key]:
            raise ValueError("Decision changed: "+key)


def verify_fmus(frozen):
    paths = frozen["fmus"]
    source = ZipFile(paths["bestest_hydronic_heat_pump"]["path"])
    target = ZipFile(paths["generalization_house_denver"]["path"])
    donor = ZipFile(paths["bestest_air"]["path"])
    if source.namelist()!=target.namelist():
        raise ValueError("Relocated house archive structure changed")
    changed = {name for name in source.namelist() if source.read(name)!=target.read(name)}
    if changed!={"resources/0/Brussels.mos","resources/weather.csv","resources/config.json"}:
        raise ValueError("Weather transfer changed physics/code")
    if target.read("resources/0/Brussels.mos")!=donor.read("resources/0/DRYCOLD.mos") or target.read("resources/weather.csv")!=donor.read("resources/weather.csv"):
        raise ValueError("Actual and forecast weather were not both replaced")
    for archive in [source,target,donor]:
        archive.close()
    return {"only_three_house_resources_changed":True,"binary_and_model_description_identical":True,
            "both_actual_and_forecast_weather_copied_from_pinned_donor":True}


def audit_episode(root,s,frozen):
    directory = root/f"{s['scenario']}__{s['controller']}"
    for name,expected in s["sha256"].items():
        if digest(directory/name)!=expected:
            raise ValueError("Trajectory source changed: "+str(directory/name))
    if load(directory/"summary.json")!=s:
        raise ValueError("Aggregate/episode summary mismatch")
    native,forecast = csv(directory/"native.csv.gz"),csv(directory/"exogenous.csv.gz")
    io,config = interface(s["domain"]),frozen["policy_config"]
    records = [json.loads(line) for line in (directory/"steps.jsonl").read_text().splitlines()]
    expected_times = s["day"]*86400+np.arange(len(records)+1)*900
    if len(records)!=s["transitions"] or native.time.iloc[0]!=expected_times[0] or native.time.iloc[-1]!=expected_times[-1]:
        raise ValueError("Trajectory boundaries incomplete")
    if s["complete"] and len(records)!=int(s["days"]*96):
        raise ValueError("Completed episode has missing actions")
    for key in io["electric"]+io["gas"]:
        if native[key].min() < -1e-6:
            raise ValueError("Negative physical consumption")
    metrics = integrate(native,forecast,io,s["domain"]["area_m2"]) if len(native)>1 else {}
    for key,value in metrics.items():
        np.testing.assert_allclose(value,s["metrics"][key],atol=1e-9,rtol=1e-10)
    full = integrate(csv(directory/"including_prelude.csv.gz"),forecast,io,s["domain"]["area_m2"])
    kpis = load(directory/"api_kpis.json")
    differences = {k:abs(full[m]-kpis[k]) for m,k in [("official_energy_kWh_m2","ener_tot"),("discomfort_Kh","tdis_tot")]}
    if max(differences.values())>1e-6:
        raise ValueError("Official KPI does not match independent full-period integration")
    if not records:
        if s["complete"] or not s["failure"]:
            raise ValueError("Empty episode without an explicit failure")
        return ({"scenario":s["scenario"],"domain":s["domain"]["id"],"day":s["day"],"weather":s["weather"],
            "controller":s["controller"],"temperature_kind":io["temperature_kind"],"complete":False},
            {"scenario":s["scenario"],"controller":s["controller"],"decisions_replayed":0,"failure":s["failure"],
             "modes":{},"reasons":{},"native_stop_band_pass":False,"official_kpi_max_difference":max(differences.values())},[],[])
    model = joblib.load(FREEZE/"model.joblib")
    policy = SolarController(model,config,s["controller"],s["weather"])
    inputs = load(directory/"api_metadata.json")["inputs"]
    policy.reset(records[0]["before"],{"oveTSet_u":inputs[io["setpoint"]]})
    prelude = load(directory/"prelude.json")
    if len(prelude)!=96 or prelude[-1]["after"]["time"]!=expected_times[0]:
        raise ValueError("Common prelude incomplete")
    for record in prelude:
        if record["action"]!={io["setpoint"]:295.15,io["active"]:1}:
            raise ValueError("Common prelude action changed")
    origins = np.arange(16,len(records)-32+1,frozen["config"]["origin_stride"])
    snapshots,heat,error,interval_energy = [],[],[],[]
    modes,reasons = Counter(),Counter()
    for index,r in enumerate(records):
        if r["before"]["time"]!=expected_times[index] or r["after"]["time"]!=expected_times[index+1]:
            raise ValueError("Step discontinuity")
        for observed in [r["before"],r["after"]]:
            mapped = canonical(observed,io)
            if any(observed[k]!=mapped[k] for k in ["reaTZon_y","reaQFloHea_y","oveTSet_u"]):
                raise ValueError("Saved policy aliases differ from physical measurements")
        if index in origins:
            snapshots.append(policy.state.copy()); heat.append(policy.q); error.append(policy.error)
        action,info = policy.decide(index,r["before"],forecast)
        if action!=r["action"]:
            np.testing.assert_allclose(action["oveTSet_u"],r["action"]["oveTSet_u"],atol=1e-10,rtol=0)
        same_info(info,r["control"])
        actual = {io["setpoint"]:action["oveTSet_u"],io["active"]:1}
        if actual!=r["actual_action"]:
            raise ValueError("Mapped actuator differs")
        command = action["oveTSet_u"]-273.15
        if not config["action_min_C"]<=command<=config["action_max_C"] or abs(action["oveTSet_u"]-r["before"]["oveTSet_u"])>config["slew_K_per_step"]+1e-9:
            raise ValueError("Action or slew bound violated")
        segment = native.loc[native.time.between(expected_times[index],expected_times[index+1])]
        if segment.time.iloc[0]!=expected_times[index] or segment.time.iloc[-1]!=expected_times[index+1]:
            raise ValueError("Missing native interval boundary")
        np.testing.assert_allclose(segment[io["temperature"]].iloc[[0,-1]],
            [r["before"]["reaTZon_y"],r["after"]["reaTZon_y"]],rtol=0,atol=1e-10)
        applied = segment[segment.time>expected_times[index]]
        np.testing.assert_allclose(applied[io["setpoint"]],action["oveTSet_u"],rtol=0,atol=1e-8)
        if not (applied[io["active"]]==1).all():
            raise ValueError("Override not applied")
        interval_energy.append(float(np.trapezoid(segment[io["electric"]].sum(axis=1),segment.time)/3600000))
        observed = pd.DataFrame({"time":segment.time,"reaQFloHea_y":segment[io["heat"]]*io["heat_factor"]})
        try:
            policy.update(r["before"],r["after"],observed,schedule_at(forecast,[expected_times[index]]).iloc[0])
        except ValueError:
            if s["complete"] or index!=len(records)-1:
                raise
        modes[info["mode"]]+=1; reasons[info.get("reason",info["mode"])]+=1
    if s["complete"] and (metrics["temperature_min_C"]<config["abort_min_C"] or metrics["temperature_max_C"]>config["abort_max_C"]):
        # Native excursions between decision boundaries are reported, not hidden by coarse sampling.
        native_stop_band = False
    else:
        native_stop_band = True
    prediction_rows = []
    if len(origins):
        actions = np.array([[records[o+k]["action"]["oveTSet_u"]-273.15 for k in range(32)] for o in origins])
        actual_temperature = np.array([r["after"]["reaTZon_y"]-273.15 for r in records])
        for view in ["perfect","supplied"]:
            future = []
            for o in origins:
                part = schedule_at(forecast,expected_times[o]+np.arange(32)*900)
                part = weather_view(part,expected_times[o],config,"perfect" if view=="perfect" else s["weather"])
                future.append(exogenous(part))
            predicted = greybox.rollout(model,np.array(snapshots),np.array(heat),np.array(error),actions,np.array(future))
            for h in frozen["config"]["prediction_horizons"]:
                temperature_error = predicted[:,h-1,0]-actual_temperature[origins+h-1]
                persistence = np.array([records[o]["before"]["reaTZon_y"]-273.15 for o in origins])-actual_temperature[origins+h-1]
                energy_error = predicted[:,:h,3].sum(axis=1)-np.array([sum(interval_energy[o:o+h]) for o in origins])
                prediction_rows.append({"scenario":s["scenario"],"controller":s["controller"],"view":view,
                    "horizon_steps":h,"origins":len(origins),"temperature_RMSE_C":float(np.sqrt(np.mean(temperature_error**2))),
                    "temperature_bias_C":float(temperature_error.mean()),"persistence_RMSE_C":float(np.sqrt(np.mean(persistence**2))),
                    "margin_coverage":float(np.mean(abs(temperature_error)<=model["margin_K"][h-1])),
                    "cumulative_electricity_RMSE_kWh":float(np.sqrt(np.mean(energy_error**2))),
                    "cumulative_electricity_bias_kWh":float(energy_error.mean())})
    schedule = schedule_at(forecast,native.time.to_numpy())
    weather_error = {key:float(np.max(abs(native[io["weather_prefix"]+key+"_y"].to_numpy()-schedule[key].to_numpy())))
                     for key in ["TDryBul","HGloHor"]}
    native_lat = float(native[io["weather_prefix"]+"Lat_y"].iloc[0])
    native_lon = float(native[io["weather_prefix"]+"Lon_y"].iloc[0])
    with ZipFile(frozen["fmus"][s["domain"]["testcase"]]["path"]) as archive:
        weather_source = pd.read_csv(archive.open("resources/weather.csv"))
    np.testing.assert_allclose(native[io["weather_prefix"]+"Lat_y"],weather_source.lat.iloc[0],rtol=0,atol=1e-10)
    np.testing.assert_allclose(native[io["weather_prefix"]+"Lon_y"],weather_source.lon.iloc[0],rtol=0,atol=1e-10)
    geometry = {}
    for api_key,source_key in [("SolZen","solZen"),("SolTim","solTim")]:
        actual = np.array([r["before"][io["weather_prefix"]+api_key+"_y"] for r in records])
        expected = np.interp(expected_times[:-1],weather_source.time,weather_source[source_key])
        geometry[api_key] = float(np.max(abs(actual-expected)))
    count = len(records); operational = max(count-config["initial_rule_steps"],0)
    audit = {"scenario":s["scenario"],"controller":s["controller"],"decisions_replayed":count,"modes":dict(modes),
        "reasons":dict(reasons),"fallback_fraction_all":modes["fallback"]/count,
        "fallback_fraction_after_initialization":modes["fallback"]/operational if operational else None,
        "mpc_fraction_all":modes["mpc"]/count,"early_coast_fraction_all":modes["early_coast"]/count,
        "weather_measurement_forecast_max_abs":weather_error,"native_lat_rad":native_lat,"native_lon_rad":native_lon,
        "solar_geometry_vs_weather_resource_max_abs":geometry,
        "native_stop_band_pass":native_stop_band,"official_kpi_max_difference":max(differences.values())}
    daily = []
    for d in range(int(s["days"])):
        a = s["day"]*86400+d*86400
        part = native[native.time.between(a,a+86400)]
        if len(part)>1:
            daily.append({"scenario":s["scenario"],"controller":s["controller"],"evaluation_day":d,
                          **integrate(part,forecast,io,s["domain"]["area_m2"])})
    return {"scenario":s["scenario"],"domain":s["domain"]["id"],"day":s["day"],"weather":s["weather"],
            "controller":s["controller"],"temperature_kind":io["temperature_kind"],"complete":s["complete"],**metrics},audit,prediction_rows,daily


def verify_grid(root,days,frozen):
    manifest = load(root/"manifest.json")
    if manifest["freeze_sha256"]!=digest(FREEZE/"freeze.json") or manifest["frozen"]!=frozen or manifest["days"]!=days:
        raise ValueError("Run does not use frozen protocol")
    summaries = load(root/"summary.json")
    config = frozen["config"]
    expected = {(f"{d['id']}_day{day}_{w}",k) for d in config["domains"] for day in config["start_days"]
                for w in config["weather_modes"] for k in config["controllers"]}
    if {(s["scenario"],s["controller"]) for s in summaries}!=expected or len(summaries)!=len(expected):
        raise ValueError("Incomplete or duplicate comparison grid")
    for s in summaries:
        domain = next(d for d in config["domains"] if d["id"]==s["domain"]["id"])
        if s["domain"]!=domain or s["days"]!=days or s["scenario"]!=f"{domain['id']}_day{s['day']}_{s['weather']}":
            raise ValueError("Episode domain or protocol metadata differs")
    return summaries


def verify_pairs(root,summaries):
    for scenario in dict.fromkeys(s["scenario"] for s in summaries):
        directories = [root/f"{scenario}__{kind}" for kind in ["guarded_rule","solar_mpc"]]
        if load(directories[0]/"prelude.json")!=load(directories[1]/"prelude.json"):
            raise ValueError("Unpaired initialization/prelude")
        pd.testing.assert_frame_equal(csv(directories[0]/"exogenous.csv.gz"),csv(directories[1]/"exogenous.csv.gz"),check_exact=True)
        records = [[json.loads(s) for s in (p/"steps.jsonl").read_text().splitlines()] for p in directories]
        for a,b in zip(records[0][:8],records[1][:8]):
            if any(a[k]!=b[k] for k in ["before","action","after"]):
                raise ValueError("Common first eight steps differ")
    for domain in {s["domain"]["id"] for s in summaries}:
        for day in {s["day"] for s in summaries}:
            reference = root/f"{domain}_day{day}_perfect__guarded_rule"
            for weather in ["warm_bias","cold_bias"]:
                other = root/f"{domain}_day{day}_{weather}__guarded_rule"
                pd.testing.assert_frame_equal(csv(reference/"native.csv.gz"),csv(other/"native.csv.gz"),check_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow",action="store_true",help="Audit completed episodes while the fixed grid is still running")
    args = parser.parse_args()
    frozen = check_freeze(); root = Path("outputs/generalization-control"); repeat = Path("outputs/generalization-repeat")
    fmus = verify_fmus(frozen)
    rows,audits,predictions,daily = [],[],[],[]
    config = frozen["config"]
    planned = [f"{d['id']}_day{day}_{w}__{k}" for d in config["domains"] for day in config["start_days"]
               for w in config["weather_modes"] for k in config["controllers"]]
    if not args.follow:
        verify_grid(root,7,frozen)
        verify_grid(repeat,.25,frozen)
    deadline = time.monotonic()+7200
    for name in planned:
        path = root/name/"summary.json"
        while args.follow and not path.exists() and time.monotonic()<deadline:
            time.sleep(5)
        s = load(path)
        row,audit,prediction,days = audit_episode(root,s,frozen)
        rows.append(row); audits.append(audit); predictions.extend(prediction); daily.extend(days)
        print("Verified",s["scenario"],s["controller"],flush=True)
    summaries = verify_grid(root,7,frozen)
    verify_pairs(root,summaries)
    frame = pd.DataFrame(rows)
    gates = []
    for scenario,group in frame.groupby("scenario",sort=False):
        rule,mpc = [group[group.controller==k].iloc[0].to_dict() for k in ["guarded_rule","solar_mpc"]]
        gate = {"scenario":scenario,"domain":rule["domain"],"day":rule["day"],"weather":rule["weather"],
                **acceptance(rule,mpc,rule["complete"] and mpc["complete"])}
        gate["native_stop_band_pass"] = all(a["native_stop_band_pass"] for a in audits if a["scenario"]==scenario)
        gate["passed"] &= gate["native_stop_band_pass"]
        gates.append(gate)
    repeat_rows = []
    while args.follow and time.monotonic()<deadline:
        if (repeat/"summary.json").exists() and len(load(repeat/"summary.json"))==len(planned):
            break
        time.sleep(5)
    repeated = verify_grid(repeat,.25,frozen)
    for s in repeated:
        name = f"{s['scenario']}__{s['controller']}"
        a,b = csv(root/name/"native.csv.gz"),csv(repeat/name/"native.csv.gz")
        pd.testing.assert_frame_equal(a[a.time<=b.time.iloc[-1]].reset_index(drop=True),b,check_exact=True)
        a,b = load(root/name/"prelude.json"),load(repeat/name/"prelude.json")
        if a!=b: raise ValueError("Repeated prelude differs")
        full = [json.loads(x) for x in (root/name/"steps.jsonl").read_text().splitlines()]
        short = [json.loads(x) for x in (repeat/name/"steps.jsonl").read_text().splitlines()]
        for x,y in zip(full,short):
            for key in ["before","action","after","actual_action"]:
                if x[key]!=y[key]: raise ValueError("Repeated control prefix differs")
            same_info({k:v for k,v in x["control"].items() if k!="decision_seconds"},y["control"])
        repeat_rows.append({"episode":name,"repeated_steps":len(short),"native_samples":s["native_samples"],"exact_match":True})
    frame.to_csv(root/"comparison.csv",index=False,lineterminator="\n")
    pd.DataFrame(predictions).to_csv(root/"prediction_metrics.csv",index=False,lineterminator="\n")
    pd.DataFrame(daily).to_csv(root/"daily_metrics.csv",index=False,lineterminator="\n")
    save_json(root/"decision_audit.json",audits)
    save_json(root/"acceptance.json",{"groups":gates,"all_passed":all(r["passed"] for r in gates)})
    save_json(root/"verification.json",{"frozen_model_config_source_pass":True,"target_training_rows":0,"fmu_checks":fmus,
        "paired_initialization_and_rule_weather_invariance":True,"formal_episodes":len(summaries),
        "complete_episodes":sum(s["complete"] for s in summaries),"formal_transitions":sum(s["transitions"] for s in summaries),
        "native_samples":sum(s["native_samples"] for s in summaries),"decisions_replayed":sum(a["decisions_replayed"] for a in audits),
        "official_kpi_max_difference":max(a["official_kpi_max_difference"] for a in audits),
        "repeat_prefixes":repeat_rows,"full_seven_day_repeat_verified":False})
    check_freeze()


if __name__=="__main__":
    main()
