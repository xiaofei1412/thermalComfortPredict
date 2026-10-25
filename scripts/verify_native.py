"""Independent units, replay, data-role, refit and gate audit for native development."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import native as n, native_model as nm, terminal as t
from thermal_comfort.generalization import integrate
from thermal_comfort.simulation import digest, save_json, schedule_at
from thermal_comfort.solar import SolarController
from scripts.verify_generalization import same_info


REPEAT=Path("outputs/native-rule-repeat")


def frame(path):return pd.read_csv(path,dtype=float,float_precision="round_trip")


def repeat():
    t.check_final();REPEAT.mkdir(parents=True,exist_ok=False);summaries=[]
    save_json(REPEAT/"manifest.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),"role":"development_repeat",
        "code_sha256":n.hashes(),"days":.25,"repeated_full_days":False})
    for plan in n.configuration()["development"]:
        day=plan["day"];policy=SolarController(joblib.load(t.FREEZE/"model.joblib"),t.policy_config(),"guarded_rule")
        summaries.append(t.run_episode(n.configuration(),"bestest_air",day,.25,"guarded_rule","perfect",policy,REPEAT/f"day{day}__guarded_rule"))
        save_json(REPEAT/"summary.json",summaries)


def verify():
    t.check_final();config=n.configuration();n.validate_roles(config)
    assert not n.CALIBRATION.exists() and not n.LOCK.exists()
    manifest=t.load(n.MODEL/"manifest.json");assessment=t.load(n.MODEL/"development_assessment.json")
    assert t.load(n.RULE/"manifest.json")["config"]==config
    assert assessment["code_sha256"]==manifest["code_sha256"]==n.hashes()
    assert manifest["training_days"]==config["training_days"]
    assert manifest["source_summary_sha256"]==digest(t.DEVELOPMENT/"summary.json")
    for name,h in manifest["artifacts_sha256"].items():
        assert digest(n.MODEL/name)==h==digest(Path("outputs/native-refit")/name)
    model_hashes={name:digest(n.MODEL/(name+".joblib")) for name in ["native_one_step","native_joint"]}
    initial=Path("outputs/native-model-initial-assessment")
    if initial.exists():
        assert all(digest(initial/(name+".joblib"))==h for name,h in model_hashes.items())
    sources=[];maximum_unit_error=0.;causal_checks=0
    development=n.source_episodes("development")
    for s in t.load(n.RULE/"summary.json"):
        development.append(n.read_native(n.RULE/f"day{s['day']}__guarded_rule",s,"development"))
    for e in n.source_episodes("train")+development:
        starts=nm.origin_indices(e,3600);y=nm.truth(e,starts,[3600,21600]);raw=frame(e["root"]/"native.csv.gz")
        for j,c in enumerate(["fcu_reaPHea_y","fcu_reaPCoo_y","fcu_reaPFan_y"]):
            observed=np.array([np.trapezoid(raw[c].iloc[k:k+721],raw.time.iloc[k:k+721])/3600000 for k in starts])
            error=float(np.max(abs(observed-y[:,-1,4+j])));maximum_unit_error=max(maximum_unit_error,error)
            assert error<1e-10
        for name in model_hashes:
            parameters=joblib.load(n.MODEL/(name+".joblib"))["parameters"]
            k=np.array([480]);before=nm.prediction(parameters,e,k,config["horizons_seconds"])
            changed={**e,"temperature":e["temperature"].copy(),"power":e["power"].copy(),"fan":e["fan"].copy()}
            changed["temperature"][481:]+=50;changed["power"][481:]+=10;changed["fan"][481:]=1
            np.testing.assert_array_equal(before,nm.prediction(parameters,changed,k,config["horizons_seconds"]));causal_checks+=1
        sources.append({"id":e["id"],"role":e["role"],"day":e["day"],"native_intervals":len(e["actions"]),
            "native_samples":len(e["temperature"]),"summary_sha256":digest(e["root"]/"summary.json"),
            "native_sha256":digest(e["root"]/"native.csv.gz"),
            "minimum_raw_mass_flow_kg_s":float(raw.fcu_reaFloSup_y.min()),"maximum_raw_mass_flow_kg_s":float(raw.fcu_reaFloSup_y.max())})
    # Recompute metrics and gates, then require exact agreement; no refitting or threshold updates.
    saved={name:digest(n.MODEL/name) for name in ["development_metrics.json","capability_gates.json"]}
    n.evaluate()
    assert all(digest(n.MODEL/name)==h for name,h in saved.items())
    current=t.load(n.MODEL/"development_assessment.json")
    for key in ["summary","activity","response"]:assert current[key]==assessment[key]
    assert not any(row["all_capability_gates_passed"] for row in current["summary"].values())
    try:n.lock_development()
    except ValueError as error:
        assert "capability gate failed" in str(error);lock_rejection=str(error)
    else:raise AssertionError("A failing model reached calibration")
    replay=[];max_kpi_error=0.
    for summary in t.load(n.RULE/"summary.json"):
        day=summary["day"];root=n.RULE/f"day{day}__guarded_rule"
        n.read_native(root,summary,"development");native=frame(root/"native.csv.gz");forecast=frame(root/"exogenous.csv.gz")
        records=[json.loads(line) for line in (root/"steps.jsonl").read_text().splitlines()]
        policy=SolarController(joblib.load(t.FREEZE/"model.joblib"),t.policy_config(),"guarded_rule")
        policy.reset(records[0]["before"],{"oveTSet_u":t.load(root/"api_metadata.json")["inputs"][t.IO["setpoint"]]})
        for index,r in enumerate(records):
            action,info=policy.decide(index,r["before"],forecast)
            assert action==r["action"];same_info(info,r["control"])
            assert r["actual_action"]=={t.IO["setpoint"]:action["oveTSet_u"],t.IO["active"]:1}
            segment=native[native.time.between(r["before"]["time"],r["after"]["time"])].copy()
            assert len(segment)==31
            np.testing.assert_allclose(segment[t.IO["setpoint"]].iloc[1:],action["oveTSet_u"],atol=1e-8,rtol=0)
            assert abs(action["oveTSet_u"]-r["before"]["oveTSet_u"])<=.5+1e-9
            segment["reaQFloHea_y"]=segment.fcu_reaPHea_y*.9
            policy.update(r["before"],r["after"],segment,schedule_at(forecast,[r["before"]["time"]]).iloc[0])
        assert (native[["con_oveTSetCoo_activate","fcu_oveFan_activate","fcu_oveTSup_activate"]]==0).all().all()
        official=t.load(root/"api_kpis.json");whole=integrate(frame(root/"including_prelude.csv.gz"),forecast,t.IO,48)
        delta=max(abs(whole[m]-official[k]) for m,k in [("official_energy_kWh_m2","ener_tot"),("discomfort_Kh","tdis_tot")]);assert delta<1e-6
        max_kpi_error=max(max_kpi_error,delta)
        prelude=t.load(root/"prelude.json");assert len(prelude)==96
        assert all(p["action"]=={t.IO["setpoint"]:295.15,t.IO["active"]:1} for p in prelude)
        repeated=REPEAT/root.name;short=t.load(repeated/"summary.json");assert short["complete"] and short["transitions"]==24
        # Validate repeat hashes, then compare the full prefix including native power and weather.
        n.read_native(repeated,short,"development")
        pd.testing.assert_frame_equal(frame(repeated/"native.csv.gz"),native.iloc[:721].reset_index(drop=True),check_exact=True)
        assert t.load(repeated/"prelude.json")==prelude
        repeated_records=[json.loads(line) for line in (repeated/"steps.jsonl").read_text().splitlines()]
        for r,again in zip(records,repeated_records):
            for k in ["before","after","action","actual_action"]:assert r[k]==again[k]
        replay.append({"day":day,"decisions_replayed":len(records),"native_samples":len(native),
            "prefix_steps_repeated":24,"prefix_native_samples":721,"complete_three_days_repeated":False,"KPI_max_error":delta})
    result={"created_UTC":datetime.now(timezone.utc).isoformat(),"integrity_passed":True,"capability_passed":False,
        "code_sha256":n.hashes(),"model_sha256":model_hashes,"independent_refit_byte_identical":True,
        "source_episodes":sources,"causal_future_label_mutations":causal_checks,"native_unit_max_error_kWh":maximum_unit_error,
        "rule_decision_audits":replay,"native_API_KPI_max_error":max_kpi_error,
        "calibration_not_generated":not n.CALIBRATION.exists(),"lock_not_created":not n.LOCK.exists(),
        "new_scenario_not_run":True,"lock_rejection":lock_rejection}
    save_json(n.MODEL/"audit.json",result);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["repeat","audit"])
    if parser.parse_args().mode=="repeat":repeat()
    else:verify()
