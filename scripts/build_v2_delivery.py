"""Derive a small, read-only replay from all frozen development cases; no simulation."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_v2_formal import lines
from thermal_comfort.rl_interface import native_score
from thermal_comfort.simulation import POWER
from thermal_comfort.v2_protocol import ROOT,read,digest,verify_files


METRICS = ["energy_kWh",*[f"{kind}_{metric}" for kind in ["cold","hot"] for metric in ["Kh","duration_s","max_K"]]]


def build():
    lock = read(ROOT/"reports/v2_4_safe/lock.json")
    for name in [*lock["parent_locks_sha256"],"reports/v2_4_safe/lock.json"]:
        verify_files(ROOT,read(ROOT/name)["files_sha256"])
    old = read(ROOT/"reports/v2_safe_formal/archive-audit.json")["seeds"]
    residual = read(ROOT/"reports/v2_4_safe/archive-audit.json")["base_audit"]["seeds"]
    baselines = read(ROOT/"reports/v2_2/summary.json")["runs"]
    cases,sources = [],{}
    for episode in ["development084","development168","development189"]:
        entries = [(kind,None,ROOT/"outputs/v2-2/live-registered"/(episode+"__"+kind),
            baselines[episode+"__"+kind]["metrics"],None) for kind in ["guarded_rule","solar_mpc","minimum_heat"]]
        for family,group,directory in [(None,old,"outputs/v2-safe-formal/live-registered"),
                ("residual_SAC",residual,"outputs/v2-4-safe/live-registered")]:
            for key,value in group.items():
                algorithm,seed = key.split("-")
                entries.append((family or algorithm,int(seed),ROOT/directory/key/"development"/episode,
                    value["development"][episode]["metrics"],value["development"][episode]))
        for family,seed,folder,expected,comparison in entries:
            summary=read(folder/"summary.json")
            required=["native.csv.gz","schedule.csv.gz","steps.jsonl.gz" if seed is not None else "steps.jsonl"]
            verify_files(folder,{p:summary["sha256"][p] for p in required})
            native=pd.read_csv(folder/"native.csv.gz",float_precision="round_trip")
            schedule=pd.read_csv(folder/"schedule.csv.gz",float_precision="round_trip")
            actual=native_score(native,schedule)
            assert len(native)==20161 and all(abs(actual[k]-expected[k])<1e-7 for k in METRICS)
            records=lines(folder/required[-1])
            assert len(records)==672
            t=native.time.to_numpy()
            power=sum(native[p].to_numpy() for p in POWER)
            cold=np.maximum(schedule['LowerSetp[1]'].to_numpy()-native.reaTZon_y.to_numpy(),0)
            hot=np.maximum(native.reaTZon_y.to_numpy()-schedule['UpperSetp[1]'].to_numpy(),0)
            def cumulative(values,scale):
                return np.r_[0,np.cumsum((values[:-1]+values[1:])*np.diff(t)/2/scale)]
            energy,ckh,hkh=cumulative(power,3600000),cumulative(cold,3600),cumulative(hot,3600)
            frames=[]
            for index in range(673):
                state=records[index-1]['after'] if index else records[0]['before']
                control=records[index-1]['control'] if index else {}
                row=index*30
                frames.append([round(float(v),6) if v is not None else None for v in [
                    state['reaTZon_y']-273.15,state['oveTSet_u']-273.15,sum(state[p] for p in POWER)/1000,
                    energy[row],ckh[row],hkh[row],schedule['LowerSetp[1]'].iloc[row]-273.15,
                    schedule['UpperSetp[1]'].iloc[row]-273.15,schedule.TDryBul.iloc[row]-273.15,
                    schedule.HGloHor.iloc[row],control.get('raw_actor_normalized'),control.get('rule_anchor_C')]])
            for k,last in [("energy_kWh",energy[-1]),("cold_Kh",ckh[-1]),("hot_Kh",hkh[-1])]:
                assert abs(last-expected[k])<1e-7
            name=f"{episode}__{family}"+(f"__{seed}" if seed is not None else "")
            sources[name]={"folder":str(folder.relative_to(ROOT)).replace('\\','/'),
                "summary_sha256":digest(folder/"summary.json"),"files_sha256":{p:summary['sha256'][p] for p in required}}
            cases.append({"id":name,"episode":episode,"start_day":summary['episode']['start_day'],"family":family,"seed":seed,
                "metrics":{k:expected[k] for k in METRICS},"comparison":comparison,
                "fallback_fraction":summary['fallback_fraction_excluding_prefix'],"frames":frames})
    assert len(cases)==45
    bundle={"schema":1,"mode":"ARCHIVED_DEVELOPMENT_REPLAY_ONLY","steps":672,"step_s":900,"prefix_steps":8,
        "frame_columns":["operative_C","setpoint_C","electric_kW","cumulative_energy_kWh","cumulative_cold_Kh",
            "cumulative_hot_Kh","lower_C","upper_C","outdoor_C","solar_W_m2","library_action","rule_anchor_C"],
        "frame_precision":"display values rounded to 6 decimals; decisions use full precision 30-second archived metrics",
        "heldout_initializations":0,"cases":cases}
    output=ROOT/"examples/v2"
    output.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
    path=output/"replay.json.gz"
    path.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
    manifest={"schema":1,"cases":45,"frames":45*673,"data_sha256":digest(path),
        "uncompressed_sha256":hashlib.sha256(data).hexdigest(),"bytes":path.stat().st_size,
        "sources":sources,"source_lock":"reports/v2_4_safe/lock.json","source_lock_sha256":digest(ROOT/"reports/v2_4_safe/lock.json"),
        "native_points_recomputed":45*20161,"new_model_experiments":0,"FMU_calls":0,
        "scope":"all three development dates, all methods/seeds; display derivation, not a new control evaluation"}
    (output/"manifest.json").write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},ensure_ascii=False))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
