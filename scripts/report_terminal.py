"""Compact evidence for seasonal terminal models and strictly held-out calibration."""

import json
from pathlib import Path
import platform
import shutil
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import terminal as t, office as o
from thermal_comfort.generalization import FREEZE
from thermal_comfort.mpc import limited_command
from thermal_comfort.recovery import RecoveryController, weather_view
from thermal_comfort.simulation import digest, schedule_at
from scripts.verify_terminal import directory, records


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def main():
    frozen=t.check_final();root=Path("outputs/terminal-control");report=Path("reports/terminal_dynamics");report.mkdir(parents=True,exist_ok=True)
    verification=t.load(root/"verification.json");assert verification["planned_episodes"]==27 and len(verification["prefixes"])==27
    raw={}
    for source,names,prefix in [
        (t.DEVELOPMENT,["manifest.json","summary.json"],"development_data_"),
        (t.MODEL,["manifest.json","selection.json","development_margins.json"],"fit_"),
        (t.LOCK,["development_lock.json"],""),
        (t.CALIBRATION,["manifest.json","summary.json","calibration_manifest.json","margins.json","audit.json","prediction_metrics.csv"],"calibration_"),
        (t.FINAL,["freeze.json"],""),
        (Path("outputs/terminal-validation"),["manifest.json","summary.json","development_audit.json","prediction_metrics.csv"],"control_development_"),
        (Path("outputs/terminal-response"),["manifest.json","summary.json"],"response_"),
        (Path("outputs/terminal-validation-reset-failed"),["summary.json"],"reset_failed_"),
        (Path("outputs/terminal-scenario"),["scenario.json"],""),
        (root,["manifest.json","summary.json","comparison.csv","daily_metrics.csv","prediction_metrics.csv","acceptance.json","decision_audit.json","verification.json"],""),
        (Path("outputs/terminal-repeat"),["manifest.json","summary.json"],"repeat_")]:
        for name in names:
            raw[str(source/name)]=digest(source/name)
            if name.endswith(".json"):write(report/(prefix+name),t.load(source/name))
            else:shutil.copyfile(source/name,report/(prefix+name))
    write(report/"autoregressive_development_failure.json",{"selected":t.load(Path("outputs/terminal-model-ar-development/selection.json")),
        "source_sha256":digest(Path("outputs/terminal-model-ar-development/source_terminal_model_ar.py")),
        "note":"Unconstrained memory-feature development attempt; replaced before calibration; selected static model remained byte-identical"})
    modelmeta={}
    for name in ["static_0","static_1","dynamic_0","dynamic_1"]:
        m=joblib.load(t.MODEL/(name+".joblib"))
        d={k:m[k].tolist() for k in ["parameters","A","B","terminal","fan","support_low","support_high"]}
        d.update(family=m["family"],residual_weight=m["residual_weight"],fit_diagnostics=m["fit_diagnostics"])
        if "lag_parameters" in m:d["lag_parameters"]=[p.tolist() for p in m["lag_parameters"]]
        modelmeta[name]=d
    write(report/"model_parameters.json",modelmeta)
    c=t.configuration();margins=t.load(t.CALIBRATION/"margins.json");margin_rows=[]
    for r,label in enumerate(c["regimes"]):
        for h in c["horizons"]:
            margin_rows.append({"regime":label,"horizon_steps":h,"origins":margins["counts"][r][h-1],
                "pooled_fallback":margins["pooled_fallback"][r][h-1],"lower_K":margins["lower_K"][r][h-1],"upper_K":margins["upper_K"][r][h-1]})
    pd.DataFrame(margin_rows).to_csv(report/"margin_table.csv",index=False,lineterminator="\n")
    frame=pd.read_csv(root/"comparison.csv");gates=t.load(root/"acceptance.json");summaries=t.load(root/"summary.json")
    table=["# 新办公室气候组合全网格","","每组依次规则 / 旧办公室MPC / 新方案；能量kWh，空气温度越界K·h。若中止，量仅为中止前累计，不作整周比较。","",
        "| day/预报 | 完成步数（目标672） | 电量 | 燃气 | 越界 | 新方案对规则/旧模型门槛 |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for day in c["test_days"]:
        for weather in c["weather_modes"]:
            rows=[next(s for s in summaries if (s["day"],s["weather"],s["controller"])==(day,weather,k)) for k in c["controllers"]]
            values=[" / ".join(f"{s['metrics'][key]:.6f}" for s in rows) for key in ["electricity_kWh","gas_kWh","discomfort_Kh"]]
            pair=[next(g for g in gates if (g["day"],g["weather"],g["baseline"])==(day,weather,k)) for k in ["guarded_rule","office_prior"]]
            table.append(f"| {day}/{weather} | {' / '.join(str(s['transitions']) for s in rows)} | {' | '.join(values)} | "+" / ".join("通过" if g["passed"] else "未通过" for g in pair)+" |")
    (report/"results_table.md").write_text("\n".join(table)+"\n",encoding="utf-8",newline="\n")
    pred=pd.read_csv(root/"prediction_metrics.csv")
    p=pred[(pred.controller=="guarded_rule")&(pred.weather=="perfect")&(pred["view"]=="perfect")&(pred.horizon_steps==24)&(pred.regime=="all")]
    p.to_csv(report/"rule_perfect_6h.csv",index=False,lineterminator="\n")
    diagnostics=[];pc=t.policy_config()
    for s in summaries:
        if s["controller"]!="terminal_mpc":continue
        path=directory(root,s);rs=records(path);forecast=pd.read_csv(path/"exogenous.csv.gz",dtype=float,float_precision="round_trip")
        policy=RecoveryController(None,pc,"guarded_rule");policy.previous=rs[0]["before"]["oveTSet_u"]-273.15;policy.recovering=False
        count=empty=old_empty=changed=mislabel=0
        old_margin=joblib.load(o.MODEL/"model.joblib")["margin_K"][:24]
        for k,r in enumerate(rs):
            policy.view=weather_view(forecast,r["before"]["time"],pc,s["weather"])
            target,_=policy.guarded_rule(r["before"],schedule_at(policy.view,[r["before"]["time"]]).iloc[0]);fallback=limited_command(target,policy.previous,pc)
            policy.previous=r["action"]["oveTSet_u"]-273.15
            if k<16:continue
            count+=1;different=abs(policy.previous-fallback)>1e-10;changed+=different;mislabel+=not different and r["control"]["mode"]=="mpc"
            lo,hi=t.tm.margins_for(margins,o.current_hvac(r["before"],o.configuration()),r["before"]["time"],24,c)
            band=schedule_at(forecast,r["before"]["time"]+np.arange(1,25)*900)
            empty+=bool((band["UpperSetp[1]"].to_numpy()-band["LowerSetp[1]"].to_numpy()<lo+hi).any())
            old_empty+=bool((band["UpperSetp[1]"].to_numpy()-band["LowerSetp[1]"].to_numpy()<2*old_margin).any())
        diagnostics.append({"day":s["day"],"weather":s["weather"],"operational_steps":count,"empty_band_steps":empty,
            "empty_band_fraction":empty/count if count else None,"actual_action_differs_from_rule_fraction":changed/count if count else None,
            "same_schedule_old_symmetric_empty_fraction":old_empty/count if count else None,
            "logged_mpc_but_same_limited_rule_steps":mislabel})
    pd.DataFrame(diagnostics).to_csv(report/"constraint_diagnostics.csv",index=False,lineterminator="\n")
    x=np.arange(3);fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for j,(name,color) in enumerate([("office_prior","#e49d28"),("selected","#1976b8")]):
        part=p[p.model==name].set_index("day").loc[c["test_days"]]
        for ax,key in zip(axes,["temperature_RMSE_C","gas_RMSE_kWh","cooling_RMSE_kWh"]):
            ax.bar(x+(j-.5)*.3,part[key],width=.29,color=color,label=name)
    for ax,label in zip(axes,["6 h air-temperature RMSE (C)","6 h gas RMSE (kWh)","6 h cooling-electricity RMSE (kWh)"]):
        ax.set_ylabel(label);ax.set_xticks(x,[f"day{d}" for d in c["test_days"]]);ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(report/"prediction.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for j,(kind,color) in enumerate(zip(c["controllers"],["#777777","#e49d28","#1976b8"])):
        part=frame[(frame.weather=="perfect")&(frame.controller==kind)].set_index("day").loc[c["test_days"]]
        for ax,key in zip(axes,["electricity_kWh","gas_kWh","discomfort_Kh"]):
            values=part[key].to_numpy().copy();values[~part.complete.to_numpy()]=np.nan
            ax.bar(x+(j-1)*.25,values,width=.24,color=color,label=kind)
            for i,complete in enumerate(part.complete):
                if not complete:ax.text(i+(j-1)*.25,0,"aborted",rotation=90,ha="center",va="bottom",fontsize=7)
    for ax,label in zip(axes,["Electricity (kWh)","Gas (kWh)","Air-temperature violation (K h)"]):
        ax.set_ylabel(label);ax.set_xticks(x,[f"day{d}" for d in c["test_days"]]);ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(report/"control.png",dpi=150);plt.close(fig)
    write(report/"provenance.json",{"baseline_commit":c["baseline_commit"],"scope":"Seasonal development; untouched calibration after development lock; previously unused office/Brussels TMY combination, not a new weather year or real building",
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "model_sha256":frozen["model_sha256"],"old_office_model_sha256":digest(o.MODEL/"model.joblib"),
        "source_sha256":{n:digest(Path(n)) for n in [*t.CORE,"scripts/verify_terminal.py","scripts/report_terminal.py","tests/test_terminal.py"]},
        "original_artifacts_sha256":raw,"json_archive_encoding":"UTF-8/LF; frozen hashes refer to unchanged source outputs; archived hashes below",
        "archive_sha256":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ["REPORT.md","provenance.json"]}})
    print(json.dumps({"verification":verification,"gates":gates,"predictions":p.to_dict("records"),"constraints":diagnostics},indent=2))


if __name__=="__main__":main()
