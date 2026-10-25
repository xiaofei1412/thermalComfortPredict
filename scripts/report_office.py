"""Archive compact office calibration evidence after the frozen experiment audit."""

import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import office as o
from thermal_comfort.generalization import FREEZE
from thermal_comfort.mpc import limited_command
from thermal_comfort.recovery import RecoveryController, weather_view
from thermal_comfort.simulation import digest, schedule_at


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def main():
    frozen=o.check_office_freeze();root=Path("outputs/office-control");report=Path("reports/office_calibration")
    verified=load(root/"verification.json")
    assert verified["formal_episodes"]==27 and len(verified["repeat_prefixes"])==27
    assert all(r["steps"]==24 and r["exact_match"] for r in verified["repeat_prefixes"])
    report.mkdir(parents=True,exist_ok=True)
    source_artifacts={}
    for source,names,prefix in [
        (Path("outputs/office-freeze"),["freeze.json"],""),
        (o.DATA,["manifest.json","summary.json"],"data_"),
        (o.MODEL,["manifest.json","selection.json","split_manifest.json"],"fit_"),
        (Path("outputs/office-validation"),["manifest.json","summary.json","development_verification.json","prediction_metrics.csv"],"development_"),
        (Path("outputs/office-response"),["manifest.json","summary.json","gate.json","gate-monotonicity-failed.json"],"response_"),
        (Path("outputs/office-response-aborted-low18"),["summary.json"],"aborted_low18_"),
        (root,["manifest.json","summary.json","comparison.csv","prediction_metrics.csv","daily_metrics.csv","acceptance.json","decision_audit.json","verification.json"],""),
        (Path("outputs/office-repeat"),["manifest.json","summary.json"],"repeat_")]:
        for name in names:
            source_artifacts[str(source/name)]=digest(source/name)
            if name.endswith(".json"):write(report/(prefix+name),load(source/name))
            else:shutil.copyfile(source/name,report/(prefix+name))
    iterations=[]
    for name in ["office-model-v1","office-model-multistart","office-model-before-radiant"]:
        p=Path("outputs")/name
        code=list(p.glob("source_*.py"))
        iterations.append({"local_directory":str(p),"selected":load(p/"selection.json")["selected"],
            "model_sha256":digest(p/"model.joblib"),"source_snapshots_sha256":{v.name:digest(v) for v in code}})
    write(report/"development_iterations.json",{"controlled_ablation":False,
        "note":"Development iterations changed input/physics assumptions; only final 3d vs 9d is a matched-data-budget comparison",
        "iterations":iterations})
    frame=pd.read_csv(root/"comparison.csv"); predictions=pd.read_csv(root/"prediction_metrics.csv")
    audits=load(root/"decision_audit.json");gates=load(root/"acceptance.json")
    table=["# 全新办公室时间留出对照","","均为672个15分钟转移；电/燃气单位kWh，空气温度越界单位K·h。三列依次为规则 / 原冻结MPC / 校准MPC。门槛含冷热分项、时长、最大偏差、电力、燃气与总终端能量。","",
        "| day/预报 | 电量 | 燃气 | 温度越界 | 校准对规则节电 | 对规则/原冻结全部门槛 |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for day in frozen["config"]["test_days"]:
        for weather in frozen["config"]["weather_modes"]:
            group=frame[(frame.day==day)&(frame.weather==weather)]
            rows=[group[group.controller==k].iloc[0] for k in frozen["config"]["controllers"]]
            pair=[next(g for g in gates if g["day"]==day and g["weather"]==weather and g["baseline"]==k) for k in ["guarded_rule","solar_mpc"]]
            values=[" / ".join(f"{r[k]:.6f}" for r in rows) for k in ["electricity_kWh","gas_kWh","discomfort_Kh"]]
            table.append(f"| {day}/{weather} | {' | '.join(values)} | {pair[0]['energy_saved_percent']:.6f}% | "+" / ".join("通过" if p["passed"] else "未通过" for p in pair)+" |")
    (report/"results_table.md").write_text("\n".join(table)+"\n",encoding="utf-8",newline="\n")
    perfect=predictions[(predictions.controller=="guarded_rule")&(predictions.weather=="perfect")&(predictions["view"]=="perfect")&(predictions.horizon_steps==24)]
    perfect.to_csv(report/"rule_perfect_6h.csv",index=False,lineterminator="\n")
    fallback=[]
    for a in audits:
        if a["episode"].endswith("__guarded_rule"):continue
        fallback.append({k:a[k] for k in ["episode","fallback_after_initialization","reference_selected_after_initialization","mpc_after_initialization"]})
    pd.DataFrame(fallback).to_csv(report/"fallback.csv",index=False,lineterminator="\n")
    constraints=[]
    import joblib
    margin=joblib.load(o.MODEL/"model.joblib")["margin_K"][:24]
    for s in load(root/"summary.json"):
        if s["controller"]!="office_mpc":continue
        p=root/f"day{s['day']}_{s['weather']}__office_mpc"
        records=[json.loads(v) for v in (p/"steps.jsonl").read_text().splitlines()]
        forecast=pd.read_csv(p/"exogenous.csv.gz",dtype=float,float_precision="round_trip")
        empty=[];empty_rejected=[];same_applied=[];logged_mpc_same=[]
        pc=load(FREEZE/"freeze.json")["policy_config"]
        shadow=RecoveryController(None,pc,"guarded_rule")
        shadow.previous=records[0]["before"]["oveTSet_u"]-273.15;shadow.recovering=False
        for index,r in enumerate(records):
            shadow.view=weather_view(forecast,r["before"]["time"],pc,s["weather"])
            target,_=shadow.guarded_rule(r["before"],schedule_at(shadow.view,[r["before"]["time"]]).iloc[0])
            rule_action=limited_command(target,shadow.previous,pc)
            shadow.previous=r["action"]["oveTSet_u"]-273.15
            if index<8:continue
            same=abs(shadow.previous-rule_action)<=1e-10
            same_applied.append(same);logged_mpc_same.append(same and r["control"]["mode"]=="mpc")
            schedule=schedule_at(forecast,r["before"]["time"]+np.arange(1,25)*900)
            impossible=bool((schedule["UpperSetp[1]"].to_numpy()-schedule["LowerSetp[1]"].to_numpy()<2*margin).any())
            empty.append(impossible)
            empty_rejected.append(impossible and r["control"]["reason"]=="no_margin_feasible_plan")
        constraints.append({"day":s["day"],"weather":s["weather"],"operational_steps":len(empty),
            "tightened_band_empty_steps":sum(empty),"empty_and_no_feasible_reason_steps":sum(empty_rejected),
            "tightened_band_empty_fraction":float(np.mean(empty)),
            "same_as_limited_rule_action_fraction":float(np.mean(same_applied)),
            "logged_mpc_but_same_applied_rule_action_steps":sum(logged_mpc_same)})
    pd.DataFrame(constraints).to_csv(report/"constraint_diagnostics.csv",index=False,lineterminator="\n")
    models=["frozen","calibrated_3d","calibrated_9d"];colors=["#e49d28","#1976b8","#479c75"]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    x=np.arange(3)
    for j,name in enumerate(models):
        selected=perfect[perfect.model==name].set_index("day").loc[frozen["config"]["test_days"]]
        axes[0].bar(x+(j-1)*.25,selected.temperature_RMSE_C,width=.24,color=colors[j],label=name)
        axes[1].bar(x+(j-1)*.25,selected.margin_coverage*100,width=.24,color=colors[j])
    for ax in axes:
        ax.set_xticks(x,[f"day{d}" for d in frozen["config"]["test_days"]]);ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("6 h air-temperature RMSE (C)");axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Endpoint empirical-margin coverage (%)");axes[1].set_ylim(0,105)
    fig.savefig(report/"prediction.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for j,kind in enumerate(frozen["config"]["controllers"]):
        selected=frame[(frame.weather=="perfect")&(frame.controller==kind)].set_index("day").loc[frozen["config"]["test_days"]]
        for ax,key in zip(axes,["electricity_kWh","gas_kWh","discomfort_Kh"]):
            ax.bar(x+(j-1)*.25,selected[key],width=.24,label=kind,color=["#777777",colors[0],colors[1]][j])
    for ax,title in zip(axes,["Electricity (kWh)","Gas (kWh)","Air-temperature violation (K h)"]):
        ax.set_ylabel(title);ax.set_xticks(x,[f"day{d}" for d in frozen["config"]["test_days"]]);ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(report/"control.png",dpi=150);plt.close(fig)
    # Preserve effective parameters and learning-bound diagnostics without model binaries.
    parameters={}
    for days in [1,3]:
        m=joblib.load(o.MODEL/f"budget_{days}.joblib")
        parameters[f"{days*3}d"]={k:m[k].tolist() for k in ["parameters","A","B","terminal","fan","margin_K","support_low","support_high"]}
        parameters[f"{days*3}d"].update(residual_weight=m["residual_weight"],fit_diagnostics=m["fit_diagnostics"])
        scaler=m["residual"].named_steps["standardscaler"];ridge=m["residual"].named_steps["ridge"]
        parameters[f"{days*3}d"]["residual"]={"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),
            "coefficient":ridge.coef_.tolist(),"intercept":float(ridge.intercept_)}
    write(report/"effective_parameters.json",parameters)
    write(report/"provenance.json",{"baseline_commit":frozen["config"]["baseline_commit"],
        "scope":"Few-shot office simulation calibration; air temperature, heating/cooling and electricity/gas; no real-building validation",
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "frozen_source_model_sha256":load(FREEZE/"freeze.json")["model_sha256"],"office_model_sha256":frozen["model_sha256"],
        "boptest_upstream_revision":load(FREEZE/"freeze.json")["source_revision"],
        "office_fmu_sha256":load(FREEZE/"freeze.json")["fmus"]["bestest_air"]["sha256"],
        "physical_contract_source_sha256":{name:digest(Path("outputs/boptest-v0.9.0/testcases/bestest_air/models/BESTESTAir/BaseClasses")/name)
            for name in ["FanCoilUnit_T.mo","Thermostat_T.mo"]},
        "json_archive_encoding":"UTF-8/LF; source hashes refer to unchanged outputs, archive hashes to normalized report files",
        "source_artifacts_sha256":source_artifacts,
        "source_sha256":{n:digest(Path(n)) for n in [*o.SOURCE_FILES,"scripts/verify_office.py","scripts/report_office.py","tests/test_office.py"]},
        "report_sha256":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ["provenance.json","REPORT.md"]}})
    print(json.dumps({"verification":verified,"acceptance":gates,"rule_6h":perfect.to_dict("records")},indent=2))


if __name__=="__main__":main()
