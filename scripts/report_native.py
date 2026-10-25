"""Compact, reproducible evidence for the failed native-time development gate."""

import json
from pathlib import Path
import platform
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import native as n, native_model as nm, terminal as t
from thermal_comfort.simulation import digest


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def main():
    t.check_final();report=Path("reports/native_terminal");report.mkdir(parents=True,exist_ok=True)
    audit=t.load(n.MODEL/"audit.json");assert audit["integrity_passed"] and not audit["capability_passed"]
    assert audit["code_sha256"]==n.hashes() and not n.CALIBRATION.exists() and not n.LOCK.exists()
    raw={}
    for root,names,prefix in [(n.MODEL,["manifest.json","native_one_step.json","native_joint.json","development_assessment.json","audit.json"],""),
        (n.RULE,["manifest.json","summary.json"],"rule_development_"),
        (Path("outputs/native-rule-repeat"),["manifest.json","summary.json"],"rule_repeat_"),
        (t.DEVELOPMENT,["manifest.json","summary.json"],"source_development_"),
        (Path("outputs/terminal-response"),["manifest.json","summary.json"],"source_response_")]:
        for name in names:
            raw[str(root/name)]=digest(root/name);write(report/(prefix+name),t.load(root/name))
    metrics=pd.DataFrame(t.load(n.MODEL/"development_metrics.json"));gates=pd.DataFrame(t.load(n.MODEL/"capability_gates.json"))
    season={p["day"]:p["season"] for p in n.configuration()["development"]}
    for data,name in [(metrics,"prediction_metrics.csv"),(gates,"capability_gates.csv")]:
        data["season"]=data.day.map(season);data.to_csv(report/name,index=False,lineterminator="\n")
    raw[str(n.MODEL/"development_metrics.json")]=digest(n.MODEL/"development_metrics.json")
    raw[str(n.MODEL/"capability_gates.json")]=digest(n.MODEL/"capability_gates.json")
    gates["group"]=gates.target.str.removesuffix("_energy")
    matrix=gates.groupby(["model","season","group"],sort=False).passed.agg(["sum","count"]).reset_index()
    matrix.to_csv(report/"capability_matrix.csv",index=False,lineterminator="\n")
    labels=["R_out_air_K_per_kW","R_air_mass_K_per_kW","C_air_kWh_per_K","C_mass_kWh_per_K",
        "solar_air_m2","solar_mass_m2","internal_gain_multiplier","PI_integral_seconds",
        "antiwindup_ratio","fan_time_constant_seconds","heating_power_gain","cooling_power_gain",
        "fan_cubic_kW","fan_linear_kW"]
    write(report/"model_parameters.json",{name:{"parameters":dict(zip(labels,joblib.load(n.MODEL/(name+".joblib"))["parameters"].tolist())),
        "model_sha256":digest(n.MODEL/(name+".joblib")),"diagnostics":joblib.load(n.MODEL/(name+".joblib"))["diagnostics"]}
        for name in ["native_one_step","native_joint"]})
    lines=["# 规则轨迹的6小时预测开发结果","","这些是既有开发日期的重跑，不是新的时间留出。每日期63个重叠起点；能源单位kWh，温度单位°C。","",
        "| 日期 | 模型 | 温度RMSE | 燃气RMSE | 制冷电RMSE | 风机电RMSE |","| --- | --- | ---: | ---: | ---: | ---: |"]
    for day in season:
        for name in ["frozen_static","native_one_step","native_joint"]:
            f=metrics[(metrics.day==day)&(metrics.model==name)&(metrics.seconds==21600)&metrics.trajectory.str.endswith("guarded_rule")]
            values=[float(f[f.target==target].RMSE.iloc[0]) for target in ["temperature",*[c+"_energy" for c in n.CARRIERS]]]
            lines.append(f"| {day}/{season[day]} | {name} | "+" | ".join(f"{v:.6f}" for v in values)+" |")
    (report/"results_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    # Plot the same development-rule forecast origin for all candidates, without future-state insertion.
    fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
    trace=[]
    for col,day in enumerate(season):
        root=n.RULE/f"day{day}__guarded_rule";e=n.read_native(root,t.load(root/"summary.json"),"development")
        start=960;seconds=np.arange(30,21601,30);hours=np.r_[0,seconds/3600]
        y=nm.truth(e,np.array([start]),seconds)[0];initial=np.r_[e["temperature"][start],e["power"][start]]
        observed=np.vstack([initial,y[:,:4]])
        predictions={name:nm.prediction(joblib.load(n.MODEL/(name+".joblib"))["parameters"],e,np.array([start]),seconds)[0]
            for name in ["native_one_step","native_joint"]}
        for row,(label,unit) in enumerate(zip(["Air temperature","Gas power","Cooling electric power","Fan electric power"],["C","kW","kW","kW"])):
            ax=axes[row,col];ax.plot(hours,observed[:,row],label="BOPTEST rule",color="black",lw=1.3)
            for name,p in predictions.items():ax.plot(hours,np.r_[initial[row],p[:,row]],label=name,lw=1,alpha=.85)
            ax.set_ylabel(label+" ("+unit+")");ax.grid(alpha=.25)
            if row==0:ax.set_title(f"Day {day}: {season[day]}, origin 08:00")
            if row==3:ax.set_xlabel("Hours after origin")
        for j,h in enumerate(seconds):
            trace.append({"day":day,"seconds":int(h),**{f"observed_{label}":float(y[j,k]) for k,label in enumerate(["air_C","gas_kW","cooling_kW","fan_kW"])},
                **{f"{name}_{label}":float(p[j,k]) for name,p in predictions.items() for k,label in enumerate(["air_C","gas_kW","cooling_kW","fan_kW"])}})
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(report/"native_forecast.png",dpi=150);plt.close(fig)
    pd.DataFrame(trace).to_csv(report/"forecast_example.csv",index=False,lineterminator="\n")
    # A compact data ledger keeps repeated and reused simulation effort distinct.
    write(report/"data_ledger.json",{"training":{"reused_development_days":12,"native_intervals":34560,"native_samples":34564},
        "excitation_validation":{"reused_development_days":6,"native_intervals":17280,"native_samples":17282},
        "rule_validation":{"fresh_runs_at_existing_development_dates_days":6,"decision_intervals":576,"native_intervals":17280,"native_samples":17282},
        "response":{"reused_paired_development_days":2,"decision_intervals":192,"native_intervals":5760,"native_samples":5764},
        "rule_repeat":{"additional_days":.5,"decision_intervals":48,"native_samples":1442,"complete_three_days_repeated":False},
        "new_calibration_days":0,"new_scenario_test_days":0,"selected_deployable_candidate":None,"primary_control":"guarded_rule"})
    write(report/"environment.json",{"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,
        "scipy":scipy.__version__,"numba":numba.__version__,"boptest":"0.9.0","upstream_commit":"9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9"})
    write(report/"development_failures.json",{"initial_rule_initialization":"RecoveryController(None) rejected during reset; switched to unchanged frozen SolarController rule implementation before evaluation",
        "source_check":"Collection snapshot hashes verified against archived collection source; frozen current protocol added response fields after original collection, so compare collection-affecting fields and original role plan",
        "native_flow":"Negative solver mass-flow tails down to -4.96645e-8 kg/s; normalized-flow tolerance 1e-6, original files preserved, energy thresholds unchanged",
        "candidate_failures":"Neither predeclared candidate passed seasonal/carrier qualification; no loss, parameter-bound or threshold adjustment. Final data-role/coverage guards were followed by byte-identical fitting and independent refitting",
        "local_evidence":["outputs/native-rule-development-initialization-failed","outputs/native-rule-initialization-failed.log","outputs/native-fit-source-check-failed.log"]})
    # Do not label a failed development result as a deployment freeze.
    write(report/"development_result.json",{"integrity_passed":True,"capability_passed":False,"eligible_candidates":[],
        "primary_control":"guarded_rule","calibration_generated":False,"unused_scenario_evaluated":False,
        "assessment_sha256":digest(n.MODEL/"development_assessment.json"),"audit_sha256":digest(n.MODEL/"audit.json")})
    source=[*n.CORE,"scripts/verify_native.py","scripts/report_native.py","tests/test_native.py"]
    write(report/"provenance.json",{"raw_output_sha256":raw,"source_sha256":{p:digest(Path(p)) for p in source},
        "archived_sha256":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ["REPORT.md","provenance.json"]},
        "note":"JSON archives normalized to UTF-8 LF; raw output checksums remain separate. Development evidence is not a deployment freeze."})


if __name__=="__main__":main()
