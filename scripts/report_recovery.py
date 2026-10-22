"""Archive compact recovery, idle-power and weather-error evidence and figures."""

import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort.simulation import digest, save_json
from verify_recovery import comfort_breakdown


if __name__=="__main__":
    root=Path("outputs/recovery-control")
    report=Path("reports/recovery_mpc")
    report.mkdir(parents=True,exist_ok=True)
    verification=json.loads((root/"verification.json").read_text())
    if not verification["paired_action_response_gate_rebuilt"]:
        raise ValueError("Verification is required before archiving")
    for source,prefix,files in [
        (root,"",["manifest.json","summary.json","verification.json","comparison.csv","daily_comparison.csv","acceptance.json","decision_audit.csv"]),
        (Path("outputs/recovery-model"),"model_",["manifest.json","selection.json","split_manifest.json","metrics.csv","power_metrics.csv","coverage.csv","weather_prediction_metrics.csv"]),
        (Path("outputs/recovery-response"),"response_",["manifest.json","summary.json","response_gate.json","response_metrics.csv"]),
        (Path("outputs/recovery-calibration"),"calibration_",["manifest.json","summary.json"]),
        (Path("outputs/recovery-validation"),"validation_",["manifest.json","summary.json"])]:
        for name in files:
            target=report/(prefix+name)
            if name.endswith(".json"):
                target.write_text((source/name).read_text(encoding="utf-8"),encoding="utf-8",newline="\n")
            else:
                shutil.copyfile(source/name,target)
    comparison=pd.read_csv(root/"comparison.csv")
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout="constrained")
    recovery_rows=[]
    for row,day in enumerate([98,119,140]):
        for kind,color in [("guarded_rule","tab:blue"),("mpc","tab:orange")]:
            directory=root/f"day{day}_perfect__{kind}"
            f=pd.read_csv(directory/"transitions.csv.gz")
            times=(f.time_end_s-day*86400)/86400
            axes[row,0].plot(times,f.operative_end_C,label=kind,color=color,linewidth=.8)
            axes[row,1].plot(times,f.interval_energy_kWh.cumsum(),label=kind,color=color)
            if kind=="guarded_rule":
                axes[row,0].step(times,f.lower_start_C,where="pre",color="gray",linestyle="--",linewidth=.5,label="comfort limits")
                axes[row,0].step(times,f.upper_start_C,where="pre",color="gray",linestyle="--",linewidth=.5)
        if day in [98,119]:
            for kind in ["guarded_rule","mpc"]:
                old=Path("outputs/mpc")/f"day{day}__{kind}"
                s=json.loads((old/"summary.json").read_text())
                b=comfort_breakdown(pd.read_csv(old/"native.csv.gz"),pd.read_csv(old/"exogenous.csv.gz"))
                new=comparison[(comparison.scenario==f"day{day}_perfect")&(comparison.controller==kind)].iloc[0]
                recovery_rows.append({"start_day":day,"controller":kind,"previous_energy_kWh":s["metrics"]["energy_kWh"],
                    "current_energy_kWh":new.energy_kWh,"previous_discomfort_Kh":s["metrics"]["discomfort_Kh"],
                    "current_discomfort_Kh":new.discomfort_Kh,"previous_cold_Kh":b["cold_Kh"],"current_cold_Kh":new.cold_Kh,
                    "previous_hot_Kh":b["hot_Kh"],"current_hot_Kh":new.hot_Kh})
        for col in [0,1]:
            axes[row,col].set_title(f"Day {day}: perfect weather / 7-day evaluation")
            axes[row,col].set_xlabel("Elapsed days")
            axes[row,col].grid(alpha=.2)
        axes[row,0].set_ylabel("Operative temperature (°C)")
        axes[row,1].set_ylabel("Cumulative electricity (kWh)")
    axes[0,0].legend(fontsize=7,ncol=3)
    axes[0,1].legend(fontsize=8)
    fig.savefig(report/"closed_loop.png",dpi=150)
    plt.close(fig)
    pd.DataFrame(recovery_rows).to_csv(report/"previous_stage_comparison.csv",index=False,lineterminator="\n")
    acceptance=json.loads((root/"acceptance.json").read_text())
    a=pd.DataFrame(acceptance["scenarios"])
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    x=np.arange(len(a))
    axes[0].bar(x,a.energy_saved_percent,color="tab:green")
    axes[0].set_ylabel("Electricity saved vs new rule (%)")
    axes[1].bar(x,a.discomfort_delta_Kh,color=np.where(a.discomfort_delta_Kh>1e-6,"tab:red","tab:blue"))
    axes[1].set_ylabel("Discomfort difference vs new rule (K·h)")
    for ax in axes:
        ax.axhline(0,color="black",linewidth=.7)
        ax.set_xticks(x,a.scenario,rotation=55,ha="right",fontsize=7)
        ax.grid(axis="y",alpha=.2)
    fig.savefig(report/"weather_stress.png",dpi=150)
    plt.close(fig)
    code_files=["thermal_comfort/recovery.py","thermal_comfort/greybox.py","thermal_comfort/mpc.py","thermal_comfort/simulation.py",
                "scripts/verify_recovery.py","scripts/report_recovery.py","scripts/verify_mpc.py","configs/recovery_mpc.json","configs/mpc.json"]
    save_json(report/"provenance.json",{"runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "code_sha256":{n:digest(Path(n)) for n in code_files},
        "scope":"Single fixed FMU, synthetic weather forecast biases, no real-building closed-loop evidence",
        "text_artifacts":"Report JSON uses UTF-8 LF; embedded source manifests refer to original outputs, not renamed report copies",
        "artifacts_sha256":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ["provenance.json","REPORT.md"]}})
