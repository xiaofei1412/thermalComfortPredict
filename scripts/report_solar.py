"""Archive warm-season control comparisons, model ablations and provenance."""

import json
import platform
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort.simulation import digest,save_json


if __name__=="__main__":
    root=Path("outputs/solar-control"); report=Path("reports/solar_mpc")
    report.mkdir(parents=True,exist_ok=True)
    verification=json.loads((root/"verification.json").read_text())
    if not verification["paired_initialization_true_weather_and_reference_invariance"]:
        raise ValueError("Verify before reporting")
    for source,prefix,files in [
        (root,"",["manifest.json","summary.json","verification.json","comparison.csv","daily_comparison.csv","acceptance.json","prediction_metrics.csv","decision_audit.csv"]),
        (Path("outputs/solar-selected"),"model_",["manifest.json","selection.json","metrics.csv","split_manifest.json"]),
        (Path("outputs/solar-refit"),"refit_",["manifest.json"]),
        (Path("outputs/solar-response"),"response_",["manifest.json","summary.json","response_gate.json","response_metrics.csv"]),
        (Path("outputs/solar-data"),"data_",["manifest.json","summary.json"]),
        (Path("outputs/solar-validation-v2"),"validation_",["manifest.json","summary.json","comparison.csv","daily_comparison.csv","acceptance.json"]),
        (Path("outputs/solar-validation"),"validation_v1_",["manifest.json","summary.json"])]:
        for name in files:
            target=report/(prefix+name)
            if name.endswith(".json"):
                target.write_text((source/name).read_text(encoding="utf-8"),encoding="utf-8",newline="\n")
            else:
                shutil.copyfile(source/name,target)
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout="constrained")
    for row,day in enumerate([238,273,315]):
        for kind,color in [("guarded_rule","tab:blue"),("minimum_heat","tab:green"),("solar_mpc","tab:orange")]:
            directory=root/f"day{day}_perfect__{kind}"
            f=pd.read_csv(directory/"transitions.csv.gz")
            time=(f.time_end_s-day*86400)/86400
            axes[row,0].plot(time,f.operative_end_C,label=kind,color=color,linewidth=.8)
            axes[row,1].plot(time,f.interval_energy_kWh.cumsum(),label=kind,color=color)
            if kind=="guarded_rule":
                axes[row,0].step(time,f.lower_start_C,where="pre",linestyle="--",color="gray",linewidth=.5,label="comfort limits")
                axes[row,0].step(time,f.upper_start_C,where="pre",linestyle="--",color="gray",linewidth=.5)
        for ax in axes[row]:
            ax.set_xlabel("Elapsed evaluation days"); ax.set_title(f"Day {day}: perfect forecast"); ax.grid(alpha=.2)
        axes[row,0].set_ylabel("Operative temperature (°C)"); axes[row,1].set_ylabel("Cumulative electricity (kWh)")
    axes[0,0].legend(fontsize=7,ncol=2); axes[0,1].legend(fontsize=8)
    fig.savefig(report/"closed_loop.png",dpi=150); plt.close(fig)
    gates=pd.DataFrame(json.loads((root/"acceptance.json").read_text())["scenarios"])
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    x=np.arange(len(gates))
    axes[0].bar(x,gates.energy_saved_percent,color=np.where(gates.energy_saved_percent>=0,"tab:green","tab:red"))
    axes[0].set_ylabel("Electricity saved vs recovery rule (%)")
    axes[1].bar(x,gates.discomfort_delta_Kh,color=np.where(gates.discomfort_delta_Kh>1e-6,"tab:red","tab:blue"))
    axes[1].set_ylabel("Temperature violation difference (K·h)")
    for ax in axes:
        ax.axhline(0,color="black",linewidth=.6); ax.grid(axis="y",alpha=.2)
        ax.set_xticks(x,gates.scenario,rotation=55,ha="right",fontsize=7)
    fig.savefig(report/"weather_stress.png",dpi=150); plt.close(fig)
    parameters=[]
    for name in ["expanded_baseline","expanded_mhe","stored_solar_mhe"]:
        m=joblib.load(Path("outputs/solar-selected")/(name+".joblib")); p=m["parameters"]
        parameters.append({"model":name,"Rza_K_kW":p[0],"Rzm_K_kW":p[1],"Cz_kWh_K":p[2],"Cm_kWh_K":p[3],
            "solar_fast_m2":p[4],"solar_stored_m2":p[5] if len(p)==7 else 0.,"person_kW":p[-1],
            "residual_weight":m["residual_weight"],"observer":m.get("observer_kind","window")})
    pd.DataFrame(parameters).to_csv(report/"model_parameters.csv",index=False,lineterminator="\n")
    files=["thermal_comfort/solar.py","configs/solar_mpc.json","tests/test_solar.py","scripts/verify_solar.py","scripts/report_solar.py",
           "thermal_comfort/greybox.py","thermal_comfort/recovery.py","thermal_comfort/mpc.py","thermal_comfort/simulation.py"]
    save_json(report/"provenance.json",{"runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "code_sha256":{n:digest(Path(n)) for n in files},
        "scope":"Same annual FMU; retired old test119 for training; new chronological control tests; no real-building control",
        "text_artifacts":"UTF-8 LF report copies; embedded source manifests refer to original outputs",
        "artifacts_sha256":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ["REPORT.md","provenance.json"]}})
