"""Archive compact, verified transfer evidence and scientific plots."""

import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort.generalization import FREEZE, check_freeze
from thermal_comfort.simulation import digest, save_json


def main():
    frozen = check_freeze()
    root,report = Path("outputs/generalization-control"),Path("reports/generalization")
    verified = json.loads((root/"verification.json").read_text())
    if not verified["frozen_model_config_source_pass"]:
        raise ValueError("Verify before reporting")
    if len(verified["repeat_prefixes"])!=54 or any(r["repeated_steps"]!=24 or not r["exact_match"] for r in verified["repeat_prefixes"]):
        raise ValueError("Verified repeat grid is incomplete")
    report.mkdir(parents=True,exist_ok=True)
    repeated = json.loads(Path("outputs/generalization-repeat/summary.json").read_text())
    if len(repeated)!=54 or any(not s["complete"] or s["transitions"]!=24 for s in repeated):
        raise ValueError("Every independent prefix must complete all 24 intervals")
    for s in repeated:
        directory = Path("outputs/generalization-repeat")/f"{s['scenario']}__{s['controller']}"
        for name,expected in s["sha256"].items():
            if digest(directory/name)!=expected:
                raise ValueError("Independent repeat archive changed")
    save_json(report/"repeat_archive_checks.json",{"all_54_prefixes_complete":True,"source_hashes_match":True,
        "repeated_steps":sum(s["transitions"] for s in repeated),"native_samples":sum(s["native_samples"] for s in repeated)})
    for source,names,prefix in [
        (FREEZE,["freeze.json","policy_config.json"],""),
        (root,["manifest.json","summary.json","verification.json","acceptance.json","decision_audit.json",
               "comparison.csv","daily_metrics.csv","prediction_metrics.csv"],""),
        (Path("outputs/generalization-repeat"),["manifest.json","summary.json"],"repeat_")]:
        for name in names:
            if name.endswith(".json"):
                (report/(prefix+name)).write_text((source/name).read_text(encoding="utf-8"),encoding="utf-8",newline="\n")
            else:
                shutil.copyfile(source/name,report/(prefix+name))
    frame = pd.read_csv(root/"comparison.csv")
    summaries = json.loads((root/"summary.json").read_text())
    gates = pd.DataFrame(json.loads((root/"acceptance.json").read_text())["groups"])
    table = ["# 全部固定测试对照", "", "电量和燃气单位kWh，温度越界单位K·h；住宅为操作温度，办公室为空气温度。", "",
             "| 域/日期/预报 | 规则/MPC步数 | 规则/MPC电量 | 规则/MPC燃气 | 规则/MPC越界 | 节电 | 全部门槛 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for _,gate in gates.iterrows():
        paired = frame[frame.scenario==gate.scenario]
        a,b = [paired[paired.controller==k].iloc[0] for k in ["guarded_rule","solar_mpc"]]
        counts = [next(s["transitions"] for s in summaries if s["scenario"]==gate.scenario and s["controller"]==k)
                  for k in ["guarded_rule","solar_mpc"]]
        saving = f"{gate.energy_saved_percent:.2f}%" if pd.notna(gate.energy_saved_percent) else "零基线/不可比较"
        table.append(f"| {gate.domain}/{gate.day}/{gate.weather} | {counts[0]}/{counts[1]} | {a.electricity_kWh:.3f}/{b.electricity_kWh:.3f} | "
            f"{a.gas_kWh:.3f}/{b.gas_kWh:.3f} | {a.discomfort_Kh:.3f}/{b.discomfort_Kh:.3f} | {saving} | {'通过' if gate.passed else '未通过'} |")
    (report/"results_table.md").write_text("\n".join(table)+"\n",encoding="utf-8",newline="\n")
    fig,axes = plt.subplots(3,2,figsize=(12,10),layout="constrained")
    for i,domain in enumerate(frozen["config"]["domains"]):
        group = gates[gates.domain==domain["id"]]
        x = np.arange(len(group))
        saving = group.energy_saved_percent.to_numpy(dtype=float)
        colors = ["tab:green" if p else "tab:red" for p in group.passed]
        axes[i,0].bar(x,np.nan_to_num(saving),color=colors)
        for n,value in enumerate(saving):
            if not np.isfinite(value): axes[i,0].text(n,0,"zero base",rotation=90,ha="center",va="bottom",fontsize=7)
        axes[i,1].bar(x,group.discomfort_Kh_delta,color=np.where(group.discomfort_Kh_delta>1e-6,"tab:red","tab:blue"))
        labels = [f"{d}/{w.replace('_bias','')}" for d,w in zip(group.day,group.weather)]
        for ax in axes[i]:
            ax.set_xticks(x,labels,rotation=45,ha="right",fontsize=8)
            ax.axhline(0,color="black",linewidth=.6); ax.grid(axis="y",alpha=.2)
            ax.set_title(domain["id"])
        axes[i,0].set_ylabel("Electricity saved vs rule (%)")
        axes[i,1].set_ylabel("Temperature violation delta (K·h)")
    fig.savefig(report/"transfer_outcomes.png",dpi=150); plt.close(fig)
    predictions = pd.read_csv(root/"prediction_metrics.csv")
    audits = json.loads((root/"decision_audit.json").read_text())
    domain_rows = []
    for domain in frozen["config"]["domains"]:
        group = gates[gates.domain==domain["id"]]
        selected = frame[(frame.domain==domain["id"])&(frame.weather=="perfect")&(frame.controller=="guarded_rule")]
        p = predictions[predictions.scenario.isin(selected.scenario)&(predictions.controller=="guarded_rule")&
                        (predictions.view=="perfect")&(predictions.horizon_steps==24)]
        a = [r for r in audits if r["scenario"].startswith(domain["id"]+"_") and r["controller"]=="solar_mpc"]
        fractions = [r["fallback_fraction_after_initialization"] for r in a if r.get("fallback_fraction_after_initialization") is not None]
        domain_rows.append({"domain":domain["id"],"groups":len(group),"passed":int(group.passed.sum()),
            "nonzero_electricity_base_groups":int(group.energy_saved_percent.notna().sum()),
            "electricity_lower_groups":int(group.electricity_lower.sum()),"comfort_not_increased_groups":int(group.comfort_not_increased.sum()),
            "electricity_saved_percent_min":group.energy_saved_percent.min(),"electricity_saved_percent_max":group.energy_saved_percent.max(),
            "rule_perfect_6h_RMSE_min_C":p.temperature_RMSE_C.min(),"rule_perfect_6h_RMSE_max_C":p.temperature_RMSE_C.max(),
            "rule_perfect_6h_margin_coverage_min":p.margin_coverage.min(),"rule_perfect_6h_margin_coverage_max":p.margin_coverage.max(),
            "fallback_after_initialization_min":min(fractions) if fractions else None,
            "fallback_after_initialization_max":max(fractions) if fractions else None})
    pd.DataFrame(domain_rows).to_csv(report/"domain_summary.csv",index=False,lineterminator="\n")
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    colors = ["tab:blue","tab:orange","tab:green"]
    x = np.arange(9)
    labels = []
    for i,domain in enumerate(frozen["config"]["domains"]):
        subset = frame[(frame.domain==domain["id"])&(frame.weather=="perfect")&(frame.controller=="guarded_rule")]
        for j,(_,row) in enumerate(subset.iterrows()):
            index = i*3+j; labels.append(f"{domain['id']}\n{row.day}")
            selected = predictions[(predictions.scenario==row.scenario)&(predictions.controller=="guarded_rule")&
                                  (predictions.view=="perfect")&(predictions.horizon_steps==24)]
            if len(selected):
                p = selected.iloc[0]
                axes[0].bar(index-.17,p.temperature_RMSE_C,width=.34,color=colors[i])
                axes[0].bar(index+.17,p.persistence_RMSE_C,width=.34,color="lightgray")
            else:
                axes[0].text(index,0,"aborted",rotation=90,ha="center",va="bottom",fontsize=7)
            a = next(a for a in audits if a["scenario"]==row.scenario and a["controller"]=="solar_mpc")
            fraction = a.get("fallback_fraction_after_initialization")
            if fraction is not None:
                axes[1].bar(index,fraction*100,color=colors[i])
            else:
                axes[1].text(index,0,"aborted",rotation=90,ha="center",va="bottom",fontsize=7)
    axes[0].set_title("6 h conditional prediction: model / persistence")
    axes[0].set_ylabel("Temperature RMSE (°C)")
    axes[0].legend(handles=[Patch(color="tab:blue",label="Frozen model (domain colors)"),
                           Patch(color="lightgray",label="Persistence")],fontsize=7)
    axes[1].set_title("MPC fallback after shared initialization")
    axes[1].set_ylabel("Rule fallback (%)"); axes[1].set_ylim(0,105)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=55,ha="right",fontsize=7); ax.grid(axis="y",alpha=.2)
    fig.savefig(report/"prediction_and_fallback.png",dpi=150); plt.close(fig)
    files = ["thermal_comfort/generalization.py","configs/generalization.json","docs/experiments/generalization.md",
             "scripts/verify_generalization.py","scripts/report_generalization.py","tests/test_generalization.py"]
    for path in report.glob("*.json"):
        path.write_text(path.read_text(encoding="utf-8"),encoding="utf-8",newline="\n")
    save_json(report/"provenance.json",{"baseline_commit":frozen["config"]["baseline_commit"],
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                   "scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "scope":"Frozen zero-target-training simulation transfer; house operative and office air temperature kept distinct",
        "frozen_model_sha256":frozen["model_sha256"],"code_sha256":{n:digest(Path(n)) for n in files},
        "text_artifacts":"UTF-8 LF report copies; embedded hashes refer to original output files",
        "artifacts_sha256":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ["REPORT.md","provenance.json"]}})
    path = report/"provenance.json"
    path.write_text(path.read_text(encoding="utf-8"),encoding="utf-8",newline="\n")


if __name__=="__main__":
    main()
