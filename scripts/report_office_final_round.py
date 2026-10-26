"""Archive the stopped office campaign without authorizing more model development."""

import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import office_final_round as f, native as n, terminal as t
from thermal_comfort.simulation import digest


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    closure=f.check_closure();report=Path('reports/office_final_round');report.mkdir(parents=True,exist_ok=True)
    raw={}
    for name in ['protocol_lock.json','model/manifest.json','model/fit.json','assessment.json','gates.json','audit.json','closure.json','metrics.csv']:
        source=f.ROOT/name;raw[str(source)]=digest(source);target=report/name.replace('/','_')
        if source.suffix=='.json':write(target,t.load(source))
        else:target.write_bytes(source.read_bytes())
    entry=Path('outputs/office-final-entrypoint-check.json');raw[str(entry)]=digest(entry)
    write(report/'closed_entrypoint_checks.json',t.load(entry))
    model=joblib.load(f.ROOT/'model/model.joblib')
    labels=['R_out_air_K_per_kW','R_air_mass_K_per_kW','C_air_kWh_per_K','C_mass_kWh_per_K','solar_air_m2','solar_mass_m2',
        'internal_gain_multiplier','heating_power_gain','cooling_power_gain','fan_cubic_kW','fan_linear_kW']
    write(report/'model_parameters.json',{'candidate':f.configuration()['candidate'],'parameters':dict(zip(labels,model['parameters'].tolist())),
        'model_sha256':digest(f.ROOT/'model/model.joblib'),'source_prior':model['source_prior'],'oracle_fields_used':False,
        'note':'Effective parameters fitted in this simulator; not measured building parameters or real-office system identification.'})
    metrics=pd.read_csv(f.ROOT/'metrics.csv',float_precision='round_trip');gates=pd.DataFrame(t.load(f.ROOT/'gates.json'))
    gates.to_csv(report/'gates.csv',index=False,lineterminator='\n')
    gates[~gates.passed].to_csv(report/'failed_gates.csv',index=False,lineterminator='\n')
    gates['group']=gates.target.str.removesuffix('_energy')
    gates.groupby(['day','trajectory','group'],sort=False).passed.agg(['sum','count']).to_csv(report/'capability_matrix.csv',lineterminator='\n')
    old=pd.read_csv('reports/native_terminal/prediction_metrics.csv',float_precision='round_trip');old=old[old.model=='native_joint'].copy()
    old['model']='previous_native_joint'
    compare=pd.concat([metrics,old],ignore_index=True)
    compare.to_csv(report/'comparison_metrics.csv',index=False,lineterminator='\n')
    targets=['temperature','gas_energy','cooling_electricity_energy','fan_electricity_energy']
    models=['frozen_static','previous_native_joint',f.configuration()['candidate']]
    lines=['# 规则轨迹上的6小时开发RMSE','','这些是既有开发日期，未开展新校准或盲评；单位°C、kWh。','',
        '| 日期 | 模型 | 温度 | 燃气 | 制冷电 | 风机电 |','| --- | --- | ---: | ---: | ---: | ---: |']
    fig,axes=plt.subplots(2,2,figsize=(11,8));colors=['#8c8c8c','#e69f00','#0072b2']
    for day in [62,150]:
        for name in models:
            sub=compare[(compare.day==day)&compare.trajectory.str.endswith('guarded_rule')&(compare.model==name)&(compare.seconds==21600)]
            values=[float(sub[sub.target==target].RMSE.iloc[0]) for target in targets]
            lines.append(f'| {day} | {name} | '+' | '.join(f'{value:.6f}' for value in values)+' |')
    (report/'results_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    for ax,target in zip(axes.flat,targets):
        for j,name in enumerate(models):
            sub=compare[(compare.target==target)&(compare.model==name)&(compare.seconds==21600)&compare.trajectory.str.endswith('guarded_rule')]
            ax.bar(np.arange(2)+(j-1)*.22,[float(sub[sub.day==day].RMSE.iloc[0]) for day in [62,150]],.22,label=name,color=colors[j])
        ax.set_xticks([0,1],['Day 62 cold','Day 150 warm']);ax.set_title(target);ax.set_ylabel('RMSE (C)' if target=='temperature' else 'RMSE (kWh)');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Existing development rule trajectories; office campaign closed negative')
    fig.tight_layout();fig.savefig(report/'development_comparison.png',dpi=140);plt.close(fig)
    write(report/'stop_decision.json',{'status':closure['status'],'development_passed':False,'new_calibration_runs':0,'new_blind_runs':0,
        'new_development_collection_runs':0,'maximum_candidates':1,'actual_candidates':1,'independent_refit_is_reproduction':True,
        'primary_controller':'guarded_rule','next_office_development_iteration_scheduled':False,
        'scope':'This office migration campaign is ended; the overall project and unfinished milestones remain distinct.'})
    write(report/'budget_notes.json',{'configured_max_nfev':f.configuration()['fit_max_nfev'],'reported_nfev':model['diagnostics']['nfev'],
        'reported_nfev_definition':'SciPy counter excludes calls for numerical Jacobian approximation; these extra calls were not separately counted.',
        'independent_reproduction_same_budget':True,'training_origins':model['diagnostics']['training_origins'],
        'training_native_rows':model['diagnostics']['training_rows'],'training_days':12,
        'no_loss_initialization_threshold_or_additional_candidate_search':True,
        'comparison_limit':'Previous native joint used 128 training origins and 14 free parameters; this round uses 32 origins and 11 parameters. Do not attribute all changes exclusively to controller structure.'})
    source=[*f.CORE,'scripts/verify_office_final_round.py','scripts/report_office_final_round.py','tests/test_office_final_round.py']
    write(report/'provenance.json',{'raw_output_sha256':raw,'source_sha256':{name:digest(Path(name)) for name in source},
        'archived_sha256':{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ['REPORT.md','BOUNDARY.md','provenance.json']},
        'note':'JSON normalized to UTF-8 LF; original sealed outputs stay unchanged. closed_negative is a stop, not a successful migration or deployment freeze.'})


if __name__=='__main__':main()
