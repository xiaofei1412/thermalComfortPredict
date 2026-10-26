"""Small archive of component evidence and explicitly non-deployable oracle experiments."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import terminal_diagnostic as d, native as n, terminal as t
from thermal_comfort.simulation import digest


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    parameters=d.check_baseline();report=Path('reports/terminal_diagnostic');report.mkdir(parents=True,exist_ok=True)
    audit=t.load(d.OUTPUT/'audit.json');assert audit['integrity_passed'] and audit['diagnostic_only']
    raw={}
    for root,names,prefix in [(d.DATA,['manifest.json','summary.json'],'collection_'),
        (d.OUTPUT,['manifest.json','audit.json','component_checks.json','metrics.csv','paired_6h_errors.csv'],'')]:
        for name in names:
            source=root/name;raw[str(source)]=digest(source)
            if source.suffix=='.json':write(report/(prefix+name),t.load(source))
            else:(report/(prefix+name)).write_bytes(source.read_bytes())
    summaries=t.load(d.DATA/'summary.json')
    write(report/'compiled_parameters.json',{f"day{s['day']}__{s['kind']}":t.load(d.DATA/f"day{s['day']}__{s['kind']}"/'parameters.json') for s in summaries})
    metrics=pd.read_csv(d.OUTPUT/'metrics.csv',float_precision='round_trip')
    targets=['temperature','gas_energy','cooling_electricity_energy','fan_electricity_energy']
    lines=['# 6小时开发诊断RMSE','','含内部真值/未来信息的单因素替换，均不可作为部署或能力门成绩。温度单位°C，能源单位kWh。','',
        '| 日期/轨迹 | 诊断替换 | 温度 | 燃气 | 制冷电 | 风机电 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for s in summaries:
        trajectory=f"day{s['day']}__{s['kind']}"
        for variant in d.VARIANTS:
            rows=metrics[(metrics.trajectory==trajectory)&(metrics.model==variant)&(metrics.seconds==21600)]
            values=[float(rows[rows.target==target].RMSE.iloc[0]) for target in targets]
            lines.append(f"| {s['day']}/{s['kind']} | {variant} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    (report/'results_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for ax,target in zip(axes.flat,targets):
        ratios=[]
        for s in summaries:
            sub=metrics[(metrics.day==s['day'])&(metrics.trajectory==f"day{s['day']}__{s['kind']}")&(metrics.seconds==21600)&(metrics.target==target)]
            base=float(sub[sub.model=='baseline'].RMSE.iloc[0])
            ratios.append([float(sub[sub.model==variant].RMSE.iloc[0])/base for variant in d.VARIANTS[1:]])
        data=np.array(ratios);im=ax.imshow(data,vmin=0,vmax=2,cmap='RdBu_r')
        for row in range(4):
            for col in range(4):ax.text(col,row,f'{data[row,col]:.3f}',ha='center',va='center',color='black' if .3<data[row,col]<1.7 else 'white')
        ax.set_title(target+' RMSE / baseline')
        ax.set_xticks(range(4),['Initial PI','Future mode','Future flow','Future enthalpy'],rotation=20,ha='right')
        ax.set_yticks(range(4),[f"Day{s['day']} {s['kind']}" for s in summaries])
    fig.suptitle('Diagnostic oracle substitutions only: below 1 improves RMSE',fontsize=13)
    fig.tight_layout(rect=[0,0,.9,.96]);cax=fig.add_axes([.92,.15,.02,.7]);fig.colorbar(im,cax=cax)
    fig.savefig(report/'oracle_ratios.png',dpi=140);plt.close(fig)
    # A fixed warm-rule example exposes the difference between integral reconstruction and forward replay.
    s=next(s for s in summaries if s['day']==150 and s['kind']=='guarded_rule');e,f,p=d.load_episode(s)
    state=d.states(parameters,e);pi=d.pi_replay(e['temperature'],e['actions'],e['settings'],e['integrals'][0],120.,.9)
    command=f['mod.con.yFan'].to_numpy()
    one=d.fan_replay(command,np.repeat(e['fan'][0],2),parameters[9],0.,False)
    two=d.fan_replay(command,f[['mod.fcu.fan.filter.x[1]','mod.fcu.fan.filter.x[2]']].iloc[0].to_numpy(),parameters[9],-p['mod.fcu.fan.filter.r[1]'],True)
    time=(f.time.to_numpy()-150*86400)/3600;mask=(time>=4)&(time<=24)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for j,label in enumerate(['Heating PI integral','Cooling PI integral']):
        axes[j].plot(time[mask],e['integrals'][mask,j],label='FMU internal truth',color='black',lw=1.5)
        axes[j].plot(time[mask],state[mask,j+2],label='Existing algebraic inverse',lw=1)
        axes[j].plot(time[mask],pi[mask,j],label='Source PI replay with observed T',ls='--',lw=1)
        axes[j].set_ylabel(label);axes[j].grid(alpha=.25)
    axes[2].plot(time[mask],e['fan'][mask],label='FMU flow / 0.55',color='black',lw=1.5)
    axes[2].plot(time[mask],one[mask],label='Learned first-order, true command',lw=1)
    axes[2].plot(time[mask],two[mask],label='Source second-order, true command',ls='--',lw=1)
    axes[2].set_ylabel('Normalized air mass flow');axes[2].set_xlabel('Hour of first development day 150');axes[2].grid(alpha=.25)
    axes[2].set_xlim(4,24)
    axes[0].legend(fontsize=8);axes[2].legend(fontsize=8);fig.suptitle('Teacher-driven components (PI integral scaled by k=0.1); not autonomous forecasts',fontsize=11)
    fig.tight_layout();fig.savefig(report/'component_replay.png',dpi=140);plt.close(fig)
    events=[]
    for s in summaries:
        e,f,p=d.load_episode(s);scheduled=np.r_[e['settings'],e['settings'][-1]];actual=f['mod.con.oveTSetCoo.y'].to_numpy()-273.15
        times=f.time.to_numpy();wrong=(abs(scheduled-actual)>1e-9)&(np.arange(len(f))>=480)
        flow=e['fan'];fan_power=parameters[12]*flow**3+parameters[13]*flow
        valid=np.arange(len(f))>=480
        events.append({'day':s['day'],'kind':s['kind'],'schedule_mismatch_times_seconds':times[wrong].tolist(),
            'fan_power_given_true_flow_RMSE_kW':float(np.sqrt(np.mean((fan_power[valid]-e['power'][valid,2])**2))),
            'fan_power_given_true_flow_bias_kW':float(np.mean(fan_power[valid]-e['power'][valid,2]))})
    write(report/'additional_equation_checks.json',events)
    write(report/'decision.json',{'diagnostic_round_complete':True,'retrained_model':False,'deployment_candidate_selected':None,
        'new_calibration_generated':False,'new_scenario_evaluated':False,'primary_controller':'guarded_rule',
        'next_priority':'Validate a causal source-consistent PI and second-order air-flow forward model on development data before thermal/energy joint refitting',
        'hypotheses':{'H1':'Integral reconstruction error supported, but origin-only oracle has little 6h benefit',
            'H2':'Only sparse boundary mismatches under observed temperature; future-mode oracle has little overall benefit',
            'H3':'First-order mismatch supported; future-flow oracle strongly reduces fan energy error, not all thermal/energy error',
            'H4':'Stream energy identity closes; constant-cp error is small here and its correction has little 6h benefit'}})
    write(report/'data_ledger.json',{'registered_development_dates':[62,150],'registered_policies':['excitation','guarded_rule'],
        'full_three_day_replays':4,'replayed_evaluation_days':12,'unique_weather_calendar_days':6,
        'decision_intervals':1152,'native_intervals':34560,'native_samples':34564,'additional_initialization_days_per_replay':7,
        'failed_collection_attempts_counted_as_independent_evidence':False,'new_training_rows':0,'new_calibration_rows':0,'new_test_rows':0,
        'overlapping_forecast_origins':252,'oracle_variants':4,'main_metric_rows':700,'paired_6h_rows':1260,
        'note':'Failed collection attempts and initialization have additional computational cost; 12 evaluation days are not total simulation cost or independent climates.'})
    write(report/'collection_failures.json',{'cwd':'Initial standalone TestCase required version.txt in its working directory; preserved outputs/terminal-diagnostic-data-cwd-failed',
        'array_filter':'PyFMI wildcard filters interpreted Modelica array brackets; escaped literal brackets; preserved outputs/terminal-diagnostic-data-filter-failed',
        'first_endpoint':'First captured simulate result was after applying a new action; original store uses previous left endpoint. Read pre-action FMU state and discard each new interval start; round-trip CSV parsing used',
        'failed_directory':'outputs/terminal-diagnostic-data retained in place after Windows denied renaming; final successful replay isolated in outputs/terminal-diagnostic-replay',
        'source_checks':'Coil inlet enthalpy differs from external return stream; power meter uses external return. Filter states are normalized flow, not multiplied by cr. Corrected against compiled derivatives before running fixed ablations',
        'no_algorithm_tuning':'No new candidate, loss, parameter fit or capability-threshold change was made'})
    files=[*d.CORE,'scripts/verify_terminal_diagnostic.py','scripts/report_terminal_diagnostic.py','tests/test_terminal_diagnostic.py']
    write(report/'provenance.json',{'raw_output_sha256':raw,'source_sha256':{name:digest(Path(name)) for name in files},
        'source_fmu_sha256':digest(t.SOURCE/'testcases/bestest_air/models/wrapped.fmu'),
        'unchanged_native_model_sha256':digest(n.MODEL/'native_joint.joblib'),
        'archived_sha256':{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name not in ['REPORT.md','provenance.json']},
        'note':'JSON archives normalized to UTF-8 LF. Internal trajectories and oracle forecasts are diagnostic evidence, not deployment qualification.'})


if __name__=='__main__':main()
