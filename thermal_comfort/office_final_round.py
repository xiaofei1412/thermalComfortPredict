"""One final causal candidate; unchanged gates and a durable negative-result stop."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from . import native as n, native_model as nm, source_terminal as sm, terminal_diagnostic as d, terminal as t
from .simulation import digest, save_json

CONFIG=Path('configs/office_final_round.json')
ROOT=Path('outputs/office-final-round')
CLOSURE=ROOT/'closure.json'
CORE=['thermal_comfort/source_terminal.py','thermal_comfort/office_final_round.py',str(CONFIG),'docs/experiments/office_final_round.md']
CALIBRATION=Path('outputs/office-final-calibration')
BLIND=Path('outputs/office-final-blind')


def configuration():return t.load(CONFIG)
def hashes():return {name:digest(Path(name)) for name in CORE}


def development():
    episodes=n.source_episodes('development');summaries=t.load(n.RULE/'summary.json')
    if {(s['day'],s['days'],s['controller']) for s in summaries}!={(62,3,'guarded_rule'),(150,3,'guarded_rule')} or len(summaries)!=2:
        raise ValueError('Final development coverage changed')
    return episodes+[n.read_native(n.RULE/f"day{s['day']}__guarded_rule",s,'development') for s in summaries]


def responses():
    result=[]
    for day in configuration()['development_days']:
        pair=[]
        for setting in [20.5,23.25]:
            root=Path(f'outputs/terminal-response/day{day}_perfect__response_{setting:g}')
            pair.append(n.read_native(root,t.load(root/'summary.json'),'development'))
        result.append(pair)
    return result


def activity(episodes):
    reference=n.configuration();result=[]
    for e in episodes:
        starts=nm.origin_indices(e,reference['validation_stride_seconds']);y=nm.truth(e,starts,reference['horizons_seconds'])
        result.append({'day':e['day'],'trajectory':e['id'],'origins':len(starts),
            'active_origins':(y[:,-1,4:7]/6>reference['gates']['active_mean_kW']).sum(axis=0).tolist()})
    return result


def prepare():
    if CLOSURE.exists():raise ValueError('Office migration round is closed; no new development protocol')
    d.check_baseline();config=configuration();reference=n.configuration();n.validate_roles(reference)
    if config['maximum_candidates']!=1 or config['training_days']!=reference['training_days']:
        raise ValueError('Final-round candidate/data budget changed')
    if CALIBRATION.exists() or BLIND.exists():raise ValueError('No final calibration/blind data may exist before development')
    compiled=Path('reports/terminal_diagnostic/compiled_parameters.json')
    for p in t.load(compiled).values():
        assert p['mod.con.heaPI.k']==config['PI_k']==.1 and p['mod.con.heaPI.Ti']==config['PI_Ti_seconds']==120
        assert p['mod.con.heaPI.Ni']==config['PI_Ni']==.9
        assert -p['mod.fcu.fan.filter.r[1]']==config['fan_pole_per_second']==sm.RATE
        assert p['mod.fcu.fan.filter.r[1]']==p['mod.fcu.fan.filter.r[2]']
    train=n.source_episodes('train');dev=development();paired=responses();coverage=activity(dev)
    files={str(e['root']/'summary.json'):digest(e['root']/'summary.json') for e in [*train,*dev,*[e for pair in paired for e in pair]]}
    ROOT.mkdir(parents=True,exist_ok=False)
    save_json(ROOT/'protocol_lock.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'config':config,
        'code_sha256':hashes(),'reference_gate_sha256':digest(n.CONFIG),'reference_configuration':reference,
        'original_model_sha256':digest(n.MODEL/'native_joint.joblib'),'compiled_prior_sha256':digest(compiled),
        'data_summary_sha256':files,'preflight_activity':coverage,
        'preflight_support_failures':[{'day':r['day'],'trajectory':r['trajectory'],'carrier':c,'active_origins':r['active_origins'][j]}
            for r in coverage for j,c in enumerate(n.CARRIERS) if r['active_origins'][j]<reference['gates']['minimum_active_origins']],
        'no_new_development_data':True,'maximum_candidates':1})


def check_protocol(allow_closed=False):
    if CLOSURE.exists() and not allow_closed:raise ValueError('Office migration round is closed; no further fitting or development')
    d.check_baseline();lock=t.load(ROOT/'protocol_lock.json')
    if lock['code_sha256']!=hashes() or lock['reference_gate_sha256']!=digest(n.CONFIG):raise ValueError('Final protocol/gates changed')
    if lock['original_model_sha256']!=digest(n.MODEL/'native_joint.joblib'):raise ValueError('Original checkpoint changed')
    for name,h in lock['data_summary_sha256'].items():
        if digest(Path(name))!=h:raise ValueError('Registered final data changed')
    if CALIBRATION.exists() or BLIND.exists():raise ValueError('Final round has no authorization from all development gates')
    return lock


def fit(independent=False):
    check_protocol();output=ROOT/('refit' if independent else 'model')
    output.mkdir(parents=True,exist_ok=False)
    config=configuration();reference=n.configuration();old=d.check_baseline()
    start=np.r_[old[:7],old[10:14]]
    model=sm.fit(n.source_episodes('train'),start,config,reference)
    joblib.dump(model,output/'model.joblib');save_json(output/'fit.json',model['diagnostics'])
    save_json(output/'manifest.json',{'code_sha256':hashes(),'protocol_lock_sha256':digest(ROOT/'protocol_lock.json'),
        'candidate':config['candidate'],'independent_reproduction':independent,
        'artifacts_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(model['diagnostics'],indent=2),flush=True)


def existing_gates(rows,coverage):
    # The frozen gate function names its candidate native_joint. This adapter changes names only.
    gates=n.capability_gates(rows,coverage,n.configuration())
    return [{**g,'model':configuration()['candidate']} for g in gates if g['model']=='native_joint']


def evaluate():
    lock=check_protocol();reference=n.configuration();config=configuration()
    model=joblib.load(ROOT/'model/model.joblib');rows=[];dev=development();coverage=activity(dev)
    assert coverage==lock['preflight_activity']
    for e in dev:
        starts=nm.origin_indices(e,reference['validation_stride_seconds']);seconds=reference['horizons_seconds']
        p=sm.prediction(model['parameters'],e,starts,seconds);y=nm.truth(e,starts,seconds)
        if not np.isfinite(p).all() or (p[:,:,1:]<0).any():raise ValueError('Invalid final free-running prediction')
        rows.extend(n.metric_rows('native_joint',e,p,y,seconds))
        base=n.baseline_prediction(e,starts);yy=nm.truth(e,starts,[3600,21600])
        rows.extend(r for r in n.metric_rows('frozen_static',e,base,yy,[3600,21600]) if r['target']=='temperature' or r['target'].endswith('_energy'))
    gates=existing_gates(rows,coverage);response=[]
    for pair in responses():
        origin=np.array([480]);seconds=[3600,21600]
        ys=[nm.truth(e,origin,seconds) for e in pair];ps=[sm.prediction(model['parameters'],e,origin,seconds) for e in pair]
        actual=ys[1]-ys[0];predicted=ps[1]-ps[0];active=abs(actual[0,0,4])>.001
        direction=active and actual[0,0,0]*predicted[0,0,0]>0 and actual[0,0,4]*predicted[0,0,4]>0
        limit=max(reference['gates']['response_absolute_floor_kWh'],reference['gates']['response_relative_error']*abs(actual[0,1,4]))
        error=abs(predicted[0,1,4]-actual[0,1,4])
        response.append({'day':pair[0]['day'],'active':bool(active),'direction_passed':bool(direction),
            'actual_gas_delta_6h_kWh':float(actual[0,1,4]),'predicted_gas_delta_6h_kWh':float(predicted[0,1,4]),
            'absolute_error_kWh':float(error),'limit_kWh':float(limit),'passed':bool(direction and error<=limit)})
    eligible=all(g['passed'] for g in gates) and all(r['passed'] for r in response) and model['diagnostics']['success']
    if eligible and lock['preflight_support_failures']:raise ValueError('Eligibility contradicts fixed data support; integrity failure')
    for row in rows:
        if row['model']=='native_joint':row['model']=config['candidate']
    pd.DataFrame(rows).to_csv(ROOT/'metrics.csv',index=False,lineterminator='\n')
    save_json(ROOT/'gates.json',gates)
    save_json(ROOT/'assessment.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'code_sha256':hashes(),
        'model_sha256':digest(ROOT/'model/model.joblib'),'candidate':config['candidate'],'activity':coverage,'response':response,
        'accuracy_and_support_gates_passed':sum(g['passed'] for g in gates),'accuracy_and_support_gates_total':len(gates),
        'response_gates_passed':sum(r['passed'] for r in response),'response_gates_total':len(response),
        'optimizer_converged':model['diagnostics']['success'],'eligible_for_calibration':bool(eligible),
        'failed_gates':[g for g in gates if not g['passed']],
        'decision':'eligible_for_once_only_calibration_and_blind' if eligible else 'close_negative_no_calibration_or_blind'})
    print(json.dumps({'eligible':bool(eligible),'passed':sum(g['passed'] for g in gates),'total':len(gates),'response':response},indent=2),flush=True)


def seal():
    check_protocol();assessment=t.load(ROOT/'assessment.json');audit=t.load(ROOT/'audit.json')
    if assessment['eligible_for_calibration']:raise ValueError('A passing candidate cannot be sealed as a negative result')
    assert assessment['code_sha256']==hashes() and audit['integrity_passed'] and audit['code_sha256']==hashes()
    assert assessment['model_sha256']==audit['model_sha256']==digest(ROOT/'model/model.joblib')
    evidence=['protocol_lock.json','model/model.joblib','model/manifest.json','model/fit.json','metrics.csv','gates.json','assessment.json','audit.json']
    save_json(CLOSURE,{'sealed_UTC':datetime.now(timezone.utc).isoformat(),'status':'closed_negative',
        'scope':'this_office_simulation_migration_campaign','code_sha256':hashes(),
        'artifacts_sha256':{name:digest(ROOT/name) for name in evidence},'primary_controller':'guarded_rule',
        'calibration_generated':False,'blind_evaluation_run':False,'further_automatic_development_allowed':False,
        'boundary':'Offline modeling research in the fixed 48m2 bestest_air interface only; no cross-season energy-saving control qualification or real-building claim'})


def check_closure():
    check_protocol(allow_closed=True);closure=t.load(CLOSURE)
    assert closure['status']=='closed_negative' and closure['code_sha256']==hashes()
    for name,h in closure['artifacts_sha256'].items():assert digest(ROOT/name)==h
    return closure


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','fit','refit','evaluate','seal','check'])
    mode=parser.parse_args().mode
    if mode=='prepare':prepare()
    elif mode=='fit':fit()
    elif mode=='refit':fit(True)
    elif mode=='evaluate':evaluate()
    elif mode=='seal':seal()
    else:print(json.dumps(check_closure(),indent=2))


if __name__=='__main__':main()
