"""Audit the one allowed candidate and unchanged gates before negative-result sealing."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thermal_comfort import office_final_round as f, source_terminal as sm, native as n, native_model as nm, terminal as t, terminal_diagnostic as d
from thermal_comfort.simulation import digest, save_json


def main():
    lock=f.check_protocol();model=joblib.load(f.ROOT/'model/model.joblib');parameters=model['parameters']
    assert model['oracle_fields_used'] is False and len(parameters)==11 and np.isfinite(parameters).all()
    assert model['diagnostics']['nfev']<=f.configuration()['fit_max_nfev']
    assert model['diagnostics']['training_origins']==32 and model['diagnostics']['training_rows']==34560
    for output,independent in [('model',False),('refit',True)]:
        root=f.ROOT/output;manifest=t.load(root/'manifest.json')
        assert manifest['code_sha256']==f.hashes() and manifest['protocol_lock_sha256']==digest(f.ROOT/'protocol_lock.json')
        assert manifest['independent_reproduction']==independent
        for name,h in manifest['artifacts_sha256'].items():assert digest(root/name)==h
    for name in ['model.joblib','fit.json']:assert digest(f.ROOT/'model'/name)==digest(f.ROOT/'refit'/name)
    before={name:digest(f.ROOT/name) for name in ['metrics.csv','gates.json']};old=t.load(f.ROOT/'assessment.json')
    f.evaluate()
    assert all(digest(f.ROOT/name)==h for name,h in before.items())
    assessment=t.load(f.ROOT/'assessment.json')
    assert {k:v for k,v in old.items() if k!='created_UTC'}=={k:v for k,v in assessment.items() if k!='created_UTC'}
    assert not assessment['eligible_for_calibration'] and len(t.load(f.ROOT/'gates.json'))==72
    assert assessment['activity']==lock['preflight_activity']
    assert [(r['day'],r['active_origins'][0]) for r in assessment['activity'] if r['day']==150]==[(150,2),(150,0)]
    causal_checks=[];unit_error=0.
    for e in [*n.source_episodes('train'),*f.development()]:
        origin=np.array([480]);seconds=n.configuration()['horizons_seconds']
        prediction=sm.prediction(parameters,e,origin,seconds);changed=copy.deepcopy(e)
        changed['temperature'][481:]+=100;changed['power'][481:]+=100;changed['fan'][481:]=1
        changed['integrals']=np.full((len(e['temperature']),2),100.);changed['mode']=np.ones(len(e['temperature']))
        changed['enthalpy_correction']=np.full(len(e['temperature']),100000.)
        np.testing.assert_array_equal(prediction,sm.prediction(parameters,changed,origin,seconds))
        assert np.isfinite(prediction).all() and (prediction[:,:,1:]>=0).all()
        np.testing.assert_allclose(prediction[0,0,4:7],(e['power'][480]+prediction[0,0,1:4])/2*nm.DH,atol=1e-14,rtol=0)
        starts=nm.origin_indices(e,3600);truth=nm.truth(e,starts,[21600]);native=pd.read_csv(e['root']/'native.csv.gz',float_precision='round_trip')
        for j,column in enumerate(['fcu_reaPHea_y','fcu_reaPCoo_y','fcu_reaPFan_y']):
            integrated=np.array([np.trapezoid(native[column].iloc[k:k+721],native.time.iloc[k:k+721])/3600000 for k in starts])
            error=float(np.max(abs(integrated-truth[:,0,4+j])));unit_error=max(error,unit_error);assert error<1e-10
        causal_checks.append({'role':e['role'],'episode':e['id'],'future_label_and_oracle_mutations_ignored':True})
    # Internal truth is used only here, after fitting, to inspect past-only observer accuracy.
    state_checks=[]
    for summary in t.load(d.DATA/'summary.json'):
        e,_,_=d.load_episode(summary);history=sm.history(e);valid=np.arange(len(history))>=480
        state_checks.append({'day':e['day'],'trajectory':e['id'],
            'PI_integral_RMSE_output_units':np.sqrt(np.mean((history[valid,:2]-e['integrals'][valid])**2,axis=0)).tolist(),
            'causal_flow_RMSE_fraction':float(np.sqrt(np.mean((history[valid,3]-e['fan'][valid])**2)))})
    metrics=pd.read_csv(f.ROOT/'metrics.csv',float_precision='round_trip')
    original=pd.read_csv('reports/native_terminal/prediction_metrics.csv',float_precision='round_trip')
    a=metrics[metrics.model=='frozen_static'];b=original[original.model=='frozen_static']
    joined=a.merge(b,on=['day','trajectory','seconds','target'],suffixes=('_a','_b'),validate='one_to_one')
    assert len(joined)==len(a)==len(b)==32
    for column in ['RMSE','bias','truth_RMS']:np.testing.assert_array_equal(joined[column+'_a'],joined[column+'_b'])
    failed=Counter(g['criterion'] for g in assessment['failed_gates'])
    result={'created_UTC':datetime.now(timezone.utc).isoformat(),'integrity_passed':True,'code_sha256':f.hashes(),
        'model_sha256':digest(f.ROOT/'model/model.joblib'),'protocol_lock_sha256':digest(f.ROOT/'protocol_lock.json'),
        'independent_refit_byte_identical':True,'gate_recompute_byte_identical':True,'baseline_rows_exact':32,
        'causal_checks':causal_checks,'native_energy_unit_max_error_kWh':unit_error,
        'postfit_observer_checks_against_internal_truth':state_checks,'internal_truth_entered_fitting_or_prediction':False,
        'failed_gate_counts':dict(failed),'all_development_gates_passed':False,
        'assessment_sha256':digest(f.ROOT/'assessment.json'),'gates_sha256':digest(f.ROOT/'gates.json'),
        'calibration_generated':f.CALIBRATION.exists(),'blind_evaluation_run':f.BLIND.exists(),
        'original_native_model_sha256':digest(n.MODEL/'native_joint.joblib'),'old_freezes_valid':True}
    assert not result['calibration_generated'] and not result['blind_evaluation_run']
    save_json(f.ROOT/'audit.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
