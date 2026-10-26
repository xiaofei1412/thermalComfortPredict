"""Fixed-parameter oracle substitutions for development-only terminal diagnosis."""

from datetime import datetime, timezone
import argparse
from pathlib import Path
import json

import joblib
from numba import njit
import numpy as np
import pandas as pd

from . import native as n, native_model as nm, terminal as t
from .simulation import digest, save_json

DATA=Path('outputs/terminal-diagnostic-replay')
OUTPUT=Path('outputs/terminal-diagnostic-analysis')
VARIANTS=['baseline','initial_PI_oracle','cooling_mode_oracle','flow_oracle','enthalpy_correction_oracle']
SECONDS=[30,300,900,3600,21600]
CORE=['thermal_comfort/terminal_diagnostic.py','scripts/collect_terminal_diagnostic.py','docs/experiments/terminal_diagnostic.md']


def check_baseline():
    t.check_final();manifest=t.load(n.MODEL/'manifest.json')
    if manifest['code_sha256']!=n.hashes():raise ValueError('Original native source changed')
    for name,h in manifest['artifacts_sha256'].items():
        if digest(n.MODEL/name)!=h:raise ValueError('Original native model changed')
    if n.CALIBRATION.exists() or n.LOCK.exists():raise ValueError('Unexpected new calibration/deployment lock')
    return joblib.load(n.MODEL/'native_joint.joblib')['parameters']


def states(parameters,e):
    a,b=nm.operators(parameters)
    return nm.observer(parameters,a,b,e['temperature'],e['power'],e['fan'],e['actions'],e['exog'],e['settings'],
        np.r_[e['initial_action'],e['actions']],np.r_[e['initial_setting'],e['settings']])


def load_episode(summary):
    root=DATA/f"day{summary['day']}__{summary['kind']}"
    if not summary['complete'] or not summary['public_bit_identical']:raise ValueError('Diagnostic replay differs from original development')
    if digest(root/'internal.csv.gz')!=summary['internal_sha256'] or digest(root/'parameters.json')!=summary['parameters_sha256']:
        raise ValueError('Diagnostic internal evidence changed')
    source=Path(summary['source'].replace('/workspace/',''))
    if digest(source/'summary.json')!=summary['source_summary_sha256']:raise ValueError('Original development identity changed')
    e=n.read_native(source,t.load(source/'summary.json'),'development')
    frame=pd.read_csv(root/'internal.csv.gz',float_precision='round_trip')
    e['integrals']=frame[['mod.con.heaPI.I.y','mod.con.cooPI.I.y']].to_numpy()*.1
    e['mode']=frame['mod.con.hysCoo.y'].to_numpy(dtype=float)
    # The FMU power meter uses the external return stream, not the coil inlet after fan dynamics.
    e['enthalpy_delta']=frame['mod.fcu.supplyAir.h_outflow'].to_numpy()-frame['mod.zon.returnAir.h_outflow'].to_numpy()
    supply=frame['mod.con.TSup'].to_numpy()-273.15
    support=e['fan']>1e-4
    e['enthalpy_correction']=np.where(support,e['enthalpy_delta']-1006*(supply-e['temperature']),0)
    return e,frame,t.load(root/'parameters.json')


@njit(cache=True)
def corrected_power(parameters,temperature,heat,cool,fan,mode,correction):
    supply=24-12*cool if mode else 21+19*heat
    net=nm.CP_FLOW*fan*(supply-temperature)+.55*fan*correction/1000
    return np.array([parameters[10]*max(net,0)/.9,parameters[11]*max(-net,0)/3,
        parameters[12]*fan**3+parameters[13]*fan])


@njit(cache=True)
def diagnostic_step(parameters,a,b,state,power,action,x,setting,variant,true_mode,true_fan,correction):
    if variant<2:return nm.transition(parameters,a,b,state,power,action,x,setting)
    temperature=state[0];eh=action-temperature;ec=temperature-setting;mode=state[5]
    if ec>0:mode=1.
    elif ec < -1:mode=0.
    if variant==2:mode=true_mode
    rh=nm.KP*eh+state[2];rc=nm.KP*ec+state[3]
    ih=state[2]+30/parameters[7]*(nm.KP*eh+(nm.clipped(rh)-rh)/parameters[8])
    ic=state[3]+30/parameters[7]*(nm.KP*ec+(nm.clipped(rc)-rc)/parameters[8])
    heat=nm.clipped(nm.KP*eh+ih);cool=nm.clipped(nm.KP*ec+ic)
    fan=nm.clipped(state[4]+(1-np.exp(-30/parameters[9]))*(nm.clipped(heat+cool)-state[4]))
    if variant==3:fan=true_fan
    delta=correction if variant==4 else 0.
    end=corrected_power(parameters,temperature,heat,cool,fan,mode,delta)
    mean=(power+end)/2;zone=nm.thermal(a,b,state[:2],.9*mean[0]-3*mean[1],x)
    end=corrected_power(parameters,zone[0],heat,cool,fan,mode,delta)
    mean=(power+end)/2;zone=nm.thermal(a,b,state[:2],.9*mean[0]-3*mean[1],x)
    return np.array([zone[0],zone[1],ih,ic,fan,mode]),end


@njit(cache=True)
def oracle_run(parameters,a,b,initial,initial_power,actions,exog,settings,mode,fan,correction,indices,variant):
    result=np.empty((len(initial),len(indices),7))
    for j in range(len(initial)):
        state=initial[j].copy();power=initial_power[j].copy();energy=np.zeros(3);index=0
        for h in range(indices[-1]):
            state,end=diagnostic_step(parameters,a,b,state,power,actions[j,h],exog[j,h],settings[j,h],variant,
                mode[j,h],fan[j,h],correction[j,h])
            energy+=(power+end)/2*nm.DH;power=end
            if h+1==indices[index]:
                result[j,index,0]=state[0];result[j,index,1:4]=power;result[j,index,4:7]=energy;index+=1
    return result


def predict(parameters,e,origins,variant,seconds=SECONDS):
    if variant not in VARIANTS:raise ValueError('Unregistered diagnostic variant')
    kind=VARIANTS.index(variant);indices=np.asarray(seconds,dtype=np.int64)//30
    if not len(origins) or np.any(origins<1) or np.any(origins+indices[-1]>len(e['actions'])):
        raise ValueError('Diagnostic origin outside episode')
    initial=states(parameters,e)[origins].copy()
    if kind==1:initial[:,2:4]=e['integrals'][origins]
    at=origins[:,None]+np.arange(indices[-1]);a,b=nm.operators(parameters)
    return oracle_run(parameters,a,b,initial,e['power'][origins],e['actions'][at],e['exog'][at],e['settings'][at],
        e['mode'][at+1],e['fan'][at+1],e['enthalpy_correction'][at+1],indices,kind)


@njit(cache=True)
def pi_replay(temperature,actions,settings,initial,ti,ni):
    """Teacher temperature, 1s midpoint integration; no future-state reset."""
    result=np.empty((len(temperature),2));state=initial.copy();result[0]=state
    for k in range(len(actions)):
        for sub in range(30):
            value=temperature[k]+(temperature[k+1]-temperature[k])*(sub+.5)/30
            errors=np.array([actions[k]-value,value-settings[k]])
            raw=.1*errors+state
            derivative=np.array([(.1*errors[j]+(nm.clipped(raw[j])-raw[j])/ni)/ti for j in range(2)])
            midpoint=state+.5*derivative;raw=.1*errors+midpoint
            state+=np.array([(.1*errors[j]+(nm.clipped(raw[j])-raw[j])/ni)/ti for j in range(2)])
        result[k+1]=state
    return result


@njit(cache=True)
def fan_replay(command,initial,tau,rate,two_poles):
    result=np.empty(len(command));one,two=initial[0],initial[1];result[0]=two if two_poles else one
    for k in range(len(command)-1):
        for sub in range(30):
            u=command[k]+(command[k+1]-command[k])*(sub+.5)/30
            if two_poles:
                d1=rate*(u-one);d2=rate*(one-two)
                mid1=one+.5*d1;mid2=two+.5*d2
                one+=rate*(u-mid1);two+=rate*(mid1-mid2)
            else:one+= (1-np.exp(-1/tau))*(u-one)
        result[k+1]=two if two_poles else one
    return result


def component_checks(parameters,e,f,p):
    state=states(parameters,e);valid=np.arange(len(state))>=480;support=(e['fan']>1e-4)&valid
    truth=e['integrals'];pi=pi_replay(e['temperature'],e['actions'],e['settings'],truth[0],120.,.9)
    hidden_zero=pi_replay(e['temperature'],e['actions'],e['settings'],np.zeros(2),120.,.9)
    one=fan_replay(f['mod.con.yFan'].to_numpy(),np.repeat(e['fan'][0],2),parameters[9],0.,False)
    x1=f['mod.fcu.fan.filter.x[1]'].to_numpy();x2=f['mod.fcu.fan.filter.x[2]'].to_numpy()
    rate=-p['mod.fcu.fan.filter.r[1]'];omega=2*np.pi*p['mod.fcu.fan.filter.f_cut']
    two=fan_replay(f['mod.con.yFan'].to_numpy(),np.array([x1[0],x2[0]]),parameters[9],rate,True)
    scheduled=np.r_[e['settings'],e['settings'][-1]]
    true_setting=f['mod.con.oveTSetCoo.y'].to_numpy()-273.15
    mode=np.zeros(len(f));previous=float(e['mode'][0])
    for k,(temperature,setting) in enumerate(zip(e['temperature'],scheduled)):
        if temperature-setting>0:previous=1.
        elif temperature-setting < -1:previous=0.
        mode[k]=previous
    flow=f['fcu_reaFloSup_y'].to_numpy()
    actual=.9*e['power'][:,0]-3*e['power'][:,1]
    exact=flow*e['enthalpy_delta']/1000
    approximate=flow*1006*(f['mod.con.TSup'].to_numpy()-273.15-e['temperature'])/1000
    return {'day':e['day'],'trajectory':e['id'],'observed_samples':int(valid.sum()),'active_flow_samples':int(support.sum()),
        'PI_inverse_RMSE_output_units':np.sqrt(np.mean((state[valid,2:4]-truth[valid])**2,axis=0)).tolist(),
        'PI_source_teacher_RMSE_output_units':np.sqrt(np.mean((pi[valid]-truth[valid])**2,axis=0)).tolist(),
        'PI_zero_initial_teacher_RMSE_output_units':np.sqrt(np.mean((hidden_zero[valid]-truth[valid])**2,axis=0)).tolist(),
        'PI_inverse_error_above_0_05_fraction':float(np.mean(np.abs(state[valid,2:4]-truth[valid])>.05)),
        'cooling_mode_mismatch_samples':int(np.sum(mode[valid]!=e['mode'][valid])),
        'schedule_mismatch_samples':int(np.sum(abs(scheduled[valid]-true_setting[valid])>1e-9)),
        'first_order_fan_teacher_RMSE_fraction':float(np.sqrt(np.mean((one[valid]-e['fan'][valid])**2))),
        'source_second_order_fan_teacher_RMSE_fraction':float(np.sqrt(np.mean((two[valid]-e['fan'][valid])**2))),
        'source_filter_output_closure_max_fraction':float(np.max(abs(x2-flow/.55))),
        'source_filter_dx1_closure_max':float(np.max(abs(f['der(mod.fcu.fan.filter.x[1])']-(p['mod.fcu.fan.filter.r[1]']*x1+rate*f['mod.fcu.fan.filter.u']/.55)))),
        'source_filter_dx2_closure_max':float(np.max(abs(f['der(mod.fcu.fan.filter.x[2])']-(p['mod.fcu.fan.filter.r[2]']*x2+rate*x1)))),
        'source_filter_pole_parameter_closure':float(abs(rate-omega*p['mod.fcu.fan.filter.cr[1]'])),
        'stream_enthalpy_balance_max_kW':float(np.max(abs(exact-actual))),
        'constant_cp_temperature_balance_RMSE_kW':float(np.sqrt(np.mean((approximate[support]-actual[support])**2))),
        'constant_cp_temperature_balance_bias_kW':float(np.mean(approximate[support]-actual[support])),
        'coil_inlet_water_mass_fraction_range':[float(f['mod.fcu.coi.Xi_in_a[1]'].min()),float(f['mod.fcu.coi.Xi_in_a[1]'].max())],
        'coil_outlet_water_mass_fraction_range':[float(f['mod.fcu.coi.Xi_internal_b[1]'].min()),float(f['mod.fcu.coi.Xi_internal_b[1]'].max())]}


def main(output=OUTPUT):
    parameters=check_baseline();manifest=t.load(DATA/'manifest.json')
    if manifest['protocol_sha256']!=digest(Path(CORE[-1])) or manifest['collector_sha256']!=digest(Path(CORE[1])):
        raise ValueError('Collection source/protocol changed')
    summaries=t.load(DATA/'summary.json')
    if {(s['day'],s['kind']) for s in summaries}!={(d,k) for d in [62,150] for k in ['excitation','guarded_rule']} or len(summaries)!=4:
        raise ValueError('Incomplete registered diagnostic roles')
    output.mkdir(parents=True,exist_ok=False);metrics=[];paired=[];components=[]
    for summary in summaries:
        e,f,p=load_episode(summary);origins=nm.origin_indices(e,3600);y=nm.truth(e,origins,SECONDS)
        components.append(component_checks(parameters,e,f,p))
        reference=nm.prediction(parameters,e,origins,SECONDS)
        for variant in VARIANTS:
            result=predict(parameters,e,origins,variant)
            if variant=='baseline':np.testing.assert_array_equal(result,reference)
            if not np.isfinite(result).all() or (result[:,:,1:]<0).any():raise ValueError('Invalid diagnostic rollout')
            metrics.extend(n.metric_rows(variant,e,result,y,SECONDS))
            for i,origin in enumerate(origins):
                paired.append({'day':e['day'],'trajectory':e['id'],'origin_seconds':float(e['times'][origin]),'variant':variant,
                    **{name:float(result[i,-1,j]-y[i,-1,j]) for j,name in enumerate(['temperature_error_K','gas_power_error_kW','cooling_power_error_kW','fan_power_error_kW','gas_error_kWh','cooling_error_kWh','fan_error_kWh'])}})
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False,lineterminator='\n')
    pd.DataFrame(paired).to_csv(output/'paired_6h_errors.csv',index=False,lineterminator='\n')
    save_json(output/'component_checks.json',components)
    save_json(output/'manifest.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'purpose':'diagnostic_only_no_deployment',
        'code_sha256':{name:digest(Path(name)) for name in CORE},'model_sha256':digest(n.MODEL/'native_joint.joblib'),
        'collection_manifest_sha256':digest(DATA/'manifest.json'),'collection_summary_sha256':digest(DATA/'summary.json'),
        'variants':VARIANTS,'refitted_parameters':False,'future_oracles':VARIANTS[2:],'initial_state_oracle':VARIANTS[1],
        'calibration_not_generated':not n.CALIBRATION.exists(),'new_scenario_run':False})
    print(json.dumps(components,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args().output)
