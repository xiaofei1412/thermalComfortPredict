"""Causal source-PI and two-pole flow model for the final bounded office round."""

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from . import native_model as nm

RATE=.259487839385516
DECAY=np.exp(-RATE)


def operators(parameters,seconds):
    rz,rm,ca,cm,sa,sm,gain=parameters[:7]
    a=np.array([[-(1/rz+1/rm)/ca,1/rm/ca],[1/rm/cm,-1/rm/cm]])
    b=np.array([[1/rz/ca,1/ca,sa/ca,gain/ca,0],[0,0,sm/cm,0,gain/cm]])
    ad=np.linalg.inv(np.eye(2)-seconds/3600*a)
    return ad,ad@b*(seconds/3600)


@njit(inline='always')
def terminal_step(ih,ic,f1,f2,mode,temperature,action,setting):
    eh=action-temperature;ec=temperature-setting
    if ec>0:mode=1.
    elif ec < -1:mode=0.
    rh=.1*eh+ih;rc=.1*ec+ic
    mh=ih+.5*(.1*eh+(nm.clipped(rh)-rh)/.9)/120
    mc=ic+.5*(.1*ec+(nm.clipped(rc)-rc)/.9)/120
    heat=nm.clipped(.1*eh+mh);cool=nm.clipped(.1*ec+mc)
    ih+=(.1*eh+(heat-(.1*eh+mh))/.9)/120
    ic+=(.1*ec+(cool-(.1*ec+mc))/.9)/120
    command=nm.clipped(heat+cool)
    second=command+(f2-command+RATE*(f1-command))*DECAY
    first=command+(f1-command)*DECAY
    return ih,ic,nm.clipped(first),nm.clipped(second),mode,heat,cool


@njit(inline='always')
def power(parameters,temperature,heat,cool,fan,mode):
    supply=24-12*cool if mode else 21+19*heat
    net=.55*1.006*fan*(supply-temperature)
    return parameters[7]*max(net,0)/.9,parameters[8]*max(-net,0)/3,parameters[9]*fan**3+parameters[10]*fan


@njit(cache=True)
def control_history(temperature,actions,settings,initial_fan):
    history=np.empty((len(temperature),5));ih=0.;ic=0.;f1=initial_fan;f2=initial_fan;mode=0.
    history[0]=np.array([ih,ic,f1,f2,mode])
    for k in range(len(actions)):
        for sub in range(30):
            current=temperature[k]+(temperature[k+1]-temperature[k])*(sub+.5)/30
            ih,ic,f1,f2,mode,_,_=terminal_step(ih,ic,f1,f2,mode,current,actions[k],settings[k])
        history[k+1]=np.array([ih,ic,f1,f2,mode])
    return history


@njit(cache=True)
def observer(parameters,a,b,temperature,powers,exog,history):
    states=np.empty((len(temperature),7));mass=temperature[0]
    for k in range(len(temperature)):
        states[k,0]=temperature[k];states[k,1]=mass;states[k,2:]=history[k]
        if k<len(exog):
            mean=(powers[k]+powers[k+1])/2
            net=.9*mean[0]-3*mean[1]
            inputs=np.array([exog[k,0],net,exog[k,1]/1000,exog[k,5],exog[k,6]])
            predicted=a@states[k,:2]+b@inputs
            mass=predicted[1]+.2*(temperature[k+1]-predicted[0])
    return states


@njit(cache=True)
def rollout(parameters,a,b,states,initial_power,actions,exog,settings,indices):
    result=np.empty((len(states),len(indices),7))
    for j in range(len(states)):
        air,mass,ih,ic,f1,f2,mode=states[j]
        gas,cooling,fan=initial_power[j];eg=0.;ec=0.;ef=0.;index=0
        for h in range(indices[-1]):
            oldg,oldc,oldf=gas,cooling,fan
            x=exog[j,h];ambient=x[0];solar=x[1]/1000;conv=x[5];rad=x[6]
            base0=b[0,0]*ambient+b[0,2]*solar+b[0,3]*conv+b[0,4]*rad
            base1=b[1,0]*ambient+b[1,2]*solar+b[1,3]*conv+b[1,4]*rad
            for _ in range(30):
                ih,ic,f1,f2,mode,heat,cool=terminal_step(ih,ic,f1,f2,mode,air,actions[j,h],settings[j,h])
                g,c,f=power(parameters,air,heat,cool,f2,mode)
                net=(.9*(gas+g)-3*(cooling+c))/2
                predicted=a[0,0]*air+a[0,1]*mass+base0+b[0,1]*net
                g,c,f=power(parameters,predicted,heat,cool,f2,mode)
                net=(.9*(gas+g)-3*(cooling+c))/2
                new_air=a[0,0]*air+a[0,1]*mass+base0+b[0,1]*net
                new_mass=a[1,0]*air+a[1,1]*mass+base1+b[1,1]*net
                air,mass=new_air,new_mass
                gas,cooling,fan=power(parameters,air,heat,cool,f2,mode)
            # Match the original 30-second metering convention, not hidden 1s energy labels.
            eg+=(oldg+gas)/2*nm.DH;ec+=(oldc+cooling)/2*nm.DH;ef+=(oldf+fan)/2*nm.DH
            if h+1==indices[index]:
                result[j,index]=np.array([air,gas,cooling,fan,eg,ec,ef]);index+=1
    return result


def history(e):return control_history(e['temperature'],e['actions'],e['settings'],float(e['fan'][0]))


def _prediction(parameters,e,origins,seconds,controls):
    a30,b30=operators(parameters,30)
    states=observer(parameters,a30,b30,e['temperature'],e['power'],e['exog'],controls)
    indices=np.asarray(seconds,dtype=np.int64)//30
    if np.any(indices<1) or np.any(np.diff(indices)<=0) or np.any(np.asarray(seconds)%30):raise ValueError('Invalid native horizons')
    if np.any(origins<1) or np.any(origins+indices[-1]>len(e['actions'])):raise ValueError('Origin outside episode')
    at=origins[:,None]+np.arange(indices[-1]);a,b=operators(parameters,1)
    return rollout(parameters,a,b,states[origins],e['power'][origins],e['actions'][at],e['exog'][at],e['settings'][at],indices)


def prediction(parameters,e,origins,seconds):return _prediction(parameters,e,origins,seconds,history(e))


def fit(episodes,start,config,reference):
    if not episodes or any(e['role']!='train' for e in episodes):raise ValueError('Only registered training roles may fit final candidate')
    seconds=reference['horizons_seconds'];controls=[history(e) for e in episodes]
    origins=[nm.origin_indices(e,config['training_stride_seconds']) for e in episodes]
    targets=[nm.truth(e,k,seconds) for e,k in zip(episodes,origins)]
    scale=np.array(reference['power_scales_kW']);duration=np.array(seconds)/3600
    def objective(log_parameters):
        parameters=np.exp(log_parameters);errors=[]
        for e,k,y,c in zip(episodes,origins,targets,controls):
            error=_prediction(parameters,e,k,seconds,c)-y
            errors.extend([error[:,:,0].ravel(),(error[:,:,1:4]/scale).ravel(),(error[:,:,4:7]/duration[None,:,None]/scale).ravel()])
        return np.concatenate(errors)
    lower=[.1,.01,.005,.1,.001,.001,.1,.5,.5,.001,.00001]
    upper=[50,10,20,200,100,100,3,1.5,1.5,1,.1]
    result=least_squares(objective,np.log(np.clip(start,lower,upper)),bounds=(np.log(lower),np.log(upper)),
        max_nfev=config['fit_max_nfev'],diff_step=1e-4)
    return {'parameters':np.exp(result.x),'diagnostics':{'success':bool(result.success),'message':result.message,
        'cost':float(result.cost),'nfev':int(result.nfev),'training_origins':sum(len(k) for k in origins),
        'training_rows':sum(len(e['actions']) for e in episodes),'horizons_seconds':seconds},
        'source_prior':{'k':.1,'Ti_seconds':120.,'Ni':.9,'two_flow_poles_per_second':[-RATE,-RATE]},'oracle_fields_used':False}
