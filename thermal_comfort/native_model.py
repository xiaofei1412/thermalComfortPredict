"""Thirty-second thermal/PI/fan grey box with joint free-running identification."""

import numpy as np
from numba import njit
from scipy.optimize import least_squares

DT = 30.
DH = DT/3600
KP = .1
CP_FLOW = .55*1.006  # kW/K at nominal air flow; known FMU prior.


def operators(parameters):
    rz,rm,ca,cm,sa,sm,gain=parameters[:7]
    a=np.array([[-(1/rz+1/rm)/ca,1/rm/ca],[1/rm/cm,-1/rm/cm]])
    b=np.array([[1/rz/ca,1/ca,sa/ca,gain/ca,0],[0,0,sm/cm,0,gain/cm]])
    ad=np.linalg.inv(np.eye(2)-DH*a)
    return ad,ad@b*DH


@njit(cache=True)
def thermal(a,b,state,net,x):
    return a@state+b@np.array([x[0],net,x[1]/1000,x[5],x[6]])


@njit(cache=True)
def clipped(value,lo=0.,hi=1.):
    return min(hi,max(lo,value))


@njit(cache=True)
def powers(parameters,temperature,heat,cool,fan,mode):
    supply=24-12*cool if mode else 21+19*heat
    net=CP_FLOW*fan*(supply-temperature)
    return np.array([parameters[10]*max(net,0)/.9,
                     parameters[11]*max(-net,0)/3,
                     parameters[12]*fan**3+parameters[13]*fan])


@njit(cache=True)
def transition(parameters,a,b,state,power,action,x,setting):
    """state=[air,mass,heating integral,cooling integral,fan fraction,cooling mode]."""
    temperature=state[0];eh=action-temperature;ec=temperature-setting
    mode=state[5]
    if ec>0:mode=1.
    elif ec < -1:mode=0.
    raw_h=KP*eh+state[2];raw_c=KP*ec+state[3]
    ih=state[2]+DT/parameters[7]*(KP*eh+(clipped(raw_h)-raw_h)/parameters[8])
    ic=state[3]+DT/parameters[7]*(KP*ec+(clipped(raw_c)-raw_c)/parameters[8])
    heat=clipped(KP*eh+ih);cool=clipped(KP*ec+ic)
    fan=clipped(state[4]+(1-np.exp(-DT/parameters[9]))*(clipped(heat+cool)-state[4]))
    end=powers(parameters,temperature,heat,cool,fan,mode)
    mean=(power+end)/2
    zone=thermal(a,b,state[:2],.9*mean[0]-3*mean[1],x)
    end=powers(parameters,zone[0],heat,cool,fan,mode)
    mean=(power+end)/2
    zone=thermal(a,b,state[:2],.9*mean[0]-3*mean[1],x)
    return np.array([zone[0],zone[1],ih,ic,fan,mode]),end


@njit(cache=True)
def observer(parameters,a,b,temperature,power,fan,actions,exog,settings,previous_actions,previous_settings):
    """Causal state reconstruction. Only preceding intervals update the mass state."""
    states=np.zeros((len(temperature),6));mass=temperature[0];ih=0.;ic=0.;mode=0.
    for k in range(len(temperature)):
        t=temperature[k];f=clipped(fan[k]);action=previous_actions[k]
        setting=previous_settings[k];ec=t-setting
        if ec>0:mode=1.
        elif ec < -1:mode=0.
        if f>1e-4:
            supply=t+(.9*power[k,0]/parameters[10]-3*power[k,1]/parameters[11])/(CP_FLOW*f)
            if mode:
                cool=clipped((24-supply)/12);heat=clipped(f-cool)
            else:
                heat=clipped((supply-21)/19);cool=clipped(f-heat)
            ih=heat-KP*(action-t);ic=cool-KP*ec
        states[k]=np.array([t,mass,ih,ic,f,mode])
        if k<len(actions):
            mean=(power[k]+power[k+1])/2
            predicted=thermal(a,b,states[k,:2],.9*mean[0]-3*mean[1],exog[k])
            # Correct only when the next observation has actually arrived.
            mass=predicted[1]+.2*(temperature[k+1]-predicted[0])
            provisional,_=transition(parameters,a,b,states[k],power[k],actions[k],exog[k],settings[k])
            ih,ic=provisional[2],provisional[3]
    return states


@njit(cache=True)
def free_run(parameters,a,b,states,initial_power,actions,exog,settings,indices):
    """Return [air C, gas/cooling/fan endpoint kW, gas/cooling/fan cumulative kWh]."""
    result=np.empty((len(states),len(indices),7))
    for j in range(len(states)):
        state=states[j].copy();power=initial_power[j].copy();energy=np.zeros(3);n=0
        for h in range(indices[-1]):
            state,end=transition(parameters,a,b,state,power,actions[j,h],exog[j,h],settings[j,h])
            energy+=(power+end)/2*DH;power=end
            if h+1==indices[n]:
                result[j,n,0]=state[0];result[j,n,1:4]=power;result[j,n,4:7]=energy;n+=1
    return result


def origin_indices(episode,stride_seconds,horizon_seconds=21600):
    return np.arange(480,len(episode["actions"])-int(horizon_seconds/DT)+1,int(stride_seconds/DT))


def prediction(parameters,episode,origins,seconds):
    indices=np.asarray(seconds,dtype=np.int64)//30
    if np.any(indices<1) or np.any(np.diff(indices)<=0) or np.any(np.asarray(seconds)%30):
        raise ValueError("Horizons must be increasing positive multiples of 30 seconds")
    if np.any(origins<1) or np.any(origins+indices[-1]>len(episode["actions"])):
        raise ValueError("Prediction origin outside observed episode")
    a,b=operators(parameters)
    states=observer(parameters,a,b,episode["temperature"],episode["power"],episode["fan"],
        episode["actions"],episode["exog"],episode["settings"],np.r_[episode["initial_action"],episode["actions"]],
        np.r_[episode["initial_setting"],episode["settings"]])
    at=origins[:,None]+np.arange(indices[-1])
    return free_run(parameters,a,b,states[origins],episode["power"][origins],episode["actions"][at],
        episode["exog"][at],episode["settings"][at],indices)


def truth(episode,origins,seconds):
    indices=np.asarray(seconds,dtype=np.int64)//30
    cumulative=np.vstack([np.zeros(3),np.cumsum((episode["power"][:-1]+episode["power"][1:])/2*DH,axis=0)])
    at=origins[:,None]+indices
    return np.concatenate([episode["temperature"][at,None],episode["power"][at],
        cumulative[at]-cumulative[origins,None]],axis=2)


def fit(episodes,start,config,joint):
    if not episodes or any(e["role"]!="train" for e in episodes):
        raise ValueError("Only training episodes may identify native parameters")
    seconds=config["horizons_seconds"] if joint else [30]
    stride=config["training_stride_seconds"] if joint else 300
    origins=[origin_indices(e,stride,seconds[-1]) for e in episodes]
    targets=[truth(e,k,seconds) for e,k in zip(episodes,origins)]
    scale=np.array(config["power_scales_kW"])
    duration=np.array(seconds)/3600
    def objective(log_parameters):
        parameters=np.exp(log_parameters);errors=[]
        for e,k,y in zip(episodes,origins,targets):
            p=prediction(parameters,e,k,seconds);error=p-y
            errors.extend([error[:,:,0].ravel(),(error[:,:,1:4]/scale).ravel(),
                (error[:,:,4:7]/duration[None,:,None]/scale).ravel()])
        return np.concatenate(errors)
    lower=[.1,.01,.005,.1,.001,.001,.1,30,.2,5,.5,.5,.001,.00001]
    upper=[50,10,20,200,100,100,3,600,2,300,1.5,1.5,1,.1]
    result=least_squares(objective,np.log(np.clip(start,lower,upper)),bounds=(np.log(lower),np.log(upper)),
        max_nfev=config["fit_max_nfev"],diff_step=1e-4)
    parameters=np.exp(result.x)
    if not np.isfinite(parameters).all():raise ValueError("Nonfinite native identification")
    return {"parameters":parameters,"joint":joint,"diagnostics":{"success":bool(result.success),
        "message":result.message,"nfev":int(result.nfev),"cost":float(result.cost),
        "origins":sum(len(k) for k in origins),"horizons_seconds":seconds}}
