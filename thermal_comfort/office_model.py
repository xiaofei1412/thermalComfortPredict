"""Few-shot air-node thermal model with heating, cooling and distinct energy carriers."""

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cooling_setpoint(times):
    hour = np.asarray(times)%86400/3600
    return np.where((hour>=8)&(hour<18),24.,30.)


def discretize(parameters):
    rz,rm,ca,cm,solar_air,solar_mass,gain = parameters
    a = np.array([[-(1/rz+1/rm)/ca,1/rm/ca],[1/rm/cm,-1/rm/cm]])
    b = np.array([[1/rz/ca,1/ca,solar_air/ca,gain/ca,0],[0,0,solar_mass/cm,0,gain/cm]])
    # Interval-average heat and a fast air node: implicit energy balance, not ZOH input physics.
    ad = np.linalg.inv(np.eye(2)-.25*a)
    return ad,ad@b*.25


def thermal_step(model,state,net_heat,exog):
    x = np.column_stack([exog[:,0],net_heat,exog[:,1]/1000,exog[:,5],exog[:,6]])
    result = state@model["A"].T+x@model["B"].T
    if model.get("residual_weight",0):
        result[:,0] += model["residual_weight"]*np.clip(model["residual"].predict(exog),-model["residual_cap_K"],model["residual_cap_K"])
    return result


def observe(model,state,temperature,hvac,exog):
    prediction = thermal_step(model,np.asarray(state)[None],np.array([hvac[0]-hvac[1]]),np.asarray(exog)[None])[0]
    return np.array([temperature,prediction[1]+model["observer_gain"]*(temperature-prediction[0])])


def observer_states(model,episode):
    states = [np.repeat(episode["temperature"][0],2)]
    for k,hvac in enumerate(episode["hvac"]):
        states.append(observe(model,states[-1],episode["temperature"][k+1],hvac,episode["exog"][k]))
    return np.array(states)


def fan_features(previous,heat,cool):
    return np.column_stack([previous,heat,cool,heat**2,cool**2,np.minimum(heat/.5,1),np.minimum(cool/.5,1)])


def rollout(model,initial,hvac,actions,exog,cool_settings):
    """Free run: [air C, latent C, heat kW, cool kW, fan kW, electric kWh, gas kWh]."""
    state = np.asarray(initial).copy(); flow = np.asarray(hvac).copy()
    outputs = []
    for k in range(actions.shape[1]):
        flow[:,:2] = terminal_heat(model,state,actions[:,k],exog[:,k],cool_settings[:,k])
        flow[:,2] = np.clip(fan_features(flow[:,2],flow[:,0],flow[:,1])@model["fan"],0,model["fan_cap_kW"])
        state = thermal_step(model,state,flow[:,0]-flow[:,1],exog[:,k])
        electric = .25*(flow[:,1]/model["cooling_COP"]+flow[:,2])
        gas = .25*flow[:,0]/model["heat_efficiency"]
        outputs.append(np.column_stack([state,flow,electric,gas]))
    result = np.stack(outputs,axis=1)
    if not np.isfinite(result).all() or (result[:,:,2:]<0).any():
        raise ValueError("Invalid office prediction")
    return result


def terminal_heat(model,state,actions,exog,cool_settings):
    passive = thermal_step(model,state,np.zeros(len(state)),exog)[:,0]
    heat_gain,cool_gain,heat_offset,cool_offset = model["terminal"]
    heat = heat_gain*np.maximum(actions+heat_offset-passive,0)/model["B"][0,1]
    cool = cool_gain*np.maximum(passive-(cool_settings-cool_offset),0)/model["B"][0,1]
    return np.clip(np.column_stack([heat,cool]),0,model["thermal_cap_kW"])


def episode_rollout(model,e,origins,horizon):
    states = observer_states(model,e)
    previous_hvac = e["hvac"][origins-1]
    actions = np.array([e["actions"][o:o+horizon] for o in origins])
    future = np.array([e["exog"][o:o+horizon] for o in origins])
    settings = np.array([cooling_setpoint(e["times"][o:o+horizon]) for o in origins])
    return rollout(model,states[origins],previous_hvac,actions,future,settings)


def fit(episodes,config):
    if any(e["role"]!="train" for e in episodes):
        raise ValueError("Only training episodes may fit office parameters")
    def objective(log_parameters):
        a,b = discretize(np.exp(log_parameters))
        model = {"A":a,"B":b,"observer_gain":config["observer_gain"]}
        errors = []
        for e in episodes:
            states = observer_states(model,e)
            origins = np.arange(16,len(e["actions"])-24+1,8)
            state = states[origins].copy()
            for h in range(24):
                state = thermal_step(model,state,e["hvac"][origins+h,0]-e["hvac"][origins+h,1],e["exog"][origins+h])
                if h in (0,3,15,23): errors.extend(state[:,0]-e["temperature"][origins+h+1])
        return np.array(errors)
    lower = [.1,.01,.005,.1,.001,.001,.1]; upper = [50,10,20,200,100,100,3.]
    # The 48 m² office has a fast air node; retain the slower initialisation as a train-only check.
    fits = [least_squares(objective,np.log(start),bounds=(np.log(lower),np.log(upper)),
        max_nfev=config["fit_max_nfev"]) for start in [[3,2,.0432,5,1,6,1],[3,.5,.3,5,1,6,1]]]
    result = min((r for r in fits if r.success),key=lambda r:r.cost,default=None)
    if result is None: raise RuntimeError("Office thermal fit did not converge")
    if not result.success: raise RuntimeError("Office thermal fit failed: "+result.message)
    parameters = np.exp(result.x); a,b = discretize(parameters)
    model = {"parameters":parameters,"A":a,"B":b,"observer_gain":config["observer_gain"],"residual_weight":0.,
        **{k:config[k] for k in ["thermal_cap_kW","fan_cap_kW","heat_efficiency","cooling_COP","residual_cap_K"]}}
    fans=[]; fan_y=[]; residual_x=[]; residual_y=[]; terminal_rows=[]
    for e in episodes:
        k = np.arange(1,len(e["actions"]))
        fans.append(fan_features(e["hvac"][k-1,2],e["hvac"][k,0],e["hvac"][k,1])); fan_y.append(e["hvac"][k,2])
        states = observer_states(model,e)
        terminal_rows.append((states[:-1][16:],e["actions"][16:],e["exog"][16:],cooling_setpoint(e["times"][16:]),
            e["temperature"][17:],e["hvac"][16:,:2]))
        p = thermal_step(model,states[:-1],e["hvac"][:,0]-e["hvac"][:,1],e["exog"])
        residual_x.append(e["exog"][16:]); residual_y.append(e["temperature"][17:]-p[16:,0])
    def terminal_objective(parameters):
        trial={**model,"terminal":parameters}; errors=[]
        for state,actions,x,settings,temperature,hvac in terminal_rows:
            q=terminal_heat(trial,state,actions,x,settings)
            p=thermal_step(trial,state,q[:,0]-q[:,1],x)
            errors.extend(p[:,0]-temperature)
            errors.extend((.2*(q-hvac)).ravel())
        return np.array(errors)
    terminal_fit=least_squares(terminal_objective,[1.,1.,0.,.05],bounds=([.1,.1,-.5,0.],[1.,1.,.5,1.]))
    if not terminal_fit.success:raise RuntimeError("Office terminal fit failed")
    model["terminal"]=terminal_fit.x
    fan_fit = lsq_linear(np.concatenate(fans),np.concatenate(fan_y),bounds=(np.zeros(7),[1,*([np.inf]*6)]))
    if not fan_fit.success: raise RuntimeError("Office fan fit failed")
    model["fan"] = fan_fit.x
    model["residual"] = make_pipeline(StandardScaler(),Ridge(alpha=config["residual_alpha"]))
    model["residual"].fit(np.concatenate(residual_x),np.concatenate(residual_y))
    support = np.concatenate([np.column_stack([e["temperature"][:-1],e["exog"][:,:2]]) for e in episodes])
    model["support_low"],model["support_high"] = support.min(axis=0),support.max(axis=0)
    model["fit_diagnostics"] = {"cost":float(result.cost),"nfev":int(result.nfev),"training_rows":sum(len(e["actions"]) for e in episodes),
        "initialisation_train_costs":[float(r.cost) for r in fits]}
    return model
