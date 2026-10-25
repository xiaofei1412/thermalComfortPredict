"""Separate interval-average energy from endpoint actuator memory."""

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from . import office_model as om


def lagged_flow(parameters,nominal,previous,change,cap):
    gain,bias,tau,impulse=parameters
    active=(nominal>0)|(previous>.01)
    target=np.where(active,np.clip(gain*nominal+bias+impulse*change,0,cap),0)
    decay=np.exp(-900/tau);weight=tau/900*(1-decay)
    # Both average and endpoint are convex mixtures; no free autoregressive gain.
    mean=target+(previous-target)*weight
    end=target+(previous-target)*decay
    return mean,end


def bounded_fit(x,y,lower,upper):
    result=lsq_linear(x,y,bounds=(lower,upper))
    if not result.success:raise RuntimeError("Bounded terminal fit failed")
    return result.x


def fan_x(previous,heat,cool):
    return np.column_stack([om.fan_features(previous,heat,cool),np.ones(len(heat))])


def fit_dynamic(base,episodes):
    if any(e["role"]!="train" for e in episodes):raise ValueError("Only training episodes may fit terminal dynamics")
    model={**base,"family":"dynamic"};features=[[],[]];targets=[[],[]];ends=[[],[]]
    mean_fan=[];end_fan=[];mean_y=[];end_y=[]
    for e in episodes:
        states=om.observer_states(base,e)[:-1]
        previous=np.vstack([e["initial_hvac"],e["end_hvac"][:-1]])
        old_actions=np.r_[e["initial_action"],e["actions"][:-1]]
        settings=om.cooling_setpoint(e["times"])
        nominal=om.terminal_heat(base,states,e["actions"],e["exog"],settings)
        changes=np.column_stack([e["actions"]-old_actions,om.cooling_setpoint(e["times"]-900)-settings])
        for j in range(2):
            features[j].append(np.column_stack([nominal[16:,j],previous[16:,j],changes[16:,j]]))
            targets[j].append(e["hvac"][16:,j]);ends[j].append(e["end_hvac"][16:,j])
        mean_fan.append(fan_x(previous[16:,2],e["hvac"][16:,0],e["hvac"][16:,1]))
        end_fan.append(fan_x(previous[16:,2],e["end_hvac"][16:,0],e["end_hvac"][16:,1]))
        mean_y.append(e["hvac"][16:,2]);end_y.append(e["end_hvac"][16:,2])
    model["lag_parameters"]=[]
    for j in range(2):
        x=np.concatenate(features[j]);y=np.concatenate(targets[j]);end=np.concatenate(ends[j])
        def objective(parameters):
            predicted_mean,predicted_end=lagged_flow(parameters,x[:,0],x[:,1],x[:,2],model["thermal_cap_kW"])
            return np.r_[predicted_mean-y,predicted_end-end]
        result=least_squares(objective,[1,0,120,.1],bounds=([.1,-2,10,0],[2,2,1800,5]))
        if not result.success:raise RuntimeError("Terminal lag fit failed")
        model["lag_parameters"].append(result.x)
    for name,xs,ys in [("mean_fan",mean_fan,mean_y),("end_fan",end_fan,end_y)]:
        model[name]=bounded_fit(np.concatenate(xs),np.concatenate(ys),np.zeros(8),[1,*([np.inf]*6),.01])
    return model


def rollout(model,initial,hvac,actions,old_actions,exog,settings,old_settings):
    """[air, mass, mean heat/cool/fan, electric kWh, gas kWh, endpoint heat/cool/fan]."""
    if model["family"]=="static":
        p=om.rollout(model,initial,hvac,actions,exog,settings)
        return np.concatenate([p,p[:,:,2:5]],axis=2)
    state=np.array(initial,copy=True);flow=np.array(hvac,copy=True);outputs=[]
    previous_action=np.array(old_actions,copy=True);previous_setting=np.array(old_settings,copy=True)
    for k in range(actions.shape[1]):
        nominal=om.terminal_heat(model,state,actions[:,k],exog[:,k],settings[:,k])
        changes=np.column_stack([actions[:,k]-previous_action,previous_setting-settings[:,k]])
        mean=[];end=[]
        for j in range(2):
            q,qe=lagged_flow(model["lag_parameters"][j],nominal[:,j],flow[:,j],changes[:,j],model["thermal_cap_kW"])
            mean.append(q);end.append(qe)
        heat,cool=mean;end_heat,end_cool=end
        fan=np.clip(fan_x(flow[:,2],heat,cool)@model["mean_fan"],0,model["fan_cap_kW"])
        end_fan=np.clip(fan_x(flow[:,2],end_heat,end_cool)@model["end_fan"],0,model["fan_cap_kW"])
        state=om.thermal_step(model,state,heat-cool,exog[:,k])
        flow=np.column_stack([end_heat,end_cool,end_fan])
        outputs.append(np.column_stack([state,heat,cool,fan,.25*(cool/model["cooling_COP"]+fan),.25*heat/model["heat_efficiency"],flow]))
        previous_action=actions[:,k];previous_setting=settings[:,k]
    result=np.stack(outputs,axis=1)
    if not np.isfinite(result).all() or (result[:,:,2:]<0).any():raise ValueError("Invalid terminal prediction")
    return result


def episode_rollout(model,e,origins,horizon,future=None):
    states=om.observer_states(model,e)
    actions=np.array([e["actions"][k:k+horizon] for k in origins])
    if future is None:future=np.array([e["exog"][k:k+horizon] for k in origins])
    settings=np.array([om.cooling_setpoint(e["times"][k:k+horizon]) for k in origins])
    return rollout(model,states[origins],e["end_hvac"][origins-1],actions,e["actions"][origins-1],future,settings,
        om.cooling_setpoint(e["times"][origins]-900))


def regime_indices(hvac,start,horizon,config):
    """Causal grouping: current measured endpoint and the known future schedule only."""
    hvac=np.atleast_2d(hvac);start=np.broadcast_to(start,(len(hvac),))
    heat=hvac[:,0]>config["active_threshold_kW"];cool=hvac[:,1]>config["active_threshold_kW"]
    index=np.where(heat,0,np.where(cool,1,2))
    transitions=np.any(om.cooling_setpoint(start[:,None]+np.arange(horizon+1)*900)!=om.cooling_setpoint(start)[:,None],axis=1)
    index=np.where(transitions|(heat&cool),3,index)
    return index


def calibrate(model,episodes,config,expected_role="calibration"):
    if expected_role not in ["selection","calibration"] or any(e["role"]!=expected_role for e in episodes):
        raise ValueError("Margin data role mismatch")
    errors=[];regimes=[]
    for e in episodes:
        starts=np.arange(16,len(e["actions"])-32+1,config["origin_stride"])
        p=episode_rollout(model,e,starts,32)
        errors.append(p[:,:,0]-e["temperature"][starts[:,None]+np.arange(1,33)])
        regimes.append(np.column_stack([regime_indices(e["end_hvac"][starts-1],e["times"][starts],h,config) for h in range(1,33)]))
    error=np.concatenate(errors);group=np.concatenate(regimes);lower=np.zeros((4,32));upper=lower.copy();counts=np.zeros((4,32),int);fallback=np.zeros((4,32),bool)
    for r in range(4):
        for h in range(32):
            selected=error[group[:,h]==r,h];counts[r,h]=len(selected)
            if len(selected)<config["minimum_regime_origins"]:selected=error[:,h];fallback[r,h]=True
            lower[r,h]=max(0,float(np.quantile(selected,config["margin_quantile"])))+config["sensor_margin_K"]
            upper[r,h]=max(0,float(np.quantile(-selected,config["margin_quantile"])))+config["sensor_margin_K"]
    return {"lower_K":lower.tolist(),"upper_K":upper.tolist(),"counts":counts.tolist(),"pooled_fallback":fallback.tolist(),"origins":len(error)}


def margins_for(margins,hvac,start,horizon,config):
    index=np.array([regime_indices(hvac,start,h,config)[0] for h in range(1,horizon+1)])
    return np.array(margins["lower_K"])[index,np.arange(horizon)],np.array(margins["upper_K"])[index,np.arange(horizon)]
