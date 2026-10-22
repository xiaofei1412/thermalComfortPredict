"""Positive 2R2C thermal network, bounded actuator and nonnegative power model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import expm
from scipy.optimize import least_squares, nnls
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dynamics import common_origins, load_episodes
from .simulation import digest, save_json


def load_heat_episodes(root, config):
    episodes, splits = load_episodes(root, config)
    for episode in episodes.values():
        native = pd.read_csv(root / (episode["id"] + "__excitation") / "native.csv.gz")
        if native.reaQFloHea_y.min() < -1e-8:
            raise ValueError("Heating-only model encountered negative delivered heat")
        cumulative = cumulative_trapezoid(native.reaQFloHea_y, native.time, initial=0)
        boundaries = np.append(episode["times"], episode["times"][-1] + 900)
        episode["heat"] = np.diff(np.interp(boundaries, native.time, cumulative)) / 900000
        episode["initial_heat"] = float(native.reaQFloHea_y.iloc[0] / 1000)
    return episodes, splits


def discretize(parameters):
    """R in K/kW, C in kWh/K, effective solar area in m², person gain in kW."""
    rz, rm, cz, cm, solar, person = parameters
    a = np.array([[-(1/rz + 1/rm)/cz, 1/rm/cz], [1/rm/cm, -1/rm/cm]])
    # Inputs: outdoor °C, delivered heat kW, irradiance kW/m², people.
    b = np.array([[1/rz/cz, 0, solar/cz, person/cz], [0, 1/cm, 0, 0]])
    discrete_a = expm(a * .25)
    discrete_b = np.linalg.solve(a, (discrete_a - np.eye(2)) @ b)
    return discrete_a, discrete_b


def thermal_step(model, state, heat, exog):
    inputs = np.column_stack([exog[:, 0], heat, exog[:, 1] / 1000, exog[:, 2]])
    result = state @ model["A"].T + inputs @ model["B"].T
    if model.get("residual_weight", 0):
        residual = np.clip(model["residual"].predict(exog), -model["residual_cap_K"], model["residual_cap_K"])
        result[:, 0] += model["residual_weight"] * residual
    return result


def initial_state(model, temperature, previous_heat):
    return np.array([temperature, temperature + model["parameters"][1] * previous_heat])


def observe(model, state, actual_temperature, actual_heat, exog):
    predicted = thermal_step(model, np.asarray(state)[None], np.array([actual_heat]), np.asarray(exog)[None])[0]
    error = actual_temperature - predicted[0]
    return np.array([actual_temperature, predicted[1] + model["observer_gain"] * error])


def observer_states(model, episode):
    states = [initial_state(model, episode["states"][0, 0], episode["initial_heat"])]
    for k, heat in enumerate(episode["heat"]):
        states.append(observe(model, states[-1], episode["states"][k + 1, 0], heat, episode["exog"][k]))
    return np.array(states)


def heat_prediction(model, previous_heat, error, previous_error, temperature, outdoor):
    x = np.column_stack([previous_heat, error, error - previous_error, temperature - outdoor, np.ones_like(error)])
    return np.clip(x @ model["actuator"], 0, model["heat_max_kW"])


def energy_prediction(model, heat, outdoor):
    if "idle_power_kW" in model:
        gate = np.minimum(heat / model["power_transition_kW"], 1)
        x = np.column_stack([gate, heat, heat * np.maximum(20 - outdoor, 0) / 20])
        return .25 * (model["idle_power_kW"] + x @ model["power"])
    x = np.column_stack([np.ones_like(heat), heat, heat * np.maximum(20 - outdoor, 0) / 20])
    return .25 * (x @ model["power"])


def rollout(model, initial, previous_heat, previous_error, actions, exog):
    """Free-running [batch,horizon,4]: operative, latent mass, heat kW, energy kWh."""
    state = np.asarray(initial).copy()
    q, error = np.asarray(previous_heat).copy(), np.asarray(previous_error).copy()
    outputs = []
    for k in range(actions.shape[1]):
        new_error = actions[:, k] - state[:, 0]
        q = heat_prediction(model, q, new_error, error, state[:, 0], exog[:, k, 0])
        state = thermal_step(model, state, q, exog[:, k])
        energy = energy_prediction(model, q, exog[:, k, 0])
        outputs.append(np.column_stack([state, q, energy]))
        error = new_error
    result = np.stack(outputs, axis=1)
    if not np.isfinite(result).all() or (result[:, :, 3] < 0).any():
        raise ValueError("Invalid grey-box rollout")
    return result


def episode_rollout(model, episode, origins, horizon):
    estimated = observer_states(model, episode)
    actions = np.stack([episode["actions"][k:k+horizon] for k in origins])
    exog = np.stack([episode["exog"][k:k+horizon] for k in origins])
    previous_heat = episode["heat"][origins-1]
    previous_error = episode["actions"][origins-1] - episode["states"][origins-1, 0]
    return rollout(model, estimated[origins], previous_heat, previous_error, actions, exog)


def fit(episodes, config):
    train = [e for e in episodes.values() if e["split"] == "train"]
    if not train:
        raise ValueError("No training episodes")
    def objective(log_parameters):
        parameters = np.exp(log_parameters)
        a, b = discretize(parameters)
        model = {"parameters": parameters, "A": a, "B": b, "observer_gain": config["observer_mass_gain"]}
        errors = []
        for episode in train:
            estimated = observer_states(model, episode)
            origins = np.arange(16, len(episode["actions"]) - 16, 16)
            state = estimated[origins].copy()
            for h in range(16):
                state = thermal_step(model, state, episode["heat"][origins+h], episode["exog"][origins+h])
                if h in (0, 3, 7, 15):
                    errors.extend(state[:, 0] - episode["states"][origins+h+1, 0])
        return np.array(errors)
    lower = np.array([.1, .01, .5, 1., .1, .001])
    upper = np.array([20., 5., 100., 500., 100., .5])
    result = least_squares(objective, np.log([5., .5, 3., 15., 8., .1]),
                           bounds=(np.log(lower), np.log(upper)), max_nfev=config["fit_max_nfev"])
    if not result.success:
        raise RuntimeError(f"RC parameter fit did not converge: {result.message}")
    parameters = np.exp(result.x)
    a, b = discretize(parameters)
    model = {"parameters": parameters, "A": a, "B": b, "observer_gain": config["observer_mass_gain"],
             "heat_max_kW": config["heat_max_kW"], "residual_weight": 0.0}
    xs, qs, px, py, rx, ry = [], [], [], [], [], []
    for episode in train:
        k = np.arange(1, len(episode["actions"]))
        error = episode["actions"] - episode["states"][:-1, 0]
        xs.append(np.column_stack([episode["heat"][k-1], error[k], error[k]-error[k-1],
                                   episode["states"][k, 0]-episode["exog"][k, 0], np.ones(len(k))]))
        qs.append(episode["heat"][k])
        heat, outdoor = episode["heat"], episode["exog"][:, 0]
        px.append(np.column_stack([np.ones_like(heat), heat, heat*np.maximum(20-outdoor, 0)/20]))
        py.append(episode["states"][1:, 3] / .25)
        estimated = observer_states(model, episode)
        prediction = thermal_step(model, estimated[:-1], heat, episode["exog"])
        rx.append(episode["exog"][16:])
        ry.append(episode["states"][17:, 0] - prediction[16:, 0])
    x, q = np.concatenate(xs), np.concatenate(qs)
    actuator = least_squares(lambda theta: np.clip(x @ theta, 0, config["heat_max_kW"]) - q,
                             [.9, .5, 1., .05, 0.], bounds=([0, 0, 0, 0, -10], [1, 10, 10, 2, 10]))
    if not actuator.success:
        raise RuntimeError("Actuator fit did not converge")
    model["actuator"] = actuator.x
    model["power"] = nnls(np.concatenate(px), np.concatenate(py))[0]
    residual_x, residual_y = np.concatenate(rx), np.concatenate(ry)
    model["residual"] = make_pipeline(StandardScaler(), Ridge(alpha=config["residual_alpha"])).fit(residual_x, residual_y)
    model["residual_cap_K"] = float(np.quantile(abs(residual_y), .95))
    support = np.concatenate([np.column_stack([e["states"][:-1, 0], e["exog"][:, :2]]) for e in train])
    model["support_low"], model["support_high"] = support.min(axis=0), support.max(axis=0)
    model["fit_diagnostics"] = {"rc_cost": float(result.cost), "rc_nfev": result.nfev,
                                "actuator_RMSE_kW": float(np.sqrt(np.mean(actuator.fun**2)))}
    return model


def metrics(prediction, episode, origins, horizons):
    rows = []
    for h in horizons:
        error = prediction[:, h-1, 0] - episode["states"][origins+h, 0]
        truth_energy = np.array([episode["states"][k+1:k+h+1, 3].sum() for k in origins])
        energy_error = prediction[:, :h, 3].sum(axis=1) - truth_energy
        rows.append({"horizon_steps": h, "origins": len(origins),
                     "temperature_RMSE_C": float(np.sqrt(np.mean(error**2))),
                     "temperature_abs_p95_C": float(np.quantile(abs(error), .95)),
                     "cumulative_energy_RMSE_kWh": float(np.sqrt(np.mean(energy_error**2))),
                     "negative_energy_fraction": float(np.mean(prediction[:, :h, 3] < 0))})
    return rows


def run(config_path, source, output):
    config = json.loads(config_path.read_text())
    dynamics_config = json.loads(Path(config["dynamics_config"]).read_text())
    episodes, splits = load_heat_episodes(source, dynamics_config)
    output.mkdir(parents=True, exist_ok=False)
    model = fit(episodes, config)
    validation = next(e for e in episodes.values() if e["split"] == "validation")
    origins = common_origins(validation, dynamics_config["model"])
    selection_origins = origins[origins + 16 <= config["validation_selection_days"] * 96]
    calibration_origins = np.arange(config["validation_calibration_start_day"] * 96,
                                     len(validation["actions"]) - config["horizon_steps"] + 1, 4)
    if len(selection_origins) == 0 or len(calibration_origins) == 0 or selection_origins.max()+16 >= calibration_origins.min():
        raise ValueError("Validation selection and margin calibration must be separated in time")
    scores = []
    for weight in config["residual_weights"]:
        candidate = {**model, "residual_weight": weight}
        prediction = episode_rollout(candidate, validation, selection_origins, 16)
        score = float(np.mean([m["temperature_RMSE_C"] for m in metrics(prediction, validation, selection_origins, [1,4,8])]))
        scores.append({"weight": weight, "validation_score_C": score})
        print(f"residual weight={weight}: validation score={score:.6f}", flush=True)
    model["residual_weight"] = min(scores, key=lambda s: s["validation_score_C"])["weight"]
    prediction = episode_rollout(model, validation, calibration_origins, config["horizon_steps"])
    errors = np.column_stack([prediction[:, h-1, 0] - validation["states"][calibration_origins+h, 0]
                              for h in range(1, config["horizon_steps"]+1)])
    model["margin_K"] = np.quantile(abs(errors), config["margin_quantile"], axis=0) + config["sensor_margin_K"]
    joblib.dump(model, output / "model.joblib")
    save_json(output / "selection.json", {"candidates": scores, "selected_weight": model["residual_weight"],
              "margin_K": model["margin_K"].tolist(), "margin_source": validation["id"],
              "selection_origin_indices": selection_origins.tolist(), "calibration_origin_indices": calibration_origins.tolist(),
              "parameters_Rza_Rzm_Cz_Cm_solar_person": model["parameters"].tolist(),
              "actuator": model["actuator"].tolist(), "nonnegative_power": model["power"].tolist(),
              "residual_cap_K": model["residual_cap_K"], "fit_diagnostics": model["fit_diagnostics"]})
    rows = []
    for episode in episodes.values():
        if episode["split"] == "train":
            continue
        origins = common_origins(episode, dynamics_config["model"])
        for name, weight in [("rc", 0.0), ("rc_residual", model["residual_weight"])]:
            prediction = episode_rollout({**model, "residual_weight": weight}, episode, origins, 16)
            rows.extend([{"episode": episode["id"], "split": episode["split"], "model": name, **m}
                         for m in metrics(prediction, episode, origins, [1,4,8,16])])
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    save_json(output / "split_manifest.json", splits)
    save_json(output / "manifest.json", {"config": config, "source": str(source),
              "source_summary_sha256": digest(source / "summary.json"), "config_sha256": digest(config_path),
              "code_sha256": digest(Path(__file__)),
              "artifacts_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mpc.json"))
    parser.add_argument("--source", type=Path, default=Path("outputs/excitation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/greybox-mpc"))
    args = parser.parse_args()
    run(args.config, args.source, args.output)
