"""Episode-isolated autoregressive dynamics and free-running rollout baselines."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .excitation import validate_plan
from .simulation import digest, save_json


STATE_NAMES = ["operative_C", "return_C", "supply_C", "previous_energy_kWh"]
EXOG_NAMES = ["outdoor_C", "solar_W_m2", "occupancy", "sin_day", "cos_day"]
MEASUREMENTS = ["reaTZon_y", "reaTRet_y", "reaTSup_y"]


def load_episodes(root, config):
    manifest = json.loads((root / "manifest.json").read_text())
    collected = manifest["config"]
    validate_plan(collected)
    if config["step_s"] != 900:
        raise ValueError("These baseline metrics require 900-second control intervals")
    for key in ["episodes", "days", "warmup_days", "step_s", "version", "testcase", "excitation"]:
        if collected[key] != config[key]:
            raise ValueError(f"Collection plan differs from model config: {key}")
    summaries = json.loads((root / "summary.json").read_text())
    if len(summaries) != len(config["episodes"]):
        raise ValueError("Collection is incomplete")
    episodes, splits = {}, []
    for plan, summary in zip(config["episodes"], summaries):
        if summary["scenario"] != plan["id"] or summary["config"]["episode"] != plan:
            raise ValueError("Episode identity mismatch")
        directory = root / (plan["id"] + "__excitation")
        for name, expected in summary["sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Artifact checksum mismatch: {directory / name}")
        frame = pd.read_csv(directory / "transitions.csv.gz")
        records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        n = int(config["days"] * 86400 / config["step_s"])
        if len(frame) != n or len(records) != n or summary["transitions"] != n:
            raise ValueError("Incomplete episode")
        times = frame.time_start_s.to_numpy()
        expected_start = plan["start_day"] * 86400
        if not np.array_equal(times, expected_start + np.arange(n) * config["step_s"]):
            raise ValueError("Episode has gaps or incorrect time limits")
        states = np.empty((n + 1, 4))
        states[0, :3] = [records[0]["before"][name] - 273.15 for name in MEASUREMENTS]
        states[0, 3] = np.nan  # No observed previous interval at initialization; never used.
        for i, record in enumerate(records):
            if record["before"]["time"] != times[i] or record["after"]["time"] != times[i] + config["step_s"]:
                raise ValueError("JSONL times and transition table disagree")
            previous = np.array([record["before"][name] - 273.15 for name in MEASUREMENTS])
            if not np.array_equal(previous, states[i, :3]):
                raise ValueError("State continuity was broken")
            states[i + 1, :3] = [record["after"][name] - 273.15 for name in MEASUREMENTS]
            states[i + 1, 3] = frame.interval_energy_kWh.iloc[i]
        actions = frame.setpoint_requested_C.to_numpy()
        if not np.allclose(actions, frame.setpoint_applied_end_C, rtol=0, atol=1e-10):
            raise ValueError("Requested and applied setpoints differ")
        angle = 2 * np.pi * times / 86400
        exog = np.column_stack([frame.outdoor_start_C, frame.solar_start_W_m2,
                                frame.occupancy_start, np.sin(angle), np.cos(angle)])
        if not all(np.isfinite(a).all() for a in [states[1:], actions, exog]):
            raise ValueError("Non-finite modelling data")
        episodes[plan["id"]] = {"id": plan["id"], "split": plan["split"], "states": states,
                                "actions": actions, "exog": exog, "times": times}
        splits.append({**plan, "warmup_start_s": expected_start - config["warmup_days"] * 86400,
                       "start_s": expected_start, "end_s": expected_start + n * config["step_s"],
                       "transitions": n, "source_hashes": summary["sha256"]})
    return episodes, splits


def features(history, actions, exog):
    return np.concatenate([history.reshape(len(history), -1), actions, exog], axis=1)


def training_data(episodes, lag):
    xs, ys = [], []
    for episode in episodes:
        if episode["split"] != "train":
            raise ValueError("Fitting is restricted to training episodes")
        state, action, exog = episode["states"], episode["actions"], episode["exog"]
        # k >= lag excludes the unobserved previous energy in state[0].
        k = np.arange(lag, len(action))
        history = np.stack([state[i - lag + 1:i + 1] for i in k])
        action_history = np.stack([action[i - lag + 1:i + 1] for i in k])
        xs.append(features(history, action_history, exog[k]))
        target = state[k + 1].copy()
        target[:, :3] -= state[k, :3]
        ys.append(target)
    return np.concatenate(xs), np.concatenate(ys)


def fit_model(episodes, kind, lag, parameter, config):
    x, y = training_data(episodes, lag)
    if kind == "ridge":
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    elif kind == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=config["tree_estimators"], min_samples_leaf=parameter,
                                       max_depth=18, random_state=config["seed"], n_jobs=1)
    else:
        raise ValueError(f"Unknown model {kind}")
    estimator.fit(x, y)
    return {"kind": kind, "lag": lag, "parameter": parameter, "estimator": estimator,
            "training_rows": len(x), "state_names": STATE_NAMES, "exog_names": EXOG_NAMES}


def rollout(model, history, past_actions, future_actions, future_exog):
    """Predict using only initial measured history and supplied future inputs.

    Arrays: history [batch, lag, 4], past_actions [batch, lag-1],
    future_actions [batch, horizon], future_exog [batch, horizon, 5].
    No future observed states are accepted by this interface. Outputs are not clipped.
    """
    history, past_actions = history.copy(), past_actions.copy()
    if history.shape[1:] != (model["lag"], 4) or past_actions.shape != (len(history), model["lag"] - 1):
        raise ValueError("Wrong state/action history shape")
    if future_actions.shape != future_exog.shape[:2] or future_exog.shape[2] != len(EXOG_NAMES):
        raise ValueError("Wrong future input shape")
    predictions = []
    for step in range(future_actions.shape[1]):
        actions = np.column_stack([past_actions, future_actions[:, step]])
        if model["kind"] == "persistence":
            next_state = history[:, -1].copy()
        else:
            next_state = model["estimator"].predict(features(history, actions, future_exog[:, step])).copy()
            next_state[:, :3] += history[:, -1, :3]
        if not np.isfinite(next_state).all():
            raise ValueError("Non-finite rollout")
        predictions.append(next_state)
        history = np.concatenate([history[:, 1:], next_state[:, None]], axis=1)
        past_actions = actions[:, 1:]
    return np.stack(predictions, axis=1)


def rollout_episode(model, episode, origins, horizon, weather_mode="perfect"):
    lag = model["lag"]
    state, action, exog = episode["states"], episode["actions"], episode["exog"]
    history = np.stack([state[k - lag + 1:k + 1] for k in origins])
    past_actions = np.stack([action[k - lag + 1:k] for k in origins])
    future_actions = np.stack([action[k:k + horizon] for k in origins])
    future_exog = np.stack([exog[k:k + horizon] for k in origins])
    if weather_mode == "hold":
        future_exog[:, :, :2] = future_exog[:, :1, :2]
    elif weather_mode != "perfect":
        raise ValueError("Unknown weather information condition")
    prediction = rollout(model, history, past_actions, future_actions, future_exog)
    truth = np.stack([state[k + 1:k + horizon + 1] for k in origins])
    return prediction, truth


def score_rollouts(prediction, truth, horizons):
    rows = []
    for h in horizons:
        error = prediction[:, h - 1, 0] - truth[:, h - 1, 0]
        energy_error = prediction[:, :h, 3].sum(axis=1) - truth[:, :h, 3].sum(axis=1)
        rows.append({"horizon_steps": h, "horizon_hours": h / 4,
                     "origins": len(truth), "temperature_RMSE_C": float(np.sqrt(np.mean(error**2))),
                     "temperature_MAE_C": float(np.mean(abs(error))),
                     "temperature_abs_p95_C": float(np.quantile(abs(error), .95)),
                     "temperature_max_abs_C": float(abs(error).max()),
                     "return_RMSE_C": float(np.sqrt(np.mean((prediction[:, h - 1, 1] - truth[:, h - 1, 1])**2))),
                     "supply_RMSE_C": float(np.sqrt(np.mean((prediction[:, h - 1, 2] - truth[:, h - 1, 2])**2))),
                     "cumulative_energy_RMSE_kWh": float(np.sqrt(np.mean(energy_error**2))),
                     "cumulative_energy_MAE_kWh": float(np.mean(abs(energy_error))),
                     "negative_energy_fraction": float(np.mean(prediction[:, :h, 3] < 0))})
    return rows


def coverage_table(episodes):
    names = STATE_NAMES + ["setpoint_C"] + EXOG_NAMES[:3]
    arrays = {e["id"]: np.column_stack([e["states"][1:], e["actions"], e["exog"][:, :3]]) for e in episodes.values()}
    train = np.concatenate([arrays[e["id"]] for e in episodes.values() if e["split"] == "train"])
    low, high = train.min(axis=0), train.max(axis=0)
    rows = []
    for episode in episodes.values():
        values = arrays[episode["id"]]
        for i, name in enumerate(names):
            rows.append({"episode": episode["id"], "split": episode["split"], "variable": name,
                         "train_min": low[i], "train_max": high[i],
                         "observed_min": values[:, i].min(), "observed_max": values[:, i].max(),
                         "fraction_outside_train_range": np.mean((values[:, i] < low[i]) | (values[:, i] > high[i]))})
    return pd.DataFrame(rows)


def common_origins(episode, config):
    return np.arange(max(config["lags"] + [config["tree_lag"]]),
                     len(episode["actions"]) - max(config["horizons_steps"]) + 1,
                     config["origin_stride_steps"])


def spectral_radius(model):
    """Linear state-history recursion with externally fixed actions and disturbances."""
    lag = model["lag"]
    pipeline = model["estimator"]
    coefficient = pipeline[-1].coef_ / pipeline[0].scale_
    matrix = np.zeros((4 * lag, 4 * lag))
    matrix[:-4, 4:] = np.eye(4 * (lag - 1))
    matrix[-4:] = coefficient[:, :4 * lag]
    matrix[-4:-1, -4:-1] += np.eye(3)  # Temperature residuals; energy is a level.
    return float(np.max(abs(np.linalg.eigvals(matrix))))


def run(config_path, source, output):
    config = json.loads(config_path.read_text())
    episodes, splits = load_episodes(source, config)
    model_config = config["model"]
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "split_manifest.json", splits)
    train = [e for e in episodes.values() if e["split"] == "train"]
    validation = [e for e in episodes.values() if e["split"] == "validation"]
    models, candidates = {}, []
    specifications = [("ridge", lag, alpha) for lag in model_config["lags"] for alpha in model_config["ridge_alphas"]]
    specifications += [("extra_trees", model_config["tree_lag"], leaf) for leaf in model_config["tree_min_samples_leaf"]]
    for kind, lag, parameter in specifications:
        model = fit_model(train, kind, lag, parameter, model_config)
        scores = []
        for episode in validation:
            prediction, truth = rollout_episode(model, episode, common_origins(episode, model_config),
                                                 max(model_config["horizons_steps"]))
            scores += [r["temperature_RMSE_C"] for r in score_rollouts(prediction, truth,
                                                                       model_config["selection_horizons_steps"])]
        score = float(np.mean(scores))
        candidates.append({"kind": kind, "lag": lag, "parameter": parameter,
                           "validation_temperature_score_C": score, "training_rows": model["training_rows"]})
        if kind not in models or score < models[kind]["validation_score"]:
            model["validation_score"] = score
            models[kind] = model
        print(f"validation {kind}, lag={lag}, parameter={parameter}: {score:.6f} °C", flush=True)
    models["persistence"] = {"kind": "persistence", "lag": 1}
    # Freeze selection before any test scores are computed. Validation is never refitted.
    save_json(output / "selection.json", {"criterion": "Mean endpoint temperature RMSE at 1, 4 and 8 hours on validation only",
              "candidates": candidates, "selected": {k: {key: m[key] for key in ["lag", "parameter", "validation_score", "training_rows"]}
                                                       for k, m in models.items() if k != "persistence"}})
    joblib.dump(models, output / "models.joblib")
    rows, samples, per_origin = [], [], []
    evaluation_plan = []
    for episode in episodes.values():
        if episode["split"] == "train":
            continue
        origins = common_origins(episode, model_config)
        evaluation_plan.append({"episode": episode["id"], "split": episode["split"],
                                "origin_indices": origins.tolist(), "max_horizon_steps": max(model_config["horizons_steps"])})
        for name, model in models.items():
            for mode in ["perfect", "hold"]:
                prediction, truth = rollout_episode(model, episode, origins, max(model_config["horizons_steps"]), mode)
                identity = {"episode": episode["id"], "split": episode["split"], "model": name, "weather": mode}
                rows.extend([{**identity, **r} for r in score_rollouts(prediction, truth, model_config["horizons_steps"])])
                for h in model_config["horizons_steps"]:
                    for i, k in enumerate(origins):
                        per_origin.append({**identity, "origin_s": int(episode["times"][k]), "horizon_steps": h,
                                           "temperature_error_C": float(prediction[i, h - 1, 0] - truth[i, h - 1, 0]),
                                           "energy_error_kWh": float((prediction[i, :h, 3] - truth[i, :h, 3]).sum())})
                if mode == "perfect":
                    # First common origin is selected by index, before inspecting errors.
                    for i in range(prediction.shape[1]):
                        samples.append({**identity, "step": i + 1, "origin_s": int(episode["times"][origins[0]]),
                                        "predicted_C": prediction[0, i, 0], "true_C": truth[0, i, 0],
                                        "predicted_energy_kWh": prediction[0, i, 3], "true_energy_kWh": truth[0, i, 3]})
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    coverage_table(episodes).to_csv(output / "coverage.csv", index=False, lineterminator="\n")
    pd.DataFrame(samples).to_csv(output / "rollout_examples.csv", index=False, lineterminator="\n")
    pd.DataFrame(per_origin).to_csv(output / "errors.csv.gz", index=False, lineterminator="\n",
                                compression={"method": "gzip", "mtime": 0})
    save_json(output / "evaluation_origins.json", evaluation_plan)
    save_json(output / "manifest.json", {"config": config, "source": str(source), "python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
              "code_sha256": {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("excitation.py"),
                                                         Path(__file__).with_name("simulation.py")]},
              "config_sha256": digest(config_path), "source_manifest_sha256": digest(source / "manifest.json"),
              "source_summary_sha256": digest(source / "summary.json"),
              "ridge_spectral_radius": spectral_radius(models["ridge"]),
              "artifacts_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dynamics.json"))
    parser.add_argument("--source", type=Path, default=Path("outputs/excitation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dynamics-baseline"))
    args = parser.parse_args()
    run(args.config, args.source, args.output)


if __name__ == "__main__":
    main()
