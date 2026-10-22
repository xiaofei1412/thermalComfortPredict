"""Replay grey-box identification outputs and every closed-loop policy decision."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_comfort.dynamics import common_origins
from thermal_comfort.greybox import episode_rollout, load_heat_episodes, metrics
from thermal_comfort.mpc import Controller, ResponseController, response_checks
from thermal_comfort.simulation import digest, save_json, schedule_at
from verify_simulation import compare_repeat, verify as verify_trajectories


def assert_same_info(expected, saved):
    saved = {k: v for k, v in saved.items() if k != "decision_seconds"}
    if expected.keys() != saved.keys():
        raise ValueError("Decision metadata fields differ")
    for key, value in expected.items():
        other = saved[key]
        if isinstance(value, (int, float, list, np.number)) and not isinstance(value, bool):
            if not np.allclose(value, other, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Decision numeric mismatch: {key}")
        elif value != other:
            raise ValueError(f"Decision metadata mismatch: {key}")


def verify_model(directory, source, model, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Grey-box artifact changed: {name}")
    if manifest["source_summary_sha256"] != digest(source / "summary.json"):
        raise ValueError("Identification source changed")
    if manifest["config"] != config:
        raise ValueError("Model/control configuration mismatch")
    if not (np.all(model["parameters"] > 0) and np.all(model["power"] >= 0) and
            np.all(model["actuator"][:4] >= 0) and model["actuator"][0] <= 1):
        raise ValueError("Model physical parameter constraints failed")
    if max(abs(np.linalg.eigvals(model["A"]))) >= 1:
        raise ValueError("Thermal network is unstable")
    dynamics = json.loads(Path(config["dynamics_config"]).read_text())
    episodes, splits = load_heat_episodes(source, dynamics)
    if splits != json.loads((directory / "split_manifest.json").read_text()):
        raise ValueError("Split manifest mismatch")
    selection = json.loads((directory / "selection.json").read_text())
    validation = next(e for e in episodes.values() if e["split"] == "validation")
    selected = np.array(selection["selection_origin_indices"])
    calibration = np.array(selection["calibration_origin_indices"])
    if selected.max()+16 >= calibration.min() or calibration.max()+config["horizon_steps"] > len(validation["actions"]):
        raise ValueError("Selection/calibration windows overlap or leave the validation episode")
    scores = []
    for weight in config["residual_weights"]:
        prediction = episode_rollout({**model, "residual_weight": weight}, validation, selected, 16)
        scores.append(float(np.mean([r["temperature_RMSE_C"] for r in metrics(prediction, validation, selected, [1,4,8])])))
    np.testing.assert_allclose(scores, [r["validation_score_C"] for r in selection["candidates"]], rtol=1e-12)
    if model["residual_weight"] != config["residual_weights"][int(np.argmin(scores))]:
        raise ValueError("Residual was not selected on validation only")
    prediction = episode_rollout(model, validation, calibration, config["horizon_steps"])
    calibration_error = prediction[:, :, 0] - validation["states"][calibration[:, None] + np.arange(1,config["horizon_steps"]+1), 0]
    np.testing.assert_allclose(model["margin_K"], np.quantile(abs(calibration_error), config["margin_quantile"], axis=0) + config["sensor_margin_K"], rtol=1e-12)
    rows, coverage = [], []
    for episode in episodes.values():
        if episode["split"] == "train":
            continue
        origins = common_origins(episode, dynamics["model"])
        for name, weight in [("rc", 0.), ("rc_residual", model["residual_weight"])]:
            prediction = episode_rollout({**model, "residual_weight": weight}, episode, origins, 16)
            rows.extend([{"episode": episode["id"], "split": episode["split"], "model": name, **r}
                         for r in metrics(prediction, episode, origins, [1,4,8,16])])
            if name == "rc_residual":
                for h in [1,4,8]:
                    error = prediction[:, h-1, 0] - episode["states"][origins+h, 0]
                    coverage.append({"episode": episode["id"], "horizon_steps": h,
                                     "margin_K": model["margin_K"][h-1],
                                     "observed_coverage": np.mean(abs(error) <= model["margin_K"][h-1])})
    pd.testing.assert_frame_equal(pd.DataFrame(rows), pd.read_csv(directory / "metrics.csv"), check_dtype=False, rtol=1e-12, atol=1e-12)
    pd.DataFrame(coverage).to_csv(directory / "margin_coverage.csv", index=False, lineterminator="\n")
    return {"physical_constraints_pass": True, "reloaded_metrics_match": True,
            "selection_and_calibration_separated": True, "selection_origins": len(selected),
            "calibration_origins": len(calibration), "margin_rebuilt_from_validation_only": True}


def verify_policies(root, model, config, response=False):
    checks, decisions = [], []
    summaries = json.loads((root / "summary.json").read_text())
    for summary in summaries:
        directory = root / f"{summary['scenario']}__{summary['controller']}"
        records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        native = pd.read_csv(directory / "native.csv.gz")
        forecast = pd.read_csv(directory / "exogenous.csv.gz")
        inputs = json.loads((directory / "api_metadata.json").read_text())["inputs"]
        policy = (ResponseController(model, config, float(summary["controller"].split("_")[-1])) if response
                  else Controller(model, config, summary["controller"]))
        policy.reset(records[0]["before"], inputs)
        modes, reasons, seconds, errors = Counter(), Counter(), [], []
        for index, record in enumerate(records):
            before, after = record["before"], record["after"]
            action, info = policy.decide(index, before, forecast)
            if not np.isclose(action["oveTSet_u"], record["action"]["oveTSet_u"], rtol=0, atol=1e-9):
                raise ValueError("Policy replay changed the applied action")
            assert_same_info(info, record["control"])
            current = schedule_at(forecast, np.array([before["time"]])).iloc[0]
            interval = native.loc[native.time.between(before["time"], after["time"]), ["time", "reaQFloHea_y"]]
            policy.update(before, after, interval, current)
            command_C = action["oveTSet_u"]-273.15
            if abs(action["oveTSet_u"]-before["oveTSet_u"]) > config["slew_K_per_step"]+1e-9 or not config["action_min_C"] <= command_C <= config["action_max_C"]:
                raise ValueError("Actual command/slew bound failed")
            modes[info["mode"]] += 1
            reasons[info.get("reason", info["mode"])] += 1
            seconds.append(record["control"]["decision_seconds"])
            if info["mode"] == "mpc":
                errors.append(info["predicted_temperature_C"][0] - (after["reaTZon_y"]-273.15))
            decisions.append({"scenario": summary["scenario"], "controller": summary["controller"], "time_s": before["time"],
                              "mode": info["mode"], "reason": info.get("reason", info["mode"]), "action_C": command_C})
        temperature = native.reaTZon_y - 273.15
        if temperature.min() < config["abort_min_C"] or temperature.max() > config["abort_max_C"]:
            raise ValueError("Native samples crossed the stop band")
        checks.append({"scenario": summary["scenario"], "controller": summary["controller"], "decisions_replayed": len(records),
                       "modes": dict(modes), "reasons": dict(reasons), "operative_min_C": float(temperature.min()),
                       "operative_max_C": float(temperature.max()), "decision_mean_s": float(np.mean(seconds)),
                       "decision_p95_s": float(np.quantile(seconds,.95)), "decision_max_s": float(max(seconds)),
                       "one_step_mpc_temperature_RMSE_C": float(np.sqrt(np.mean(np.array(errors)**2))) if errors else None,
                       "one_step_mpc_margin_coverage": float(np.mean(abs(np.array(errors)) <= model["margin_K"][0])) if errors else None,
                       "action_and_slew_pass": True, "native_stop_band_pass": True})
    if not response:
        for day in config["closed_loop"]["start_days"]:
            directories = [root / f"day{day:g}__{kind}" for kind in config["closed_loop"]["controllers"]]
            a, b = [[json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()] for directory in directories]
            if a[0]["before"] != b[0]["before"]:
                raise ValueError("Controller comparison started from different FMU states")
            pd.testing.assert_frame_equal(pd.read_csv(directories[0]/"exogenous.csv.gz"), pd.read_csv(directories[1]/"exogenous.csv.gz"), check_exact=True)
            n = config["initial_rule_steps"]
            if any(x["action"] != y["action"] or x["after"] != y["after"] for x,y in zip(a[:n],b[:n])):
                raise ValueError("Common observer-initialization prefix differs")
    pd.DataFrame(decisions).to_csv(root / "decision_audit.csv", index=False, lineterminator="\n")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mpc.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/greybox-mpc"))
    parser.add_argument("--response", type=Path, default=Path("outputs/action-response-final"))
    parser.add_argument("--closed-loop", type=Path, default=Path("outputs/mpc"))
    parser.add_argument("--repeat-model", type=Path)
    parser.add_argument("--repeat-sim", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model = joblib.load(args.model_dir / "model.joblib")
    result = {"model": verify_model(args.model_dir, Path("outputs/excitation"), model, config),
              "response_trajectories": verify_trajectories(args.response),
              "control_trajectories": verify_trajectories(args.closed_loop),
              "response_policy_replay": verify_policies(args.response, model, config, True),
              "control_policy_replay": verify_policies(args.closed_loop, model, config)}
    original_gate = json.loads((args.response / "response_gate.json").read_text())
    original_response = pd.read_csv(args.response / "response_metrics.csv")
    response_checks(args.response, config, args.model_dir / "model.joblib", args.config)
    if original_gate != json.loads((args.response / "response_gate.json").read_text()) or not original_gate["passed"]:
        raise ValueError("Action response gate failed to reproduce")
    pd.testing.assert_frame_equal(original_response, pd.read_csv(args.response / "response_metrics.csv"), check_exact=True)
    result["paired_action_response_gate_rebuilt"] = True
    if args.repeat_model:
        repeated = joblib.load(args.repeat_model / "model.joblib")
        verify_model(args.repeat_model, Path("outputs/excitation"), repeated, config)
        for name in ["metrics.csv", "margin_coverage.csv"]:
            pd.testing.assert_frame_equal(pd.read_csv(args.model_dir / name), pd.read_csv(args.repeat_model / name), check_exact=True)
        if json.loads((args.model_dir / "selection.json").read_text()) != json.loads((args.repeat_model / "selection.json").read_text()):
            raise ValueError("Independent grey-box fitting changed")
        result["independent_model_repeat_exact"] = True
    if args.repeat_sim:
        result["short_simulation_repeat"] = compare_repeat(args.closed_loop, args.repeat_sim)
    save_json(args.closed_loop / "verification.json", result)
    print(json.dumps(result["control_policy_replay"], indent=2))
