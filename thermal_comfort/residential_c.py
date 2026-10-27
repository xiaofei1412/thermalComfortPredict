"""Read-only residential coverage audit and pre-registered development contracts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .simulation import digest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/residential_c.json"
CASE = "bestest_hydronic_heat_pump"


def save_snapshot(path, value):
    # Canonical LF keeps snapshot hashes valid after Git checkout on Windows/Linux.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def merged_windows(windows):
    merged = []
    for low, high in sorted(windows):
        if high <= low:
            raise ValueError("无效时间区间")
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(high, merged[-1][1])
        else:
            merged.append([low, high])
    return merged


def historical_registry(root):
    """Only interval metadata from old results; never scores or new holdout labels."""
    records, failed = [], []
    for parent in sorted((root / "outputs").iterdir()):
        if not parent.is_dir() or parent.name.startswith(("boptest", "storage")):
            continue
        for path in sorted(parent.glob("*/summary.json")):
            s = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(s, dict):
                continue
            cfg = s.get("config", {})
            if cfg.get("testcase", s.get("domain", {}).get("testcase")) != CASE:
                continue
            start = s.get("start_time_s", s.get("day", -999) * 86400) / 86400
            end = s.get("end_time_s", (start + s.get("days", cfg.get("days", 0))) * 86400) / 86400
            if start < 0 or end <= start:
                raise ValueError(f"无法确认历史时间：{path}")
            records.append({"source": path.relative_to(root).as_posix(), "sha256": digest(path),
                            "start_day": start, "end_day": end,
                            "warmup_start_day": start - cfg.get("warmup_days", 7),
                            "warmup_is_conservative": "warmup_days" not in cfg})
        for path in sorted(parent.glob("*/api_metadata.json")):
            if not (path.parent / "summary.json").exists():
                failed.append(path.parent.relative_to(root).as_posix())
    for path in sorted((root / "outputs/workbench/sessions").glob("*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        p = s["provenance"]
        if p["case"] == CASE:
            start = p["start_s"] / 86400
            end = s["frames"][-1]["time_s"] / 86400 if s["frames"] else start
            records.append({"source": path.relative_to(root).as_posix(), "sha256": digest(path),
                            "start_day": start, "end_day": max(end, start + 900/86400),
                            "warmup_start_day": start - p["warmup_s"] / 86400,
                            "warmup_is_conservative": False})
    # The only failed original-house attempt lacks summary; reserve its full old planned window.
    if "outputs/mpc-smoke/day98__guarded_rule" in failed:
        records.append({"source": "outputs/mpc-smoke/day98__guarded_rule", "sha256": None,
                        "start_day": 98, "end_day": 105, "warmup_start_day": 91,
                        "warmup_is_conservative": True})
    known_other_cases = {"outputs/native-rule-development-initialization-failed/day62__guarded_rule",
                         "outputs/office-data-failed-api/day14__excitation",
                         "outputs/terminal-validation-reset-failed/day62_perfect__terminal_mpc"}
    unknown = sorted(set(failed) - known_other_cases - {"outputs/mpc-smoke/day98__guarded_rule"})
    if unknown:
        raise ValueError(f"失败采集时间/域未核对，不能声明新留出：{unknown}")
    return {"sources": records, "failed_attempts": failed,
            "used_windows_days": merged_windows([(r["warmup_start_day"], r["end_day"]) for r in records])}


def validate_plan(config, registry):
    roles = {"train": 0, "development": 1, "calibration": 2, "test": 3}
    if config["case"] != CASE or config["step_s"] != 900 or config["episode_days"] != 2:
        raise ValueError("本轮只支持固定住宅/900秒/两天回合")
    windows = []
    for p in config["reserved_episodes"]:
        low, start, end = p["start_day"] - config["warmup_days"], p["start_day"], p["start_day"] + config["episode_days"]
        if p["role"] not in roles or low < 0 or end > 365:
            raise ValueError("角色或年度范围无效")
        if any(low < b and a < end for a, b in registry["used_windows_days"]):
            raise ValueError(f"预留与旧已用/预热窗口重叠：{p['id']}")
        windows.append({**p, "warmup_start_day": low, "end_day": end, "start_day": start})
    ordered = sorted(windows, key=lambda w: w["warmup_start_day"])
    if {w["role"] for w in windows} != set(roles) or len({w["id"] for w in windows}) != len(windows):
        raise ValueError("角色缺失或回合ID重复")
    for a, b in zip(ordered, ordered[1:]):
        if a["end_day"] > b["warmup_start_day"] or roles[a["role"]] > roles[b["role"]]:
            raise ValueError("跨角色时间/预热泄漏或角色顺序错误")
    return windows


def regime_masks(heat, config):
    t = config["regime_thresholds"]
    on = heat >= t["active_heat_kW_min"]
    changed = np.r_[False, on[1:] != on[:-1]]
    recent = pd.Series(changed).rolling(t["transition_lookback_steps"], min_periods=1).max().to_numpy(dtype=bool)
    transition = recent | ((heat > t["idle_heat_kW_max"]) & ~on)
    return {"active": on & ~transition, "idle": (heat <= t["idle_heat_kW_max"]) & ~transition,
            "transition": transition}


def physical_coverage(frame, native, config):
    times = frame.time_start_s.to_numpy()
    if len(frame) != 672 or not np.array_equal(np.diff(times), np.full(len(times)-1, 900)):
        raise ValueError("旧研究回合不完整或时间不连续")
    values = frame[["time_start_s", "time_end_s", "setpoint_requested_C", "occupancy_start", "solar_start_W_m2", "outdoor_start_C", "operative_start_C"]]
    if (not np.isfinite(values.to_numpy()).all() or not np.isfinite(native.to_numpy()).all()
            or not np.all(np.diff(native.time) == 30) or native.time.iloc[0] != times[0]
            or native.time.iloc[-1] != times[-1]+900 or not np.all(frame.time_end_s-times == 900)):
        raise ValueError("原生覆盖/有限值/区间边界不符，不能插值外推")
    actions = frame.setpoint_requested_C.to_numpy()
    change = np.diff(actions)
    heat = np.interp(times, native.time, native.reaQFloHea_y) / 1000
    t = config["regime_thresholds"]
    masks = regime_masks(heat, config)
    return {"transitions": len(frame), "rise_steps": int((change > t["action_change_K_min"]).sum()),
            "fall_steps": int((change < -t["action_change_K_min"]).sum()),
            "outside_current_action_steps": int(((actions < 18-1e-8) | (actions > 23.25+1e-8)).sum()),
            "outside_current_slew_steps": int((np.abs(change) > .5+1e-8).sum()),
            **{key + "_steps": int(value.sum()) for key, value in masks.items()},
            "occupied_steps": int((frame.occupancy_start > 0).sum()),
            "high_solar_steps": int((frame.solar_start_W_m2 >= t["high_solar_W_m2_min"]).sum()),
            "outdoor_range_C": [float(frame.outdoor_start_C.min()), float(frame.outdoor_start_C.max())],
            "operative_range_C": [float(frame.operative_start_C.min()), float(frame.operative_start_C.max())],
            "setpoint_range_C": [float(actions.min()), float(actions.max())]}


def audit_sources(root, config):
    rows = []
    for p in config["source_episodes"]:
        parent = root / p["root"]
        summaries = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
        chosen = [s for s in summaries if s["config"]["episode"]["start_day"] == p["day"]]
        if len(chosen) != 1:
            raise ValueError("回合身份不唯一/缺失")
        summary = chosen[0]
        directory = parent / (summary["scenario"] + "__" + summary["controller"])
        hashes = {name: digest(directory / name) for name in ("transitions.csv.gz", "native.csv.gz", "steps.jsonl")}
        if any(value != summary["sha256"][name] for name, value in hashes.items()):
            raise ValueError("旧数据摘要不符")
        frame = pd.read_csv(directory / "transitions.csv.gz")
        native = pd.read_csv(directory / "native.csv.gz", usecols=["time", "reaQFloHea_y"])
        steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        if any("demand_snapshot" in s for s in steps):
            raise ValueError("旧数据结构变化，请复核语义后再审计")
        rows.append({**p, "source": directory.relative_to(root).as_posix(), "sha256": hashes,
                     **physical_coverage(frame, native, config), "demand_ledger_present": False,
                     "semantic_event_counts": {k: 0 for k in ("replace", "withdraw", "expire", "conflict_resolved", "infeasible_resolved", "return_to_base")}})
    return rows


def semantic_coverage(session):
    counts = Counter({k: 0 for k in ("submit", "replace", "withdraw", "expire", "conflict_resolved", "infeasible_resolved", "return_to_base")})
    previous = {}
    for e in session.get("demand_events", []):
        r, action = e["request"], e["action"]
        if action == "submit" and previous.get(r["person"], {}).get("status") == "active":
            counts["replace"] += 1
        counts[{"submit": "submit", "withdraw": "withdraw", "expire": "expire"}[action]] += 1
        previous[r["person"]] = r
    had_active = False
    for event in session.get("events", []):
        d = event.get("decision", {})
        if d.get("demand_handling") != "adopted":
            continue
        r = d["demand"]
        if r["status"] in ("conflict", "infeasible"):
            counts[r["status"] + "_resolved"] += 1
        if had_active and not r["active_requests"]:
            counts["return_to_base"] += 1
        had_active = bool(r["active_requests"])
    return dict(counts)


def qualify_development(rows, checks, config):
    """Fail closed on missing cells/coverage; no model fitting or holdout reads."""
    failed = []
    indexed = {(r["group"], r["horizon_steps"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("开发评分重复")
    if set(indexed) - {(g, h) for g in config["minimum_coverage"]["groups"] for h in config["horizons_steps"]}:
        raise ValueError("未注册的分组或时域")
    def valid(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    for group in config["minimum_coverage"]["groups"]:
        for j, horizon in enumerate(config["horizons_steps"]):
            key = f"{group}/{horizon}"
            row = indexed.get((group, horizon), {})
            for field, minimum in [("origins", config["minimum_coverage"]["origins_per_group"]),
                                   ("distinct_days", config["minimum_coverage"]["distinct_days_per_group"])]:
                if type(row.get(field)) is not int or row[field] < minimum:
                    failed.append(f"{key}/{field}")
            for field in ("temperature_RMSE_K", "temperature_absolute_error_q95_K", "energy_RMSE_kWh", "energy_absolute_bias_kWh", "power_MAE_kW"):
                value = row.get(field)
                if not valid(value) or not 0 <= value <= config["gates"][field + "_max"][j]:
                    failed.append(f"{key}/{field}")
            for field in ("nonfinite_predictions", "negative_power_predictions"):
                if type(row.get(field)) is not int or row[field] != 0:
                    failed.append(f"{key}/{field}")
            if group == "idle":
                v = row.get("power_absolute_bias_kW")
                if not valid(v) or not 0 <= v <= config["gates"]["idle_power_absolute_bias_kW_max"]:
                    failed.append(f"{key}/idle_bias")
            if group == "all" and horizon in (16, 24):
                for field, reference, limit in [
                    ("temperature_RMSE_K", "persistence_temperature_RMSE_K", lambda v: max(config["gates"]["temperature_persistence_floor_K"], config["gates"]["temperature_persistence_ratio_max"]*v)),
                    ("energy_RMSE_kWh", "persistence_energy_RMSE_kWh", lambda v: config["gates"]["energy_persistence_ratio_max"]*v+config["gates"]["energy_persistence_slack_kWh"])]:
                    v = row.get(reference)
                    if not valid(v) or v < 0 or not valid(row.get(field)) or row[field] > limit(v):
                        failed.append(f"{key}/{reference}")
    for name in ("causal", "native_integration", "action_limits", "response_direction_amplitude", "support_coverage", "provenance"):
        if checks.get(name) is not True:
            failed.append(name)
    return {"qualified": not failed, "failed": failed, "scope": "development_only_not_control_or_energy_efficacy"}


def eligible_origins(length, horizon, config):
    if horizon not in config["horizons_steps"]:
        raise ValueError("未预注册的预测时域")
    return np.arange(config["history_steps"], length-horizon+1, config["origin_stride_steps"])


def response_gate(actual_temperature, predicted_temperature, actual_energy, predicted_energy, config):
    values = (actual_temperature, predicted_temperature, actual_energy, predicted_energy)
    if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) for v in values):
        return False
    g = config["gates"]
    return bool(actual_temperature >= g["response_temperature_min_K"] and actual_energy >= g["response_energy_min_kWh"]
            and predicted_temperature > 0 and predicted_energy > 0
            and abs(predicted_temperature-actual_temperature) <= max(g["response_temperature_error_floor_K"], g["response_relative_error_max"]*actual_temperature)
            and abs(predicted_energy-actual_energy) <= max(g["response_energy_error_floor_kWh"], g["response_relative_error_max"]*actual_energy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/residential-c-audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = historical_registry(ROOT)
    windows = validate_plan(config, registry)
    sources = audit_sources(ROOT, config)
    demos = []
    for p in sorted((ROOT / "outputs/workbench/sessions").glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("schema") == 2 and not s.get("comparison_of"):
            demos.append({"id": s["id"], "steps": len(s["events"]), "source_sha256": digest(p),
                          "role": "engineering_only", "semantic_events": semantic_coverage(s)})
    save_snapshot(args.output / "registry.json", registry)
    result = {"protocol_sha256": digest(CONFIG), "status": "audit_and_preregistration_only",
              "sources": sources, "engineering_demos": demos, "reserved_not_collected": windows,
              "new_model_fits": 0, "new_calibration_rounds": 0, "new_blind_rounds": 0,
              "model_control_qualified": False}
    save_snapshot(args.output / "coverage.json", result)
    files = ["configs/residential_c.json", "docs/experiments/residential_c.md", "thermal_comfort/residential_c.py",
             "thermal_comfort/demand_rule.py", "thermal_comfort/demands.py", "thermal_comfort/workbench.py", "scripts/verify_common_rule.py"]
    save_snapshot(args.output / "protocol_lock.json", {
        "created_UTC": datetime.now(timezone.utc).isoformat(), "status": "preregistered_not_model_qualified",
        "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {p: digest(ROOT / p) for p in files},
        "registry_sha256": digest(args.output / "registry.json"), "coverage_sha256": digest(args.output / "coverage.json"),
        "frozen_reference_model_sha256": digest(ROOT / "outputs/solar-selected/model.joblib")})
    print(json.dumps({"sources": len(sources), "transitions": sum(s["transitions"] for s in sources),
                      "engineering_demos": len(demos), "reserved_windows": len(windows),
                      "status": result["status"]}))


if __name__ == "__main__":
    main()
