"""Frozen C2 train/development-only collection and coverage-first qualification."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import demands, greybox, solar
from .demand_rule import common_action
from .residential_c import (eligible_origins, historical_registry, qualify_development, regime_masks,
                            response_gate, save_snapshot, validate_plan)
from .simulation import Client, FORECAST, POINTS, POWER, digest, integrate, schedule_at, transitions_frame, validate_kpis
from .workbench import Conflict, LIMITS


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/residential_c.json"
EVENTS = ROOT / "configs/residential_c2_events.json"
C1 = ROOT / "reports/residential_c/protocol_lock.json"
RUNS = [("train224", "train", 224, 192, None), ("development287", "development", 287, 192, None),
        ("response_low287", "response", 287, 40, 18.), ("response_high287", "response", 287, 40, 23.25)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class EventScript:
    def __init__(self, config):
        self.config = config
        self.demands = {"A": None, "B": None}
        self.revision = 0
        self.events = []
        self.previous_active = False

    def event(self, action, request, now):
        self.revision += 1
        self.events.append({"action": action, "time_s": now, "revision": self.revision,
                            "request": copy.deepcopy(request)})

    def tick(self, step, before, response_target=None):
        now, labels, rejections = before["time"], [], []
        for r in self.demands.values():
            if r and r["status"] == "active" and r["expires_s"] <= now:
                r["status"] = "expired"
                self.event("expire", r, now)
                labels.append("expire")
        phase = self.config["phases"][(step % self.config["cycle_steps"]) // 4]
        if response_target is None and step % self.config["cycle_steps"] == phase["offset"]:
            for item in phase["events"]:
                old = self.demands[item["person"]]
                if item["action"] == "submit":
                    if old and old["status"] == "active":
                        labels.append("replace")
                    body = {k: v for k, v in item.items() if k != "action"}
                    value = demands.submission({**body, "revision": self.revision}, now, self.revision)
                    self.demands[item["person"]] = value
                    self.event("submit", value, now)
                else:
                    if not old or old["status"] != "active":
                        raise ValueError("脚本撤销的请求必须有效")
                    old["status"], old["withdrawn_s"] = "withdrawn", now
                    self.event("withdraw", old, now)
                    labels.append("withdraw")
        base = phase["base_C"] if response_target is None else (21. if step < 16 else response_target)
        request = {"mode": "needs", "target_C": base, "preferences": {"A": "hold", "B": "hold"},
                   "resolution": phase["resolution"] if response_target is None else "unresolved", "demand_revision": self.revision}
        snapshot = {"demands": copy.deepcopy(self.demands), "revision": self.revision, "time_s": now}
        if response_target is None and step % self.config["cycle_steps"] == phase["offset"] and phase.get("check_rejection"):
            try:
                common_action({**request, "resolution": "unresolved"}, snapshot, before)
            except Conflict:
                rejections.append({"time_s": now, "revision": self.revision, "reason": "unresolved", "advance_called": False})
            else:
                raise ValueError("脚本要求先拒绝未裁决请求")
        action, decision = common_action(request, snapshot, before)
        if rejections:
            labels.append(decision["demand"]["status"] + "_resolved")
        active = bool(decision["demand"]["active_requests"])
        if self.previous_active and not active:
            labels.append("return_to_base")
        self.previous_active = active
        delta = action["oveTSet_u"] - before["oveTSet_u"]
        if delta > .1:
            labels.append("rise")
        elif delta < -.1:
            labels.append("fall")
        return {"intent": request, "demand_snapshot": snapshot, "action": action, "decision": decision,
                "labels": sorted(set(labels)), "rejections": rejections}


def verify_lock(output):
    lock = read(output / "freeze.json")
    for name, expected in lock["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"C2冻结源已变化：{name}")
    for name, expected in read(C1)["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"C1锁变化：{name}")
    if digest(ROOT / "outputs/solar-selected/model.joblib") != lock["model_sha256"]:
        raise ValueError("冻结模型变化")
    if digest(output / "registry_before.json") != lock["registry_before_sha256"] or digest(C1) != lock["c1_lock_sha256"]:
        raise ValueError("首采前时间登记/C1锁变化")
    return lock


def require_open(output):
    if (output / "closure.json").exists():
        raise ValueError("C2已收尾封存，禁止新增采集/拟合")
    return verify_lock(output)


def freeze(output):
    for name, expected in read(C1)["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"C1锁变化：{name}")
    if digest(ROOT/"outputs/solar-selected/model.joblib") != read(C1)["frozen_reference_model_sha256"]:
        raise ValueError("参照权重与C1锁不同")
    if digest(ROOT/"reports/residential_c/coverage.json") != read(C1)["coverage_sha256"]:
        raise ValueError("C1训练来源清单变化")
    coverage = read(ROOT/"reports/residential_c/coverage.json")
    for source in coverage["sources"]:
        if source["role"].startswith("legacy_train"):
            for name, expected in source["sha256"].items():
                if digest(ROOT/source["source"]/name) != expected:
                    raise ValueError("原训练来源与C1摘要不同")
    registry = historical_registry(ROOT)
    windows = validate_plan(read(PLAN), registry)
    output.mkdir(parents=True, exist_ok=False)
    files = ["configs/residential_c.json", "configs/residential_c2_events.json", "docs/experiments/residential_c2.md",
             "thermal_comfort/residential_c2.py", "tests/test_residential_c2.py", "thermal_comfort/greybox.py",
             "thermal_comfort/solar.py", "configs/solar_mpc.json", "configs/recovery_mpc.json", "configs/mpc.json", "configs/dynamics.json"]
    save_snapshot(output / "registry_before.json", registry)
    save_snapshot(output / "freeze.json", {"created_UTC": datetime.now(timezone.utc).isoformat(),
        "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {p: digest(ROOT / p) for p in files}, "c1_lock_sha256": digest(C1),
        "model_sha256": digest(ROOT / "outputs/solar-selected/model.joblib"),
        "registry_before_sha256": digest(output / "registry_before.json"), "reserved_windows": windows,
        "allowed_runs": [list(r) for r in RUNS], "advance_limit": 464})
    print("C2 frozen before new initialization/labels; only train224/development287 and two responses allowed", flush=True)


def check_workbench():
    with urlopen("http://127.0.0.1:8768/api/state", timeout=10) as response:
        state = json.load(response)
    if state["busy"] or state.get("can_end"):
        raise ValueError("工作台存在活跃会话，不能占用唯一worker")


def collect(output, url):
    require_open(output)
    if (output / "operations.json").exists():
        raise ValueError("本轮已尝试采集，不重复初始化或advance")
    client, operations, attempts = Client(url), [], 0
    def call(method, path, data=None):
        nonlocal attempts
        if path.startswith("initialize/"):
            if data["start_time"] not in (224*86400, 287*86400) or data["warmup_period"] != 604800:
                raise ValueError("禁止初始化校准/测试或改变预热")
        if path.startswith("advance/"):
            attempts += 1
            if attempts > 464:
                raise ValueError("已达464次动作预算")
        operations.append({"method": method, "path": path, "data": data, "attempted_actions": attempts})
        save_snapshot(output / "operations.json", operations)
        return client.call(method, path, data)
    if call("GET", "version")["version"] != "0.9.0":
        raise ValueError("引擎版本不符")
    for name, role, day, count, target in RUNS:
        check_workbench()
        directory = output / name
        directory.mkdir()
        script, records, segments = EventScript(read(EVENTS)), [], []
        testid = call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
        try:
            inputs, measurements = call("GET", f"inputs/{testid}"), call("GET", f"measurements/{testid}")
            if inputs["oveTSet_u"]["Unit"] != "K" or any(measurements[p]["Unit"] != "W" for p in POWER) or measurements["reaTZon_y"]["Unit"] != "K":
                raise ValueError("单位不符")
            save_snapshot(directory / "interface.json", {"inputs": inputs, "measurements": measurements})
            call("PUT", f"scenario/{testid}", {"electricity_price": "constant"})
            before = call("PUT", f"initialize/{testid}", {"start_time": day*86400, "warmup_period": 604800})
            call("PUT", f"step/{testid}", {"step": 900})
            forecast = pd.DataFrame(call("PUT", f"forecast/{testid}", {"point_names": FORECAST, "horizon": count*900, "interval": 30}))
            if forecast.time.iloc[0] != day*86400 or forecast.time.iloc[-1] != day*86400+count*900:
                raise ValueError("预报越出本回合时间窗")
            forecast.to_csv(directory / "forecast.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            for k in range(count):
                item = script.tick(k, before, target)
                after = call("POST", f"advance/{testid}", item["action"])
                raw = pd.DataFrame(call("PUT", f"results/{testid}", {"point_names": POINTS, "start_time": before["time"], "final_time": after["time"]}))
                record = {"before": before, "after": after, **item, "native": raw.to_dict("list")}
                transitions_frame([record], raw, forecast, {"step_s": 900, "area_m2": 192})
                if len(raw) != 31 or not np.all(np.diff(raw.time) == 30) or (raw[POWER].to_numpy() < -1e-6).any():
                    raise ValueError("原生点/功率不符")
                record["metrics"] = integrate(raw, forecast, 192)
                records.append(record); segments.append(raw)
                with (directory / "steps.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                save_snapshot(directory / "demand_events.json", script.events)
                if not LIMITS["stop_min_C"] <= after["reaTZon_y"]-273.15 <= LIMITS["stop_max_C"]:
                    raise ValueError("实际温度越过停止带，保留前缀并终止")
                before = after
                if (k+1) % 24 == 0 or k+1 == count:
                    print(f"{name}: {k+1}/{count}; total attempted actions={attempts}", flush=True)
            native = pd.concat(segments, ignore_index=True).drop_duplicates("time", keep="first")
            native.to_csv(directory / "native.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            sums = {p: sum(r["metrics"][p] for r in records) for p in records[0]["metrics"]}
            kpis = call("GET", f"kpi/{testid}")
            save_snapshot(directory / "api_kpis.json", kpis)
            differences = validate_kpis(sums, kpis)
            summary = {"name": name, "role": role, "config": {"testcase": read(PLAN)["case"], "warmup_days": 7},
                "start_time_s": day*86400, "end_time_s": before["time"], "transitions": len(records),
                "native_samples": len(native), "metrics": sums, "kpi_absolute_difference": differences,
                "sha256": {p.name: digest(p) for p in directory.iterdir() if p.is_file()}}
            save_snapshot(directory / "summary.json", summary)
        except Exception as error:
            save_snapshot(output / "collection_failure.json", {"run": name, "error": str(error), "completed_steps": len(records), "attempted_actions": attempts})
            save_snapshot(output / "closure.json", {"status": "closed_incomplete", "calibration_rounds": 0, "blind_rounds": 0, "reason": "collection_failure"})
            raise
        finally:
            call("PUT", f"stop/{testid}")
    save_snapshot(output / "collection_complete.json", {"attempted_actions": attempts, "runs": [r[0] for r in RUNS]})


def load_episode(directory):
    summary = read(directory / "summary.json")
    for name, expected in summary["sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("采集摘要变化")
    records = [json.loads(line) for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    n = len(records)
    if n != summary["transitions"] or n not in (192, 40):
        raise ValueError("回合长度不符")
    forecast = pd.read_csv(directory / "forecast.csv.gz")
    times = np.array([r["before"]["time"] for r in records])
    if not np.all(np.diff(times) == 900):
        raise ValueError("步序不连续")
    states = np.zeros((n+1, 4)); states[0, 0] = records[0]["before"]["reaTZon_y"]-273.15; states[0, 3] = np.nan
    states[1:, 0] = [r["after"]["reaTZon_y"]-273.15 for r in records]
    states[1:, 3] = [r["metrics"]["energy_kWh"] for r in records]
    rows = schedule_at(forecast, times)
    heat = np.array([np.trapezoid(r["native"]["reaQFloHea_y"], r["native"]["time"])/900000 for r in records])
    return {"id": directory.name, "split": summary["role"], "states": states, "times": times,
            "actions": np.array([r["action"]["oveTSet_u"]-273.15 for r in records]), "heat": heat,
            "initial_heat": records[0]["before"]["reaQFloHea_y"]/1000,
            "exog": np.column_stack([rows.TDryBul-273.15, rows.HGloHor, rows["Occupancy[1]"], np.sin(2*np.pi*times/86400), np.cos(2*np.pi*times/86400)]),
            "records": records, "observed_heat": np.array([r["before"]["reaQFloHea_y"]/1000 for r in records])}


def group_masks(episode, config):
    masks = {"all": np.ones(len(episode["actions"]), dtype=bool), **regime_masks(episode["observed_heat"], config)}
    for name in config["minimum_coverage"]["groups"]:
        if name in masks:
            continue
        stamps = [r["before"]["time"] for r in episode["records"] if name in r["labels"]]
        masks[name] = np.array([any(0 <= now-t <= 3600 for t in stamps) for now in episode["times"]])
    return masks


def coverage_rows(episode, config):
    masks = group_masks(episode, config)
    return [{"group": group, "horizon_steps": h, "origins": int(mask[eligible_origins(len(episode["actions"]), h, config)].sum()),
             "distinct_days": len(set((episode["times"][o]//86400).tolist()))}
            for group, mask in masks.items() for h in config["horizons_steps"]
            for o in [eligible_origins(len(episode["actions"]), h, config)[mask[eligible_origins(len(episode["actions"]), h, config)]]]]


def score_model(model, episode, config):
    masks, rows = group_masks(episode, config), []
    support = np.column_stack([episode["states"][:-1, 0], episode["exog"][:, :2]])
    tolerance = np.array(list(config["support_tolerance"].values()))
    supported = ((support >= model["support_low"]-tolerance) & (support <= model["support_high"]+tolerance)).all(axis=1)
    for h in config["horizons_steps"]:
        origins = eligible_origins(len(episode["actions"]), h, config)
        prediction = greybox.episode_rollout(model, episode, origins, h)
        actual = episode["states"][origins[:, None]+np.arange(1, h+1), :]
        te = prediction[:, -1, 0]-actual[:, -1, 0]
        ee = prediction[:, :, 3].sum(axis=1)-actual[:, :, 3].sum(axis=1)
        pe = (prediction[:, :, 3]-actual[:, :, 3])*4
        tp = episode["states"][origins, 0]-actual[:, -1, 0]
        ep = episode["states"][origins, 3]*h-actual[:, :, 3].sum(axis=1)
        for group, mask in masks.items():
            take = mask[origins]; count = int(take.sum())
            row = {"group": group, "horizon_steps": h, "origins": count,
                   "distinct_days": len(set((episode["times"][origins[take]]//86400).tolist())),
                   "supported_origins": int(supported[origins[take]].sum()),
                   "nonfinite_predictions": int((~np.isfinite(prediction[take])).sum()),
                   "negative_power_predictions": int((prediction[take, :, 3] < 0).sum())}
            row.update({"temperature_RMSE_K": float(np.sqrt(np.mean(te[take]**2))) if count else None,
                "temperature_absolute_error_q95_K": float(np.quantile(abs(te[take]), .95)) if count else None,
                "energy_RMSE_kWh": float(np.sqrt(np.mean(ee[take]**2))) if count else None,
                "energy_absolute_bias_kWh": float(abs(np.mean(ee[take]))) if count else None,
                "power_MAE_kW": float(np.mean(abs(pe[take]))) if count else None,
                "power_absolute_bias_kW": float(abs(np.mean(pe[take]))) if count else None,
                "persistence_temperature_RMSE_K": float(np.sqrt(np.mean(tp[take]**2))) if count else None,
                "persistence_energy_RMSE_kWh": float(np.sqrt(np.mean(ep[take]**2))) if count else None})
            rows.append(row)
    return rows


def refit_model(config, new_train):
    train = {new_train["id"]: new_train}
    for p in config["source_episodes"]:
        if not p["role"].startswith("legacy_train"):
            continue
        parent = ROOT / p["root"]
        s = next(s for s in read(parent / "summary.json") if s["config"]["episode"]["start_day"] == p["day"])
        e, _ = solar.read_episode(parent/(s["scenario"]+"__"+s["controller"]), s, "train")
        train[e["id"]] = e
    old_config = solar.configuration()
    model = greybox.fit(train, {**old_config, "fit_max_nfev": 150, "residual_alpha": 10.})
    heat = np.concatenate([e["heat"] for e in train.values()]); outdoor = np.concatenate([e["exog"][:, 0] for e in train.values()])
    power = np.concatenate([e["states"][1:, 3]*4 for e in train.values()])
    model["idle_power_kW"] = float(power[heat < .05].mean()); model["power_transition_kW"] = .5
    x = np.column_stack([np.minimum(heat/.5, 1), heat, heat*np.maximum(20-outdoor, 0)/20])
    model["power"] = nnls(x, np.maximum(power-model["idle_power_kW"], 0))[0]
    model.update(residual_weight=1., observer_kind="recursive")
    return model


def response_metrics(model, low, high, config):
    predictions = [greybox.episode_rollout(model, e, np.array([16]), 24)[0] for e in (low, high)]
    actual_t = float(high["states"][40, 0]-low["states"][40, 0])
    actual_e = float(high["states"][17:41, 3].sum()-low["states"][17:41, 3].sum())
    pred_t = float(predictions[1][-1, 0]-predictions[0][-1, 0]); pred_e = float(predictions[1][:, 3].sum()-predictions[0][:, 3].sum())
    return {"actual_temperature_difference_K": actual_t, "predicted_temperature_difference_K": pred_t,
            "actual_energy_difference_kWh": actual_e, "predicted_energy_difference_kWh": pred_e,
            "passed": response_gate(actual_t, pred_t, actual_e, pred_e, config)}


def causal_check(model, episode):
    k, h = 16, 24
    baseline = greybox.episode_rollout(model, episode, np.array([k]), h)
    changed = copy.deepcopy(episode)
    changed["states"][k+1:, 0] += 100; changed["states"][k+1:, 3] += 100
    changed["heat"][k:] += 100
    return bool(np.array_equal(baseline, greybox.episode_rollout(model, changed, np.array([k]), h)))


def verify_data(output):
    verify_lock(output)
    operations = read(output / "operations.json")
    initializations = [o["data"]["start_time"] for o in operations if o["path"].startswith("initialize/")]
    if initializations != [224*86400, 287*86400, 287*86400, 287*86400]:
        raise ValueError("初始化未严格符合四回合计划")
    if sum(o["path"].startswith("advance/") for o in operations) != 464:
        raise ValueError("完整预算/动作计数不符")
    windows, trial = {}, 0
    for op in operations:
        method, _, testid = op["path"].partition("/")
        if method == "initialize":
            start = op["data"]["start_time"]
            windows[testid] = (start, start+RUNS[trial][3]*900)
            trial += 1
        elif method in ("forecast", "results"):
            start, end = windows[testid]
            d = op["data"]
            if method == "forecast" and d["horizon"] != end-start:
                raise ValueError("预报读取越过登记时间窗")
            if method == "results" and not (start <= d["start_time"] < d["final_time"] <= end and d["final_time"]-d["start_time"] == 900):
                raise ValueError("原生数据读取越过登记时间窗")
    episodes = {name: load_episode(output/name) for name, *_ in RUNS}
    max_difference = 0.
    for name, role, day, count, target in RUNS:
        episode = episodes[name]; script = EventScript(read(EVENTS)); forecast = pd.read_csv(output/name/"forecast.csv.gz")
        if episode["times"][0] != day*86400 or len(episode["actions"]) != count or episode["split"] != role:
            raise ValueError("数据角色/日期不符")
        for k, r in enumerate(episode["records"]):
            expected = script.tick(k, r["before"], target)
            if any(expected[key] != r[key] for key in expected):
                raise ValueError("需求/裁决/动作重放不一致")
            raw = pd.DataFrame(r["native"])
            transitions_frame([r], raw, forecast, {"step_s": 900, "area_m2": 192})
            recomputed = integrate(raw, forecast, 192)
            if any(abs(v-r["metrics"][key]) > 1e-10 for key, v in recomputed.items()):
                raise ValueError("原生积分不符")
        s = read(output/name/"summary.json")
        max_difference = max(max_difference, *validate_kpis(s["metrics"], read(output/name/"api_kpis.json")).values())
    low, high = episodes["response_low287"], episodes["response_high287"]
    for a, b in zip(low["records"][:16], high["records"][:16], strict=True):
        if any(a[key] != b[key] for key in ("before", "after", "action", "native")):
            raise ValueError("响应共同预序列不一致")
    return episodes, {"actions": 464, "native_interval_points": 464*31,
                      "kpi_max_absolute_difference": max_difference, "calibration_test_initializations": 0,
                      "same_response_prelude": True, "all_decisions_replayed": True}


def assess_candidate(name, model, episodes, config):
    dev = episodes["development287"]
    try:
        rows = score_model(model, dev, config)
        response = response_metrics(model, episodes["response_low287"], episodes["response_high287"], config)
        checks = {"causal": causal_check(model, dev), "native_integration": True, "action_limits": True,
                  "response_direction_amplitude": response["passed"], "provenance": True,
                  "support_coverage": all(r["supported_origins"] == r["origins"] for r in rows)}
    except (ValueError, FloatingPointError) as error:
        rows, response, checks = [], {"error": str(error), "passed": False}, {}
    return {"candidate": name, "metrics": rows, "response": response, "checks": checks,
            **qualify_development(rows, checks, config)}


def evaluate(output):
    require_open(output)
    if (output / "evaluation_started.json").exists():
        raise ValueError("已经评估，不重复开发")
    episodes, verification = verify_data(output)
    save_snapshot(output / "evaluation_started.json", {"created_UTC": datetime.now(timezone.utc).isoformat()})
    config = read(PLAN); dev = episodes["development287"]
    coverage = coverage_rows(dev, config)
    missing = [r for r in coverage if r["origins"] < 12 or r["distinct_days"] < 2]
    save_snapshot(output / "coverage_development.json", coverage)
    models = [("frozen_expanded_baseline", joblib.load(ROOT/"outputs/solar-selected/model.joblib"))]
    refits, fit_failure = 0, None
    if not missing:
        save_snapshot(output / "fit_attempt.json", {"refits": 1, "train_roles_only": True})
        refits = 1
        try:
            model = refit_model(config, episodes["train224"])
            joblib.dump(model, output/"refit.joblib")
            models.append(("refit_expanded_baseline", model))
        except RuntimeError as error:
            fit_failure = str(error)
    results = [assess_candidate(name, model, episodes, config) for name, model in models]
    passing = [r for r in results if r["qualified"]]
    selected = min(passing, key=lambda r: next(m["temperature_RMSE_K"] for m in r["metrics"] if m["group"] == "all" and m["horizon_steps"] == 24))["candidate"] if passing else None
    assessment = {"coverage_passed": not missing, "missing_coverage": missing, "candidates": results,
                  "refits": refits, "fit_failure": fit_failure, "selected": selected, "verification": verification,
                  "calibration_rounds": 0, "blind_rounds": 0}
    save_snapshot(output / "assessment.json", assessment)
    closure = {"status": "development_passed" if passing else "closed_negative", "selected": selected,
               "reason": "passed" if passing else "missing_coverage" if missing else "model_gates_failed",
               "refits": refits, "calibration_rounds": 0, "blind_rounds": 0,
               "assessment_sha256": digest(output/"assessment.json"), "further_collection_or_fit_allowed": False}
    closure["data_sha256"] = {str((output/name/"summary.json").relative_to(output)): digest(output/name/"summary.json") for name, *_ in RUNS}
    closure["data_sha256"]["operations.json"] = digest(output/"operations.json")
    if (output/"refit.joblib").exists():
        closure["data_sha256"]["refit.joblib"] = digest(output/"refit.joblib")
    save_snapshot(output / "closure.json", closure)
    print(json.dumps({k: v for k, v in closure.items() if k != "assessment_sha256"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "collect", "evaluate", "verify"])
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/residential-c2")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    if args.mode == "freeze": freeze(args.output)
    elif args.mode == "collect": collect(args.output, args.url)
    elif args.mode == "evaluate": evaluate(args.output)
    else:
        episodes, result = verify_data(args.output)
        if (args.output/"closure.json").exists():
            closure = read(args.output/"closure.json")
            if digest(args.output/"assessment.json") != closure["assessment_sha256"]:
                raise ValueError("封存评估摘要变化")
            for name, expected in closure["data_sha256"].items():
                if digest(args.output/name) != expected:
                    raise ValueError("封存轨迹或权重变化")
            for saved in read(args.output/"assessment.json")["candidates"]:
                path = ROOT/"outputs/solar-selected/model.joblib" if saved["candidate"] == "frozen_expanded_baseline" else args.output/"refit.joblib"
                recomputed = assess_candidate(saved["candidate"], joblib.load(path), episodes, read(PLAN))
                if recomputed != saved:
                    raise ValueError("开发指标/门/响应独立重算不一致")
            result["development_scores_recomputed"] = True
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
