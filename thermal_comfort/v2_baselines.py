"""V2-2 frozen residential baselines; development only, no learning."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

from .mpc import limited_command
from .rl_interface import bounded_forecast, native_score
from .simulation import Client, FORECAST, POINTS, POWER, digest, save_json, validate_kpis
from .solar import SolarController
from .v2_protocol import ROOT, read, validate_split, verify_files


def protocol():
    plan, parent = read(ROOT/"configs/v2_2.json"), read(ROOT/"configs/v2_0.json")
    validate_split(parent)
    episodes = [e for e in parent["episodes"] if e["id"] in plan["episodes"]]
    if (plan["training_authorized"] or plan["weather"] != "perfect"
            or plan["episodes"] != ["development084", "development168", "development189"]
            or [e["role"] for e in episodes] != ["development"]*3
            or plan["controllers"] != ["guarded_rule", "solar_mpc", "minimum_heat"]):
        raise ValueError("V2-2 only permits the registered frozen development baselines")
    for name in plan["parent_locks"]:
        lock = read(ROOT/name)
        verify_files(ROOT, lock["files_sha256"])
    v1 = read(ROOT/plan["parent_locks"][0])["v1_baseline"]
    verify_files(ROOT, v1["files_sha256"])
    for name, expected in [(parent["baseline_model"], v1["model_sha256"]),
            ("outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu", v1["fmu_sha256"]),
            ("reports/v2_0/audit.json", read(ROOT/"reports/v2_0/lock.json")["audit_sha256"])]:
        if digest(ROOT/name) != expected:
            raise ValueError(f"Frozen asset changed: {name}")
    return plan, episodes, v1["effective_config"]


class ScopedClient:
    """All attempts consume budget; reject dates and future-result requests before I/O."""

    def __init__(self, client, plan, days, output):
        if set(days) != {84, 168, 189}:
            raise ValueError("V2-2 scope must be the three registered development dates")
        self.client, self.plan, self.days, self.output = client, plan, set(days), output
        self.counts = {"initialize": 0, "advance": 0}
        self.now, self.starts = {}, {}
        self.started = time.perf_counter()

    def call(self, method, path, data=None):
        kind, _, testid = path.partition("/")
        budget = self.plan["live_budget"]
        if kind != "stop" and time.perf_counter()-self.started > budget["wall_seconds_max"]:
            raise RuntimeError("V2-2 wall budget exhausted")
        if kind == "initialize":
            if data.get("start_time") not in {d*86400 for d in self.days} or data.get("warmup_period") != 604800:
                raise ValueError("Only registered development initializations are allowed")
        if kind == "forecast" and (testid not in self.now or data != {
                "point_names": FORECAST, "horizon": 43200, "interval": 30}):
            raise ValueError("Only current..12h forecasts are allowed")
        if kind == "results" and (testid not in self.now or data["start_time"] < self.starts[testid]
                or data["final_time"] > self.now[testid] or data["start_time"] > data["final_time"]):
            raise ValueError("Results must be within completed scored observations")
        if kind == "advance" and (testid not in self.now or self.now[testid] >= self.starts[testid]+604800):
            raise ValueError("Cannot advance an uninitialized or completed development episode")
        if kind in self.counts:
            limit = "initializations_max" if kind == "initialize" else "advance_attempts_max"
            if self.counts[kind] >= budget[limit]:
                raise RuntimeError("V2-2 interaction budget exhausted")
            self.counts[kind] += 1
        entry = {"method": method, "path": path, "request": data, "counts": dict(self.counts), "ok": False}
        with (self.output/"attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry)+"\n")
        started = time.perf_counter()
        try:
            result = self.client.call(method, path, data)
            entry["ok"] = True
            if kind == "initialize":
                self.now[testid] = self.starts[testid] = data["start_time"]
            if kind == "advance":
                self.now[testid] = result["time"]
            return result
        finally:
            entry["seconds"] = time.perf_counter()-started
            with (self.output/"calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, allow_nan=False)+"\n")


def score(native, schedule):
    metrics = native_score(native, schedule)
    metrics["heat_kWh"] = float(np.trapezoid(native.reaQFloHea_y, native.time)/3600000)
    return metrics


def command(policy, index, before, view, target=None):
    if target is None or index < 8:
        return policy.decide(index, before, view)
    value = limited_command(target, policy.previous, policy.config)
    return {"oveTSet_u": value+273.15, "oveTSet_activate": 1}, {
        "mode": "intervention", "reason": "fixed_response_target", "target_C": target, "requested_C": value}


def check_action(action, before, native=None):
    value = action["oveTSet_u"]-273.15
    if (not np.isfinite(value) or not 18-1e-7 <= value <= 23.25+1e-7
            or abs(action["oveTSet_u"]-before["oveTSet_u"]) > .5+1e-7
            or action["oveTSet_activate"] != 1):
        raise ValueError("Mechanical action violation")
    if native is not None:
        if not np.array_equal(native.time, before["time"]+np.arange(31)*30):
            raise ValueError("Incomplete native interval")
        if (not np.isfinite(native.to_numpy()).all() or not (native.oveTSet_activate.iloc[1:] == 1).all()
                or not np.allclose(native.oveTSet_u.iloc[1:], action["oveTSet_u"], rtol=0, atol=1e-7)):
            raise ValueError("Native values or actual action invalid")
        if ((native.reaTZon_y-273.15 < 18) | (native.reaTZon_y-273.15 > 35)).any():
            raise ValueError("Engineering temperature abort; no energy-saving qualification")


def stitch(frames):
    return pd.concat([f.iloc[:-1] for f in frames]+[frames[-1].iloc[-1:]], ignore_index=True)


def run_episode(client, episode, kind, model, config, output, target=None):
    output.mkdir()
    started = time.perf_counter()
    policy = SolarController(model, config, kind, "perfect")
    steps = 672 if target is None else 24
    testid = client.call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
    natives, schedules, records = [], [], []
    try:
        inputs = client.call("GET", f"inputs/{testid}")
        measurements = client.call("GET", f"measurements/{testid}")
        if inputs["oveTSet_u"]["Unit"] != "K" or measurements["reaTZon_y"]["Unit"] != "K":
            raise ValueError("Unexpected temperature units")
        if any(measurements[p]["Unit"] != "W" for p in [*POWER, "reaQFloHea_y"]):
            raise ValueError("Unexpected power units")
        save_json(output/"inputs.json", inputs)
        client.call("PUT", f"scenario/{testid}", {"electricity_price": "constant"})
        initial = client.call("PUT", f"initialize/{testid}", {
            "start_time": episode["start_day"]*86400, "warmup_period": 604800})
        if initial["time"] != episode["start_day"]*86400:
            raise ValueError("Unexpected initialization time")
        policy.reset(initial, inputs)
        client.call("PUT", f"step/{testid}", {"step": 900})
        if client.call("GET", f"step/{testid}") != 900:
            raise ValueError("Unexpected control step")
        before = initial
        with (output/"steps.jsonl").open("w", encoding="utf-8") as log, gzip.open(
                output/"forecasts.jsonl.gz", "wt", encoding="utf-8") as forecasts:
            for index in range(steps):
                view = bounded_forecast(client.call("PUT", f"forecast/{testid}", {
                    "point_names": FORECAST, "horizon": 43200, "interval": 30}), before["time"])
                forecasts.write(json.dumps(view.to_dict("list"), allow_nan=False)+"\n")
                decision_started = time.perf_counter()
                action, control = command(policy, index, before, view, target)
                control["decision_seconds"] = time.perf_counter()-decision_started
                check_action(action, before)
                after = client.call("POST", f"advance/{testid}", action)
                if after["time"] != before["time"]+900:
                    raise ValueError("Unexpected advance timestamp")
                native = pd.DataFrame(client.call("PUT", f"results/{testid}", {
                    "point_names": POINTS, "start_time": before["time"], "final_time": after["time"]}))
                schedule = view.iloc[:31].reset_index(drop=True)
                natives.append(native)
                schedules.append(schedule)
                record = {"index": index, "before": before, "action": action, "after": after,
                          "control": control, "metrics": score(native, schedule)}
                log.write(json.dumps(record, allow_nan=False)+"\n")
                log.flush()
                records.append(record)
                check_action(action, before, native)
                policy.update(before, after, native, view.iloc[0])
                before = after
                if (index+1) % 96 == 0:
                    print(f"{output.name}: {index+1}/{steps}", flush=True)
        native, schedule = stitch(natives), stitch(schedules)
        metrics = score(native, schedule)
        kpis = client.call("GET", f"kpi/{testid}")
        native.to_csv(output/"native.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        schedule.to_csv(output/"schedule.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        modes = Counter(r["control"]["mode"] for r in records[8:])
        decisions = [r["control"]["decision_seconds"] for r in records[8:]]
        summary = {"episode": episode, "kind": kind, "target_C": target, "transitions": steps,
            "native_points": len(native), "initial": initial, "metrics": metrics, "kpis": kpis,
            "kpi_absolute_difference": validate_kpis(metrics, kpis), "end_state": before,
            "post_prefix_modes": dict(modes), "fallback_fraction_excluding_prefix": modes["fallback"]/(steps-8),
            "decision_seconds": {"mean": float(np.mean(decisions)), "p95": float(np.percentile(decisions, 95))},
            "wall_seconds": time.perf_counter()-started,
            "sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
        save_json(output/"summary.json", summary)
        print(f"{output.name}: complete, {metrics['energy_kWh']:.6f}kWh", flush=True)
        return summary
    except Exception as error:
        if natives:
            stitch(natives).to_csv(output/"partial_native.csv.gz", index=False)
        save_json(output/"failure.json", {"error": str(error), "completed_records": len(records),
                                        "episode": episode, "kind": kind, "target_C": target})
        raise
    finally:
        client.call("PUT", f"stop/{testid}")


def weather_coverage(schedule, limits):
    t = schedule.time.to_numpy()
    outdoor = schedule.TDryBul.to_numpy()-273.15
    solar = schedule.HGloHor.to_numpy()
    hour = (t % 86400)/3600
    hours = lambda mask: float(np.trapezoid(np.asarray(mask, dtype=float), t)/3600)
    return {"outdoor_min_C": float(min(outdoor)), "outdoor_max_C": float(max(outdoor)),
        "cold_hours": hours(outdoor <= limits["cold_outdoor_max_C"]),
        "warm_hours": hours(outdoor >= limits["warm_outdoor_min_C"]),
        "sunny_hours": hours(solar >= limits["sunny_min_W_m2"]),
        "daytime_low_solar_hours": hours((hour >= 8) & (hour < 18) & (solar <= 50)),
        "night_hours": hours((hour < 8) | (hour >= 18)),
        "solar_peak_W_m2": float(max(solar)), "solar_total_kWh_m2": float(np.trapezoid(solar, t)/3600000)}


def response_check(low, high, limits):
    de = high["energy_kWh"]-low["energy_kWh"]
    dq = high["heat_kWh"]-low["heat_kWh"]
    dt = high["operative_end_C"]-low["operative_end_C"]
    responsive = (de >= limits["response_energy_difference_min_kWh"]
        and dq >= limits["response_heat_difference_min_kWh"]
        and dt >= limits["response_end_temperature_difference_min_K"])
    reversed_response = (de < -limits["response_energy_difference_min_kWh"]
        or dq < -limits["response_heat_difference_min_kWh"]
        or dt < -limits["response_end_temperature_difference_min_K"])
    return {"delta_energy_kWh": de, "delta_heat_kWh": dq, "delta_end_temperature_K": dt,
            "responsive": responsive, "reversed": reversed_response}


def comparison(rule, candidate, gates):
    saving = rule["energy_kWh"]-candidate["energy_kWh"]
    thermal = all(candidate[f"{kind}_{metric}"]-rule[f"{kind}_{metric}"] <= tolerance
        for kind in ["cold", "hot"] for metric, tolerance in [
            ("Kh", gates["cold_and_hot_degree_hour_increase_tolerance"]),
            ("duration_s", gates["cold_and_hot_duration_increase_s_tolerance"]),
            ("max_K", gates["cold_and_hot_max_deviation_increase_K_tolerance"])] )
    active = rule["energy_kWh"] >= gates["active_reference_energy_kWh_min"]
    energy = (saving >= gates["saving_absolute_kWh_min"] and saving/rule["energy_kWh"] >= gates["saving_relative_min"]
              if active else -saving <= gates["low_load_energy_increase_kWh_tolerance"])
    return {"active": active, "saving_kWh": saving,
        "saving_fraction": saving/rule["energy_kWh"] if active else None,
        "thermal_nonincrease": thermal, "energy_gate_passed": energy,
        "joint_gate_passed": bool(thermal and energy)}


def coverage_gate(coverage, active, responses, limits):
    checks = {"cold": any(c["cold_hours"] >= limits["cold_hours_min_in_one_episode"] for c in coverage.values()),
        "warm": any(c["warm_hours"] >= limits["warm_hours_min_in_one_episode"] for c in coverage.values()),
        "sunny": any(c["sunny_hours"] >= limits["sunny_hours_min_in_one_episode"] for c in coverage.values()),
        "active_dates": sum(active.values()) >= limits["active_development_dates_min"],
        "responsive_active_dates": sum(active[k] and r["responsive"] for k, r in responses.items()) >= limits["responsive_active_dates_min"],
        "no_reversed_response": not any(r["reversed"] for r in responses.values())}
    return {"checks": checks, "passed": all(checks.values()), "training_authorized": False}


def assert_same_values(left, right):
    # CSV inference can type an all-zero short trace as int and its longer peer as float.
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)


def analyze(directory, summaries, plan):
    coverage, active, responses, comparisons = {}, {}, {}, {}
    gates = read(ROOT/"configs/v2_0.json")["comparison_gates"]
    for episode in plan["episodes"]:
        runs = {kind: summaries[f"{episode}__{kind}"] for kind in plan["controllers"]}
        rule = runs["guarded_rule"]
        folder = directory/f"{episode}__guarded_rule"
        schedule = pd.read_csv(folder/"schedule.csv.gz")
        native = pd.read_csv(folder/"native.csv.gz")
        coverage[episode] = weather_coverage(schedule, plan["coverage"])
        active[episode] = rule["metrics"]["energy_kWh"] >= gates["active_reference_energy_kWh_min"]
        comparisons[episode] = {}
        for kind in ["solar_mpc", "minimum_heat"]:
            other = runs[kind]
            other_folder = directory/f"{episode}__{kind}"
            if rule["initial"] != other["initial"]:
                raise ValueError("Baseline initial states differ")
            assert_same_values(schedule, pd.read_csv(other_folder/"schedule.csv.gz"))
            other_native = pd.read_csv(other_folder/"native.csv.gz")
            assert_same_values(native.iloc[:241], other_native.iloc[:241])
            pair = comparison(rule["metrics"], other["metrics"], gates)
            pair["different_actor_setpoint_samples"] = int((np.abs(native.oveTSet_u.iloc[241:].to_numpy()-other_native.oveTSet_u.iloc[241:].to_numpy()) > 1e-6).sum())
            pair["end_temperature_difference_K"] = other["end_state"]["reaTZon_y"]-rule["end_state"]["reaTZon_y"]
            pair["end_water_difference_K"] = {p: other["end_state"][p]-rule["end_state"][p] for p in ["reaTRet_y", "reaTSup_y"]}
            warning = plan["terminal_state_warning"]
            pair["terminal_storage_warning"] = (abs(pair["end_temperature_difference_K"]) >= warning["operative_difference_K"]
                or any(abs(v) >= warning["water_difference_K"] for v in pair["end_water_difference_K"].values()))
            comparisons[episode][kind] = pair
        branches = []
        for target in plan["response"]["targets_C"]:
            name = f"{episode}__response_{target:g}"
            branch = summaries[name]
            branch_native = pd.read_csv(directory/name/"native.csv.gz")
            assert_same_values(native.iloc[:241], branch_native.iloc[:241])
            assert_same_values(schedule.iloc[:721], pd.read_csv(directory/name/"schedule.csv.gz"))
            if rule["initial"] != branch["initial"]:
                raise ValueError("Response initial states differ")
            tail = branch_native.iloc[240:]
            branches.append({"energy_kWh": float(np.trapezoid(tail[POWER].sum(axis=1), tail.time)/3600000),
                "heat_kWh": float(np.trapezoid(tail.reaQFloHea_y, tail.time)/3600000),
                "operative_end_C": float(tail.reaTZon_y.iloc[-1]-273.15)})
        responses[episode] = response_check(*branches, plan["coverage"])
    return {"coverage": coverage, "active_dates": active, "responses": responses,
            "comparisons_vs_rule": comparisons, "development_gate": coverage_gate(coverage, active, responses, plan["coverage"])}


def run(output):
    plan, episodes, config = protocol()
    output.mkdir(parents=True, exist_ok=False)
    files = ["configs/v2_2.json", "docs/experiments/v2_2.md", "thermal_comfort/v2_baselines.py", *plan["parent_locks"]]
    save_json(output/"registration.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol": plan,
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": {p: digest(ROOT/p) for p in files}})
    model = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
    client = ScopedClient(Client(plan["live_engine"]), plan, [e["start_day"] for e in episodes], output)
    summaries = {}
    try:
        if client.call("GET", "version")["version"] != "0.9.0":
            raise ValueError("BOPTEST version changed")
        for episode in episodes:
            for kind in plan["controllers"]:
                name = f"{episode['id']}__{kind}"
                summaries[name] = run_episode(client, episode, kind, model, config, output/name)
        for episode in episodes:
            for target in plan["response"]["targets_C"]:
                name = f"{episode['id']}__response_{target:g}"
                summaries[name] = run_episode(client, episode, "guarded_rule", model, config, output/name, target)
        result = {"status": "completed_development_only_not_training_authorized", "training_runs": 0,
            "counts": client.counts, "wall_seconds": time.perf_counter()-client.started,
            "summaries": summaries, **analyze(output, summaries, plan)}
        save_json(output/"result.json", result)
        print(json.dumps({k: result[k] for k in ["status", "counts", "wall_seconds", "development_gate"]}), flush=True)
    except Exception as error:
        save_json(output/"failure.json", {"error": str(error), "counts": client.counts,
                                        "completed_runs": list(summaries)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
