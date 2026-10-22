"""Fixed BOPTEST single-zone experiments; no learned comfort model is assumed."""

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


POWER = ["reaPHeaPum_y", "reaPFan_y", "reaPPumEmi_y"]
POINTS = ["reaTZon_y", "reaTSetHea_y", "reaTSetCoo_y", "reaQFloHea_y",
          "reaTRet_y", "reaTSup_y", "oveTSet_u", "oveTSet_activate", *POWER]
FORECAST = ["LowerSetp[1]", "UpperSetp[1]", "Occupancy[1]", "TDryBul", "HGloHor",
            "PriceElectricPowerConstant", "EmissionsElectricPower"]


class Client:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def call(self, method, path, data=None):
        if data is None and method in ("POST", "PUT"):
            data = {}
        request = urllib.request.Request(
            self.url + "/" + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}, method=method)
        # Mutations must never be retried: a timeout can follow an applied action.
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
            if path.startswith("stop/"):
                return None  # The service returns plain text, rather than a JSON envelope.
            result = json.loads(body)
        if isinstance(result, dict) and "status" in result:
            if result["status"] != 200:
                raise RuntimeError(f"BOPTEST {path}: {result}")
            return result["payload"]
        return result


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule_action(lower_K, occupancy, limits, config):
    offset = config["occupied_offset_K"] if occupancy > 0 else config["unoccupied_offset_K"]
    return {"oveTSet_u": float(np.clip(lower_K + offset, limits["Minimum"], limits["Maximum"])),
            "oveTSet_activate": 1}


def schedule_at(forecast, times):
    """BOPTEST non-weather schedules use a zero-order hold, not linear interpolation."""
    index = np.searchsorted(forecast["time"].to_numpy(), times, side="right") - 1
    if (index < 0).any() or np.max(times) > forecast["time"].iloc[-1]:
        raise ValueError("Forecast does not cover the requested interval")
    return forecast.iloc[index].reset_index(drop=True)


def integrate(native, forecast, area):
    t = native["time"].to_numpy()
    if len(t) < 2 or not np.isfinite(native.to_numpy(dtype=float)).all() or (np.diff(t) < 0).any():
        raise ValueError("Native trajectory is incomplete, non-finite or unordered")
    schedule = schedule_at(forecast, t)
    power = native[POWER].sum(axis=1).to_numpy()
    violation = (np.maximum(schedule["LowerSetp[1]"].to_numpy() - native["reaTZon_y"], 0)
                 + np.maximum(native["reaTZon_y"] - schedule["UpperSetp[1]"].to_numpy(), 0))
    energy_ws = float(np.trapezoid(power, t))
    return {"energy_kWh": energy_ws / 3600000,
            "energy_kWh_m2": energy_ws / 3600000 / area,
            # Official v0.9.0 uses this rounded conversion factor.
            "boptest_energy_kWh_m2": energy_ws * 2.77778e-7 / area,
            "discomfort_Kh": float(np.trapezoid(violation, t) / 3600),
            "cost_EUR_m2": float(np.trapezoid(power * schedule["PriceElectricPowerConstant"], t)
                                 * 2.77778e-7 / area),
            "emissions_kg_m2": float(np.trapezoid(power * schedule["EmissionsElectricPower"], t)
                                     * 2.77778e-7 / area)}


def validate_kpis(metrics, kpis):
    differences = {}
    for local, official in [("boptest_energy_kWh_m2", "ener_tot"),
                            ("discomfort_Kh", "tdis_tot"), ("cost_EUR_m2", "cost_tot"),
                            ("emissions_kg_m2", "emis_tot")]:
        difference = abs(metrics[local] - kpis[official])
        if not np.isclose(metrics[local], kpis[official], rtol=1e-6, atol=1e-6):
            raise ValueError(f"KPI mismatch: {local}={metrics[local]}, API={kpis[official]}")
        differences[official] = difference
    return differences


def transitions_frame(records, native, forecast, config):
    rows = []
    for record in records:
        start, end = record["before"]["time"], record["after"]["time"]
        if end - start != config["step_s"]:
            raise ValueError("Unexpected control interval")
        segment = native.loc[native["time"].between(start, end)]
        if segment.empty or segment["time"].iloc[0] != start or segment["time"].iloc[-1] != end:
            raise ValueError("Missing native interval boundary")
        action = record["action"]
        # The initial boundary belongs to the previous action; subsequent samples must match.
        applied = segment.loc[segment["time"] > start, "oveTSet_u"]
        active = segment.loc[segment["time"] > start, "oveTSet_activate"]
        if not (active == action["oveTSet_activate"]).all():
            raise ValueError("Activation was not applied")
        if action["oveTSet_activate"] and not np.allclose(applied, action["oveTSet_u"], atol=1e-7, rtol=0):
            raise ValueError("Setpoint was not applied")
        current = schedule_at(forecast, np.array([start])).iloc[0]
        metrics = integrate(segment, forecast, config["area_m2"])
        rows.append({"time_start_s": start, "time_end_s": end,
                     "operative_start_C": record["before"]["reaTZon_y"] - 273.15,
                     "operative_end_C": record["after"]["reaTZon_y"] - 273.15,
                     "lower_start_C": current["LowerSetp[1]"] - 273.15,
                     "upper_start_C": current["UpperSetp[1]"] - 273.15,
                     "occupancy_start": current["Occupancy[1]"],
                     "outdoor_start_C": current["TDryBul"] - 273.15,
                     "solar_start_W_m2": current["HGloHor"],
                     "setpoint_requested_C": action.get("oveTSet_u", None) - 273.15
                         if "oveTSet_u" in action else None,
                     "override_active": action["oveTSet_activate"],
                     "setpoint_applied_end_C": float(applied.iloc[-1]) - 273.15,
                     "interval_energy_kWh": metrics["energy_kWh"],
                     "interval_discomfort_Kh": metrics["discomfort_Kh"]})
    frame = pd.DataFrame(rows)
    if len(frame) > 1 and not np.array_equal(frame.time_start_s.to_numpy()[1:], frame.time_end_s.to_numpy()[:-1]):
        raise ValueError("Control trajectory has gaps or overlaps")
    return frame


def run_episode(client, config, scenario, controller, output):
    output.mkdir(parents=True, exist_ok=False)
    testid = client.call("POST", f"testcases/{config['testcase']}/select")["testid"]
    started = time.perf_counter()
    try:
        inputs = client.call("GET", f"inputs/{testid}")
        measurements = client.call("GET", f"measurements/{testid}")
        forecast_points = client.call("GET", f"forecast_points/{testid}")
        if inputs["oveTSet_u"]["Unit"] != "K" or measurements["reaTZon_y"]["Unit"] != "K":
            raise ValueError("Unexpected temperature units")
        for point in POWER:
            if measurements[point]["Unit"] != "W":
                raise ValueError("Unexpected power units")
        save_json(output / "api_metadata.json", {"inputs": inputs, "measurements": measurements,
                                                "forecast_points": forecast_points})
        if controller == "excitation":
            client.call("PUT", f"scenario/{testid}", {"electricity_price": "constant"})
            initial = client.call("PUT", f"initialize/{testid}",
                                  {"start_time": config["episode"]["start_day"] * 86400,
                                   "warmup_period": config["warmup_days"] * 86400})
            from .excitation import SafeExcitation
            policy = SafeExcitation(config, inputs["oveTSet_u"], initial)
        else:
            initial = client.call("PUT", f"scenario/{testid}",
                                  {"electricity_price": "constant", "time_period": scenario})["time_period"]
        client.call("PUT", f"step/{testid}", {"step": config["step_s"]})
        if client.call("GET", f"step/{testid}") != config["step_s"]:
            raise ValueError("Control step was not set")
        duration = config["days"] * 86400
        if duration <= 0 or duration % config["step_s"] or duration > 14 * 86400:
            raise ValueError("Duration must be whole control intervals within the official 14-day period")
        forecast = pd.DataFrame(client.call("PUT", f"forecast/{testid}",
                                {"point_names": FORECAST, "horizon": duration, "interval": 30}))
        forecast.to_csv(output / "exogenous.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        before, records = initial, []
        with (output / "steps.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(int(duration // config["step_s"])):
                current = schedule_at(forecast, np.array([before["time"]])).iloc[0]
                if controller == "rule":
                    action = rule_action(current["LowerSetp[1]"], current["Occupancy[1]"], inputs["oveTSet_u"], config)
                elif controller == "embedded":
                    action = {"oveTSet_activate": 0}
                elif controller == "excitation":
                    action, safety = policy.action(index, before, current)
                else:
                    raise ValueError(f"Unknown controller: {controller}")
                after = client.call("POST", f"advance/{testid}", action)
                record = {"before": before, "action": action, "after": after}
                if controller == "excitation":
                    record["safety"] = safety
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                if controller == "excitation":
                    policy.check_temperature(after["reaTZon_y"] - 273.15)
                if after["time"] - before["time"] != config["step_s"]:
                    raise ValueError("Advance returned an unexpected time")
                before = after
                if (index + 1) % 96 == 0:
                    print(f"{scenario}/{controller}: {(index + 1) // 96} days", flush=True)
        raw = client.call("PUT", f"results/{testid}", {"point_names": POINTS,
                          "start_time": initial["time"], "final_time": before["time"]})
        native = pd.DataFrame(raw)
        native.to_csv(output / "native.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        kpis = client.call("GET", f"kpi/{testid}")
        save_json(output / "api_kpis.json", kpis)
        metrics = integrate(native, forecast, config["area_m2"])
        differences = validate_kpis(metrics, kpis)
        transitions = transitions_frame(records, native, forecast, config)
        if not np.isclose(transitions.interval_energy_kWh.sum(), metrics["energy_kWh"], rtol=1e-10):
            raise ValueError("Interval energy does not sum to episode energy")
        transitions.to_csv(output / "transitions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        summary = {"scenario": scenario, "controller": controller, "config": config,
                   "start_time_s": initial["time"], "end_time_s": before["time"],
                   "transitions": len(records), "native_samples": len(native),
                   "metrics": metrics, "kpis": kpis, "kpi_absolute_difference": differences,
                   "wall_seconds": time.perf_counter() - started,
                   "sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
        save_json(output / "summary.json", summary)
        return summary
    finally:
        client.call("PUT", f"stop/{testid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    parser.add_argument("--days", type=float, help="Short smoke run; recorded in effective config")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.days is not None:
        config["days"] = args.days
    client = Client(args.url)
    version = client.call("GET", "version")
    if version["version"] != config["version"]:
        raise ValueError(f"Expected BOPTEST {config['version']}, got {version}")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"url": args.url, "version": version, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "config": config,
                "code_sha256": digest(Path(__file__)), "config_sha256": digest(args.config)}
    save_json(args.output / "manifest.json", metadata)
    summaries = []
    for scenario in config["scenarios"]:
        for controller in config["controllers"]:
            summaries.append(run_episode(client, config, scenario, controller,
                                         args.output / f"{scenario}__{controller}"))
    save_json(args.output / "summary.json", summaries)
    print(json.dumps([{k: s[k] for k in ["scenario", "controller", "transitions", "metrics"]}
                      for s in summaries], indent=2))


if __name__ == "__main__":
    main()
