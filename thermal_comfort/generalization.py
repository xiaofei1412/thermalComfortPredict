"""Frozen, zero-target-training transfer tests; original controller stays unchanged."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from .simulation import Client, FORECAST, POWER, digest, save_json, schedule_at
from .solar import SolarController, configuration


FREEZE = Path("outputs/generalization-freeze")
SOURCE = Path("outputs/boptest-v0.9.0")
CONFIG = Path("configs/generalization.json")
POLICY_FILES = [f"thermal_comfort/{n}.py" for n in ["solar", "greybox", "mpc", "recovery", "simulation"]]
POLICY_FILES += [f"configs/{n}.json" for n in ["solar_mpc", "recovery_mpc", "mpc", "dynamics"]]
FROZEN_FILES = POLICY_FILES + [str(CONFIG), "thermal_comfort/generalization.py", "docs/experiments/generalization.md"]


def interface(domain):
    if domain["interface"] == "house":
        return {"temperature": "reaTZon_y", "setpoint": "oveTSet_u", "active": "oveTSet_activate",
                "heat": "reaQFloHea_y", "heat_factor": 1., "electric": POWER, "gas": [],
                "weather_prefix": "weaSta_reaWea", "temperature_kind": "operative"}
    if domain["interface"] == "office":
        return {"temperature": "zon_reaTRooAir_y", "setpoint": "con_oveTSetHea_u", "active": "con_oveTSetHea_activate",
                "heat": "fcu_reaPHea_y", "heat_factor": .9,
                "electric": ["fcu_reaPCoo_y", "fcu_reaPFan_y"], "gas": ["fcu_reaPHea_y"],
                "weather_prefix": "zon_weaSta_reaWea", "temperature_kind": "air"}
    raise ValueError("Unknown physical interface")


def canonical(raw, io):
    # Internal policy aliases only; saved native data and KPI names retain physical meaning.
    return {**raw, "reaTZon_y": raw[io["temperature"]], "oveTSet_u": raw[io["setpoint"]],
            "reaQFloHea_y": raw[io["heat"]] * io["heat_factor"]}


def prepare():
    config = json.loads(CONFIG.read_text())
    if subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() != config["boptest_revision"]:
        raise ValueError("Unpinned BOPTEST source")
    for name in POLICY_FILES:
        original = subprocess.check_output(["git", "show", config["baseline_commit"]+":"+name])
        if Path(name).read_bytes().replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise ValueError("Source controller/config changed: "+name)
    old_splits = json.loads(Path("outputs/solar-selected/split_manifest.json").read_text())
    old_tests = [{"warmup_start_s": (d-7)*86400, "end_s": (d+7)*86400}
                 for d in configuration()["test_start_days"]]
    windows = [(d-7,d+config["days"]) for d in config["start_days"]]
    for start,end in windows:
        if any(start*86400 < s["end_s"] and end*86400 > s["warmup_start_s"] for s in old_splits+old_tests):
            raise ValueError("New evaluation/warmup overlaps a previous role")
    if any(a[1] > b[0] for a,b in zip(windows,windows[1:])):
        raise ValueError("New evaluation/warmup windows overlap")
    expected = json.loads(Path("reports/solar_mpc/model_manifest.json").read_text())["artifacts_sha256"]["model.joblib"]
    if digest(Path(config["model"])) != expected:
        raise ValueError("Source model differs from previous published model")
    FREEZE.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config["model"], FREEZE/"model.joblib")
    save_json(FREEZE/"policy_config.json", configuration())
    house = SOURCE/"testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"
    office = SOURCE/"testcases/bestest_air/models/wrapped.fmu"
    targets = {}
    for case, source in [("bestest_air", office), ("generalization_house_denver", house)]:
        target = FREEZE/"testcases"/case/"models/wrapped.fmu"
        target.parent.mkdir(parents=True)
        if case == "bestest_air":
            shutil.copyfile(source, target)
        else:
            with ZipFile(source) as old, ZipFile(office) as donor, ZipFile(target, "w") as new:
                for info in old.infolist():
                    content = old.read(info.filename)
                    if info.filename == "resources/0/Brussels.mos":
                        content = donor.read("resources/0/DRYCOLD.mos")
                    elif info.filename == "resources/weather.csv":
                        content = donor.read(info.filename)
                    elif info.filename == "resources/config.json":
                        embedded = json.loads(content)
                        embedded["name"] = case
                        content = (json.dumps(embedded, indent=2)+"\n").encode()
                    new.writestr(info, content)
        targets[case] = {"path": str(target), "sha256": digest(target)}
    targets["bestest_hydronic_heat_pump"] = {"path": str(house), "sha256": digest(house)}
    save_json(FREEZE/"freeze.json", {"created_UTC": datetime.now(timezone.utc).isoformat(),
        "config": config, "policy_config": configuration(), "model_sha256": expected,
        "files_sha256": {n: digest(Path(n)) for n in FROZEN_FILES}, "fmus": targets,
        "target_training_rows": 0, "source_revision": config["boptest_revision"],
        "new_windows_days":windows, "source_model_manifest_sha256":digest(Path("reports/solar_mpc/model_manifest.json")),
        "office_heat_mapping_source_sha256":digest(SOURCE/"testcases/bestest_air/models/BESTESTAir/BaseClasses/FanCoilUnit_T.mo")})
    print("Controller, model, protocol and FMUs frozen before target evaluation", flush=True)


def check_freeze(root=FREEZE):
    frozen = json.loads((root/"freeze.json").read_text())
    for name, expected in frozen["files_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Frozen source changed: "+name)
    if digest(root/"model.joblib") != frozen["model_sha256"]:
        raise ValueError("Frozen model changed")
    if json.loads((root/"policy_config.json").read_text()) != frozen["policy_config"]:
        raise ValueError("Frozen policy config changed")
    for item in frozen["fmus"].values():
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Frozen FMU changed")
    return frozen


def integrate(native, forecast, io, area):
    t = native.time.to_numpy()
    if len(t) < 2 or not np.isfinite(native.to_numpy(dtype=float)).all() or (np.diff(t) <= 0).any():
        raise ValueError("Invalid native trajectory")
    schedule = schedule_at(forecast, t)
    temperature = native[io["temperature"]].to_numpy()
    cold = np.maximum(schedule["LowerSetp[1]"].to_numpy()-temperature, 0)
    hot = np.maximum(temperature-schedule["UpperSetp[1]"].to_numpy(), 0)
    result = {}
    for key, names in [("electricity_kWh", io["electric"]), ("gas_kWh", io["gas"])]:
        result[key] = float(np.trapezoid(native[names].sum(axis=1), t)/3600000)
    result.update(total_energy_kWh=result["electricity_kWh"]+result["gas_kWh"],
        cold_Kh=float(np.trapezoid(cold,t)/3600), hot_Kh=float(np.trapezoid(hot,t)/3600),
        outside_hours=float(np.trapezoid(((cold+hot)>1e-8).astype(float),t)/3600),
        maximum_cold_K=float(cold.max()), maximum_hot_K=float(hot.max()),
        temperature_min_C=float(temperature.min()-273.15), temperature_max_C=float(temperature.max()-273.15))
    result["discomfort_Kh"] = result["cold_Kh"]+result["hot_Kh"]
    result["official_energy_kWh_m2"] = result["total_energy_kWh"]*3600000*2.77778e-7/area
    return result


def run_episode(client, frozen, domain, day, weather, kind, output, days):
    config, policy_config = frozen["config"], frozen["policy_config"]
    io = interface(domain)
    name = f"{domain['id']}_day{day}_{weather}"
    output.mkdir(parents=True, exist_ok=False)
    testid = client.call("POST", f"testcases/{domain['testcase']}/select")["testid"]
    started = time.perf_counter()
    records, failure = [], None
    try:
        inputs = client.call("GET", f"inputs/{testid}")
        measurements = client.call("GET", f"measurements/{testid}")
        if inputs[io["setpoint"]]["Unit"] != "K" or measurements[io["temperature"]]["Unit"] != "K":
            raise ValueError("Unexpected temperature units")
        for point in set(io["electric"]+io["gas"]+[io["heat"]]):
            if measurements[point]["Unit"] != "W":
                raise ValueError("Unexpected power units")
        save_json(output/"api_metadata.json", {"inputs": inputs, "measurements": measurements, "interface": io})
        client.call("PUT", f"scenario/{testid}", {"electricity_price": "constant"})
        before = client.call("PUT", f"initialize/{testid}", {
            "start_time": (day-config["common_prelude_days"])*86400,
            "warmup_period": config["embedded_warmup_days"]*86400})
        client.call("PUT", f"step/{testid}", {"step": 900})
        forecast = pd.DataFrame(client.call("PUT", f"forecast/{testid}", {
            "point_names": FORECAST, "horizon": (days+config["common_prelude_days"])*86400+48*900, "interval": 30}))
        forecast.to_csv(output/"exogenous.csv.gz", index=False, compression={"method":"gzip","mtime":0})
        prelude = []
        common_action = {io["setpoint"]: config["common_prelude_setpoint_C"]+273.15, io["active"]: 1}
        for _ in range(config["common_prelude_days"]*96):
            after = client.call("POST", f"advance/{testid}", common_action)
            prelude.append({"before": before, "action": common_action, "after": after})
            before = after
        save_json(output/"prelude.json", prelude)
        if before["time"] != day*86400:
            raise ValueError("Prelude misaligned")
        policy = SolarController(joblib.load(FREEZE/"model.joblib"), policy_config, kind, weather)
        policy.reset(canonical(before,io), {"oveTSet_u": inputs[io["setpoint"]]})
        start = before["time"]
        with (output/"steps.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(int(days*96)):
                canonical_before = canonical(before, io)
                try:
                    decision_start = time.perf_counter()
                    action, info = policy.decide(index, canonical_before, forecast)
                    info["decision_seconds"] = time.perf_counter()-decision_start
                except ValueError as error:
                    failure = str(error)
                    break
                actual = {io["setpoint"]: action["oveTSet_u"], io["active"]: action["oveTSet_activate"]}
                after = client.call("POST", f"advance/{testid}", actual)
                record = {"before": canonical_before, "action": action, "actual_action": actual,
                          "after": canonical(after,io), "control": info}
                stream.write(json.dumps(record,allow_nan=False)+"\n"); stream.flush()
                records.append(record)
                interval = pd.DataFrame(client.call("PUT", f"results/{testid}", {
                    "point_names": [io["heat"]], "start_time": before["time"], "final_time": after["time"]}))
                interval["reaQFloHea_y"] = interval[io["heat"]]*io["heat_factor"]
                try:
                    policy.update(canonical_before,canonical(after,io),interval,schedule_at(forecast,[before["time"]]).iloc[0])
                except ValueError as error:
                    failure = str(error)
                    before = after
                    break
                before = after
                if (index+1)%96 == 0:
                    print(f"{name}/{kind}: {(index+1)//96} days",flush=True)
        points = list(dict.fromkeys([io["temperature"],io["setpoint"],io["active"],io["heat"]]+io["electric"]+io["gas"]+
                                   [io["weather_prefix"]+n+"_y" for n in ["TDryBul","HGloHor","Lat","Lon"]]))
        native = pd.DataFrame(client.call("PUT",f"results/{testid}",{
            "point_names": points,"start_time":start,"final_time":before["time"]}))
        native.to_csv(output/"native.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        kpis = client.call("GET",f"kpi/{testid}")
        save_json(output/"api_kpis.json",kpis)
        # API KPI includes the common prelude. Integrate that same entire period independently.
        all_native = pd.DataFrame(client.call("PUT",f"results/{testid}",{
            "point_names":points,"start_time":prelude[0]["before"]["time"],"final_time":before["time"]}))
        all_native.to_csv(output/"including_prelude.csv.gz",index=False,compression={"method":"gzip","mtime":0})
        full_metrics = integrate(all_native,forecast,io,domain["area_m2"])
        differences = {official:abs(full_metrics[local]-kpis[official]) for local,official in [
            ("official_energy_kWh_m2","ener_tot"),("discomfort_Kh","tdis_tot")]}
        if max(differences.values()) > 1e-6:
            raise ValueError("Native full-period/API KPI mismatch: "+str(differences))
        metrics = integrate(native,forecast,io,domain["area_m2"]) if len(native)>1 else None
        summary = {"scenario":name,"domain":domain,"day":day,"weather":weather,"controller":kind,
            "days":days,"complete":failure is None and len(records)==int(days*96),"failure":failure,
            "transitions":len(records),"native_samples":len(native),"metrics":metrics,"wall_seconds":time.perf_counter()-started,
            "api_full_period_difference":differences,
            "sha256":{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
        save_json(output/"summary.json",summary)
        return summary
    finally:
        client.call("PUT",f"stop/{testid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare","run"])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--days",type=float)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
        return
    frozen = check_freeze()
    config = frozen["config"]
    days = config["days"] if args.days is None else args.days
    if days <= 0 or days*96 != int(days*96) or days > config["days"]:
        raise ValueError("Invalid evaluation length")
    args.output.mkdir(parents=True,exist_ok=False)
    client = Client("http://127.0.0.1:8080")
    if client.call("GET","version")["version"] != "0.9.0":
        raise ValueError("BOPTEST version mismatch")
    save_json(args.output/"manifest.json",{"freeze_sha256":digest(FREEZE/"freeze.json"),"days":days,
        "started_UTC":datetime.now(timezone.utc).isoformat(),"frozen":frozen})
    summaries = []
    for domain in config["domains"]:
        for day in config["start_days"]:
            for weather in config["weather_modes"]:
                for kind in config["controllers"]:
                    check_freeze()
                    name = f"{domain['id']}_day{day}_{weather}__{kind}"
                    summaries.append(run_episode(client,frozen,domain,day,weather,kind,args.output/name,days))
                    save_json(args.output/"summary.json",summaries)
    check_freeze()


if __name__ == "__main__":
    main()
