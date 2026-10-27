"""One development189 diagnostic round; archived actions, no policy training."""

import argparse
from decimal import Decimal, localcontext
import gzip
import json
from pathlib import Path
import subprocess
import time
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .rl_interface import bounded_forecast
from .simulation import Client, FORECAST, POINTS, digest, save_json, validate_kpis
from .v2_baselines import ScopedClient, assert_same_values, check_action, score, stitch
from .v2_protocol import ROOT, read, verify_files


ARCHIVE = ROOT/"outputs/v2-2/live-registered"
EXTRA = ["reaQHeaPumCon_y", "reaQHeaPumEva_y", "weaSta_reaWeaTDryBul_y",
         "weaSta_reaWeaHGloHor_y", "oveHeaPumY_activate", "oveFan_activate", "ovePum_activate"]


class DiagnosticClient(ScopedClient):
    """Narrow the frozen development client to one day and four selects."""

    def __init__(self, client, plan, output):
        super().__init__(client, plan, [84, 168, 189], output)
        self.counts["select"] = 0

    def call(self, method, path, data=None):
        if path.startswith("initialize/") and data != {"start_time": 189*86400, "warmup_period": 604800}:
            raise ValueError("Diagnostic scope is development189 only")
        if path.startswith("testcases/"):
            if path != "testcases/bestest_hydronic_heat_pump/select" or method != "POST":
                raise ValueError("Only the original case may be selected")
            if self.counts["select"] >= self.plan["live_budget"]["select_attempts_max"]:
                raise RuntimeError("Diagnostic select budget exhausted")
            self.counts["select"] += 1
        return super().call(method, path, data)


def load_plan():
    plan = read(ROOT/"configs/v2_3_numerics.json")
    if (plan["episode"] != "development189" or plan["start_day"] != 189
            or plan["training_authorized"] or plan["tolerance_changes_authorized"]
            or plan["scored_steps"] != 672 or plan["step_s"] != 900 or plan["native_interval_s"] != 30
            or plan["warmup_s"] != 604800 or plan["runs"] != [
                {"id": "rule_A", "archived_kind": "guarded_rule"},
                {"id": "rule_B", "archived_kind": "guarded_rule"},
                {"id": "mpc_replay", "archived_kind": "solar_mpc"},
                {"id": "minimum_replay", "archived_kind": "minimum_heat"}]):
        raise ValueError("Only the registered one-round development diagnostic is allowed")
    for filename in plan["parent_locks"]:
        verify_files(ROOT, read(ROOT/filename)["files_sha256"])
    v1 = read(ROOT/"reports/v2_0/lock.json")["v1_baseline"]
    verify_files(ROOT, v1["files_sha256"])
    fmu = ROOT/"outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"
    if digest(fmu) != v1["fmu_sha256"]:
        raise ValueError("FMU changed")
    metadata = read(ROOT/"reports/v2_2/summary.json")
    actions = {}
    for kind in ["guarded_rule", "solar_mpc", "minimum_heat"]:
        name = "development189__"+kind
        folder = ARCHIVE/name
        if digest(folder/"summary.json") != metadata["runs"][name]["summary_sha256"]:
            raise ValueError("Archived summary changed")
        summary = read(folder/"summary.json")
        verify_files(folder, summary["sha256"])
        if summary["episode"] != {"id": "development189", "role": "development", "start_day": 189}:
            raise ValueError("Archived role differs")
        records = [json.loads(s) for s in (folder/"steps.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(records) != 672:
            raise ValueError("Archived action waveform incomplete")
        actions[kind] = [r["action"] for r in records]
    if any(a[:8] != actions["guarded_rule"][:8] for a in actions.values()):
        raise ValueError("Archived common prefixes differ")
    return plan, actions


def source_metadata():
    folder = ROOT/"outputs/boptest-v0.9.0"
    fmu = folder/"testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"
    xml = ElementTree.fromstring(ZipFile(fmu).read("modelDescription.xml"))
    params = {v.attrib["name"]: list(v)[0].attrib for v in xml.find("ModelVariables")
              if v.attrib["name"] in ["_cs_rel_tol", "_cs_solver", "_cs_step_size", "_events_default_tol"]}
    names = ["testcase.py", "testcases/bestest_hydronic_heat_pump/models/wrapped.mo",
             "testcases/bestest_hydronic_heat_pump/models/config.json",
             "testcases/bestest_hydronic_heat_pump/models/BESTESTHydronicHeatPump/TestCase.mo"]
    return {"fmu_sha256": digest(fmu), "fmi": xml.attrib,
        "cosimulation": xml.find("CoSimulation").attrib, "default_experiment_present": xml.find("DefaultExperiment") is not None,
        "internal_parameter_metadata_not_output_error_bounds": params,
        "bootstrap_config": read(folder/names[2]),
        "source_sha256": {n: digest(folder/n) for n in names}}


def check_zero(native, plan):
    maxima = {name: float(np.max(np.abs(native[name]))) for name in plan["zero_channels"]}
    if max(maxima.values()) > plan["sampled_zero_flow_absolute_W"]:
        raise ValueError("Nonzero sampled energy flow; stop zero-load attribution")
    if (native[["oveHeaPumY_activate", "oveFan_activate", "ovePum_activate"]] != 0).any().any():
        raise ValueError("Unexpected additional actuator override")
    return maxima


def run_one(client, plan, waveform, output):
    output.mkdir()
    started = time.perf_counter()
    testid = client.call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
    frames, schedules, records = [], [], []
    try:
        inputs = client.call("GET", f"inputs/{testid}")
        measurements = client.call("GET", f"measurements/{testid}")
        for name in plan["zero_channels"]:
            if measurements[name]["Unit"] != "W":
                raise ValueError("Unexpected energy-flow unit")
        save_json(output/"api_metadata.json", {"inputs": inputs, "measurements": measurements})
        client.call("PUT", f"scenario/{testid}", {"electricity_price": "constant"})
        initial = client.call("PUT", f"initialize/{testid}", {"start_time": 189*86400, "warmup_period": 604800})
        if initial["time"] != 189*86400:
            raise ValueError("Initialization time differs")
        client.call("PUT", f"step/{testid}", {"step": 900})
        if client.call("GET", f"step/{testid}") != 900:
            raise ValueError("Control interval differs")
        before = initial
        with gzip.open(output/"native_api.jsonl.gz", "wt", encoding="utf-8") as raw_log, gzip.open(
                output/"forecasts.jsonl.gz", "wt", encoding="utf-8") as forecast_log, (
                output/"steps.jsonl").open("w", encoding="utf-8") as step_log:
            for index, action in enumerate(waveform):
                view = bounded_forecast(client.call("PUT", f"forecast/{testid}", {
                    "point_names": FORECAST, "horizon": 43200, "interval": 30}), before["time"])
                forecast_log.write(json.dumps(view.to_dict("list"), allow_nan=False)+"\n")
                check_action(action, before)
                after = client.call("POST", f"advance/{testid}", action)
                if after["time"] != before["time"]+900:
                    raise ValueError("Advance time differs")
                raw = client.call("PUT", f"results/{testid}", {"point_names": POINTS+EXTRA,
                    "start_time": before["time"], "final_time": after["time"]})
                raw_log.write(json.dumps(raw, allow_nan=False)+"\n")
                native = pd.DataFrame(raw)
                frames.append(native)
                schedules.append(view.iloc[:31].reset_index(drop=True))
                check_action(action, before, native)
                check_zero(native, plan)
                record = {"index": index, "before": before, "action": action, "after": after}
                step_log.write(json.dumps(record, allow_nan=False)+"\n")
                step_log.flush()
                records.append(record)
                before = after
                if (index+1) % 96 == 0:
                    print(f"{output.name}: {index+1}/672", flush=True)
        native, schedule = stitch(frames), stitch(schedules)
        metrics = score(native, schedule)
        kpis = client.call("GET", f"kpi/{testid}")
        native.to_csv(output/"native.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        schedule.to_csv(output/"schedule.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        summary = {"transitions": len(records), "native_points": len(native), "initial": initial,
            "metrics": metrics, "kpis": kpis, "zero_channel_max_abs_W": check_zero(native, plan),
            "kpi_absolute_difference": validate_kpis(metrics, kpis), "wall_seconds": time.perf_counter()-started,
            "sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
        save_json(output/"summary.json", summary)
        return summary
    except Exception as error:
        save_json(output/"failure.json", {"error": str(error), "completed_steps": len(records)})
        raise
    finally:
        client.call("PUT", f"stop/{testid}")


def decimal_thermal(native, schedule, precision=50):
    with localcontext() as context:
        context.prec = precision
        times = [Decimal(str(x)) for x in native.time]
        temp = [Decimal(str(x)) for x in native.reaTZon_y]
        result = {}
        for kind, name in [("cold", "LowerSetp[1]"), ("hot", "UpperSetp[1]")]:
            bound = [Decimal(str(x)) for x in schedule[name]]
            d = [max(b-t if kind == "cold" else t-b, Decimal(0)) for t, b in zip(temp, bound, strict=True)]
            integral = sum(((d[i]+d[i+1])*(times[i+1]-times[i])/2 for i in range(len(d)-1)), Decimal(0))/3600
            result[kind+"_Kh"] = str(integral)
        return result


def raw_frame(folder):
    with gzip.open(folder/"native_api.jsonl.gz", "rt", encoding="utf-8") as stream:
        return stitch([pd.DataFrame(json.loads(line)) for line in stream])


def precision_check(folder, native, schedule, plan):
    round_trip = pd.read_csv(folder/"native.csv.gz", float_precision="round_trip")
    assert_same_values(native[round_trip.columns], round_trip)
    default = pd.read_csv(folder/"native.csv.gz")
    exact = decimal_thermal(native, schedule, plan["analysis"]["decimal_precision"])
    current, parsed = score(native, schedule), score(default, schedule)
    decimal_difference = {k: abs(float(exact[k])-current[k]) for k in exact}
    csv_difference = {k: abs(parsed[k]-current[k]) for k in exact}
    return {"decimal_thermal_Kh": exact, "float_vs_decimal_abs_Kh": decimal_difference,
        "csv_default_vs_api_max_temperature_K": float(np.max(np.abs(default.reaTZon_y-native.reaTZon_y))),
        "csv_default_vs_api_abs_Kh": csv_difference,
        "api_json_to_csv_round_trip_equal": True,
        "postprocessing_below_registered_diagnostic_limit": max(*decimal_difference.values(), *csv_difference.values()) <= plan["analysis"]["postprocessing_Kh_difference_max"]}


def difference(left, right, schedule, tolerance):
    delta = right.reaTZon_y.to_numpy()-left.reaTZon_y.to_numpy()
    command_delta = right.oveTSet_u.to_numpy()-left.oveTSet_u.to_numpy()
    changed = np.flatnonzero(command_delta != 0)
    thermal_changed = np.flatnonzero(delta != 0)
    a, b = score(left, schedule), score(right, schedule)
    changes = {f"{kind}_{metric}": b[f"{kind}_{metric}"]-a[f"{kind}_{metric}"]
               for kind in ["cold", "hot"] for metric in ["Kh", "duration_s", "max_K"]}
    return {"temperature_max_abs_K": float(np.max(np.abs(delta))), "thermal_differences": changes,
        "first_action_difference_s": float(left.time.iloc[changed[0]]) if len(changed) else None,
        "first_temperature_difference_s": float(left.time.iloc[thermal_changed[0]]) if len(thermal_changed) else None,
        "hot_integral_abs_difference_over_original_tolerance": abs(changes["hot_Kh"])/tolerance,
        "original_degree_hour_nonincrease_passed": changes["cold_Kh"] <= tolerance and changes["hot_Kh"] <= tolerance}


def analyze(directory, plan):
    frames, schedules, summaries, precisions, archived = {}, {}, {}, {}, {}
    for run in plan["runs"]:
        name, kind = run["id"], run["archived_kind"]
        folder = directory/name
        summaries[name] = read(folder/"summary.json")
        verify_files(folder, summaries[name]["sha256"])
        frames[name] = raw_frame(folder)
        schedules[name] = pd.read_csv(folder/"schedule.csv.gz", float_precision="round_trip")
        check_zero(frames[name], plan)
        precisions[name] = precision_check(folder, frames[name], schedules[name], plan)
        old = pd.read_csv(ARCHIVE/f"development189__{kind}/native.csv.gz", float_precision="round_trip")
        archived[name] = {"temperature_max_abs_difference_K": float(np.max(np.abs(old.reaTZon_y-frames[name].reaTZon_y))),
            "hot_integral_difference_Kh": summaries[name]["metrics"]["hot_Kh"]-read(ARCHIVE/f"development189__{kind}/summary.json")["metrics"]["hot_Kh"]}
        assert_same_values(schedules["rule_A"], schedules[name])
        if summaries["rule_A"]["initial"] != summaries[name]["initial"]:
            raise ValueError("Diagnostic initialization differs")
        assert_same_values(frames["rule_A"].iloc[:241], frames[name].iloc[:241])
    tolerance = read(ROOT/"configs/v2_0.json")["comparison_gates"]["cold_and_hot_degree_hour_increase_tolerance"]
    pairs = {name: difference(frames["rule_A"], frames[name], schedules[name], tolerance) for name in frames if name != "rule_A"}
    limit = plan["analysis"]
    repeatable = (pairs["rule_B"]["temperature_max_abs_K"] <= limit["same_waveform_max_abs_temperature_K"]
                  and max(abs(v) for k, v in pairs["rule_B"]["thermal_differences"].items() if k.endswith("Kh")) <= limit["same_waveform_max_abs_Kh"])
    represented = all(p["postprocessing_below_registered_diagnostic_limit"] for p in precisions.values())
    input_path = (repeatable and represented and any(pairs[n]["hot_integral_abs_difference_over_original_tolerance"] > 1
        and pairs[n]["first_action_difference_s"] is not None
        and pairs[n]["first_temperature_difference_s"] >= pairs[n]["first_action_difference_s"] for n in ["mpc_replay", "minimum_replay"]))
    hot_hours = summaries["rule_A"]["metrics"]["hot_duration_s"]/3600
    return {"status": "diagnostic_complete_original_failure_preserved", "same_waveform_repeatable": repeatable,
        "postprocessing_excluded_at_registered_scale": represented, "input_associated_FMU_path_evidence": bool(input_path),
        "specific_internal_solver_cause_proven": False, "solver_convergence_test_run": False,
        "original_tolerances_changed": False, "training_authorized": False,
        "original_Kh_tolerance": tolerance, "equivalent_mean_K_over_7days": tolerance/168,
        "equivalent_mean_K_over_original_hot_hours": tolerance/hot_hours if hot_hours else None,
        "summaries": summaries, "precision_checks": precisions, "archived_reproduction": archived, "pairs_vs_rule_A": pairs}


def run(directory):
    plan, waveforms = load_plan()
    directory.mkdir(parents=True, exist_ok=False)
    files = ["configs/v2_3_numerics.json", "docs/experiments/v2_3_numerics.md", "thermal_comfort/v2_numerics.py", *plan["parent_locks"]]
    save_json(directory/"registration.json", {"plan": plan,
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files_sha256": {p: digest(ROOT/p) for p in files}, "source_metadata": source_metadata(),
        "archived_action_sha256": {kind: digest(ARCHIVE/f"development189__{kind}/steps.jsonl") for kind in waveforms}})
    client = DiagnosticClient(Client(plan["live_engine"]), plan, directory)
    try:
        if client.call("GET", "version")["version"] != "0.9.0":
            raise ValueError("BOPTEST version changed")
        for item in plan["runs"]:
            run_one(client, plan, waveforms[item["archived_kind"]], directory/item["id"])
        result = analyze(directory, plan)
        result.update(counts=client.counts, wall_seconds=time.perf_counter()-client.started)
        save_json(directory/"result.json", result)
        print(json.dumps({k: result[k] for k in ["status", "counts", "wall_seconds", "same_waveform_repeatable", "input_associated_FMU_path_evidence"]}), flush=True)
    except Exception as error:
        save_json(directory/"failure.json", {"error": str(error), "counts": client.counts})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
