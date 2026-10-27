"""Replay the B engineering timeline with a demand-adopting common rule, not a new holdout."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from thermal_comfort.demand_rule import BASELINE, common_action
from thermal_comfort.simulation import Client, FORECAST, POINTS, POWER, digest, integrate, save_json, transitions_frame, validate_kpis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--workbench-url", default="http://127.0.0.1:8768")
    parser.add_argument("--source", type=Path, default=Path("outputs/workbench/sessions/0a2e84c3cb66.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/residential-c-smoke"))
    args = parser.parse_args()
    with urlopen(args.workbench_url + "/api/state", timeout=10) as response:
        state = json.load(response)
    if state["busy"] or state.get("can_end"):
        raise RuntimeError("先正常结束工作台会话；验收不会接管活跃仿真")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source["status"] != "completed" or any(e["intent"]["mode"] != "needs" for e in source["events"]):
        raise ValueError("只复核正常结束的按需求源记录")
    args.output.mkdir(parents=True, exist_ok=False)
    client = Client(args.url)
    if client.call("GET", "version")["version"] != "0.9.0":
        raise ValueError("引擎版本不符")
    provenance = {"baseline": BASELINE, "role": "engineering_only_existing_day63",
                  "source": args.source.as_posix(), "source_sha256": digest(args.source),
                  "code_sha256": {p: digest(Path(p)) for p in ["thermal_comfort/demand_rule.py", "thermal_comfort/demands.py", "thermal_comfort/workbench.py", "thermal_comfort/simulation.py", "configs/residential_c.json"]}}
    save_json(args.output / "provenance.json", provenance)
    runs = []
    first_records = None
    for run in range(2):
        directory = args.output / f"repeat{run+1}"
        directory.mkdir()
        testid = client.call("POST", "testcases/bestest_hydronic_heat_pump/select")["testid"]
        records = []
        try:
            client.call("PUT", f"scenario/{testid}", {"electricity_price": "constant"})
            p = source["provenance"]
            before = client.call("PUT", f"initialize/{testid}", {"start_time": p["start_s"], "warmup_period": p["warmup_s"]})
            client.call("PUT", f"step/{testid}", {"step": 900})
            if not np.allclose([before[k] for k in POINTS], [source["initial"][k] for k in POINTS], atol=1e-7, rtol=0):
                raise ValueError("初态不一致")
            for index, event in enumerate(source["events"]):
                action, decision = common_action(event["intent"], event["demand_snapshot"], before)
                assert action == event["action"], "共同规则与原按需求映射不一致"
                assert decision["evaluation"] == event["demand_evaluation"]
                forecast = pd.DataFrame(client.call("PUT", f"forecast/{testid}", {"point_names": FORECAST, "horizon": 900, "interval": 30}))
                after = client.call("POST", f"advance/{testid}", action)
                raw = pd.DataFrame(client.call("PUT", f"results/{testid}", {"point_names": POINTS, "start_time": before["time"], "final_time": after["time"]}))
                record = {"before": before, "action": action, "after": after}
                transitions_frame([record], raw, forecast, {"step_s": 900, "area_m2": 192})
                assert len(raw) == 31 and np.allclose(np.diff(raw.time), 30)
                assert (raw[POWER].to_numpy() >= -1e-6).all()
                expected = pd.DataFrame(source["intervals"][index]["native"])
                assert np.allclose(raw[["time", *POINTS]], expected[["time", *POINTS]], atol=1e-7, rtol=0)
                assert np.allclose([after[k] for k in POINTS], [event["after"][k] for k in POINTS], atol=1e-7, rtol=0)
                record.update(decision=decision, demand_snapshot=event["demand_snapshot"], metrics=integrate(raw, forecast, 192),
                              native=raw.to_dict("list"), forecast=forecast.to_dict("list"))
                records.append(record)
                save_json(directory / "steps.json", records)
                before = after
            kpis = client.call("GET", f"kpi/{testid}")
            sums = {key: sum(r["metrics"][key] for r in records) for key in records[0]["metrics"]}
            differences = validate_kpis(sums, kpis)
            if first_records is None:
                first_records = records
            else:
                assert records == first_records, "独立复跑原生数据/决策不一致"
            row = {"steps": len(records), "native_interval_points": sum(len(r["native"]["time"]) for r in records),
                   "same_demand_snapshots": True, "same_actions_as_B_needs": True, "native_matches_B_atol": 1e-7,
                   "kpi_absolute_difference": differences, "metrics": sums}
            save_json(directory / "verification.json", row)
            runs.append(row)
        finally:
            client.call("PUT", f"stop/{testid}")
    assert runs[0] == runs[1]
    save_json(args.output / "verification.json", {**provenance, "runs": runs, "exact_repeat": True,
                                                "new_model_or_energy_efficacy_claim": False})
    print("Common demand rule: two independent initializations, actions/requests/native/KPI checks passed")


if __name__ == "__main__":
    main()
