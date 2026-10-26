"""Smoke-check the running workbench against real BOPTEST, using demo data only."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from thermal_comfort.simulation import integrate, transitions_frame, validate_kpis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--scenes", nargs="+", choices=["cool", "warm", "cold"], default=["cool", "warm", "cold"])
    parser.add_argument("--output", type=Path, default=Path("outputs/workbench/verification.json"))
    args = parser.parse_args()

    def api(path, data=None):
        request = Request(args.url + path, data=None if data is None else json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def wait():
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            state = api("/api/state")
            if not state["busy"]:
                if state["session"] is None or state["session"]["status"] == "error":
                    raise AssertionError(state["message"])
                return state["session"]
            time.sleep(.2)
        raise TimeoutError("Workbench job did not complete")

    def verify(sid):
        session = api("/api/download/" + sid)
        energies = []
        for event, interval in zip(session["events"], session["intervals"], strict=True):
            native, forecast = pd.DataFrame(interval["native"]), pd.DataFrame(interval["forecast"])
            assert len(native) == 31 and np.allclose(np.diff(native.time), 30)
            transitions_frame([event], native, forecast, {"step_s": 900, "area_m2": 192})
            metric = integrate(native, forecast, 192)
            for k, value in metric.items():
                assert np.isclose(value, event["metrics"][k], atol=1e-10, rtol=0), k
            assert np.isclose(session["frames"][len(energies) + 1]["temperature_C"], event["after"]["reaTZon_y"] - 273.15)
            energies.append(metric["energy_kWh"])
        assert np.isclose(sum(energies), session["frames"][-1]["energy_kWh"], atol=1e-10)
        sums = {k: sum(e["metrics"][k] for e in session["events"]) for k in session["events"][0]["metrics"]}
        differences = validate_kpis(sums, session["api_kpis"])
        return {"id": sid, "scene": session["scene"], "steps": len(energies),
                "metrics": sums, "kpi_absolute_difference": differences,
                "comparison_valid": session.get("comparison_valid")}

    state = api("/api/state")
    if state["busy"] or (state["session"] and state["session"]["status"] == "paused"):
        raise RuntimeError("Finish the interactive session before smoke-checking; never take over an active run")
    rows = []
    for scene in args.scenes:
        api("/api/create", {"scene": scene})
        wait()
        request = {"mode": "manual", "target_C": 23.25, "preferences": {"A": "warmer", "B": "cooler"}}
        api("/api/advance", {"steps": 2, "intent": request})
        wait()
        api("/api/advance", {"steps": 2, "intent": {**request, "target_C": 18.}})
        wait()
        api("/api/end", {})
        source = wait()
        rows.append(verify(source["id"]))
        api("/api/compare", {"session_id": source["id"]})
        compared = wait()
        assert compared["comparison_valid"]
        assert len(compared["frames"]) == len(source["frames"])
        rows.append(verify(compared["id"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"scope": "workbench_smoke_not_research_evaluation", "runs": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"{scene}: manual + same-start rule verified", flush=True)


if __name__ == "__main__":
    main()
