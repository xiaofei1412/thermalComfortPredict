"""B-stage functional demand lifecycle check on the existing residential demo."""

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from thermal_comfort.simulation import integrate, transitions_frame, validate_kpis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--output", type=Path, default=Path("outputs/demand-validation/verification.json"))
    args = parser.parse_args()

    def api(path, body=None, expected=200):
        request = Request(args.url + path, data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urlopen(request, timeout=10) as response:
                assert response.status == expected, (path, response.status)
                return json.load(response)
        except HTTPError as error:
            try:
                assert error.code == expected, (path, error.code, error.read())
                return json.load(error)
            finally:
                error.close()

    def wait():
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            state = api("/api/state")
            if not state["busy"]:
                assert state["session"] and state["session"]["status"] != "error", state["message"]
                return state["session"]
            time.sleep(.2)
        raise TimeoutError("Demand job did not finish")

    def submit(person, kind, ttl=4, **extra):
        s = api("/api/state")["session"]
        api("/api/demand", {"person": person, "kind": kind, "base_C": 21., "ttl_steps": ttl,
                            "revision": s["demand_revision"], **extra}, 202)

    def advance(resolution="unresolved", expected=202):
        s = api("/api/state")["session"]
        request = {"mode": "needs", "target_C": 21., "preferences": {"A": "hold", "B": "hold"},
                   "resolution": resolution, "demand_revision": s["demand_revision"]}
        api("/api/advance", {"steps": 1, "intent": request}, expected)
        return wait() if expected == 202 else s

    state = api("/api/state")
    if state["busy"] or state.get("can_end"):
        raise RuntimeError("Finish the existing session first; this check never takes over an active run")
    api("/api/create", {"scene": "cool"}, 202); wait()
    submit("A", "warmer", ttl=2)
    first = advance()
    assert first["events"][0]["decision"]["target_C"] == 21.5
    assert first["events"][0]["decision"]["rate_limited"]
    submit("B", "cooler")
    before = advance(expected=409)
    assert len(api("/api/state")["session"]["events"]) == len(before["events"])
    second = advance("base")
    assert second["demands"]["A"]["status"] == "expired"
    api("/api/demand/revoke", {"person": "B", "revision": second["demand_revision"]}, 202)
    submit("A", "target", ttl=2, target_C=26.)
    before = advance(expected=409)
    assert len(api("/api/state")["session"]["events"]) == len(before["events"])
    advance("base")
    submit("B", "range", ttl=1, lower_C=20.75, upper_C=21.25)
    fourth = advance("B")
    assert all(d["status"] == "expired" for d in fourth["demands"].values())
    fifth = advance()
    assert not fifth["events"][-1]["decision"]["demand"]["active_requests"]
    api("/api/end", {}, 202); source = wait()
    api("/api/compare", {"session_id": source["id"]}, 202); paired = wait()
    assert paired["comparison_valid"] and paired["demand_replay_valid"]
    assert paired["demand_events"] == source["demand_events"]
    assert [e["demand_snapshot"] for e in paired["events"]] == [e["demand_snapshot"] for e in source["events"]]
    rows = []
    for s in [source, paired]:
        full = api("/api/download/" + s["id"])
        for event, interval in zip(full["events"], full["intervals"], strict=True):
            raw, forecast = pd.DataFrame(interval["native"]), pd.DataFrame(interval["forecast"])
            assert len(raw) == 31 and np.allclose(np.diff(raw.time), 30)
            transitions_frame([event], raw, forecast, {"step_s": 900, "area_m2": 192})
            actual = integrate(raw, forecast, 192)
            assert all(np.isclose(actual[k], v, rtol=0, atol=1e-10) for k, v in event["metrics"].items())
        sums = {k: sum(e["metrics"][k] for e in full["events"]) for k in full["events"][0]["metrics"]}
        differences = validate_kpis(sums, full["api_kpis"])
        rows.append({"id": s["id"], "steps": len(s["events"]), "provenance": full["provenance"],
                     "demand_summary": full["demand_summary"], "demand_events": full["demand_events"],
                     "decisions": [{"target_C": e["decision"]["target_C"], "applied_C": e["decision"]["applied_C"],
                                    "status": e["decision"]["demand"]["status"], "handling": e["decision"]["demand_handling"],
                                    "unmet_people": e["demand_evaluation"]["unmet_people"]} for e in full["events"]],
                     "kpi_absolute_difference": differences})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "functional_virtual_intent_not_comfort_or_energy_efficacy",
        "conflict_and_infeasible_rejected_without_advance": True, "same_demand_timeline_verified": True,
        "runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Demand lifecycle + explicit overrides + same-start replay + native/KPI checks passed", flush=True)


if __name__ == "__main__":
    main()
