"""Offline delivery checks: archived native trajectory and frozen negative decision."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.install_demo import sample
from thermal_comfort.demand_rule import common_action
from thermal_comfort.residential_c import qualify_development
from thermal_comfort.simulation import integrate, transitions_frame, validate_kpis


ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    s, _, manifest = sample(root)
    if len(s["events"]) != 5 or len(s["intervals"]) != 5 or len(s["frames"]) != 6:
        raise ValueError("Incomplete bundled trajectory")
    for event, interval in zip(s["events"], s["intervals"], strict=True):
        action, decision = common_action(event["intent"], event["demand_snapshot"], event["before"])
        if action != event["action"] or decision["evaluation"] != event["demand_evaluation"]:
            raise ValueError("Archived demand/action mismatch")
        native, forecast = pd.DataFrame(interval["native"]), pd.DataFrame(interval["forecast"])
        transitions_frame([event], native, forecast, {"step_s": 900, "area_m2": 192})
        if len(native) != 31 or not np.all(np.diff(native.time) == 30):
            raise ValueError("Native sampling mismatch")
        metrics = integrate(native, forecast, 192)
        if any(not np.isclose(v, event["metrics"][k], atol=1e-10, rtol=0) for k, v in metrics.items()):
            raise ValueError("Native integral mismatch")
    total = {k: sum(e["metrics"][k] for e in s["events"]) for k in s["events"][0]["metrics"]}
    kpis = validate_kpis(total, s["api_kpis"])
    c2 = json.loads((root/"reports/residential_c2/assessment.json").read_text(encoding="utf-8"))
    config = json.loads((root/"configs/residential_c.json").read_text(encoding="utf-8"))
    for candidate in c2["candidates"]:
        decision = qualify_development(candidate["metrics"], candidate["checks"], config)
        if decision["qualified"] or decision["failed"] != candidate["failed"]:
            raise ValueError("Negative development decision differs")
    closure = json.loads((root/"reports/residential_c2/closure.json").read_text(encoding="utf-8"))
    if closure["status"] != "closed_negative" or any(closure[k] != 0 for k in ("refits", "calibration_rounds", "blind_rounds")):
        raise ValueError("Unsupported release boundary")
    return {"sample_id": s["id"], "source_sha256": manifest["original_session_sha256"],
            "replayed_actions": 5, "native_interval_points": 155,
            "kpi_max_absolute_difference": max(kpis.values()), "c2_stop_decision_reproduced": True,
            "new_simulations": 0, "new_model_fits": 0,
            "scope": "archived trajectory and decision audit; not model retraining or a new control evaluation"}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
