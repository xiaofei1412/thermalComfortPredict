"""Deterministic virtual setpoint intents; never estimates human comfort."""

import copy
import math


PROTOCOL = "virtual_setpoint_intent_v1"
RESOLUTIONS = ("unresolved", "base", "A", "B")


def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"需要{low}～{high}范围内的有限数值，不能是布尔值")
    return float(value)


def submission(body, now, revision):
    common = {"person", "kind", "base_C", "ttl_steps", "revision"}
    kind = body.get("kind")
    extra = {"target_C"} if kind == "target" else {"lower_C", "upper_C"} if kind == "range" else set()
    if set(body) != common | extra or kind not in ("warmer", "cooler", "hold", "target", "range"):
        raise ValueError("需求类型或字段不符")
    if body["person"] not in ("A", "B"):
        raise ValueError("只允许虚拟住户A/B")
    if type(body["revision"]) is not int or body["revision"] < 0:
        raise ValueError("需求版本必须是非负整数")
    if body["revision"] != revision:
        raise ValueError("需求版本已变化，请刷新后重新确认")
    if type(body["ttl_steps"]) is not int or not 1 <= body["ttl_steps"] <= 16:
        raise ValueError("有效期需为1～16个十五分钟步")
    base = number(body["base_C"], 18, 23.25)
    if kind == "target":
        low = high = number(body["target_C"], 5, 40)
    elif kind == "range":
        low, high = number(body["lower_C"], 5, 40), number(body["upper_C"], 5, 40)
        if low > high:
            raise ValueError("区间下界不能超过上界")
    else:
        low, high = {"warmer": (base + .5, 40.), "cooler": (5., base - .5), "hold": (base, base)}[kind]
    return {"person": body["person"], "kind": kind, "base_C": base, "lower_C": low,
            "upper_C": high, "submitted_s": now, "expires_s": now + body["ttl_steps"] * 900,
            "status": "active", "request_revision": revision + 1}


def active(demands, now):
    return {p: copy.deepcopy(d) for p, d in demands.items() if d and d["status"] == "active"
            and d["submitted_s"] <= now < d["expires_s"]}


def resolve(demands, base, now, temperature, resolution="unresolved"):
    base = number(base, 18, 23.25)
    if resolution not in RESOLUTIONS:
        raise ValueError("裁决只能为unresolved、base、A或B")
    requests = active(demands, now)
    low = max((r["lower_C"] for r in requests.values()), default=18.)
    high = min((r["upper_C"] for r in requests.values()), default=23.25)
    status = "conflict" if low > high else "infeasible" if max(low, 18.) > min(high, 23.25) else "compatible" if requests else "no_active"
    chosen_low, chosen_high = max(low, 18.), min(high, 23.25)
    chosen = "intersection"
    if status in ("conflict", "infeasible"):
        if resolution == "base":
            chosen_low = chosen_high = base
            chosen = "operator_base"
        elif resolution in requests:
            selected = requests[resolution]
            chosen_low, chosen_high = max(18., selected["lower_C"]), min(23.25, selected["upper_C"])
            chosen = "operator_" + resolution
    blocked = chosen_low > chosen_high or (status in ("conflict", "infeasible") and chosen == "intersection")
    target = None if blocked else min(max(base, chosen_low), chosen_high)
    notes = []
    if any(r["kind"] == "cooler" for r in requests.values()) or (target is not None and temperature > target):
        notes.append("heating_only_no_active_cooling")
    return {"protocol": PROTOCOL, "status": status, "blocked": blocked, "base_C": base,
            "target_C": target, "resolution": resolution, "choice": chosen,
            "common_interval_C": [low, high] if low <= high else None,
            "executable_interval_C": [chosen_low, chosen_high] if not blocked else None,
            "active_requests": requests, "capability_notes": notes}


def evaluate(resolved, applied, step_s=900):
    deviations = {p: max(r["lower_C"] - applied, applied - r["upper_C"], 0.)
                  for p, r in resolved["active_requests"].items()}
    return {"setpoint_intent_deviation_K": deviations,
            "setpoint_intent_deviation_Kh": {p: value * step_s / 3600 for p, value in deviations.items()},
            "unmet_people": [p for p, value in deviations.items() if value > 1e-7],
            "semantics": "setpoint_intent_not_human_comfort"}
