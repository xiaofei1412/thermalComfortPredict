"""Common demand-adopting baseline, equivalent to workbench needs mode."""

from . import demands
from .workbench import Conflict, command, intent


BASELINE = "demand_tracking_rule_v1"


def common_action(request, snapshot, before):
    """Only current state and confirmed current requests; no future labels or events."""
    request = intent(request)
    if request["mode"] != "needs":
        raise ValueError("共同基线只接受needs意图，不能旁路需求")
    if request["demand_revision"] != snapshot["revision"]:
        raise Conflict("需求版本已变化，请重新裁决")
    if snapshot["time_s"] != before["time"]:
        raise ValueError("需求快照与当前状态时间不一致")
    resolved = demands.resolve(snapshot["demands"], request["target_C"], before["time"],
                              before["reaTZon_y"] - 273.15, request["resolution"])
    if resolved["blocked"]:
        raise Conflict("共同需求冲突或不可执行，本步不得推进")
    control = {**request, "mode": "manual", "target_C": resolved["target_C"]}
    action, decision = command(control, before, None)
    return action, {**decision, "baseline": BASELINE, "reason": "demand_target", "demand": resolved,
                    "demand_handling": "adopted",
                    "evaluation": demands.evaluate(resolved, decision["applied_C"])}
