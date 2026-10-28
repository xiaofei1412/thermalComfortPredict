"""One rule-anchored action mapping; all frozen observation/reward/guard code stays intact."""

import numpy as np

from .rl_interface import project_action
from .rl_runner import ComparisonEnv


class RuleCache:
    def __init__(self, rule):
        self.rule, self.pending = rule, None

    def reset(self, initial, inputs):
        self.pending = None
        return self.rule.reset(initial, inputs)

    def prepare(self, index, before, forecast):
        if self.pending is not None:
            raise RuntimeError("Unconsumed rule decision")
        value = self.rule.decide(index, before, forecast)
        self.pending = (index, before["time"], value)
        return value

    def decide(self, index, before, forecast):
        if self.pending is None:
            return self.rule.decide(index, before, forecast)  # Common prefix.
        expected_index, expected_time, value = self.pending
        if (index, before["time"]) != (expected_index, expected_time):
            raise ValueError("Rule cache belongs to another step")
        self.pending = None
        return value

    def update(self, *args):
        return self.rule.update(*args)


def mapped_action(residual, previous_C, rule_C, amplitude_K):
    project_action(residual, previous_C)  # Same finite Box contract, before any rule state changes.
    target = rule_C+amplitude_K*float(np.asarray(residual)[0])
    bounded = float(np.clip(target,18,23.25))
    increment = float(np.clip((bounded-previous_C)/.5,-1,1))
    return np.array([increment],np.float64), target


class ResidualEnv(ComparisonEnv):
    def __init__(self, policy, *args, amplitude_K=.5, **kwargs):
        super().__init__(RuleCache(policy), *args, **kwargs)
        self.amplitude_K = amplitude_K

    def step(self, action):
        if self.completed or self.done or self.testid is None:
            raise RuntimeError("Residual episode inactive/exhausted")
        previous = self.before["oveTSet_u"]-273.15
        project_action(action, previous)
        shadow, shadow_info = self.policy.prepare(self.index,self.before,self.view)
        rule_C = shadow["oveTSet_u"]-273.15
        mapped, target = mapped_action(action,previous,rule_C,self.amplitude_K)
        obs,reward,done,truncated,info = super().step(mapped)
        info["control"].update(residual_normalized=float(np.asarray(action)[0]),
            rule_anchor_C=rule_C, rule_anchor_reason=shadow_info.get("reason"),
            residual_unprojected_target_C=target, mapped_increment_normalized=float(mapped[0]),
            raw_actor_normalized=float(np.asarray(action)[0]),
            anchor_deviation_after_limits_K=info["control"]["actual_command_C"]-rule_C)
        if info["control"]["mode"] == "sac":
            info["control"]["mode"] = "rule_residual_sac"
        self.records[-1]["control"].update(info["control"])
        return obs,reward,done,truncated,info
