import copy
import unittest

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from scripts.audit_v2_residual_safe import check_control,check_journal


@unittest.skipIf(stable_baselines3 is None,"Install requirements/sac.txt")
class ResidualAuditTests(unittest.TestCase):
    def record(self):
        return {"before":{"oveTSet_u":294.15},"action":{"oveTSet_u":294.45},"after":{"oveTSet_u":294.45},
            "control":{"residual_normalized":.4,"rule_anchor_C":21.1,"fallback":False,
                "raw_actor_normalized":.4,"residual_unprojected_target_C":21.3,"mapped_increment_normalized":.6,
                "actor_projection":{"raw_normalized":.6,"applied_C":21.3},"actor_target_C":21.3,
                "actual_command_C":21.3,"measured_setpoint_C":21.3,"anchor_deviation_after_limits_K":.2,
                "mode":"rule_residual_sac"}}

    def test_mapping_distinguishes_residual_from_increment(self):
        result = check_control(self.record())
        self.assertEqual(result["residual"],.4)
        self.assertAlmostEqual(result["effective_offset_K"],.2)

    def test_wrong_raw_action_label_is_rejected(self):
        record = self.record()
        record["control"]["raw_actor_normalized"] = .6
        with self.assertRaises(AssertionError):
            check_control(record)

    def test_wrong_applied_or_measured_command_is_rejected(self):
        for key in ["actual_command_C","measured_setpoint_C","mapped_increment_normalized"]:
            record = self.record()
            record["control"][key] += .1
            with self.assertRaises(AssertionError):
                check_control(record)

    def test_fallback_requires_rule_anchor_and_keeps_actor_proposal(self):
        record = self.record()
        record["control"].update(fallback=True,mode="fallback",actual_command_C=21.1,
            measured_setpoint_C=21.1,anchor_deviation_after_limits_K=0.)
        record["action"]["oveTSet_u"] = record["after"]["oveTSet_u"] = 294.25
        self.assertAlmostEqual(check_control(record)["effective_offset_K"],0.)
        corrupted = copy.deepcopy(record)
        corrupted["control"]["mode"] = "rule_residual_sac"
        with self.assertRaises(AssertionError):
            check_control(corrupted)

    def test_journal_requires_all_controls_except_separate_record_flags(self):
        record = self.record()
        action = {"episode":"train014","control":copy.deepcopy(record["control"]),
            "library_action":[.4],"buffer_action":[.4]}
        record["control"].update(common_prefix=False,actor_transition=True)
        check_journal(action,record,"train014",True)
        action["control"]["actual_command_C"] += .01
        with self.assertRaises(AssertionError):
            check_journal(action,record,"train014",True)
