import copy
import unittest

from thermal_comfort.demands import active, evaluate, resolve, submission


def request(person="A", kind="warmer", **extra):
    return submission({"person": person, "kind": kind, "base_C": 21., "ttl_steps": 4,
                       "revision": 0, **extra}, 1000, 0)


class DemandTests(unittest.TestCase):
    def test_relative_intents_are_anchored_once_and_hold_is_explicit(self):
        self.assertEqual(request()["lower_C"], 21.5)
        self.assertEqual(request(kind="cooler")["upper_C"], 20.5)
        self.assertEqual(request(kind="hold")["lower_C"], 21.)
        self.assertEqual(request(kind="hold")["upper_C"], 21.)
        self.assertEqual(resolve({"A": request()}, 23., 1000, 20.)["target_C"], 23.)
        self.assertEqual(request()["lower_C"], 21.5)

    def test_common_interval_uses_nearest_base_without_averaging(self):
        a = request(kind="range", lower_C=21.25, upper_C=22.5)
        b = request(person="B", kind="range", lower_C=21.5, upper_C=22.)
        result = resolve({"A": a, "B": b}, 21., 1000, 20.)
        self.assertEqual(result["target_C"], 21.5)
        self.assertEqual(result["executable_interval_C"], [21.5, 22.])

    def test_conflict_requires_explicit_operator_choice(self):
        records = {"A": request(), "B": request(person="B", kind="cooler")}
        self.assertTrue(resolve(records, 21., 1000, 20.)["blocked"])
        for choice, target in [("A", 21.5), ("B", 20.5), ("base", 21.)]:
            with self.subTest(choice=choice):
                result = resolve(records, 21., 1000, 20., choice)
                self.assertEqual(result["status"], "conflict")
                self.assertEqual(result["target_C"], target)
                self.assertFalse(result["blocked"])

    def test_actuator_bounds_cannot_be_overridden_by_priority(self):
        records = {"A": request(kind="target", target_C=26.)}
        for choice in ["unresolved", "A", "B"]:
            self.assertTrue(resolve(records, 21., 1000, 20., choice)["blocked"])
        result = resolve(records, 21., 1000, 20., "base")
        self.assertEqual(result["target_C"], 21.)
        self.assertEqual(evaluate(result, 21.)["unmet_people"], ["A"])

    def test_exact_deadline_and_withdrawal_are_not_active(self):
        a = request(ttl_steps=1)
        self.assertIn("A", active({"A": a}, 1899))
        self.assertNotIn("A", active({"A": a}, 1900))
        a["status"] = "withdrawn"
        self.assertFalse(active({"A": a}, 1000))

    def test_deviation_is_actuator_intent_not_comfort(self):
        result = resolve({"A": request()}, 21., 1000, 19.)
        score = evaluate(result, 21.)
        self.assertEqual(score["setpoint_intent_deviation_K"], {"A": .5})
        self.assertEqual(score["setpoint_intent_deviation_Kh"], {"A": .125})
        self.assertEqual(score["semantics"], "setpoint_intent_not_human_comfort")

    def test_cooling_capability_note_is_not_an_infeasibility_proof(self):
        result = resolve({"A": request(kind="cooler")}, 21., 1000, 26.)
        self.assertFalse(result["blocked"])
        self.assertIn("heating_only_no_active_cooling", result["capability_notes"])

    def test_invalid_numbers_types_intervals_roles_and_expiry_rejected(self):
        for value in [True, None, "21", float("nan"), float("inf"), 4., 41.]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                request(kind="target", target_C=value)
        for change in [{"person": "C"}, {"ttl_steps": True}, {"ttl_steps": 0}, {"ttl_steps": 17},
                       {"base_C": 24.}, {"revision": 1}, {"extra": 1}]:
            with self.assertRaises(ValueError):
                request(**change)
        with self.assertRaises(ValueError):
            request(kind="range", lower_C=22., upper_C=20.)

    def test_resolver_does_not_mutate_request_or_presume_clipped_goal_met(self):
        records = {"A": request(kind="range", lower_C=22., upper_C=25.)}
        saved = copy.deepcopy(records)
        result = resolve(records, 23.25, 1000, 20.)
        self.assertEqual(result["target_C"], 23.25)
        self.assertEqual(records, saved)
        self.assertEqual(result["executable_interval_C"], [22., 23.25])
        self.assertEqual(evaluate(result, 21.5)["unmet_people"], ["A"])
