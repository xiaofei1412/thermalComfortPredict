import unittest

import numpy as np
import pandas as pd

try:
    import gymnasium
except ImportError:
    gymnasium = None

if gymnasium is not None:
    from gymnasium.utils.env_checker import check_env
    from thermal_comfort.rl_interface import ResidentialEnv, START, bounded_forecast, project_action, native_score
    from thermal_comfort.simulation import FORECAST, POWER


class FakeRule:
    def reset(self, before, inputs):
        pass

    def decide(self, index, before, forecast):
        return {"oveTSet_u": before["oveTSet_u"], "oveTSet_activate": 1}, {"reason": "rule"}

    def update(self, *args):
        pass


class FakeClient:
    def __init__(self):
        self.log = []
        self.time = START if gymnasium else 0
        self.setpoint = 294.15
        self.temp = 294.15
        self.fail = None
        self.bad_native = False
        self.wrong_action = False

    def measurement(self, t):
        return {"time": t, "reaTZon_y": self.temp, "oveTSet_u": self.setpoint,
                "oveTSet_activate": 1, "reaQFloHea_y": 2000., "reaTRet_y": 300.,
                "reaTSup_y": 305., "reaTSetHea_y": 293.15, "reaTSetCoo_y": 303.15,
                **{p: 100. for p in POWER}}

    def call(self, method, path, data=None):
        kind = path.split("/")[0]
        self.log.append((kind, data))
        if kind == self.fail:
            raise TimeoutError("unknown applied status")
        if kind == "version":
            return {"version": "0.9.0"}
        if kind == "testcases":
            return {"testid": "fixture"}
        if kind == "inputs":
            return {"oveTSet_u": {"Unit": "K", "Minimum": 280, "Maximum": 310}}
        if kind == "measurements":
            return {p: {"Unit": "W" if p in [*POWER, "reaQFloHea_y"] else "K"}
                    for p in self.measurement(self.time)}
        if kind == "initialize":
            self.time, self.setpoint = START, 294.15
            return self.measurement(self.time)
        if kind == "step":
            return 900
        if kind == "forecast":
            times = self.time + np.arange(1441)*30
            values = [293.15, 303.15, 1., 280., 50., .2, .1]
            return {"time": times, **{k: np.full(len(times), v) for k, v in zip(FORECAST, values)}}
        if kind == "advance":
            self.time += 900
            self.setpoint = data["oveTSet_u"]
            return self.measurement(self.time)
        if kind == "results":
            times = np.arange(data["start_time"], data["final_time"]+1, 30)
            if self.bad_native:
                times = times[1:]
            rows = pd.DataFrame([self.measurement(t) for t in times])
            if self.wrong_action:
                rows["oveTSet_u"] += 1
            return rows.to_dict("list")
        if kind == "kpi":
            energy_ws = (self.time-START)*300
            return {"ener_tot": energy_ws*2.77778e-7/192, "tdis_tot": 0.,
                    "cost_tot": energy_ws*.2*2.77778e-7/192,
                    "emis_tot": energy_ws*.1*2.77778e-7/192}
        return None


@unittest.skipIf(gymnasium is None, "install requirements/rl-interface.txt in separate V2 env")
class InterfaceTests(unittest.TestCase):
    def env(self, **kwargs):
        client = FakeClient()
        env = ResidentialEnv(FakeRule(), client=client, **kwargs)
        self.addCleanup(env.close)
        return env, client

    def test_gymnasium_checker(self):
        env, _ = self.env()
        check_env(env, skip_render_check=True)

    def test_other_dates_rejected_before_network(self):
        for name in ["train035", "development084", "holdout210", "intent_test329"]:
            with self.assertRaises(ValueError):
                self.env(episode=name)
        env, client = self.env()
        with self.assertRaises(ValueError):
            env.reset(options={"start_day": 210})
        self.assertEqual(client.log, [])

    def test_prefix_mask_and_time_remaining(self):
        env, client = self.env()
        obs, info = env.reset(seed=11)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs["history_mask"].sum(), 9)
        np.testing.assert_equal(obs["history"][:7], 0)
        self.assertEqual(obs["history"][-1, -1], 664)
        self.assertEqual(len(env.records), 8)
        self.assertTrue(all(not r["control"]["actor_transition"] for r in env.records))
        self.assertAlmostEqual(info["prefix_energy_kWh"], .6)
        self.assertTrue(all(d["horizon"] == 43200 for k, d in client.log if k == "forecast"))

    def test_forecast_endpoint_and_mutation_copy(self):
        env, client = self.env()
        obs, _ = env.reset()
        now = env.before["time"]
        self.assertEqual(env.view.time.iloc[-1], now+43200)
        self.assertEqual(obs["forecast"].shape, (49, 7))
        obs["forecast"][:] = -999
        self.assertTrue((env.observe()["forecast"] >= 0).all())
        payload = client.call("PUT", "forecast/x", {})
        for mutation in [lambda p: p.update({"future_temperature": np.zeros(1441)}),
                         lambda p: p.update(time=now+np.arange(1441)*31),
                         lambda p: p.update(time=now-30+np.arange(1441)*30)]:
            broken = dict(payload)
            mutation(broken)
            with self.assertRaises(ValueError):
                bounded_forecast(broken, now)

    def test_action_projection_limits(self):
        self.assertAlmostEqual(project_action([1], 23.25)[0]["oveTSet_u"], 296.4)
        self.assertAlmostEqual(project_action([-1], 18)[0]["oveTSet_u"], 291.15)
        for previous in [18, 20, 23.25]:
            for a in [-1, 0, 1]:
                action, _ = project_action([a], previous)
                value = action["oveTSet_u"]-273.15
                self.assertTrue(18-1e-7 <= value <= 23.25+1e-7)
                self.assertLessEqual(abs(value-previous), .5+1e-7)

    def test_invalid_action_has_no_advance(self):
        env, client = self.env()
        env.reset()
        n = len(client.log)
        for action in [[], [0, 1], [np.nan], [np.inf], [1.01]]:
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(len(client.log), n)
        self.assertFalse(env.done)

    def test_external_cap_truncates_and_scores_prefix(self):
        env, client = self.env(max_agent_steps=1)
        env.reset()
        obs, reward, terminal, truncated, info = env.step(np.array([0], np.float32))
        self.assertFalse(terminal)
        self.assertTrue(truncated)
        self.assertEqual(obs["history"][-1, -1], 663)
        self.assertAlmostEqual(reward, -.075)
        self.assertAlmostEqual(info["score"]["metrics"]["energy_kWh"], .675)
        self.assertFalse(info["episode_complete"])
        self.assertIsNone(env.testid)
        with self.assertRaises(RuntimeError):
            env.step([0])

    def test_finite_horizon_is_terminal_even_at_external_cap(self):
        env, _ = self.env(max_agent_steps=664)
        env.reset()
        # Use the real boundary logic without an expensive 664-interval fake rollout.
        env.index = 671
        _, _, terminal, truncated, info = env.step([0])
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertEqual(info["end_reason"], "finite_horizon")

    def test_temperature_failure_is_terminal_not_success(self):
        env, client = self.env()
        env.reset()
        client.temp = 309.15
        # Official KPI is deliberately wrong for this fixture's new hot interval.
        from unittest.mock import patch
        with patch.object(env, "score", return_value={}):
            _, _, terminal, truncated, info = env.step([0])
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertTrue(info["aborted"])
        self.assertFalse(info["episode_complete"])

    def test_uncertain_advance_failure_no_retry_or_fake_transition(self):
        env, client = self.env()
        env.reset()
        client.fail = "advance"
        with self.assertRaises(TimeoutError):
            env.step([0])
        self.assertEqual(len(env.records), 8)
        self.assertEqual(sum(k == "advance" for k, _ in client.log), 9)
        self.assertEqual(client.log[-1][0], "stop")
        self.assertTrue(env.done)
        self.assertIsNone(env.testid)

    def test_missing_native_or_wrong_applied_action_invalidates(self):
        for kind in ["bad_native", "wrong_action"]:
            env, client = self.env()
            env.reset()
            setattr(client, kind, True)
            with self.assertRaises(ValueError):
                env.step([0])
            self.assertEqual(len(env.records), 8)
            self.assertTrue(env.done)
            self.assertIsNone(env.testid)

    def test_reset_releases_worker_and_erases_previous_history(self):
        env, client = self.env()
        first, _ = env.reset(seed=11)
        env.step([1])
        second, _ = env.reset(seed=11)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(sum(k == "stop" for k, _ in client.log), 1)

    def test_native_score_uses_interval_interior_and_splits_cold_hot(self):
        native = pd.DataFrame({"time": [0, 450, 900], "reaTZon_y": [292.15, 294.15, 304.15],
                               **{p: [0, 1200, 0] for p in POWER}})
        schedule = pd.DataFrame({"time": [0, 450, 900], "LowerSetp[1]": [293.15]*3,
            "UpperSetp[1]": [303.15]*3, "PriceElectricPowerConstant": [.2]*3,
            "EmissionsElectricPower": [.1]*3})
        metrics = native_score(native, schedule)
        self.assertAlmostEqual(metrics["energy_kWh"], .45)
        for name in ["cold", "hot"]:
            self.assertAlmostEqual(metrics[name+"_Kh"], .0625)
            self.assertAlmostEqual(metrics[name+"_duration_s"], 225)
            self.assertAlmostEqual(metrics[name+"_max_K"], 1)

    def test_budget_guards_dates_failed_attempts_and_limits(self):
        import tempfile
        from pathlib import Path
        from scripts.verify_rl_interface import BudgetClient
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            budget = BudgetClient(client, {"initializations_max": 1,
                "advance_attempts_max": 1, "wall_seconds_max": 60}, Path(folder))
            with self.assertRaises(ValueError):
                budget.call("PUT", "initialize/x", {"start_time": 210*86400, "warmup_period": 604800})
            self.assertEqual(client.log, [])
            budget.call("PUT", "initialize/x", {"start_time": START, "warmup_period": 604800})
            with self.assertRaises(RuntimeError):
                budget.call("PUT", "initialize/x", {"start_time": START, "warmup_period": 604800})
            client.fail = "advance"
            with self.assertRaises(TimeoutError):
                budget.call("POST", "advance/x", {})
            with self.assertRaises(RuntimeError):
                budget.call("POST", "advance/x", {})
            self.assertEqual(budget.counts, {"initialize": 1, "advance": 1})
            self.assertEqual(sum(k == "advance" for k, _ in client.log), 1)


if __name__ == "__main__":
    unittest.main()
