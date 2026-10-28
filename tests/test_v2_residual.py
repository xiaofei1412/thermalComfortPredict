import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    from scripts.run_v2_residual import authorize, close, plan, registration_for, REPLICAS
    from test_rl_interface import FakeClient, FakeRule
    from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
    from thermal_comfort.rl_residual import ResidualEnv, RuleCache, mapped_action
    from thermal_comfort.sac_env import CampaignClient
    from thermal_comfort.simulation import digest
    from thermal_comfort.v2_protocol import ROOT, read, verify_files


class CountingRule(FakeRule):
    def __init__(self):
        self.decisions, self.updates = 0, 0

    def decide(self, index, before, forecast):
        self.decisions += 1
        return {"oveTSet_u": before["oveTSet_u"]+.1, "oveTSet_activate": 1}, {"reason": "stateful"}

    def update(self, *args):
        self.updates += 1


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class ResidualTests(unittest.TestCase):
    def env(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        config = plan()
        adapter = {"train_ids": config["train_ids"], "development_ids": config["development_ids"],
            "budget_per_seed": config["budget_per_algorithm_seed"],
            "fixed_observation_scaling": config["shared"]["observation_scaling"],
            "reward": config["shared"]["reward"], "guard": config["shared"]["guard"]}
        client, rule = FakeClient(), CountingRule()
        spec = next(e for e in read(ROOT/"configs/v2_0.json")["episodes"] if e["id"] == "train014")
        env = ResidualEnv(rule, CampaignClient(client, adapter, Path(temp.name)), spec, "train", adapter,
            algorithm="SAC")
        self.addCleanup(env.close)
        return env, client, rule

    def test_mapping_zero_residual_and_physical_limits(self):
        for previous, anchor in [(21,21.2),(18,18),(23.25,23.25)]:
            mapped, target = mapped_action([0], previous, anchor, .5)
            self.assertAlmostEqual(previous+.5*mapped[0], anchor)
            self.assertEqual(target, anchor)
        for previous, anchor, raw, expected in [(23.25,23.25,1,23.25),(18,18,-1,18),
                (21,21.5,1,21.5),(21,20.5,-1,20.5)]:
            mapped, _ = mapped_action([raw], previous, anchor, .5)
            self.assertAlmostEqual(previous+.5*mapped[0], expected)

    def test_stateful_rule_exactly_once_per_actor_and_raw_trace(self):
        env, _, rule = self.env()
        obs, _ = env.reset(seed=11)
        self.assertEqual(obs.shape, (503,))
        self.assertEqual(rule.decisions, 8)
        for raw in [0,.4,-.4,0]:
            previous = env.before["oveTSet_u"]-273.15
            _, _, _, _, info = env.step([raw])
            control = info["control"]
            self.assertAlmostEqual(control["rule_anchor_C"], previous+.1)
            self.assertAlmostEqual(control["actual_command_C"], previous+.1+.5*raw)
            self.assertEqual(control["raw_actor_normalized"], raw)
            self.assertEqual(control["residual_normalized"], raw)
            self.assertAlmostEqual(control["mapped_increment_normalized"], .2+raw)
            self.assertAlmostEqual(control["actual_command_C"],control["measured_setpoint_C"])
            self.assertEqual(control["mode"], "rule_residual_sac")
            self.assertIsNone(env.policy.pending)
        self.assertEqual(rule.decisions, 12)
        self.assertEqual(rule.updates, 12)

    def test_invalid_action_and_inactive_episode_do_not_mutate_rule_or_network(self):
        env, client, rule = self.env()
        env.reset()
        before = len(client.log), rule.decisions, rule.updates
        for raw in [[],[0,1],[np.nan],[np.inf],[1.001]]:
            with self.assertRaises(ValueError):
                env.step(raw)
        self.assertEqual(before,(len(client.log),rule.decisions,rule.updates))
        env.completed = True
        with self.assertRaises(RuntimeError):
            env.step([0])
        self.assertEqual(before,(len(client.log),rule.decisions,rule.updates))

    def test_existing_hard_guard_overrides_residual(self):
        env, _, rule = self.env()
        env.reset()
        env.before["reaTZon_y"] = 292.05  # 18.9 C, original engineering low guard.
        _, _, _, _, info = env.step([-1])
        control = info["control"]
        self.assertEqual(control["mode"], "fallback")
        self.assertAlmostEqual(control["actual_command_C"],control["rule_anchor_C"])
        self.assertEqual(rule.decisions, 9)

    def test_cache_rejects_unconsumed_or_stale_decision(self):
        rule, before = CountingRule(), {"time":0,"oveTSet_u":294.15}
        cache = RuleCache(rule)
        value = cache.prepare(8,before,None)
        with self.assertRaises(RuntimeError):
            cache.prepare(8,before,None)
        with self.assertRaises(ValueError):
            cache.decide(9,before,None)
        self.assertEqual(cache.decide(8,before,None), value)
        self.assertEqual(rule.decisions, 1)

    def test_child_keeps_learning_roles_and_seed_budget_exact(self):
        child, base = plan(), read(ROOT/"configs/v2_comparison.json")
        for key in ["shared","parameters","train_ids","development_ids","seeds","budget_per_algorithm_seed"]:
            self.assertEqual(child[key],base[key])
        self.assertEqual(child["algorithms"],["SAC"])
        self.assertEqual(child["budget_total_all_algorithms_seeds"],
            {"select":225,"initialize":225,"advance":151200,"stage_wall_seconds_max":18000})


@unittest.skipIf(stable_baselines3 is None, "Install requirements/sac.txt")
class ResidualAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output, self.preflight, self.diagnostic = [self.root/name for name in ["new","preflight.json","diagnostic.json"]]
        write_snapshot(self.preflight,{"status":"passed","replicas":REPLICAS,"FMU_select_or_initialize_calls":0})
        write_snapshot(self.diagnostic,{"hypothesis_gate_passed":True,"protocol_sha256":digest(ROOT/"configs/v2_4.json"),
            "parent_lock_sha256":digest(ROOT/"reports/v2_safe_formal/lock.json")})
        def verify(directory, files):
            if set(files) == {"outputs/solar-selected/model.joblib",
                    "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"}:
                return  # Private binaries are absent from CI; source locks still really verified.
            verify_files(directory, files)
        patcher = patch("scripts.run_v2_residual.verify_files",side_effect=verify)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_failed_diagnostic_creates_no_campaign(self):
        value = read(self.diagnostic)
        value["hypothesis_gate_passed"] = False
        write_snapshot(self.diagnostic,value)
        with self.assertRaises(ValueError):
            authorize(self.output,self.preflight,self.diagnostic)
        self.assertFalse(self.output.exists())

    def test_fresh_zero_budget_dual_source_binding_and_no_overwrite(self):
        authorize(self.output,self.preflight,self.diagnostic)
        value = registration_for(self.output)
        self.assertEqual(len(value["parent_locks_sha256"]),10)
        self.assertIn("configs/v2_4.json",value["files_sha256"])
        self.assertIn("thermal_comfort/rl_residual.py",value["files_sha256"])
        self.assertFalse(read(self.output/"registration-before-authorization.json")["formal_training_authorized"])
        self.assertEqual(BudgetLedger(self.output,plan()).read()["counts"],{"select":0,"initialize":0,"advance":0})
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            authorize(self.output,self.preflight,self.diagnostic)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_closed_authorization_refuses_all_execution(self):
        authorize(self.output,self.preflight,self.diagnostic)
        close(self.output)
        for name in ["seed","dispatch"]:
            with patch("scripts.run_v2_residual.Client") as network, patch("scripts.run_v2_residual.subprocess.Popen") as spawn:
                from scripts import run_v2_residual
                with self.assertRaises(PermissionError):
                    getattr(run_v2_residual,name)(self.output,*([11,8083] if name == "seed" else []))
                network.assert_not_called()
                spawn.assert_not_called()

    def test_failed_worker_stops_pending_seeds_without_retry(self):
        authorize(self.output,self.preflight,self.diagnostic)
        class ExitedWorker:
            pid, returncode = 1, 1
            def poll(self):
                return self.returncode
        from scripts.run_v2_residual import dispatch
        with patch("scripts.run_v2_residual.subprocess.Popen",return_value=ExitedWorker()) as spawn:
            dispatch(self.output)
        self.assertEqual(spawn.call_count,1)
        state = BudgetLedger(self.output,plan()).read()
        self.assertEqual(set(state["seeds"]),{"SAC-11"})
        self.assertEqual(state["seeds"]["SAC-11"]["status"],"failed")
        self.assertEqual(state["counts"],{"select":0,"initialize":0,"advance":0})
        with (self.output/"SAC-11.log").open("ab") as stream:
            stream.write(b"closed launcher handle")
