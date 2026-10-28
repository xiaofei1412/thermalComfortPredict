from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    import stable_baselines3
except ImportError:
    stable_baselines3 = None

if stable_baselines3 is not None:
    import pandas as pd
    from scripts import run_v2_residual_safe as entry
    from test_rl_interface import FakeClient, FakeRule
    from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
    from thermal_comfort.rl_interface import START
    from thermal_comfort.simulation import digest
    from thermal_comfort.v2_protocol import ROOT, read, verify_files


@unittest.skipIf(stable_baselines3 is None,"Install requirements/sac.txt")
class SafeEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/"fresh"
        self.preflight,self.diagnostic,self.gate = [self.root/(name+".json") for name in ["preflight","diagnostic","gate"]]
        write_snapshot(self.preflight,{"status":"passed","replicas":entry.REPLICAS,"FMU_select_or_initialize_calls":0})
        write_snapshot(self.diagnostic,{"hypothesis_gate_passed":True,"protocol_sha256":digest(ROOT/"configs/v2_4.json"),
            "parent_lock_sha256":digest(ROOT/"reports/v2_safe_formal/lock.json")})
        write_snapshot(self.gate,{"status":"passed","test_returncode":0,"test_count":16,"FMU_calls":0,
            "sources_sha256":entry.source_hashes()})
        def verify(directory,files):
            if set(files) == {"outputs/solar-selected/model.joblib",
                    "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu"}:
                return  # Only absent private binaries stubbed in CI, never runtime authorization.
            verify_files(directory,files)
        patcher = patch.object(entry,"verify_files",side_effect=verify)
        patcher.start()
        self.addCleanup(patcher.stop)

    def authorize(self):
        entry.authorize(self.output,self.preflight,self.diagnostic,self.gate)
        return BudgetLedger(self.output,entry.plan())

    def test_science_unchanged_and_actual_contract_complete(self):
        failed, safe = entry.failed_plan(),entry.plan()
        for name in failed.keys()-{"protocol","budget_total_all_algorithms_seeds"}:
            self.assertEqual(safe[name],failed[name])
        self.assertEqual(safe["budget_total_all_algorithms_seeds"],
            {"select_max":225,"initialize_max":225,"advance_max":151200,"stage_wall_seconds_max":18000})

    def test_invalid_contract_rejected_before_directory_or_authorization(self):
        with patch.object(entry,"plan",return_value=entry.failed_plan()):
            with self.assertRaisesRegex(ValueError,"select_max"):
                self.authorize()
        self.assertFalse(self.output.exists())

    def test_failed_gate_rejected_before_directory(self):
        value = read(self.gate)
        value["status"] = "failed"
        write_snapshot(self.gate,value)
        with self.assertRaises(ValueError):
            self.authorize()
        self.assertFalse(self.output.exists())

    def test_stale_source_gate_rejected_before_directory(self):
        value = read(self.gate)
        value["sources_sha256"]["configs/v2_4_safe.json"] = "0"*64
        write_snapshot(self.gate,value)
        with self.assertRaises(ValueError):
            self.authorize()
        self.assertFalse(self.output.exists())

    def test_fresh_registration_and_real_reserve_of_all_counter_types(self):
        ledger = self.authorize()
        registered = entry.registration_for(self.output)
        self.assertEqual(len(registered["parent_locks_sha256"]),11)
        self.assertFalse(read(self.output/"registration-before-authorization.json")["formal_training_authorized"])
        key = ledger.start_seed("SAC",11)
        for kind in ["select","initialize","advance"]:
            ledger.reserve(key,kind)
        self.assertEqual(ledger.read()["counts"],{"select":1,"initialize":1,"advance":1})
        self.assertIsNotNone(ledger.read()["stage_start"])

    def test_existing_campaign_cannot_be_overwritten(self):
        self.authorize()
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.authorize()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_old_and_closed_registration_refused_without_network_or_process(self):
        self.output.mkdir()
        (self.output/"registration.json").write_bytes((ROOT/"reports/v2_4/registration.json").read_bytes())
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        for name,args in [("seed",[11,8083]),("dispatch",[])]:
            with patch.object(entry,"Client") as client,patch.object(entry.subprocess,"Popen") as spawn:
                with self.assertRaises(PermissionError):
                    getattr(entry,name)(self.output,*args)
                client.assert_not_called()
                spawn.assert_not_called()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_changed_acceptance_receipt_blocks_execution(self):
        self.authorize()
        value = read(self.output/"entry-gate.json")
        value["FMU_calls"] = 1
        write_snapshot(self.output/"entry-gate.json",value)
        with self.assertRaises(ValueError):
            entry.registration_for(self.output)

    def test_unknown_seed_or_port_refused_before_client_construction(self):
        self.authorize()
        for number,port in [(42,8083),(11,8080)]:
            with patch.object(entry,"Client") as client:
                with self.assertRaises(ValueError):
                    entry.seed(self.output,number,port)
                client.assert_not_called()

    def seed_fixture(self, fail_advance=False):
        ledger = self.authorize()
        observed = []
        class API(FakeClient):
            def call(inner,method,path,data=None):
                kind = path.split("/")[0]
                counter = "select" if kind == "testcases" else kind
                if counter in ["select","initialize","advance"]:
                    count = ledger.read()["seeds"]["SAC-11"]["counts"][counter]
                    self.assertGreater(count,0)
                    observed.append((counter,count))
                if fail_advance and kind == "advance":
                    raise TimeoutError("fixture unknown applied status")
                return super().call(method,path,data)
        api = API()
        prefix = self.root/"fixture-prefix"
        prefix.mkdir()
        pd.DataFrame([api.measurement(START+30*i) for i in range(241)]).to_csv(prefix/"native.csv.gz",index=False)
        class Runner:
            def __init__(inner,folder,algorithm,number,protocol,actual_ledger,factory,sources,started):
                inner.key = actual_ledger.start_seed(algorithm,number,started)
                folder.mkdir(exist_ok=False)
                inner.factory = factory
                inner.ledger = actual_ledger
            def run(inner):
                spec = next(e for e in read(ROOT/"configs/v2_0.json")["episodes"] if e["id"] == "train014")
                env = inner.factory(spec,"train")
                try:
                    obs,_ = env.reset()
                    self.assertEqual(obs.shape,(503,))
                    for action in [0,.5,-.5]:
                        _,_,_,_,info = env.step([action])
                        self.assertEqual(info["control"]["raw_actor_normalized"],action)
                        self.assertAlmostEqual(info["control"]["actual_command_C"],info["control"]["measured_setpoint_C"])
                finally:
                    env.close()
                    inner.ledger.finish(inner.key,"failed")  # Deliberately short fixture, not research data.
        with patch.object(entry,"Client",return_value=api),patch.object(entry.joblib,"load",return_value=None),\
                patch.object(entry,"SolarController",return_value=FakeRule()),patch.object(entry,"FirstGateRunner",Runner),\
                patch.object(entry,"reference",return_value=prefix):
            if fail_advance:
                with self.assertRaises(TimeoutError):
                    entry.seed(self.output,11,8083)
            else:
                entry.seed(self.output,11,8083)
        return ledger.read(),observed,api

    def test_versioned_seed_full_transport_predebits_and_residual_interface(self):
        state,observed,api = self.seed_fixture()
        self.assertEqual(state["counts"],{"select":1,"initialize":1,"advance":11})
        self.assertEqual(observed,[('select',1),('initialize',1)]+[('advance',i) for i in range(1,12)])
        self.assertEqual(sum(kind == "stop" for kind,_ in api.log),1)

    def test_unknown_advance_not_retried_or_refunded(self):
        state,observed,_ = self.seed_fixture(fail_advance=True)
        self.assertEqual(state["counts"],{"select":1,"initialize":1,"advance":1})
        self.assertEqual(observed,[('select',1),('initialize',1),('advance',1)])
        self.assertEqual(state["seeds"]["SAC-11"]["status"],"failed")

    def dispatch_fixture(self, fail_first=False):
        ledger = self.authorize()
        spawned,peak = [],[0]
        class Worker:
            def __init__(inner,args,**kwargs):
                self.assertEqual(args[2],"scripts.run_v2_residual_safe")
                number = int(args[args.index("--seed")+1])
                inner.key = ledger.start_seed("SAC",number,ledger.now())
                inner.pid,inner.returncode,inner.polls = number,None,0
                inner.log = kwargs["stdout"]
                spawned.append(inner)
                peak[0] = max(peak[0],sum(v["status"] == "running" for v in ledger.read()["seeds"].values()))
            def poll(inner):
                if inner.returncode is not None:
                    return inner.returncode
                inner.polls += 1
                if ledger.read()["seeds"][inner.key]["status"] == "failed":
                    inner.returncode = 1
                elif inner.polls >= 3:
                    if fail_first and inner.key == "SAC-11":
                        ledger.finish(inner.key,"failed")
                        inner.returncode = 1
                    else:
                        # Mark simulated completion only in this temporary process fixture.
                        ledger.reserve(inner.key,"select")
                        state = ledger.read()
                        for kind in state["counts"]:
                            cap = ledger.plan["budget_per_algorithm_seed"][kind+"_max"]
                            state["counts"][kind] += cap-state["seeds"][inner.key]["counts"][kind]
                            state["seeds"][inner.key]["counts"][kind] = cap
                        write_snapshot(ledger.path,state)
                        ledger.finish(inner.key,"completed")
                        inner.returncode = 0
                return inner.returncode
        with patch.object(entry.subprocess,"Popen",side_effect=Worker),patch.object(entry.time,"sleep"):
            entry.dispatch(self.output)
        self.assertTrue(all(w.log.closed for w in spawned))
        return ledger,spawned,peak[0]

    def test_two_slot_dispatch_success_order_counts_and_handle_closure(self):
        ledger,workers,peak = self.dispatch_fixture()
        self.assertEqual([w.key for w in workers],["SAC-11","SAC-23","SAC-37"])
        self.assertEqual(peak,2)
        self.assertEqual(ledger.read()["counts"],{"select":225,"initialize":225,"advance":151200})
        self.assertTrue(all(v["status"] == "completed" for v in ledger.read()["seeds"].values()))
        with (self.output/"SAC-11.log").open("ab") as stream:
            stream.write(b"closed parent handle")

    def test_two_slot_failure_cancels_peer_and_never_launches_third(self):
        ledger,workers,peak = self.dispatch_fixture(fail_first=True)
        self.assertEqual([w.key for w in workers],["SAC-11","SAC-23"])
        self.assertEqual(peak,2)
        self.assertTrue(all(v["status"] == "failed" for v in ledger.read()["seeds"].values()))
        self.assertEqual(ledger.read()["counts"],{"select":0,"initialize":0,"advance":0})

    def test_dispatch_claim_prevents_second_launch(self):
        self.authorize()
        (self.output/"dispatch.jsonl").write_bytes(b"existing dispatch\n")
        with patch.object(entry.subprocess,"Popen") as spawn:
            with self.assertRaises(FileExistsError):
                entry.dispatch(self.output)
            spawn.assert_not_called()
        self.assertEqual((self.output/"dispatch.jsonl").read_bytes(),b"existing dispatch\n")

    def test_spawn_failure_has_receipt_and_closes_log_without_retry(self):
        self.authorize()
        with patch.object(entry.subprocess,"Popen",side_effect=OSError("fixture spawn failure")) as spawn:
            with self.assertRaises(OSError):
                entry.dispatch(self.output)
            self.assertEqual(spawn.call_count,1)
        self.assertTrue(read(self.output/"dispatch-failure.json")["no_restart"])
        with (self.output/"SAC-11.log").open("ab") as stream:
            stream.write(b"closed after spawn failure")

    def test_close_refuses_running_seed_and_after_close_no_execution(self):
        ledger = self.authorize()
        key = ledger.start_seed("SAC",11)
        with self.assertRaises(RuntimeError):
            entry.close(self.output)
        ledger.finish(key,"failed")
        entry.close(self.output)
        with self.assertRaises(PermissionError):
            entry.registration_for(self.output)
