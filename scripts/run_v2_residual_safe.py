"""Versioned V2-4 entry with pre-authorization budget/transport/dispatch acceptance."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

PROCESS_START = {"wall":time.time(),"monotonic":time.monotonic()}

import joblib
import torch

from scripts.authorize_v2_safe import close
from scripts.run_v2_residual import PARENTS as OLD_PARENTS, SOURCES as OLD_SOURCES, plan as failed_plan
from scripts.run_sac import reference
from scripts.run_v2_formal import REPLICAS
from thermal_comfort.rl_budget_contract import corrected_residual_plan, validate_budget
from thermal_comfort.rl_budget import BudgetTransport
from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
from thermal_comfort.rl_failure_export import FirstGateRunner
from thermal_comfort.rl_residual import ResidualEnv
from thermal_comfort.rl_runner import append_json
from thermal_comfort.sac_env import CampaignClient
from thermal_comfort.simulation import Client, digest
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read, verify_files

PARENTS = [*OLD_PARENTS,"reports/v2_4/lock.json"]
SOURCES = [*OLD_SOURCES,"thermal_comfort/rl_budget_contract.py","configs/v2_4_safe.json",
    "scripts/run_v2_residual_safe.py","scripts/check_v2_residual_safe.py",
    "tests/test_v2_residual_safe.py","docs/experiments/v2_4_safe.md"]


def plan():
    settings = read(ROOT/"configs/v2_4_safe.json")
    value = corrected_residual_plan(failed_plan())
    if (settings != {"protocol":"residential_rl_v2_4_rule_residual_safe_1",
            "scientific_protocol":"configs/v2_4.json","parent_lock":PARENTS[-1],
            "variant":"rule_anchored_residual_SAC","total_select_max":225,"total_initialize_max":225,
            "total_advance_max":151200,"seed_wall_seconds_max":10800,"stage_wall_seconds_max":18000,
            "parallel_max":2,"entry_gate_required":True,"heldout_initializations_authorized":0,
            "resume_or_replace_failed_seeds":False}):
        raise ValueError("Safe entry differs from frozen V2-4 scientific scope/budget")
    value["protocol"] = settings["protocol"]
    validate_budget(value)
    expected = {"select_max":225,"initialize_max":225,"advance_max":151200,"stage_wall_seconds_max":18000}
    if value["budget_total_all_algorithms_seeds"] != expected:
        raise ValueError("Invalid residual ledger contract")
    return value


def source_hashes():
    return {p:digest(ROOT/p) for p in SOURCES}


def check_gate(gate):
    if (gate["status"] != "passed" or gate["test_returncode"] != 0 or gate["test_count"] < 12
            or gate["FMU_calls"] != 0 or gate["sources_sha256"] != source_hashes()):
        raise ValueError("Current safe entry has no intact offline acceptance gate")


def authorize(output, preflight_path, diagnostic_path, entry_gate_path):
    protocol = validate_budget(plan())  # Fail before creating any directory/authorization.
    gate = read(entry_gate_path)
    check_gate(gate)
    preflight, diagnostic = read(preflight_path),read(diagnostic_path)
    if (preflight["status"] != "passed" or preflight["replicas"] != REPLICAS
            or preflight["FMU_select_or_initialize_calls"] != 0 or not diagnostic["hypothesis_gate_passed"]
            or diagnostic["protocol_sha256"] != digest(ROOT/"configs/v2_4.json")
            or diagnostic["parent_lock_sha256"] != digest(ROOT/"reports/v2_safe_formal/lock.json")):
        raise ValueError("Frozen diagnostic/replica gate failed")
    for name in PARENTS:
        verify_files(ROOT,read(ROOT/name)["files_sha256"])
    baseline = read(ROOT/PARENTS[0])["v1_baseline"]
    assets = {"outputs/solar-selected/model.joblib":baseline["model_sha256"],
        "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu":baseline["fmu_sha256"]}
    verify_files(ROOT,assets)
    output.mkdir(parents=True,exist_ok=False)
    original = {"utc":datetime.now(timezone.utc).isoformat(),"git_base":subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),"variant":protocol["variant"],"entry_protocol":protocol["protocol"],"formal_training_authorized":False,
        "protocol":protocol,"files_sha256":source_hashes(),
        "parent_locks_sha256":{p:digest(ROOT/p) for p in PARENTS}}
    write_snapshot(output/"registration.json",original)
    BudgetLedger(output,protocol).create()
    (output/"registration-before-authorization.json").write_bytes((output/"registration.json").read_bytes())
    write_snapshot(output/"preflight.json",preflight)
    write_snapshot(output/"diagnostic.json",diagnostic)
    write_snapshot(output/"entry-gate.json",gate)
    auth = {"utc":datetime.now(timezone.utc).isoformat(),"basis":"explicit_owner_request_safe_entry_then_effect_experiment",
        "request":"先完成新版安全入口验收，再开展效果实验；保留旧失败、同一残差配置/预算，不碰留出",
        "protocol_sha256":digest(ROOT/"configs/v2_4_safe.json"),"scientific_protocol_sha256":digest(ROOT/"configs/v2_4.json"),"base_protocol_sha256":digest(ROOT/"configs/v2_comparison.json"),
        "parent_locks_sha256":original["parent_locks_sha256"],"frozen_assets_sha256":assets,"replicas":REPLICAS,
        "actual_max_parallel":2,"heldout_initializations_authorized":0,"old_trial_resumption_authorized":False,
        "initial_registration_sha256":digest(output/"registration-before-authorization.json"),
        "preflight_sha256":digest(output/"preflight.json"),"diagnostic_sha256":digest(output/"diagnostic.json"),
        "entry_gate_sha256":digest(output/"entry-gate.json")}
    write_snapshot(output/"authorization.json",auth)
    original.update(formal_training_authorized=True,authorization_sha256=digest(output/"authorization.json"))
    write_snapshot(output/"registration.json",original)
    (output/"registration-authorized.json").write_bytes((output/"registration.json").read_bytes())


def registration_for(output):
    registration = read(output/"registration.json")
    if (not registration["formal_training_authorized"] or registration.get("variant") != "rule_anchored_residual_SAC"
            or registration.get("entry_protocol") != "residential_rl_v2_4_rule_residual_safe_1"):
        raise PermissionError("No active V2-4 residual authorization; old studies cannot be resumed")
    if digest(output/"authorization.json") != registration["authorization_sha256"] or registration["protocol"] != plan():
        raise ValueError("V2-4 authorization/protocol changed")
    validate_budget(registration["protocol"])
    auth = read(output/"authorization.json")
    if digest(output/"entry-gate.json") != auth["entry_gate_sha256"]:
        raise ValueError("Acceptance receipt changed")
    check_gate(read(output/"entry-gate.json"))
    verify_files(ROOT,registration["files_sha256"])
    verify_files(ROOT,registration["parent_locks_sha256"])
    verify_files(ROOT,read(output/"authorization.json")["frozen_assets_sha256"])
    return registration


def seed(output, number, port):
    registration = registration_for(output)
    if number not in [11,23,37] or port not in [r["port"] for r in REPLICAS]:
        raise ValueError("Unknown seed/replica")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = registration["protocol"]
    ledger = BudgetLedger(output,protocol)
    key, folder = f"SAC-{number}",output/f"SAC-{number}"
    adapter = {"train_ids":protocol["train_ids"],"development_ids":protocol["development_ids"],
        "budget_per_seed":protocol["budget_per_algorithm_seed"],"fixed_observation_scaling":protocol["shared"]["observation_scaling"],
        "reward":protocol["shared"]["reward"],"guard":protocol["shared"]["guard"]}
    client = CampaignClient(BudgetTransport(Client(f"http://127.0.0.1:{port}"),ledger,key),adapter,folder)
    baseline = read(ROOT/PARENTS[0])["v1_baseline"]
    guard = joblib.load(ROOT/"outputs/solar-selected/model.joblib")
    def factory(spec, role):
        return ResidualEnv(SolarController(guard,baseline["effective_config"],"guarded_rule","perfect"),
            client,spec,role,adapter,reference(spec),algorithm="SAC",amplitude_K=protocol["residual_amplitude_K"])
    runner = FirstGateRunner(folder,"SAC",number,protocol,ledger,factory,registration["files_sha256"],started=PROCESS_START)
    write_snapshot(folder/"method.json",{"variant":protocol["variant"],"protocol_sha256":digest(ROOT/"configs/v2_4_safe.json"),
        "scientific_protocol_sha256":digest(ROOT/"configs/v2_4.json"),
        "checkpoint_base_protocol_binding":"configs/v2_comparison.json; child method/config pinned in sources_sha256"})
    runner.run()


def dispatch(output):
    registration = registration_for(output)
    ledger = BudgetLedger(output,registration["protocol"])
    journal = output/"dispatch.jsonl"
    with journal.open("x",encoding="utf-8"):
        pass  # Exclusive claim before spawning; a second dispatcher cannot launch a worker.
    pending,ports,active = [11,23,37],[r["port"] for r in REPLICAS],{}
    with ExitStack() as handles:
        try:
            while pending or active:
                state = ledger.read()
                if any(v["status"] == "failed" for v in state["seeds"].values()):
                    pending.clear()
                    for key in active:
                        if state["seeds"].get(key,{}).get("status") == "running":
                            ledger.finish(key,"failed")
                if pending and ports:
                    number,port = pending.pop(0),ports.pop(0)
                    key = f"SAC-{number}"
                    log = handles.enter_context((output/(key+".log")).open("wb"))
                    started = ledger.now()
                    process = subprocess.Popen([sys.executable,"-m","scripts.run_v2_residual_safe","seed","--output",str(output),
                        "--seed",str(number),"--port",str(port)],stdout=log,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                    active[key] = process,log,port,started
                    append_json(journal,{"event":"launch","key":key,"pid":process.pid,"port":port})
                    while key not in ledger.read()["seeds"] and process.poll() is None:
                        time.sleep(.2)
                for key,(process,log,port,started) in list(active.items()):
                    if process.poll() is not None:
                        state = ledger.read()
                        if key not in state["seeds"]:
                            ledger.start_seed("SAC",int(key.split("-")[1]),started)
                        if key not in state["seeds"] or state["seeds"][key]["status"] == "running":
                            ledger.finish(key,"failed")
                            (output/key).mkdir(exist_ok=True)
                            write_snapshot(output/key/"dispatcher-failure.json",{"error":"worker exited before completion",
                                "exit_code":process.returncode,"budget":ledger.read()})
                        if ledger.read()["seeds"][key]["status"] != "completed":
                            pending.clear()
                            for other in active:
                                if other != key and ledger.read()["seeds"].get(other,{}).get("status") == "running":
                                    ledger.finish(other,"failed")
                        append_json(journal,{"event":"exit","key":key,"exit_code":process.returncode})
                        log.close()
                        ports.append(port)
                        del active[key]
                write_snapshot(output/"progress.json",{"pending":pending,"active":{k:v[0].pid for k,v in active.items()},"budget":ledger.read()})
                if active:
                    time.sleep(2)
            write_snapshot(output/"dispatch-result.json",{"status":"finished_no_retries","budget":ledger.read()})
        except BaseException as error:
            for key in active:
                if key in ledger.read()["seeds"]:
                    ledger.finish(key,"failed")
            for process,_,_,_ in active.values():
                process.wait(timeout=30)  # Drain owned children after cooperative budget cancellation.
            write_snapshot(output/"dispatch-failure.json",{"error":str(error),"budget":ledger.read(),
                "active_pids":{k:v[0].pid for k,v in active.items()},"no_restart":True})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["authorize","seed","dispatch","close"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--diagnostic",type=Path)
    parser.add_argument("--entry-gate",type=Path)
    parser.add_argument("--seed",type=int,choices=[11,23,37])
    parser.add_argument("--port",type=int)
    args = parser.parse_args()
    if args.mode == "authorize":
        authorize(args.output,args.preflight,args.diagnostic,args.entry_gate)
    elif args.mode == "seed":
        seed(args.output,args.seed,args.port)
    elif args.mode == "dispatch":
        dispatch(args.output)
    else:
        close(args.output)
