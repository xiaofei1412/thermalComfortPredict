"""Single preregistered V2-4 rule-residual SAC campaign, using the safe ledger."""

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

PROCESS_START = {"wall":time.time(),"monotonic":time.monotonic()}

import joblib
import torch

from scripts.authorize_v2_safe import PARENTS as BASE_PARENTS, close
from scripts.run_sac import reference
from scripts.run_v2_comparison import SOURCES as BASE_SOURCES
from scripts.run_v2_formal import REPLICAS
from scripts.run_v2_safe import REPAIR_SOURCES
from thermal_comfort.rl_budget import BudgetTransport
from thermal_comfort.rl_budget_safe import BudgetLedger, write_snapshot
from thermal_comfort.rl_failure_export import FirstGateRunner
from thermal_comfort.rl_residual import ResidualEnv
from thermal_comfort.rl_runner import append_json
from thermal_comfort.sac_env import CampaignClient
from thermal_comfort.simulation import Client, digest
from thermal_comfort.solar import SolarController
from thermal_comfort.v2_protocol import ROOT, read, verify_files

PARENTS = [*BASE_PARENTS,"reports/v2_safe_formal/lock.json"]
SOURCES = [*BASE_SOURCES,*REPAIR_SOURCES,"configs/v2_4.json","thermal_comfort/rl_residual.py",
    "scripts/run_v2_residual.py","scripts/diagnose_v2_4.py","docs/experiments/v2_4.md","tests/test_v2_residual.py"]


def plan():
    settings = read(ROOT/"configs/v2_4.json")
    original = read(ROOT/settings["base_protocol"])
    if (settings["variants_max"] != 1 or settings["algorithms"] != ["SAC"] or settings["seeds"] != original["seeds"]
            or settings["cycles"] != 12 or settings["residual_amplitude_K"] != .5
            or settings["heldout_initializations_authorized"] != 0 or settings["old_trials_resumed"]):
        raise ValueError("Unregistered V2-4 variant or permissions")
    value = copy.deepcopy(original)
    value.update(protocol=settings["protocol"],variant=settings["variant"],algorithms=["SAC"],
        residual_amplitude_K=settings["residual_amplitude_K"])
    value["budget_total_all_algorithms_seeds"] = {k: original["budget_per_algorithm_seed"][k+"_max"]*3
        for k in ["select","initialize","advance"]}
    value["budget_total_all_algorithms_seeds"]["stage_wall_seconds_max"] = settings["stage_wall_seconds_max"]
    if (value["budget_total_all_algorithms_seeds"] != {"select":225,"initialize":225,"advance":151200,"stage_wall_seconds_max":18000}
            or settings["total_advance_max"] != 151200 or settings["total_select_max"] != 225
            or settings["total_initialize_max"] != 225 or settings["physical_advances_per_seed"] != 50400
            or settings["train_actor_steps_per_seed"] != 47808 or settings["seed_wall_seconds_max"] != 10800
            or settings["development"] != original["development_ids"]):
        raise ValueError("V2-4 budget/roles differ from the preregistration")
    return value


def authorize(output, preflight_path, diagnostic_path):
    protocol, preflight, diagnostic = plan(),read(preflight_path),read(diagnostic_path)
    if (preflight["status"] != "passed" or preflight["replicas"] != REPLICAS
            or preflight["FMU_select_or_initialize_calls"] != 0 or not diagnostic["hypothesis_gate_passed"]
            or diagnostic["protocol_sha256"] != digest(ROOT/"configs/v2_4.json")
            or diagnostic["parent_lock_sha256"] != digest(ROOT/PARENTS[-1])):
        raise ValueError("Frozen diagnostic/replica gate failed")
    for name in PARENTS:
        verify_files(ROOT,read(ROOT/name)["files_sha256"])
    baseline = read(ROOT/PARENTS[0])["v1_baseline"]
    assets = {"outputs/solar-selected/model.joblib":baseline["model_sha256"],
        "outputs/boptest-v0.9.0/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu":baseline["fmu_sha256"]}
    verify_files(ROOT,assets)
    output.mkdir(parents=True,exist_ok=False)
    original = {"utc":datetime.now(timezone.utc).isoformat(),"git_base":subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),"variant":protocol["variant"],"formal_training_authorized":False,
        "protocol":protocol,"files_sha256":{p:digest(ROOT/p) for p in SOURCES},
        "parent_locks_sha256":{p:digest(ROOT/p) for p in PARENTS}}
    write_snapshot(output/"registration.json",original)
    BudgetLedger(output,protocol).create()
    (output/"registration-before-authorization.json").write_bytes((output/"registration.json").read_bytes())
    write_snapshot(output/"preflight.json",preflight)
    write_snapshot(output/"diagnostic.json",diagnostic)
    auth = {"utc":datetime.now(timezone.utc).isoformat(),"basis":"explicit_owner_request_for_V2_4_in_current_chat",
        "request":"V2-4有限改进：单一规则残差SAC固定预算；保留旧结果，不碰留出",
        "protocol_sha256":digest(ROOT/"configs/v2_4.json"),"base_protocol_sha256":digest(ROOT/"configs/v2_comparison.json"),
        "parent_locks_sha256":original["parent_locks_sha256"],"frozen_assets_sha256":assets,"replicas":REPLICAS,
        "actual_max_parallel":2,"heldout_initializations_authorized":0,"old_trial_resumption_authorized":False,
        "initial_registration_sha256":digest(output/"registration-before-authorization.json"),
        "preflight_sha256":digest(output/"preflight.json"),"diagnostic_sha256":digest(output/"diagnostic.json")}
    write_snapshot(output/"authorization.json",auth)
    original.update(formal_training_authorized=True,authorization_sha256=digest(output/"authorization.json"))
    write_snapshot(output/"registration.json",original)
    (output/"registration-authorized.json").write_bytes((output/"registration.json").read_bytes())


def registration_for(output):
    registration = read(output/"registration.json")
    if not registration["formal_training_authorized"] or registration.get("variant") != "rule_anchored_residual_SAC":
        raise PermissionError("No active V2-4 residual authorization; old studies cannot be resumed")
    if digest(output/"authorization.json") != registration["authorization_sha256"] or registration["protocol"] != plan():
        raise ValueError("V2-4 authorization/protocol changed")
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
    write_snapshot(folder/"method.json",{"variant":protocol["variant"],"protocol_sha256":digest(ROOT/"configs/v2_4.json"),
        "checkpoint_base_protocol_binding":"configs/v2_comparison.json; child method/config pinned in sources_sha256"})
    runner.run()


def dispatch(output):
    registration = registration_for(output)
    ledger = BudgetLedger(output,registration["protocol"])
    journal = output/"dispatch.jsonl"
    if journal.exists():
        raise FileExistsError("Dispatcher restart forbidden")
    pending,ports,active = [11,23,37],[r["port"] for r in REPLICAS],{}
    with ExitStack() as handles:
        try:
            while pending or active:
                if pending and ports:
                    number,port = pending.pop(0),ports.pop(0)
                    key = f"SAC-{number}"
                    log = handles.enter_context((output/(key+".log")).open("wb"))
                    started = ledger.now()
                    process = subprocess.Popen([sys.executable,"-m","scripts.run_v2_residual","seed","--output",str(output),
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
        except BaseException:
            for key in active:
                if key in ledger.read()["seeds"]:
                    ledger.finish(key,"failed")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["authorize","seed","dispatch","close"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--diagnostic",type=Path)
    parser.add_argument("--seed",type=int,choices=[11,23,37])
    parser.add_argument("--port",type=int)
    args = parser.parse_args()
    if args.mode == "authorize":
        authorize(args.output,args.preflight,args.diagnostic)
    elif args.mode == "seed":
        seed(args.output,args.seed,args.port)
    elif args.mode == "dispatch":
        dispatch(args.output)
    else:
        close(args.output)
