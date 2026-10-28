"""Offline end-to-end acceptance receipt required by the new residual entry."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys

from scripts.run_v2_residual_safe import source_hashes
from thermal_comfort.rl_budget_safe import write_snapshot
from thermal_comfort.simulation import digest


def check(output):
    output.mkdir(parents=True,exist_ok=False)
    result = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests",
        "-p","test_v2_residual_safe.py","-v"],capture_output=True,text=True)
    log = result.stdout+result.stderr
    (output/"tests.log").write_bytes(log.encode("utf-8"))
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s",log)
    value = {"status":"passed" if result.returncode == 0 and match else "failed",
        "utc":datetime.now(timezone.utc).isoformat(),"test_returncode":result.returncode,
        "test_count":int(match[1]) if match else 0,"seconds":float(match[2]) if match else None,
        "sources_sha256":source_hashes(),"log_sha256":digest(output/"tests.log"),"FMU_calls":0,
        "scope":"real safe ledger + transport + versioned seed/dispatch entry; mock HTTP/subprocesses, temporary fixtures only",
        "training_or_real_trial_created":False}
    write_snapshot(output/"entry-gate.json",value)
    if value["status"] != "passed":
        raise RuntimeError("Safe entry acceptance failed; preserve tests.log")
    print(f"{value['test_count']} offline entry tests passed; no FMU calls")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    check(parser.parse_args().output)
