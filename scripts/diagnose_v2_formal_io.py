"""Isolated file-handle probe only; never repairs/restarts the registered trial."""

import argparse
import os
from pathlib import Path

from thermal_comfort.rl_budget import atomic_json, locked
from thermal_comfort.v2_protocol import read


def diagnose(directory):
    directory.mkdir(parents=True, exist_ok=False)
    target = directory/"fixture-budget.json"
    atomic_json(target, {"counter": 0})
    error = None
    with target.open("r"):
        try:
            atomic_json(target, {"counter": 1})
        except OSError as caught:
            error = {"type": type(caught).__name__, "errno": caught.errno,
                "winerror": getattr(caught, "winerror", None), "message": str(caught)}
    after_rejected_replace = read(target)
    atomic_json(target, {"counter": 2})
    with locked(directory/"fixture.lock"):
        with target.open("r") as stream:
            stream.read()
    with locked(directory/"fixture.lock"):
        atomic_json(target, {"counter": 3})
    result = {"platform": os.name, "held_reader_replace_error": error,
        "after_held_reader_attempt": after_rejected_replace, "after_close_and_serialization": read(target),
        "temporary_files_preserved": [p.name for p in directory.glob("*.tmp")],
        "matches_live_Windows_failure_pattern": os.name == "nt" and error is not None and error["winerror"] == 5,
        "trial_source_changed": False, "new_training_or_FMU_calls": 0,
        "scope": "Shows an unlocked open reader can block replacement; does not uniquely identify every live holder"}
    atomic_json(directory/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(diagnose(args.output))
