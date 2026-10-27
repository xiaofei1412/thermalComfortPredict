"""Failure fragments contain only observed data and never enter the first gate."""

import gzip
import json

import numpy as np

from scripts.run_sac import outcome_for_json
from scripts.run_v2_formal import FirstGateRunner as FrozenFirstGateRunner
from .simulation import digest, save_json
from .v2_baselines import stitch


def export_fragment(env, folder, wall_seconds):
    folder.mkdir(parents=True, exist_ok=False)
    records = env.records
    with gzip.open(folder/"steps.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps({k: v for k, v in record.items() if k not in ["native", "schedule", "forecast"]}, allow_nan=False)+"\n")
    with gzip.open(folder/"forecasts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record["forecast"].to_dict("list"), allow_nan=False)+"\n")
    if records:
        for name in ["native", "schedule"]:
            stitch([r[name] for r in records]).to_csv(folder/(name+".csv.gz"), index=False)
    observations = np.asarray(getattr(env, "observations", []), dtype=np.float32).reshape(-1, 503)
    values = {"observations": observations}
    if hasattr(env, "cached_obs"):
        values["last_returned_observation"] = env.cached_obs
    np.savez_compressed(folder/"actor_observations.npz", **values)
    if getattr(env, "view", None) is not None:
        with gzip.open(folder/"last_available_forecast.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(env.view.to_dict("list"), stream, allow_nan=False)
    summary = {"status": "incomplete_failure_fragment", "episode": env.spec, "role": env.role,
        "transitions": len(records), "actor_observations": len(observations),
        "eligible_for_comparison_or_resume": False, "wall_seconds": None,
        "cost_source": "original cumulative ledger in failure.json; no fabricated episode duration",
        "last_observation_available": "last_returned_observation" in values,
        "last_info": outcome_for_json(getattr(env, "final_info", None)),
        "sha256": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}}
    save_json(folder/"summary.json", summary)
    return summary


class FirstGateRunner(FrozenFirstGateRunner):
    def fail(self, error):
        original = self.exporter
        self.exporter = export_fragment
        try:
            super().fail(error)
        finally:
            self.exporter = original
