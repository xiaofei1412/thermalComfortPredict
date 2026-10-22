"""Recompute model scores, check serialization and compare independent repeat fitting."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_comfort.dynamics import common_origins, load_episodes, rollout_episode, score_rollouts
from thermal_comfort.excitation import verify_excitation
from thermal_comfort.simulation import digest, save_json


def verify(root, source):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Model artifact changed: {name}")
    if digest(source / "manifest.json") != manifest["source_manifest_sha256"] or digest(source / "summary.json") != manifest["source_summary_sha256"]:
        raise ValueError("Collection provenance changed")
    config = manifest["config"]
    episodes, splits = load_episodes(source, config)
    saved_splits = json.loads((root / "split_manifest.json").read_text())
    if splits != saved_splits:
        raise ValueError("Split manifest did not rebuild")
    # Only load trusted, locally generated model artifacts; joblib is not a safe data format.
    models = joblib.load(root / "models.joblib")
    selection = json.loads((root / "selection.json").read_text())
    for kind, chosen in selection["selected"].items():
        best = min([c for c in selection["candidates"] if c["kind"] == kind],
                   key=lambda c: c["validation_temperature_score_C"])
        if chosen["lag"] != best["lag"] or chosen["parameter"] != best["parameter"]:
            raise ValueError("Saved model was not chosen by validation")
    saved_metrics = pd.read_csv(root / "metrics.csv")
    rebuilt = []
    checks = []
    saved_origins = json.loads((root / "evaluation_origins.json").read_text())
    for episode in episodes.values():
        if episode["split"] == "train":
            continue
        origins = common_origins(episode, config["model"])
        declared = next(p for p in saved_origins if p["episode"] == episode["id"])
        if declared["origin_indices"] != origins.tolist():
            raise ValueError("Evaluation origins changed")
        for name, model in models.items():
            for weather in ["perfect", "hold"]:
                prediction, truth = rollout_episode(model, episode, origins, max(config["model"]["horizons_steps"]), weather)
                identity = {"episode": episode["id"], "split": episode["split"], "model": name, "weather": weather}
                rebuilt.extend([{**identity, **s} for s in score_rollouts(prediction, truth, config["model"]["horizons_steps"])])
        checks.append({"episode": episode["id"], "origins": len(origins), "full_horizon_inside_episode": True})
    pd.testing.assert_frame_equal(pd.DataFrame(rebuilt), saved_metrics, check_dtype=False, rtol=1e-12, atol=1e-12)
    summaries = json.loads((source / "summary.json").read_text())
    safety = [{"episode": s["scenario"], **verify_excitation(source / (s["scenario"] + "__excitation"), s)} for s in summaries]
    if safety != json.loads((source / "safety.json").read_text()):
        raise ValueError("Safety summary did not reproduce")
    return {"artifact_hashes_match": True, "split_manifest_rebuilt": True,
            "reloaded_model_scores_match": True, "selection_uses_validation_only": True,
            "evaluation_episodes": checks, "safety": safety}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/dynamics-baseline"))
    parser.add_argument("--source", type=Path, default=Path("outputs/excitation"))
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    result = verify(args.output, args.source)
    if args.repeat:
        verify(args.repeat, args.source)
        names = ["metrics.csv", "rollout_examples.csv", "coverage.csv", "errors.csv.gz"]
        for name in names:
            pd.testing.assert_frame_equal(pd.read_csv(args.output / name), pd.read_csv(args.repeat / name), check_exact=True)
        for name in ["selection.json", "split_manifest.json", "evaluation_origins.json"]:
            if json.loads((args.output / name).read_text()) != json.loads((args.repeat / name).read_text()):
                raise ValueError(f"Independent fitting differs: {name}")
        result["independent_repeat"] = {"exact_numeric_match": True, "files": names,
                                         "selection_and_splits_match": True}
    save_json(args.output / "verification.json", result)
    print(json.dumps(result, indent=2))
