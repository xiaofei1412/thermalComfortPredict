"""Complete cycle checkpoints: validate privately, then publish by atomic rename."""

import hashlib
import os
import shutil

import numpy as np
import torch

from .rl_budget import atomic_json
from .rl_random_state import restore_rng, runtime, save_rng
from .rl_runner import ALGORITHMS, assert_boundary
from .simulation import digest
from .v2_protocol import ROOT, read, verify_files


ARTIFACTS = ["steps.jsonl.gz", "forecasts.jsonl.gz", "native.csv.gz", "schedule.csv.gz",
    "actor_observations.npz", "terminal_forecast.json.gz"]


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def contained(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint reference escapes its seed directory")
    return path


def episode_files(seed_root, folder):
    summary = read(folder/"summary.json")
    score = summary["outcome"]["score"]
    if (summary["transitions"] != 672 or summary["actor_transitions"] != 664
            or not summary["outcome"]["episode_complete"] or summary["outcome"]["aborted"]
            or not summary["outcome"]["terminated"] or summary["outcome"]["truncated"]
            or score["transitions"] != 672 or score["native_samples"] != 20161
            or not score["kpis"] or not score["kpi_absolute_difference"]
            or any(not np.isfinite(v) or v > 1e-7 for v in score["kpi_absolute_difference"].values())):
        raise ValueError("Incomplete episode/official KPI export")
    verify_files(folder, {name: summary["sha256"][name] for name in ARTIFACTS})
    with np.load(folder/"actor_observations.npz") as values:
        if (values["observations"].shape != (664, 503) or values["terminal_observation"].shape != (503,)
                or not all(np.isfinite(values[k]).all() for k in values)):
            raise ValueError("Missing/invalid actor or terminal observation")
    return {p.relative_to(seed_root).as_posix(): digest(p)
        for p in [folder/"summary.json", *[folder/name for name in ARTIFACTS]]}


def inspect_bundle(path, env, expected_manifest_sha256):
    """Audit load only; never reconnect/restore an FMU or enable training resume."""
    if digest(path/"manifest.json") != expected_manifest_sha256:
        raise ValueError("Checkpoint manifest hash mismatch")
    manifest = read(path/"manifest.json")
    verify_files(path, manifest["files_sha256"])
    binding = manifest["bindings"]
    verify_files(ROOT, binding["sources_sha256"])
    if digest(ROOT/"configs/v2_comparison.json") != binding["protocol_sha256"]:
        raise ValueError("Checkpoint protocol mismatch")
    if manifest["runtime"] != runtime():
        raise ValueError("Checkpoint runtime mismatch")
    seed_root = path.parent.parent
    for name, expected in manifest["episodes_sha256"].items():
        if digest(contained(seed_root, name)) != expected:
            raise ValueError("Exported episode hash mismatch")
    budget = read(path/"budget.json")
    entry = budget["seeds"][binding["key"]]
    if entry["counts"] != binding["counts"] or entry["start"] != binding["seed_start"]:
        raise ValueError("Original counts/clock binding differs")
    callback = read(path/"callback.json")
    for filename, size in [("actions.jsonl", callback["actions_bytes"]), ("curve.jsonl", callback["curve_bytes"])]:
        actual = contained(seed_root, filename)
        # At an earlier retained boundary, only the original prefix is pinned.
        with actual.open("rb") as stream:
            prefix = stream.read(size)
        if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != manifest["journal_prefix_sha256"][filename]:
            raise ValueError("Action/curve log prefix differs")
    model = ALGORITHMS[binding["algorithm"]].load(path/"model.zip", device="cpu")
    if binding["algorithm"] != "PPO":
        model.load_replay_buffer(path/"replay.pkl")
    if model.num_timesteps != manifest["learning"]["actor_steps"] or model._n_updates != manifest["learning"]["updates"]:
        raise ValueError("Loaded learning counters differ")
    restore_rng(path/"rng.pkl", model, env, binding, manifest["files_sha256"]["rng.pkl"])
    return model, manifest, callback


def save_bundle(seed_root, model, env, algorithm, cycle, journal, ledger, key, plan, folders, sources):
    if cycle < 1 or cycle > 12 or len(folders) != cycle*6 or env.testid is not None or not env.completed:
        raise ValueError("A released complete six-episode cycle is required")
    learning = assert_boundary(model, algorithm, cycle*6)
    if journal.rows != learning["actor_steps"]:
        raise ValueError("Action log does not cover every training step")
    snapshots = seed_root/"checkpoints"
    snapshots.mkdir(exist_ok=True)
    partial, final = snapshots/f"cycle-{cycle:02}.partial", snapshots/f"cycle-{cycle:02}"
    if final.exists():
        raise FileExistsError("Committed checkpoint cannot be replaced")
    partial.mkdir(exist_ok=False)
    episodes = {}
    for folder in folders:
        episodes.update(episode_files(seed_root, folder))
    model.save(partial/"model.zip")
    if algorithm != "PPO":
        model.save_replay_buffer(partial/"replay.pkl")
    atomic_json(partial/"callback.json", journal.state())
    budget = ledger.reserve(key)
    entry = budget["seeds"][key]
    expected = {"select": cycle*6, "initialize": cycle*6, "advance": cycle*4032}
    if entry["counts"] != expected:
        raise ValueError("Cumulative physical attempts do not match completed cycles")
    binding = {"algorithm": algorithm, "key": key, "completed_cycles": cycle,
        "next_episode": plan["train_ids"][0] if cycle < 12 else plan["development_ids"][0],
        "protocol_sha256": digest(ROOT/"configs/v2_comparison.json"), "sources_sha256": sources,
        "model_sha256": digest(partial/"model.zip"),
        "replay_sha256": digest(partial/"replay.pkl") if algorithm != "PPO" else None,
        "counts": entry["counts"], "seed_start": entry["start"]}
    rng_hash = save_rng(partial/"rng.pkl", model, env, binding)
    restored = False
    try:
        atomic_json(partial/"budget.json", budget)
        manifest = {"bindings": binding, "runtime": runtime(), "learning": learning,
            "episodes_sha256": episodes, "training_resume_authorized": False,
            "journal_prefix_sha256": {name: digest(seed_root/name) for name in ["actions.jsonl", "curve.jsonl"]},
            "files_sha256": {p.name: digest(p) for p in partial.iterdir() if p.is_file()}}
        atomic_json(partial/"manifest.json", manifest)
        loaded, _, _ = inspect_bundle(partial, env, digest(partial/"manifest.json"))
        if not equal(model.get_parameters(), loaded.get_parameters()):
            raise ValueError("Weights/optimizer restore differs")
        for name in model._get_torch_save_params()[1]:
            if not equal(getattr(model, name), getattr(loaded, name)):
                raise ValueError("Entropy/extra tensor restore differs")
        if algorithm != "PPO":
            for name in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
                if not equal(getattr(model.replay_buffer, name), getattr(loaded.replay_buffer, name)):
                    raise ValueError("Replay restore differs")
            if (loaded.replay_buffer.pos != learning["actor_steps"] or loaded.replay_buffer.full
                    or loaded.replay_buffer.dones[:learning["actor_steps"]].sum() != cycle*6
                    or loaded.replay_buffer.timeouts[:learning["actor_steps"]].sum() != 0):
                raise ValueError("Replay position/termination differs")
        # Validation loads reseed globals; put the LIVE model's stream back LAST.
        restore_rng(partial/"rng.pkl", model, env, binding, rng_hash)
        restored = True
        for path in partial.iterdir():
            if path.is_file():
                with path.open("rb+") as stream:
                    os.fsync(stream.fileno())
        atomic_json(partial/"budget.json", ledger.reserve(key))
        manifest["files_sha256"]["budget.json"] = digest(partial/"budget.json")
        atomic_json(partial/"manifest.json", manifest)
        partial.rename(final)
    finally:
        if not restored:
            restore_rng(partial/"rng.pkl", model, env, binding, rng_hash)
    archive = seed_root/"checkpoint-manifests"
    archive.mkdir(exist_ok=True)
    atomic_json(archive/f"cycle-{cycle:02}.json", {"manifest_sha256": digest(final/"manifest.json"), **manifest})
    retain_latest(seed_root)
    return final


def retain_latest(seed_root):
    snapshots, archive = seed_root/"checkpoints", seed_root/"checkpoint-manifests"
    # Delete only resolved, verified, older complete bundles under this seed.
    for old in sorted(snapshots.glob("cycle-??"))[:-2]:
        contained(snapshots, old.name)
        pin = read(archive/(old.name+".json"))["manifest_sha256"]
        if digest(old/"manifest.json") != pin:
            raise ValueError("Refusing cleanup of an unverified checkpoint")
        shutil.rmtree(old)
