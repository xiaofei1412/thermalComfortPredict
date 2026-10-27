"""One shared six-date cycle / final-only development loop for all algorithms."""

import copy
import time

from scripts.run_sac import export
from .rl_budget import atomic_json
from .rl_checkpoint import equal, inspect_bundle, save_bundle
from .rl_runner import ActionJournal, append_json, build_model, deterministic_action, seed_local
from .sac_env import replay_hash
from .v2_protocol import ROOT, read


class TrainingRunner:
    def __init__(self, root, algorithm, seed, plan, ledger, env_factory, sources, exporter=export, started=None):
        self.key = ledger.start_seed(algorithm, seed, started)
        root.mkdir(exist_ok=False)
        self.root, self.algorithm, self.seed, self.plan, self.ledger = root, algorithm, seed, plan, ledger
        self.env_factory, self.sources, self.exporter = env_factory, sources, exporter
        self.model, self.env, self.cycle, self.failed = None, None, 0, False
        self.folders = []
        self.journal = ActionJournal(root, algorithm, lambda: ledger.reserve(self.key))
        self.cpu_started = time.process_time()
        self.specs = {e["id"]: e for e in read(ROOT/"configs/v2_0.json")["episodes"]}

    def run_cycle(self):
        if self.failed or self.cycle >= 12:
            raise RuntimeError("Stopped/exhausted runner cannot be continued")
        from .rl_runner import train_episode
        try:
            for index, name in enumerate(self.plan["train_ids"]):
                self.ledger.reserve(self.key)
                self.env = self.env_factory(self.specs[name], "train")
                if self.model is None:
                    self.model = build_model(self.algorithm, self.env, self.seed, self.plan)
                seed_local(self.model, self.env, self.seed+10*(self.cycle*6+index))
                started = time.perf_counter()
                state = train_episode(self.model, self.env, self.journal, self.cycle*6+index+1,
                    lambda: self.ledger.reserve(self.key))
                folder = self.root/"train"/f"cycle-{self.cycle+1:02}"/name
                folder.parent.mkdir(parents=True, exist_ok=True)
                summary = self.exporter(self.env, folder, time.perf_counter()-started)
                if summary["outcome"]["score"]["kpis"] is None:
                    raise ValueError("Official KPI archive is required")
                self.env.close()
                self.folders.append(folder)
                append_json(self.root/"episodes.jsonl", {"episode": name, "cycle": self.cycle+1, **state})
                self.env.records.clear()
                self.ledger.reserve(self.key)
            self.final = save_bundle(self.root, self.model, self.env, self.algorithm, self.cycle+1,
                self.journal, self.ledger, self.key, self.plan, self.folders, self.sources)
            self.cycle += 1
            self.ledger.reserve(self.key)
            return self.final
        except Exception as error:
            self.fail(error)
            raise

    def fail(self, error):
        self.failed = True
        failure = {"error": str(error), "cycle": self.cycle,
            "actor_steps": self.model.num_timesteps if self.model else 0, "budget": self.ledger.read(),
            "completed_episode_folders": [str(p.relative_to(self.root)) for p in self.folders]}
        atomic_json(self.root/"failure.json", failure)
        if self.env is not None:
            if self.env.records:
                try:
                    self.exporter(self.env, self.root/"failed-episode", 0.)
                except Exception as export_error:
                    failure["partial_export_error"] = str(export_error)
            try:
                self.env.close()
            except Exception as cleanup_error:
                failure["cleanup_error"] = str(cleanup_error)
        self.ledger.finish(self.key, "failed")
        atomic_json(self.root/"failure.json", failure)

    def evaluate_final(self):
        if self.failed or self.cycle != 12 or self.model.num_timesteps != 47808:
            raise RuntimeError("Only the complete final checkpoint may enter development")
        pin = read(self.root/"checkpoint-manifests/cycle-12.json")["manifest_sha256"]
        loaded, _, _ = inspect_bundle(self.final, self.env, pin)
        before = copy.deepcopy(loaded.get_parameters())
        replay_before = replay_hash(loaded) if self.algorithm != "PPO" else None
        results, inference = {}, 0.
        for name in self.plan["development_ids"]:
            self.env = self.env_factory(self.specs[name], "development")
            seed_local(loaded, self.env, self.seed)
            obs, _ = self.env.reset()
            started = time.perf_counter()
            for index in range(664):
                tick = time.perf_counter()
                action, raw = deterministic_action(loaded, self.algorithm, obs)
                inference += time.perf_counter()-tick
                obs, _, done, truncated, info = self.env.step(action)
                append_json(self.root/"development-actions.jsonl", {"episode": name, "actor_index": index,
                    "algorithm_raw": raw.tolist(), "library_action": action.tolist(), "control": info["control"]})
                if done or truncated:
                    break
            if index != 663 or not info["episode_complete"] or truncated:
                raise ValueError("Incomplete development episode")
            if loaded.num_timesteps != 47808 or not equal(before, loaded.get_parameters()):
                raise ValueError("Evaluation updated the frozen model/optimizers")
            folder = self.root/"development"/name
            folder.parent.mkdir(exist_ok=True)
            results[name] = self.exporter(self.env, folder, time.perf_counter()-started)
            if results[name]["outcome"]["score"]["kpis"] is None:
                raise ValueError("Development official KPI archive is required")
            self.env.close()
            self.env.records.clear()
            self.ledger.reserve(self.key)
        if self.algorithm != "PPO" and replay_hash(loaded) != replay_before:
            raise ValueError("Evaluation changed the in-memory replay state")
        return results, inference

    def run(self):
        try:
            while self.cycle < 12:
                self.run_cycle()
            results, inference = self.evaluate_final()
            self.ledger.reserve(self.key)
            atomic_json(self.root/"result.partial.json", {"status": "completed_development_only", "algorithm": self.algorithm,
                "seed": self.seed, "development": results, "inference_seconds": inference,
                "cpu_seconds_including_updates_exports_and_checkpoints": time.process_time()-self.cpu_started,
                "update_seconds": self.model.update_seconds, "budget": self.ledger.read()})
            self.ledger.finish(self.key, "completed")
            (self.root/"result.partial.json").rename(self.root/"result.json")
        except Exception as error:
            if not self.failed:
                self.fail(error)
            raise
