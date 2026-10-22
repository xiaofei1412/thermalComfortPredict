import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_comfort.dynamics import (
    common_origins, coverage_table, fit_model, load_episodes, rollout, rollout_episode,
    score_rollouts, spectral_radius, training_data,
)
from thermal_comfort.excitation import SafeExcitation, validate_plan


CONFIG = json.loads(Path("configs/dynamics.json").read_text())
LIMITS = {"Minimum": 278.15, "Maximum": 308.15}


def synthetic_episode(split="train", n=40):
    k = np.arange(n + 1, dtype=float)
    return {"id": split, "split": split, "states": np.column_stack([20 + .01 * k, 25 + .02 * k, 30 + .03 * k, .1 + .001 * k]),
            "actions": 21 + np.arange(n, dtype=float) / 100,
            "exog": np.column_stack([np.arange(n) / 10, np.zeros((n, 4))]),
            "times": np.arange(n) * 900}


class ExcitationTests(unittest.TestCase):
    def policy(self):
        config = {**CONFIG, "episode": CONFIG["episodes"][0]}
        return SafeExcitation(config, LIMITS, {"oveTSet_u": 294.35})

    def current(self):
        return pd.Series({"LowerSetp[1]": 294.15, "UpperSetp[1]": 297.15})

    def test_plan_is_strictly_temporal_including_warmup(self):
        validate_plan(CONFIG)
        for change in ["overlap", "duplicate", "role"]:
            config = copy.deepcopy(CONFIG)
            if change == "overlap":
                config["episodes"][1]["start_day"] = 20
            elif change == "duplicate":
                config["episodes"][1]["id"] = config["episodes"][0]["id"]
            else:
                config["episodes"][0]["split"] = "test"
            with self.assertRaises(ValueError):
                validate_plan(config)

    def test_temperature_guards_and_slew(self):
        for temperature, mode, direction in [(20.9, "recover_heat", 1), (24.1, "coast", -1)]:
            policy = self.policy()
            action, info = policy.action(0, {"reaTZon_y": temperature + 273.15}, self.current())
            self.assertEqual(info["mode"], mode)
            self.assertAlmostEqual(info["requested_C"] - info["previous_C"], direction * .5)
            self.assertEqual(action["oveTSet_activate"], 1)
            self.assertAlmostEqual(action["oveTSet_u"] - 273.15, info["requested_C"])

    def test_watchdog_rejects_nonfinite_and_outside_stop_band(self):
        for value in [17.9, 28.1, float("nan")]:
            with self.assertRaises(ValueError):
                self.policy().check_temperature(value)

    def test_seeded_levels_reproduce_and_supervision_is_bounded(self):
        a, b = self.policy(), self.policy()
        np.testing.assert_array_equal(a.levels, b.levels)
        actions = []
        for i in range(32):
            temperature = 24.5 if i < 16 else 20.5
            action, info = a.action(i, {"reaTZon_y": temperature + 273.15}, self.current())
            self.assertTrue(18 <= action["oveTSet_u"] - 273.15 <= 24)
            self.assertLessEqual(abs(info["requested_C"] - info["previous_C"]), .500000001)
            actions.append(action["oveTSet_u"])
        self.assertGreater(max(actions), min(actions))


class DynamicsTests(unittest.TestCase):
    def test_incomplete_collection_and_changed_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps({"config": CONFIG}))
            (root / "summary.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_episodes(root, CONFIG)
            summaries = [{"scenario": p["id"], "config": {"episode": p},
                          "sha256": {"transitions.csv.gz": "0" * 64}} for p in CONFIG["episodes"]]
            directory = root / (CONFIG["episodes"][0]["id"] + "__excitation")
            directory.mkdir()
            (directory / "transitions.csv.gz").write_bytes(b"changed source")
            (root / "summary.json").write_text(json.dumps(summaries))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_episodes(root, CONFIG)

    def test_training_windows_and_targets_align_without_crossing_episode(self):
        episode = synthetic_episode()
        episode["states"][0, 3] = np.nan
        x, y = training_data([episode], 4)
        self.assertEqual(len(x), 36)
        np.testing.assert_array_equal(x[0, :16], episode["states"][1:5].reshape(-1))
        np.testing.assert_array_equal(x[0, 16:20], episode["actions"][1:5])
        np.testing.assert_allclose(y[0, :3], episode["states"][5, :3] - episode["states"][4, :3])
        self.assertEqual(y[0, 3], episode["states"][5, 3])
        self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(ValueError):
            training_data([synthetic_episode("test")], 4)

    def test_scaler_fit_is_training_only_and_fit_repeats(self):
        train = [synthetic_episode()]
        config = CONFIG["model"]
        model = fit_model(train, "ridge", 4, .1, config)
        x, _ = training_data(train, 4)
        np.testing.assert_allclose(model["estimator"][0].mean_, x.mean(axis=0))
        repeat = fit_model(train, "ridge", 4, .1, config)
        np.testing.assert_array_equal(model["estimator"].predict(x), repeat["estimator"].predict(x))
        self.assertTrue(np.isfinite(spectral_radius(model)))

    def test_recursive_rollout_feeds_predictions_and_preserves_inputs(self):
        class Increment:
            def predict(self, x):
                return np.tile([1., 0., 0., .25], (len(x), 1))

        history = np.array([[[20., 25., 30., .1], [21., 25., 30., .1]]])
        unchanged = history.copy()
        result = rollout({"kind": "ridge", "lag": 2, "estimator": Increment()},
                         history, np.array([[21.]]), np.full((1, 3), 22.), np.zeros((1, 3, 5)))
        np.testing.assert_array_equal(result[0, :, 0], [22, 23, 24])
        np.testing.assert_array_equal(result[0, :, 3], [.25, .25, .25])
        np.testing.assert_array_equal(history, unchanged)

    def test_future_measured_states_cannot_change_predictions(self):
        episode = synthetic_episode()
        model = fit_model([episode], "ridge", 4, .1, CONFIG["model"])
        prediction, truth = rollout_episode(model, episode, np.array([8]), 8)
        changed = copy.deepcopy(episode)
        changed["states"][9:] += 1000
        altered_prediction, altered_truth = rollout_episode(model, changed, np.array([8]), 8)
        np.testing.assert_array_equal(prediction, altered_prediction)
        self.assertFalse(np.array_equal(truth, altered_truth))

    def test_weather_hold_changes_only_future_weather_inputs(self):
        class WeatherDelta:
            def predict(self, x):
                result = np.zeros((len(x), 4))
                result[:, 0] = x[:, -5]
                return result

        episode = synthetic_episode()
        model = {"kind": "ridge", "lag": 1, "estimator": WeatherDelta()}
        perfect, truth = rollout_episode(model, episode, np.array([8]), 3)
        hold, hold_truth = rollout_episode(model, episode, np.array([8]), 3, "hold")
        self.assertGreater(perfect[0, -1, 0], hold[0, -1, 0])
        np.testing.assert_array_equal(truth, hold_truth)

    def test_identical_origins_and_full_horizon_stay_inside_episode(self):
        episode = synthetic_episode(n=160)
        origins = common_origins(episode, CONFIG["model"])
        self.assertGreaterEqual(origins.min(), max(CONFIG["model"]["lags"]))
        self.assertLessEqual(origins.max() + 96, len(episode["actions"]))
        self.assertTrue((np.diff(origins) == 4).all())

    def test_energy_is_accumulated_and_negative_predictions_remain_visible(self):
        truth = np.zeros((2, 4, 4))
        prediction = truth.copy()
        prediction[:, :, 0] = 2
        prediction[:, :, 3] = -.5
        score = score_rollouts(prediction, truth, [4])[0]
        self.assertEqual(score["temperature_RMSE_C"], 2)
        self.assertEqual(score["cumulative_energy_RMSE_kWh"], 2)
        self.assertEqual(score["negative_energy_fraction"], 1)

    def test_persistence_does_not_respond_to_future_inputs(self):
        episode = synthetic_episode()
        model = {"kind": "persistence", "lag": 1}
        prediction, _ = rollout_episode(model, episode, np.array([8]), 4)
        np.testing.assert_array_equal(prediction[0], np.tile(episode["states"][8], (4, 1)))

    def test_out_of_distribution_coverage_uses_train_ranges(self):
        train, test = synthetic_episode(), synthetic_episode("test")
        test["states"][:, 0] += 10
        frame = coverage_table({"train": train, "test": test})
        row = frame[(frame.split == "test") & (frame.variable == "operative_C")].iloc[0]
        self.assertEqual(row.fraction_outside_train_range, 1)
        self.assertLess(row.train_max, row.observed_min)


if __name__ == "__main__":
    unittest.main()
