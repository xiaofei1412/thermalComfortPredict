import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_comfort.baseline import classification_metrics, physical_predictions, regression_metrics
from thermal_comfort.data import (
    FEATURES, ROOT, legacy_masks, load_snapshot, make_splits, reproduce_legacy,
)
from thermal_comfort.models import fit_neural, load_neural, predict_neural, save_neural


@unittest.skipUnless((ROOT / 'data/raw/ashrae_db2.csv').is_file()
                     and (ROOT / 'data/processed/cleaned_asds4.csv').is_file(),
                     'Private historical CSV omitted from the public release; restore the owner archive to run data checks')
class DataIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        cls.config = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
        cls.raw = load_snapshot(cls.manifest)
        cls.data = make_splits(cls.raw, cls.config)

    def test_frozen_data_shape_and_hash_guard(self):
        self.assertEqual(len(self.raw), 107584)
        incorrect = {**self.manifest, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_snapshot(incorrect)

    def test_changed_header_mapping_is_rejected(self):
        incorrect = copy.deepcopy(self.manifest)
        incorrect["schema"][0]["header"] = "Unexpected publication header"
        with self.assertRaisesRegex(ValueError, "header"):
            load_snapshot(incorrect)

    def test_legacy_cleaning_matches_the_actual_saved_csv(self):
        audit = reproduce_legacy(self.raw, ROOT / "data/processed/cleaned_asds4.csv")
        self.assertEqual(audit["complete_rows"], 71177)
        self.assertEqual(audit["removed_by_temperature_target_rules"], 8763)
        self.assertEqual(audit["removed_non_integer_targets"], 1715)
        self.assertEqual(audit["retained_rows"], 60699)
        self.assertTrue(audit["matches_stored_csv"])

    def test_targets_keep_their_meaning_and_fractional_votes(self):
        self.assertEqual(self.raw.thermal_sensation.min(), -3)
        self.assertEqual(self.raw.thermal_sensation.max(), 3)
        _, _, _, shifted = legacy_masks(self.raw)
        self.assertTrue(shifted.loc[self.raw.thermal_sensation.eq(0)].eq(4).all())
        valid = self.data.thermal_sensation.dropna()
        self.assertGreater(int((valid != np.floor(valid)).sum()), 0)
        self.assertFalse(self.raw.thermal_sensation.equals(self.raw.reported_comfort))

    def test_temperature_column_units_from_paired_values(self):
        for name in ["air_temperature", "radiant_temperature"]:
            pairs = self.raw[[name + "_c", name + "_f"]].dropna()
            self.assertGreater(len(pairs), 30000)
            error = (pairs[name + "_c"] * 1.8 + 32 - pairs[name + "_f"]).abs()
            self.assertLessEqual(error.max(), 0.15)

    def test_sources_and_reserved_climate_never_cross_splits(self):
        names = ["train", "validation", "test", "external_climate"]
        groups = {name: set(self.data.loc[self.data.split.eq(name), "source_group"]) for name in names}
        for index, first in enumerate(names):
            self.assertGreater(len(groups[first]), 0)
            for second in names[index + 1:]:
                self.assertFalse(groups[first] & groups[second])
        external = self.data.loc[self.data.split.eq("external_climate")]
        self.assertTrue(external.climate.eq("Cfa").all())
        development = self.data.loc[self.data.split.isin(["train", "validation", "test"])]
        self.assertFalse(development.climate.eq("Cfa").any())

    def test_primary_quality_control_is_label_independent(self):
        primary = self.data.loc[self.data.split.isin(["train", "validation", "test", "external_climate"])]
        self.assertTrue(primary.air_temperature_c.between(10, 40, inclusive="neither").all())
        self.assertTrue(primary.relative_humidity_pct.between(0, 100).all())
        self.assertGreater(int(self.data.split.eq("excluded_feature_quality").sum()), 0)

    def test_missing_publications_share_one_group(self):
        missing = self.raw.loc[self.raw.publication.isna()]
        self.assertEqual(len(missing), 1655)
        self.assertEqual(missing.source_group.nunique(), 1)

    def test_split_reproduction_and_label_independence(self):
        reproduced = make_splits(self.raw, self.config)
        pd.testing.assert_frame_equal(
            self.data[["row_id", "source_group", "split"]],
            reproduced[["row_id", "source_group", "split"]],
        )
        changed_labels = self.raw.copy()
        changed_labels["thermal_sensation"] = 0.0
        changed_labels["preference_class"] = 1
        without_labels = make_splits(changed_labels, self.config)
        pd.testing.assert_series_equal(self.data.split, without_labels.split)


class MetricTests(unittest.TestCase):
    def test_group_macro_metric_does_not_weight_large_studies_as_heavily(self):
        scores = regression_metrics(np.array([0, 0, 0]), np.array([0, 0, 3]), ["a", "a", "b"])
        self.assertAlmostEqual(scores["rmse"], np.sqrt(3))
        self.assertAlmostEqual(scores["source_macro_rmse"], 1.5)

    def test_perfect_class_probabilities_have_zero_calibration_error(self):
        scores = classification_metrics([0, 1, 2], np.eye(3))
        self.assertEqual(scores["accuracy"], 1)
        self.assertEqual(scores["macro_f1"], 1)
        self.assertEqual(scores["brier_multiclass"], 0)
        self.assertEqual(scores["ece_10_bins"], 0)

    def test_pmv_respects_applicability_instead_of_filling_missing_inputs(self):
        frame = pd.DataFrame({
            "air_temperature_c": [25, 50, 25],
            "radiant_temperature_c": [25, 25, np.nan],
            "relative_humidity_pct": [50, 50, 50],
            "air_speed_m_s": [0.1, 0.1, 0.1],
            "metabolic_rate_met": [1.1, 1.1, 1.1],
            "clothing_clo": [0.5, 0.5, 0.5],
        })
        predicted = physical_predictions(frame)
        self.assertTrue(np.isfinite(predicted.iloc[0]))
        self.assertTrue(np.isnan(predicted.iloc[1]))
        self.assertTrue(np.isnan(predicted.iloc[2]))


class NeuralReproductionTests(unittest.TestCase):
    def test_training_is_repeatable_scaler_is_train_only_and_checkpoint_reloads(self):
        generator = np.random.default_rng(7)
        train = generator.normal(size=(48, 4))
        validation = generator.normal(size=(12, 4)) + 10
        config = {
            "hidden_size": 8, "epochs": 2, "batch_size": 16,
            "learning_rate": 0.0005, "patience": 2, "device": "cpu",
        }
        first = fit_neural(train, train[:, 0], validation, validation[:, 0], 33, config)
        second = fit_neural(train, train[:, 0], validation, validation[:, 0], 33, config)
        np.testing.assert_allclose(first["scaler"].mean_, train.mean(axis=0))
        np.testing.assert_array_equal(predict_neural(first, validation), predict_neural(second, validation))
        test_root = ROOT / "outputs/tests"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            checkpoint = Path(folder) / "model.pt"
            save_neural(first, checkpoint, FEATURES)
            restored = load_neural(checkpoint)
            self.assertEqual(restored["features"], FEATURES)
            np.testing.assert_array_equal(predict_neural(first, validation), predict_neural(restored, validation))


if __name__ == "__main__":
    unittest.main()
