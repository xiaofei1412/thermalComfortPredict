"""Rebuild the data and repeat seed 33 to verify a completed baseline run."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thermal_comfort.data import FEATURES, ROOT, prepare, write_json
from thermal_comfort.models import fit_neural, load_neural, predict_neural


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline")
    args = parser.parse_args()
    output = args.output
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    original = json.loads((output / "data/audit.json").read_text(encoding="utf-8"))
    data, rebuilt = prepare(config, output / "verification/data")
    for key in ["raw_sha256", "split_manifest_sha256", "prepared_sha256"]:
        if original[key] != rebuilt[key]:
            raise AssertionError(f"Data reproduction failed: {key}")
    sensation = data.loc[data.sensation_eligible]
    train = sensation.loc[sensation.split.eq("train")]
    validation = sensation.loc[sensation.split.eq("validation")]
    test = sensation.loc[sensation.split.eq("test")]
    seed = config["seeds"][0]
    print(f"Repeating random forest and neural training, seed={seed}", flush=True)
    forest = make_pipeline(
        StandardScaler(), RandomForestRegressor(random_state=seed, **config["forest"])
    ).fit(train[FEATURES], train.thermal_sensation)
    original_forest = joblib.load(output / "models" / f"sensation_random_forest_{seed}.joblib")
    tree_difference = float(np.max(np.abs(
        forest.predict(test[FEATURES]) - original_forest.predict(test[FEATURES])
    )))
    if tree_difference > 1e-12:
        raise AssertionError(f"Forest prediction difference: {tree_difference}")
    neural = fit_neural(
        train[FEATURES].to_numpy(), train.thermal_sensation,
        validation[FEATURES].to_numpy(), validation.thermal_sensation,
        seed, config["neural"],
    )
    original_neural = load_neural(output / "models" / f"sensation_legacy_attention_mlp_{seed}.pt")
    repeated_predictions = predict_neural(neural, test[FEATURES].to_numpy())
    original_predictions = predict_neural(original_neural, test[FEATURES].to_numpy())
    np.testing.assert_array_equal(repeated_predictions, original_predictions)
    original_history = json.loads(
        (output / "histories" / f"legacy_attention_mlp_{seed}.json").read_text(encoding="utf-8")
    )
    if neural["history"] != original_history:
        raise AssertionError("Neural learning curves differ from the original run.")
    result = {
        "seed": seed,
        "raw_data_rebuild_exact": True,
        "split_rebuild_exact": True,
        "prepared_data_rebuild_exact": True,
        "forest_max_prediction_difference": tree_difference,
        "forest_tolerance": 1e-12,
        "neural_history_exact": True,
        "neural_predictions_exact": True,
        "neural_best_epoch": neural["best_epoch"],
        "verification_device": neural["device"],
    }
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
