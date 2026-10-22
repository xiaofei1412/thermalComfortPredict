"""Load the frozen legacy snapshot without trusting its damaged unit headers."""

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "air_temperature_c", "relative_humidity_pct", "outdoor_temperature_c", "season_code"
]
PHYSICAL_FEATURES = [
    "air_temperature_c", "radiant_temperature_c", "relative_humidity_pct",
    "air_speed_m_s", "metabolic_rate_met", "clothing_clo",
]
SEASONS = {"Winter": 1, "Autumn": 2, "Spring": 3, "Summer": 4}
PREFERENCES = {"cooler": 0, "no change": 1, "warmer": 2}
TEXT_FIELDS = {
    "publication", "season", "climate", "country", "building_type",
    "cooling_strategy", "thermal_preference",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_snapshot(manifest):
    path = ROOT / manifest["path"]
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Raw data checksum differs from the frozen manifest.")
    with path.open(encoding=manifest["encoding"], newline="") as stream:
        header = next(csv.reader(stream))
    if len(header) != manifest["columns"]:
        raise ValueError("Unexpected raw CSV width.")
    for field in manifest["schema"]:
        if header[field["index"]] != field["header"]:
            raise ValueError(f"Unexpected header at position {field['index']}.")
    frame = pd.read_csv(
        path, header=None, skiprows=1, encoding=manifest["encoding"],
        usecols=[field["index"] for field in manifest["schema"]],
        dtype=str, na_values=["NA", "NaN", "nan", "N/A", "null", "None", " "],
    )
    frame.columns = [field["name"] for field in manifest["schema"]]
    if len(frame) != manifest["rows"]:
        raise ValueError("Unexpected raw CSV row count.")
    coercions = {}
    for column in frame.columns:
        if column in TEXT_FIELDS:
            frame[column] = frame[column].str.strip().replace("", np.nan)
        else:
            before = int(frame[column].notna().sum())
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            coercions[column] = before - int(frame[column].notna().sum())
    frame.insert(0, "row_id", np.arange(len(frame)))
    frame["season_code"] = frame["season"].map(SEASONS)
    frame["preference_class"] = frame["thermal_preference"].map(PREFERENCES)
    citations = frame["publication"].fillna("__unknown_source__").map(
        lambda text: " ".join(unicodedata.normalize("NFKC", text).split())
    )
    frame["source_group"] = citations.map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    )
    frame.attrs["numeric_parse_failures"] = coercions
    return frame


def legacy_masks(frame):
    """Reproduce the original rules, including their label shift and rounding."""
    complete = frame[FEATURES + ["thermal_sensation"]].notna().all(axis=1)
    target = frame["thermal_sensation"]
    target = (target - target.min()) / (target.max() - target.min()) * 6 + 1
    temperature = frame["air_temperature_c"]
    excluded = (
        ((temperature > 29) & (target < 5))
        | ((temperature < 18) & (target > 4))
        | ((temperature > 24) & (target < 3))
        | ((temperature < 25) & (target > 5))
    )
    after_rules = complete & ~excluded
    integer = target.notna() & (target == np.floor(target))
    return complete, after_rules, after_rules & integer, target


def reproduce_legacy(frame, stored_path):
    complete, after_rules, retained, shifted = legacy_masks(frame)
    reproduced = frame.loc[retained, [
        "season_code", "thermal_sensation", "air_temperature_c",
        "relative_humidity_pct", "outdoor_temperature_c",
    ]].copy()
    reproduced["thermal_sensation"] = shifted.loc[retained]
    reproduced.columns = [
        "Season", "Thermal comfort", "Air temperature", "Relative humidity",
        "Outdoor air temperature",
    ]
    stored = pd.read_csv(stored_path)
    pd.testing.assert_frame_equal(
        reproduced.reset_index(drop=True), stored, check_dtype=False,
        check_exact=False, atol=1e-12, rtol=0,
    )
    return {
        "complete_rows": int(complete.sum()),
        "removed_by_temperature_target_rules": int((complete & ~after_rules).sum()),
        "removed_non_integer_targets": int((after_rules & ~retained).sum()),
        "retained_rows": int(retained.sum()),
        "matches_stored_csv": True,
        "stored_csv_sha256": sha256(stored_path),
    }


def make_splits(frame, config):
    """Reserve the climate and all its sources before splitting other studies."""
    complete = frame[FEATURES].notna().all(axis=1)
    data = frame.loc[complete].copy()
    # Use the upstream maintenance script's label-independent air-temperature QC.
    quality = (
        np.isfinite(data[FEATURES]).all(axis=1)
        & data["relative_humidity_pct"].between(0, 100)
        & data["air_temperature_c"].between(10, 40, inclusive="neither")
    )
    eligible = data.loc[quality]
    external = eligible["climate"].eq(config["external_climate"])
    external_sources = set(eligible.loc[external, "source_group"])
    data["split"] = "excluded_feature_quality"
    data.loc[eligible.index, "split"] = "excluded_external_source"
    data.loc[eligible.loc[external].index, "split"] = "external_climate"
    development = eligible.loc[~eligible["source_group"].isin(external_sources)]
    outer = GroupShuffleSplit(
        n_splits=1, test_size=config["test_group_fraction"],
        random_state=config["split_seed"],
    )
    train_val_idx, test_idx = next(outer.split(development, groups=development.source_group))
    train_val = development.iloc[train_val_idx]
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=config["validation_group_fraction"] / (1 - config["test_group_fraction"]),
        random_state=config["split_seed"],
    )
    train_idx, validation_idx = next(inner.split(train_val, groups=train_val.source_group))
    data.loc[train_val.iloc[train_idx].index, "split"] = "train"
    data.loc[train_val.iloc[validation_idx].index, "split"] = "validation"
    data.loc[development.iloc[test_idx].index, "split"] = "test"
    groups = {
        split: set(data.loc[data.split.eq(split), "source_group"])
        for split in ["train", "validation", "test", "external_climate"]
    }
    for first, first_groups in groups.items():
        for second, second_groups in groups.items():
            if first != second and first_groups & second_groups:
                raise AssertionError(f"Source overlap: {first}, {second}")
    return data


def prepare(config, output):
    manifest = json.loads((ROOT / config["manifest"]).read_text(encoding="utf-8"))
    raw = load_snapshot(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "raw_sha256": sha256(ROOT / manifest["path"]),
        "rows": len(raw),
        "numeric_parse_failures": raw.attrs["numeric_parse_failures"],
        "missing_publication_rows": int(raw.publication.isna().sum()),
        "non_missing_targets": {
            name: int(raw[name].notna().sum())
            for name in ["thermal_sensation", "thermal_preference", "reported_comfort", "acceptability_code"]
        },
        "legacy_reproduction": reproduce_legacy(raw, ROOT / "data/processed/cleaned_asds4.csv"),
        "temperature_pairs": {},
    }
    for name in ["air_temperature", "radiant_temperature"]:
        pairs = raw[[name + "_c", name + "_f"]].dropna()
        error = (pairs[name + "_c"] * 1.8 + 32 - pairs[name + "_f"]).abs()
        if error.max() > 0.15:
            raise ValueError(f"Unit-pair check failed: {name}")
        audit["temperature_pairs"][name] = {
            "pairs": len(pairs), "max_fahrenheit_error": float(error.max()),
            "median_fahrenheit_error": float(error.median()),
        }
    data = make_splits(raw, config)
    data["legacy_retained"] = legacy_masks(raw)[2].loc[data.index]
    data["sensation_eligible"] = data.thermal_sensation.between(-3, 3)
    data["preference_eligible"] = data.preference_class.notna()
    audit["feature_complete_rows"] = len(data)
    audit["excluded_for_features"] = len(raw) - len(data)
    audit["feature_quality_exclusions"] = int(data.split.eq("excluded_feature_quality").sum())
    audit["splits"] = {}
    for split, subset in data.groupby("split", sort=True):
        audit["splits"][split] = {
            "rows": len(subset), "sources": subset.source_group.nunique(),
            "sensation_rows": int(subset.sensation_eligible.sum()),
            "preference_rows": int(subset.preference_eligible.sum()),
        }
    exported = data.drop(columns="publication")
    exported.to_csv(output / "prepared.csv", index=False, lineterminator="\n")
    exported[["row_id", "source_group", "split"]].to_csv(
        output / "splits.csv", index=False, lineterminator="\n"
    )
    audit["split_manifest_sha256"] = sha256(output / "splits.csv")
    audit["prepared_sha256"] = sha256(output / "prepared.csv")
    write_json(output / "audit.json", audit)
    return data, audit
