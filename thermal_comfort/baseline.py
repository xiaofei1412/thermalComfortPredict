"""Fixed prediction baselines, held-out evaluation, and cleaning diagnostics."""

import importlib.metadata
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, PHYSICAL_FEATURES, ROOT, sha256, write_json
from .models import fit_neural, load_neural, predict_neural, save_neural


def regression_metrics(observed, predicted, groups):
    errors = pd.DataFrame({
        "absolute": np.abs(np.asarray(observed) - predicted),
        "squared": (np.asarray(observed) - predicted) ** 2,
        "group": np.asarray(groups),
    })
    grouped = errors.groupby("group")[["absolute", "squared"]].mean()
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "source_macro_mae": float(grouped.absolute.mean()),
        "source_macro_rmse": float(np.sqrt(grouped.squared).mean()),
    }


def classification_metrics(observed, probabilities):
    observed = np.asarray(observed, dtype=int)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == observed
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    one_hot = np.eye(3)[observed]
    return {
        "accuracy": float(accuracy_score(observed, predicted)),
        "macro_f1": float(f1_score(observed, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "log_loss": float(log_loss(observed, probabilities, labels=[0, 1, 2])),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": ece,
    }


def physical_predictions(data):
    from pythermalcomfort.clothing import clo_dynamic_iso
    from pythermalcomfort.environment import v_relative
    from pythermalcomfort.models import pmv_ppd_iso

    support = (
        data[PHYSICAL_FEATURES].notna().all(axis=1)
        & data.air_temperature_c.between(10, 30)
        & data.radiant_temperature_c.between(10, 40)
        & data.air_speed_m_s.between(0, 1)
        & data.metabolic_rate_met.between(0.8, 4)
        & data.clothing_clo.between(0, 2, inclusive="right")
    )
    subset = data.loc[support]
    speed = v_relative(v=subset.air_speed_m_s.to_numpy(), met=subset.metabolic_rate_met.to_numpy())
    clothing = clo_dynamic_iso(
        clo=subset.clothing_clo.to_numpy(), met=subset.metabolic_rate_met.to_numpy(),
        v=subset.air_speed_m_s.to_numpy(),
    )
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", UserWarning)
        result = pmv_ppd_iso(
            tdb=subset.air_temperature_c.to_numpy(), tr=subset.radiant_temperature_c.to_numpy(),
            vr=speed, rh=subset.relative_humidity_pct.to_numpy(),
            met=subset.metabolic_rate_met.to_numpy(), clo=clothing,
            model="7730-2005", limit_inputs=True, round_output=False,
        )
    predictions = pd.Series(np.nan, index=data.index, dtype=float)
    predictions.loc[subset.index] = result.pmv
    predictions.attrs["applicability_notices"] = [
        str(notice.message).split(" value", 1)[0] for notice in notices
    ]
    return predictions


def environment_snapshot():
    import torch
    packages = [
        "numpy", "pandas", "scikit-learn", "torch", "pythermalcomfort",
        "scipy", "numba", "matplotlib", "joblib",
    ]
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted((ROOT / "thermal_comfort").glob("*.py"))
        },
    }


def run(data, audit, config, output):
    output = Path(output)
    for directory in ["models", "predictions", "histories"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment_snapshot())
    records, group_records, training = [], [], []
    evaluations = ["validation", "test", "external_climate"]

    def evaluate(task, name, seed, subset, predicted, *, features=4,
                 cohort="all", protocol="source_group", probabilities=None, split=None):
        if subset.empty:
            return
        split = split or subset["split"].iloc[0]
        target = "thermal_sensation" if task == "sensation" else "preference_class"
        metric_function = regression_metrics if task == "sensation" else classification_metrics
        scores = (
            metric_function(subset[target], predicted, subset.source_group)
            if task == "sensation" else metric_function(subset[target], probabilities)
        )
        identity = {
            "task": task, "model": name, "seed": seed, "protocol": protocol,
            "cohort": cohort, "split": split, "features": features,
        }
        records.append({**identity, "rows": len(subset), "sources": subset.source_group.nunique(), **scores})
        predictions = subset[["row_id", "source_group", target]].rename(columns={target: "observed"}).copy()
        predictions["predicted"] = predicted
        if probabilities is not None:
            for index, label in enumerate(["cooler", "no_change", "warmer"]):
                predictions["probability_" + label] = probabilities[:, index]
        filename = f"{task}_{name}_{seed}_{protocol}_{cohort}_{split}.csv"
        predictions.to_csv(output / "predictions" / filename, index=False, lineterminator="\n")
        for source, positions in subset.groupby("source_group").indices.items():
            group = subset.iloc[positions]
            group_scores = (
                regression_metrics(group[target], predicted[positions], group.source_group)
                if task == "sensation" else classification_metrics(group[target], probabilities[positions])
            )
            group_records.append({**identity, "source_group": source, "rows": len(group), **group_scores})

    sensation = data.loc[data.sensation_eligible].copy()
    train = sensation.loc[sensation.split.eq("train")]
    validation = sensation.loc[sensation.split.eq("validation")]
    reference_forest = None
    for name in ["mean", "linear", "random_forest", "legacy_attention_mlp"]:
        seeds = config["seeds"] if name in ["random_forest", "legacy_attention_mlp"] else config["seeds"][:1]
        for seed in seeds:
            print(f"Training sensation/{name}, seed={seed}, rows={len(train)}", flush=True)
            if name == "legacy_attention_mlp":
                fitted = fit_neural(
                    train[FEATURES].to_numpy(), train.thermal_sensation,
                    validation[FEATURES].to_numpy(), validation.thermal_sensation,
                    seed, config["neural"],
                )
                model_path = output / "models" / f"sensation_{name}_{seed}.pt"
                save_neural(fitted, model_path, FEATURES)
                restored = load_neural(model_path)
                np.testing.assert_allclose(
                    predict_neural(fitted, validation[FEATURES].to_numpy()),
                    predict_neural(restored, validation[FEATURES].to_numpy()), rtol=0, atol=0,
                )
                write_json(output / "histories" / f"{name}_{seed}.json", fitted["history"])
                training.append({
                    "task": "sensation", "model": name, "seed": seed,
                    "train_rows": len(train), "validation_rows": len(validation),
                    "best_epoch": fitted["best_epoch"], "epochs_run": len(fitted["history"]),
                    "device": fitted["device"], "checkpoint_reload_exact": True,
                })
                predict = lambda values: predict_neural(fitted, values.to_numpy())
            else:
                estimator = {
                    "mean": DummyRegressor(strategy="mean"),
                    "linear": LinearRegression(),
                    "random_forest": RandomForestRegressor(random_state=seed, **config["forest"]),
                }[name]
                fitted = make_pipeline(StandardScaler(), estimator)
                fitted.fit(train[FEATURES], train.thermal_sensation)
                joblib.dump(fitted, output / "models" / f"sensation_{name}_{seed}.joblib", compress=3)
                predict = fitted.predict
                if name == "random_forest" and seed == config["seeds"][0]:
                    reference_forest = fitted
            for split in evaluations:
                subset = sensation.loc[sensation.split.eq(split)]
                predicted = predict(subset[FEATURES])
                evaluate("sensation", name, seed, subset, predicted)

    preference = data.loc[data.preference_eligible].copy()
    preference_train = preference.loc[preference.split.eq("train")]
    for name in ["prior", "logistic", "random_forest"]:
        seeds = config["seeds"] if name == "random_forest" else config["seeds"][:1]
        for seed in seeds:
            print(f"Training preference/{name}, seed={seed}, rows={len(preference_train)}", flush=True)
            estimator = {
                "prior": DummyClassifier(strategy="prior"),
                "logistic": LogisticRegression(max_iter=1000, random_state=seed),
                "random_forest": RandomForestClassifier(random_state=seed, **config["forest"]),
            }[name]
            fitted = make_pipeline(StandardScaler(), estimator)
            fitted.fit(preference_train[FEATURES], preference_train.preference_class.astype(int))
            if not np.array_equal(fitted.classes_, [0, 1, 2]):
                raise ValueError("Training preference classes must include cooler/no-change/warmer.")
            joblib.dump(fitted, output / "models" / f"preference_{name}_{seed}.joblib", compress=3)
            for split in evaluations:
                subset = preference.loc[preference.split.eq(split)]
                probabilities = fitted.predict_proba(subset[FEATURES])
                evaluate("preference", name, seed, subset, probabilities.argmax(axis=1), probabilities=probabilities)

    print("Evaluating PMV and matched six-input baselines", flush=True)
    pmv = physical_predictions(data)
    physical = sensation.loc[pmv.loc[sensation.index].notna()].copy()
    physical_train = physical.loc[physical.split.eq("train")]
    coverage = {
        split: {"eligible_rows": int(sensation.split.eq(split).sum()),
                "pmv_rows": int(physical.split.eq(split).sum())}
        for split in ["train", *evaluations]
    }
    seed = config["seeds"][0]
    for name, estimator in {
        "mean_6input_cohort": DummyRegressor(strategy="mean"),
        "linear_6input": LinearRegression(),
        "random_forest_6input": RandomForestRegressor(random_state=seed, **config["forest"]),
    }.items():
        fitted = make_pipeline(StandardScaler(), estimator)
        fitted.fit(physical_train[PHYSICAL_FEATURES], physical_train.thermal_sensation)
        joblib.dump(fitted, output / "models" / f"sensation_{name}_{seed}.joblib", compress=3)
        for split in evaluations:
            subset = physical.loc[physical.split.eq(split)]
            evaluate("sensation", name, seed, subset, fitted.predict(subset[PHYSICAL_FEATURES]),
                     features=6, cohort="pmv_applicable")
    for split in evaluations:
        subset = physical.loc[physical.split.eq(split)]
        evaluate("sensation", "pmv_iso_7730_2005", seed, subset, pmv.loc[subset.index].to_numpy(),
                 features=6, cohort="pmv_applicable")
    write_json(output / "pmv_coverage.json", coverage)
    write_json(output / "pmv_applicability.json", pmv.attrs)

    print("Evaluating cleaning ablation on unchanged held-out rows", flush=True)
    retained_train = train.loc[train.legacy_retained]
    cleaned_model = make_pipeline(
        StandardScaler(), RandomForestRegressor(random_state=seed, **config["forest"])
    ).fit(retained_train[FEATURES], retained_train.thermal_sensation)
    joblib.dump(cleaned_model, output / "models" / "sensation_legacy_filtered_train.joblib", compress=3)
    for name, fitted in [("unfiltered_train", reference_forest), ("legacy_filtered_train", cleaned_model)]:
        for split in ["test", "external_climate"]:
            for cohort in ["all", "legacy_retained_test"]:
                subset = sensation.loc[sensation.split.eq(split)]
                if cohort == "legacy_retained_test":
                    subset = subset.loc[subset.legacy_retained]
                evaluate("sensation", name, seed, subset, fitted.predict(subset[FEATURES]),
                         cohort=cohort, protocol="cleaning_ablation")

    print("Evaluating historical-style random-row split (diagnostic only)", flush=True)
    historical = sensation.loc[sensation.legacy_retained]
    historical_train, temporary = train_test_split(historical, test_size=0.2, random_state=config["split_seed"])
    historical_validation, historical_test = train_test_split(temporary, test_size=0.5, random_state=config["split_seed"])
    diagnostic_model = make_pipeline(
        StandardScaler(), RandomForestRegressor(random_state=seed, **config["forest"])
    ).fit(historical_train[FEATURES], historical_train.thermal_sensation)
    evaluate("sensation", "random_forest", seed, historical_test,
             diagnostic_model.predict(historical_test[FEATURES]),
             protocol="row_random_diagnostic", split="test")
    overlap = set(historical_train.source_group) & set(historical_test.source_group)
    diagnostics = {
        "train_rows": len(historical_train), "validation_rows": len(historical_validation),
        "test_rows": len(historical_test), "overlapping_train_test_sources": len(overlap),
        "test_rows_with_training_source": int(historical_test.source_group.isin(overlap).sum()),
        "warning": "Different cohort and training budget; this is not a causal estimate of the effect of grouping.",
        "cleaning_ablation_train_rows": {"unfiltered": len(train), "filtered": len(retained_train)},
    }
    write_json(output / "diagnostics.json", diagnostics)
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(group_records).to_csv(output / "source_metrics.csv", index=False, lineterminator="\n")
    write_json(output / "training.json", training)
    create_report(metrics, audit, config, coverage, diagnostics, output)
    print(f"Finished. Results: {output.resolve()}", flush=True)


def create_report(metrics, audit, config, coverage, diagnostics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = metrics.loc[metrics.protocol.eq("source_group") & metrics.cohort.eq("all")]
    table = [
        "# 数据与预测基线结果", "",
        "这是监督预测结果，不代表楼宇控制或真实节能验证。", "",
        f"原始数据：{audit['rows']:,} 行；SHA-256：`{audit['raw_sha256']}`。",
        f"切分清单 SHA-256：`{audit['split_manifest_sha256']}`。", "",
        "## 固定评测协议", "",
        "按规范化来源文献分组；缺失来源合并为一个组。该分组不能证明具体建筑或人员已完全隔离。",
        f"预先留出 `{config['external_climate']}` 气候及其所有来源；这些来源的其他气候记录不参与训练。",
        "其余来源以固定种子按组划分训练、验证、测试；实际行数比例不必等于组数比例。",
        "原始冷热感觉保留 −3～3 的含义及小数标签，不按温度与标签关系过滤。",
        "主评测沿用上游维护脚本的标签无关室温质控（10 < ta < 40°C），并要求湿度在 0～100% 内；旧流程复现不应用新质控。",
        "四输入沿用室温、湿度、室外温度与旧季节编码。缩放器只拟合训练数据。", "",
        "| 切分 | 总行数 | 来源组 | 热感觉样本 | 冷热偏好样本 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, counts in audit["splits"].items():
        table.append(f"| {split} | {counts['rows']} | {counts['sources']} | {counts['sensation_rows']} | {counts['preference_rows']} |")
    for task, columns in [
        ("sensation", ["mae", "rmse", "source_macro_rmse"]),
        ("preference", ["macro_f1", "log_loss", "brier_multiclass", "ece_10_bins"]),
    ]:
        table.extend(["", f"## {task} 主评测", "",
                      f"随机森林和旧网络运行固定的 {len(config['seeds'])} 个训练种子；± 表示训练种子间样本标准差，不是跨建筑置信区间。",
                      "", "| 切分 | 模型 | 种子数 | " + " | ".join(columns) + " |",
                      "| --- | --- | ---: | " + " | ".join(["---:"] * len(columns)) + " |"])
        for (split, model), rows in main.loc[main.task.eq(task)].groupby(["split", "model"], sort=True):
            values = []
            for column in columns:
                value = f"{rows[column].mean():.4f}"
                if len(rows) > 1:
                    value += f" ± {rows[column].std():.4f}"
                values.append(value)
            table.append(f"| {split} | {model} | {len(rows)} | " + " | ".join(values) + " |")
    table.extend(["", "## PMV 的共同测试子集", "",
                  "使用 pythermalcomfort 的 ISO 7730:2005 PMV、相对风速及动态衣着修正；外部功为 0，保留默认适用范围限制。",
                  "六输入学习模型在同一适用子集的训练来源上训练，测试行与 PMV 完全一致。此小表不能与四输入全样本表直接排名。", "",
                  "| 切分 | 全部热感觉样本 | PMV 适用样本 |", "| --- | ---: | ---: |"])
    for split, counts in coverage.items():
        table.append(f"| {split} | {counts['eligible_rows']} | {counts['pmv_rows']} |")
    table.extend(["", "| 切分 | 模型 | MAE | RMSE |", "| --- | --- | ---: | ---: |"])
    for row in metrics.loc[metrics.cohort.eq("pmv_applicable")].itertuples():
        table.append(f"| {row.split} | {row.model} | {row.mae:.4f} | {row.rmse:.4f} |")
    table.extend(["", "## 旧清洗规则消融", "",
                  "固定分组，分别改变训练集过滤与测试集过滤。仅测试集过滤会改变评测人群，不能把更低误差视为模型改进。",
                  f"训练样本：未过滤 {diagnostics['cleaning_ablation_train_rows']['unfiltered']}，过滤后 {diagnostics['cleaning_ablation_train_rows']['filtered']}。", "",
                  "| 切分 | 训练处理 | 测试人群 | 测试行数 | MAE | RMSE |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in metrics.loc[metrics.protocol.eq("cleaning_ablation")].itertuples():
        table.append(f"| {row.split} | {row.model} | {row.cohort} | {row.rows} | {row.mae:.4f} | {row.rmse:.4f} |")
    diagnostic = metrics.loc[metrics.protocol.eq("row_random_diagnostic")].iloc[0]
    table.extend(["", "## 历史风格随机按行切分诊断", "",
                  f"旧清洗人群使用 80/10/10 按行切分；随机森林测试 RMSE={diagnostic.rmse:.4f}。",
                  f"训练与测试重合 {diagnostics['overlapping_train_test_sources']} 个来源，测试中 {diagnostics['test_rows_with_training_source']}/{diagnostics['test_rows']} 行的来源在训练中出现。",
                  "该诊断改变了人群、训练样本量及切分，不用其差值估计某一个因素的因果影响；也不声称复现了缺失随机种子的历史网络成绩。", "",
                  "## 工件与限制", "",
                  "- `metrics.csv` 为逐模型、种子与切分指标，`source_metrics.csv` 为逐来源指标。",
                  "- `data/audit.json`、`data/splits.csv`、`config.json` 和 `environment.json` 保存数据、切分、配置及环境证据。",
                  "- 模型、预测与学习曲线保存在本次输出目录；大型工件不提交 Git。",
                  "- 网络保留旧四维门控和 128×128 隐层结构；新增固定种子、验证集早停、最佳权重恢复及保存后重载核验。",
                  "- 冷热偏好 Brier 为多类别平方误差之和的样本均值；ECE 为 10 个等宽置信度桶。当前仅测量概率校准质量，未做额外后处理校准。",
                  "- 原始快照未与上游发行版逐行匹配；当前来源元数据不等于对本地副本的完整历史溯源。",
                  "- 少量留出来源、非均衡人群和缺失输入限制结论；此阶段不报告控制收益。", ""])
    (output / "REPORT.md").write_text("\n".join(table), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, split in zip(axes, ["test", "external_climate"]):
        subset = main.loc[main.task.eq("sensation") & main.split.eq(split)]
        aggregated = subset.groupby("model").rmse.agg(["mean", "std"]).fillna(0)
        axis.bar(aggregated.index, aggregated["mean"], yerr=aggregated["std"], capsize=4)
        axis.set(title=split, ylabel="Thermal sensation RMSE")
        axis.tick_params(axis="x", rotation=25)
    figure.savefig(output / "sensation_rmse.png", dpi=180)
    plt.close(figure)
