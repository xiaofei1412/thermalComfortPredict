# 数据与预测基线结果

这是监督预测结果，不代表楼宇控制或真实节能验证。

原始数据：107,584 行；SHA-256：`873a83fb9c0d9a6181c663b8af19ff35380153e70ef145e22fe17a32f2727b9a`。
切分清单 SHA-256：`de08bf0fdc2a8c645743d63afee7c115173d7ea466dadf6135d69178eaeb5ad6`。

## 固定评测协议

按规范化来源文献分组；缺失来源合并为一个组。该分组不能证明具体建筑或人员已完全隔离。
预先留出 `Cfa` 气候及其所有来源；这些来源的其他气候记录不参与训练。
其余来源以固定种子按组划分训练、验证、测试；实际行数比例不必等于组数比例。
原始冷热感觉保留 −3～3 的含义及小数标签，不按温度与标签关系过滤。
主评测沿用上游维护脚本的标签无关室温质控（10 < ta < 40°C），并要求湿度在 0～100% 内；旧流程复现不应用新质控。
四输入沿用室温、湿度、室外温度与旧季节编码。缩放器只拟合训练数据。

| 切分 | 总行数 | 来源组 | 热感觉样本 | 冷热偏好样本 |
| --- | ---: | ---: | ---: | ---: |
| excluded_external_source | 206 | 2 | 206 | 206 |
| excluded_feature_quality | 217 | 11 | 213 | 50 |
| external_climate | 12915 | 13 | 12874 | 5818 |
| test | 7460 | 6 | 7388 | 4228 |
| train | 44669 | 16 | 44583 | 42881 |
| validation | 5962 | 6 | 5913 | 2626 |

## sensation 主评测

随机森林和旧网络运行固定的 5 个训练种子；± 表示训练种子间样本标准差，不是跨建筑置信区间。

| 切分 | 模型 | 种子数 | mae | rmse | source_macro_rmse |
| --- | --- | ---: | ---: | ---: | ---: |
| external_climate | legacy_attention_mlp | 5 | 0.8038 ± 0.0050 | 1.0759 ± 0.0075 | 1.2234 ± 0.0072 |
| external_climate | linear | 1 | 0.8088 | 1.0661 | 1.2641 |
| external_climate | mean | 1 | 0.8698 | 1.1736 | 1.2785 |
| external_climate | random_forest | 5 | 0.8327 ± 0.0009 | 1.1113 ± 0.0008 | 1.3185 ± 0.0039 |
| test | legacy_attention_mlp | 5 | 0.7880 ± 0.0153 | 1.0312 ± 0.0175 | 1.0852 ± 0.0074 |
| test | linear | 1 | 0.7935 | 1.0337 | 1.1172 |
| test | mean | 1 | 0.8777 | 1.1501 | 1.1689 |
| test | random_forest | 5 | 0.8695 ± 0.0013 | 1.1137 ± 0.0012 | 1.1582 ± 0.0026 |
| validation | legacy_attention_mlp | 5 | 0.9579 ± 0.0048 | 1.2931 ± 0.0067 | 1.1042 ± 0.0044 |
| validation | linear | 1 | 0.9802 | 1.3324 | 1.1062 |
| validation | mean | 1 | 0.9783 | 1.3245 | 1.1718 |
| validation | random_forest | 5 | 1.0062 ± 0.0024 | 1.3499 ± 0.0029 | 1.1128 ± 0.0013 |

## preference 主评测

随机森林和旧网络运行固定的 5 个训练种子；± 表示训练种子间样本标准差，不是跨建筑置信区间。

| 切分 | 模型 | 种子数 | macro_f1 | log_loss | brier_multiclass | ece_10_bins |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| external_climate | logistic | 1 | 0.3191 | 0.8453 | 0.4983 | 0.0906 |
| external_climate | prior | 1 | 0.2640 | 0.9467 | 0.5610 | 0.1720 |
| external_climate | random_forest | 5 | 0.3854 ± 0.0037 | 0.8774 ± 0.0024 | 0.5063 ± 0.0010 | 0.0494 ± 0.0049 |
| test | logistic | 1 | 0.4095 | 0.8928 | 0.5422 | 0.0414 |
| test | prior | 1 | 0.1788 | 1.0616 | 0.6515 | 0.1172 |
| test | random_forest | 5 | 0.4522 ± 0.0039 | 0.9604 ± 0.0044 | 0.5765 ± 0.0023 | 0.0868 ± 0.0035 |
| validation | logistic | 1 | 0.3944 | 0.8786 | 0.5297 | 0.1186 |
| validation | prior | 1 | 0.2682 | 0.9693 | 0.5703 | 0.1894 |
| validation | random_forest | 5 | 0.3784 ± 0.0040 | 0.9007 ± 0.0058 | 0.5437 ± 0.0037 | 0.0469 ± 0.0071 |

## PMV 的共同测试子集

使用 pythermalcomfort 的 ISO 7730:2005 PMV、相对风速及动态衣着修正；外部功为 0，保留默认适用范围限制。
六输入学习模型在同一适用子集的训练来源上训练，测试行与 PMV 完全一致。此小表不能与四输入全样本表直接排名。

| 切分 | 全部热感觉样本 | PMV 适用样本 |
| --- | ---: | ---: |
| train | 44583 | 3836 |
| validation | 5913 | 2853 |
| test | 7388 | 1464 |
| external_climate | 12874 | 1782 |

| 切分 | 模型 | MAE | RMSE |
| --- | --- | ---: | ---: |
| validation | mean_6input_cohort | 1.2820 | 1.6222 |
| test | mean_6input_cohort | 0.6348 | 0.8665 |
| external_climate | mean_6input_cohort | 1.1257 | 1.4950 |
| validation | linear_6input | 1.3527 | 1.6991 |
| test | linear_6input | 0.6025 | 0.7642 |
| external_climate | linear_6input | 1.0147 | 1.2820 |
| validation | random_forest_6input | 1.3640 | 1.7086 |
| test | random_forest_6input | 0.7055 | 0.8716 |
| external_climate | random_forest_6input | 0.9920 | 1.2675 |
| validation | pmv_iso_7730_2005 | 1.6693 | 2.0464 |
| test | pmv_iso_7730_2005 | 0.6346 | 0.8072 |
| external_climate | pmv_iso_7730_2005 | 1.0627 | 1.3430 |

## 旧清洗规则消融

固定分组，分别改变训练集过滤与测试集过滤。仅测试集过滤会改变评测人群，不能把更低误差视为模型改进。
训练样本：未过滤 44583，过滤后 38429。

| 切分 | 训练处理 | 测试人群 | 测试行数 | MAE | RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| test | unfiltered_train | all | 7388 | 0.8695 | 1.1142 |
| test | unfiltered_train | legacy_retained_test | 6390 | 0.7643 | 0.9658 |
| external_climate | unfiltered_train | all | 12874 | 0.8323 | 1.1114 |
| external_climate | unfiltered_train | legacy_retained_test | 11343 | 0.7364 | 0.9797 |
| test | legacy_filtered_train | all | 7388 | 0.9338 | 1.2091 |
| test | legacy_filtered_train | legacy_retained_test | 6390 | 0.7466 | 0.9356 |
| external_climate | legacy_filtered_train | all | 12874 | 0.8297 | 1.0986 |
| external_climate | legacy_filtered_train | legacy_retained_test | 11343 | 0.7087 | 0.9233 |

## 历史风格随机按行切分诊断

旧清洗人群使用 80/10/10 按行切分；随机森林测试 RMSE=0.9011。
训练与测试重合 40 个来源，测试中 6070/6070 行的来源在训练中出现。
该诊断改变了人群、训练样本量及切分，不用其差值估计某一个因素的因果影响；也不声称复现了缺失随机种子的历史网络成绩。

## 工件与限制

- `metrics.csv` 为逐模型、种子与切分指标，`source_metrics.csv` 为逐来源指标。
- `data/audit.json`、`data/splits.csv`、`config.json` 和 `environment.json` 保存数据、切分、配置及环境证据。
- 模型、预测与学习曲线保存在本次输出目录；大型工件不提交 Git。
- 网络保留旧四维门控和 128×128 隐层结构；新增固定种子、验证集早停、最佳权重恢复及保存后重载核验。
- 冷热偏好 Brier 为多类别平方误差之和的样本均值；ECE 为 10 个等宽置信度桶。当前仅测量概率校准质量，未做额外后处理校准。
- 原始快照未与上游发行版逐行匹配；当前来源元数据不等于对本地副本的完整历史溯源。
- 少量留出来源、非均衡人群和缺失输入限制结论；此阶段不报告控制收益。
