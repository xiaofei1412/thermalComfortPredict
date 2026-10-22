# 数据重建与预测基线

此阶段只验证监督预测，不运行 HVAC 仿真或控制。正式配置为 [`configs/baseline.json`](../../configs/baseline.json)。

## 安装与运行

推荐使用 Python 3.14；本机验证版本为 3.14.3。核心依赖固定在 [`pyproject.toml`](../../pyproject.toml)。在项目根目录执行以下 PowerShell 命令：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m thermal_comfort prepare
.\.venv\Scripts\python.exe -u -m thermal_comfort run
.\.venv\Scripts\python.exe -u scripts/verify_baseline.py
```

Linux/macOS 将 Python 路径替换为 `.venv/bin/python`，用 `python3.14 -m venv .venv` 创建环境。本机已有 CUDA 12.8 的 PyTorch 2.10.0，正式训练使用 GPU；上述安装示例使用 CPU。若需要 GPU，应先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 2.10.0 构建，再安装项目。配置中的 `device: auto` 自动选择可用设备。

`prepare` 只重建数据与审计；`run` 会先执行相同预处理，再训练并输出结果。可用 `--output` 指定输出目录，用 `--config` 指定另外的 JSON 配置。正式结果必须与对应配置一起报告，不能将少量种子或短训练的冒烟结果当作正式结果。

`scripts/verify_baseline.py` 会重建数据和切分，并重复第一个正式训练种子的随机森林与神经网络训练，检查原始摘要、处理数据、切分、学习曲线和预测。它也支持 `--output` 指向待验证的完整工件目录；重复训练需要与原运行相同的设备和数值环境才能要求神经网络逐位一致。

## 数据协议

- 原始文件保持不变，先验证 SHA-256、列位置和列名；损坏或替换的数据不会静默进入训练。
- 按位置读取损坏的单位列，再使用摄氏/华氏成对数据验证转换关系，不依赖 pandas 自动生成的重名列后缀。
- 主输入沿用室温、湿度、室外温度和旧季节编码：Winter=1、Autumn=2、Spring=3、Summer=4。季节编码是历史基线的兼容选择，不表示最佳特征表示。
- 主评测要求四输入完整且有限、湿度在 0～100% 内；沿用上游维护脚本的 `10 < 室温 < 40°C` 质控规则。完整输入但未通过质控的记录保留在输出表中，标记为 `excluded_feature_quality`，不用于主模型训练或评测。
- 热感觉保持 −3～3，保留小数标签。冷热偏好映射为 cooler=0、no change=1、warmer=2。原始舒适度和可接受性代码单独保留，本阶段不猜测旧版量表或将它们替换成热感觉。
- 未对缺失特征做全表插补，不依据温度与目标关系删除主评测记录。

## 切分与评测

来源文献经 Unicode 和空白规范化后生成稳定组 ID；1,655 条缺失来源的原始记录合为一个未知来源组，避免逐行伪造独立来源。本表没有可靠建筑 ID，因此结果称为跨研究来源评测，不能声称已经严格隔离了具体建筑或人员。

预先选定 Cfa 作为外部气候。先保留 Cfa 记录及其全部来源；同来源的其他气候记录标为 `excluded_external_source`。其余来源按固定种子 33 分组切分，目标组数比例约 60/20/20。不同来源样本量不同，行数比例与组数比例不同。两个预测任务共享同一切分，不根据目标或成绩重新选分组。

均值和线性模型使用固定训练集；随机森林与旧网络架构使用 33～37 五个训练种子。网络沿用四维特征门控、128×128 隐层、Adam、学习率 0.0005、batch size 64、最多 100 轮，新增验证集早停（patience=12）、恢复最佳权重和保存后重载核验。这是网络结构复现，不是缺失原始随机状态下的 2023 年成绩逐位复现。

热感觉报告 MAE、RMSE 和按来源等权的宏平均指标。偏好报告 Macro-F1、log loss、多类别 Brier 和 10 桶 ECE；本阶段仅评估概率质量，不做额外校准后处理。超参数预先固定，验证集用于网络权重选择，测试与外部气候不用于选择训练轮数。

## PMV 与清洗诊断

PMV 使用 pythermalcomfort 4.6.2、ISO 7730:2005、相对风速、ISO 动态衣着修正、外部功 0，并保留模型适用范围限制。不把缺失辐射温度等输入替换成假定值。六输入均值、线性与随机森林模型在同一适用训练子集上训练，全部方法在相同 PMV 适用测试行比较。此表与四输入全样本表的人群和输入不同，不能混为一个排行榜。

旧清洗复现将原始目标平移成 1～7，仅用于核对现存处理表，输出标签不再称为舒适度。温度—标签规则与整数过滤均按旧逻辑复现。清洗消融固定来源切分，分别改变训练过滤和测试过滤，报告 2×2 结果。测试过滤导致人群改变，误差更低不能直接说明模型更好。

另运行旧清洗人群的 80/10/10 随机按行切分，记录来源重叠，仅作诊断。它同时改变数据人群、训练量和切分方法，不能用其分数差值估计单一因素的因果影响。

## 输出与复用

默认输出到被 Git 忽略的 `outputs/baseline/`：

- `data/prepared.csv`、`data/splits.csv`、`data/audit.json`：数据、行级切分及审计。
- `config.json`、`environment.json`：有效配置、包版本、设备及 Python 源文件摘要。
- `models/`、`histories/`、`predictions/`：训练产物、曲线与逐行预测。
- `metrics.csv`、`source_metrics.csv`：完整指标及逐来源指标。
- `pmv_coverage.json`、`pmv_applicability.json`：物理基线覆盖率与适用范围提示摘要。
- `diagnostics.json`、`training.json`：清洗诊断、训练轮数与重载校验。
- `REPORT.md`、`sensation_rmse.png`：可读报告与静态对照图。

正式运行的紧凑报告归档到 [`reports/baseline/`](../../reports/baseline/REPORT.md)。模型和逐行数据不上传 Git；重新执行同一配置可生成。机器和库构建变化可能带来数值差异，环境快照用于解释差异。

重载网络进行预测：

```python
import pandas as pd
from thermal_comfort.models import load_neural, predict_neural

fitted = load_neural("outputs/baseline/models/sensation_legacy_attention_mlp_33.pt")
data = pd.read_csv("outputs/baseline/data/prepared.csv")
inputs = data.loc[data["split"].eq("test"), fitted["features"]].to_numpy()
predictions = predict_neural(fitted, inputs)
```

预测含义是 −3～3 的冷热感觉，不是舒适程度或控制动作。
