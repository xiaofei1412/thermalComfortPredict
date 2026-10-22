# 数据说明

| 路径 | 当前用途 | 状态 |
| --- | --- | --- |
| `raw/ashrae_db2.csv` | 107,584 行历史原始快照 | 原文件不变；SHA-256 与字段映射已固定，准确上游发行版仍未确定 |
| `processed/cleaned_asds4.csv` | 60,699 行历史五列表 | 已由新流程逐值复核；不作为新主评测的最终训练表 |
| `manifest.json` | 原始摘要、位置式字段、单位、质控和来源依据 | 用于每次重建的前置校验 |

冷热感觉、舒适程度、冷热偏好及可接受性分别保留。本阶段训练冷热感觉和冷热偏好任务，保留小数热感觉标签。损坏的单位列采用固定位置读取，并通过摄氏/华氏成对数值核验。主评测使用上游维护脚本的标签无关室温质控，不套用旧温度—标签过滤规则。

重新生成的数据、切分和审计默认位于 `outputs/baseline/data/`。紧凑审计归档在 [`reports/baseline/data/audit.json`](../reports/baseline/data/audit.json)，协议见 [实验说明](../docs/experiments/baseline.md)，来源与许可证据见 [核查记录](../docs/experiments/data-provenance.md)。

参考来源：

- [ASHRAE Global Thermal Comfort Database II 官方维护仓库](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II)
- [官方数据档案](https://datadryad.org/dataset/doi:10.6078/D1F671)

现存文件不视为官方最新版，也不视为已恢复原实验室的完整数据。当前 Dryad 元数据声明 CC0-1.0，但本地历史副本的完整版本链仍需核实。后续新增数据继续记录来源、版本、校验值与使用条件；代码许可与第三方数据许可分别处理。
