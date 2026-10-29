# 历史数据清单

最终公开main不再分发历史原始CSV和清洗CSV。原快照为107,584行，旧清洗表60,699行；字段、单位、版本依据和SHA保留在[manifest.json](manifest.json)，不是用当前上游最新版替代冻结数据。

所有者可从V1最终私有归档恢复`data/raw/ashrae_db2.csv`及`data/processed/cleaned_asds4.csv`后运行原数据完整性测试；公开环境缺少这些文件时，9项依赖该快照的检查明确跳过。其他单元测试、V1小样例和V2公开回放仍可运行。历史文件仍可在Git历史`fe646d7`中查阅，本次没有重写历史或重新授权。

热感觉、热舒适程度、冷热偏好和可接受性不混用。新流程的源分组、标签无关质控和冻结切分见[实验协议](../docs/experiments/baseline.md)，结果见[基线报告](../reports/baseline/REPORT.md)。

来源：[官方维护仓库](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II)与[数据档案](https://datadryad.org/dataset/doi:10.6078/D1F671)。历史快照的精确发行链仍有缺口；不宣称它是最新数据或原实验室完整数据。
