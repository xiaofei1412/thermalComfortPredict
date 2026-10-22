# 现存数据的来源与单位核查

核查日期：2026-10-22。

## 当前可确认的资产

本阶段使用仓库原有的 `data/raw/ashrae_db2.csv`，共 107,584 行、70 列，原始文件保持不变。其 SHA-256、编码、所用列位置与原始列名固定在 [`data/manifest.json`](../../data/manifest.json)。

该表采用 ASHRAE Global Thermal Comfort Database II 的旧版字段结构。现有证据不足以确定它与哪个上游发行文件逐字节对应，因此标记为“本地历史快照”，不强行标记成官方最新版。

已核查官方维护仓库提交 `c527d6115ebadb4725a4e1406b2439a7a1ffd214`。其中 `v2.1.0/source_data/db_measurements_v2.0.1.csv.gz` 是 110,070 行、49 列的规范化中间表，与本地文件形态不同，不能直接假定两者逐行对应。官方新版带建筑元数据也不意味着本地旧表已经恢复建筑 ID。

## 单位与标签

原始 CSV 有损坏及重复的温度单位列名。本地从 0 开始计数：29/30 列为空气温度配对，39/40 列为辐射温度配对。前者有 99,911 对、后者有 32,472 对可比较值。以 `F = 1.8*C + 32` 校验，最大误差均为 0.12°F，与记录精度相容；新流程固定这些位置并校验成对数据。

主评测的 `10 < ta < 40°C` 室温质控来自 [官方维护脚本](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II/blob/c527d6115ebadb4725a4e1406b2439a7a1ffd214/v2.1.0/main.py)，不使用目标标签。PMV 的适用范围另行处理。

冷热感觉、主观舒适度、冷热偏好与可接受性分别保留。旧版主观舒适度字段有量表异质性及一个无法解析为数值的值，本阶段不把它强制当作统一的 1～6 目标。主预测任务使用含义清楚的冷热感觉和明确字符串编码的冷热偏好。

## 引用与许可证据

- 数据集入口：[Dryad，DOI 10.6078/D1F671](https://doi.org/10.6078/D1F671)。
- 官方维护仓库：[CenterForTheBuiltEnvironment/ashrae-db-II](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II)。
- 原始介绍论文：[Development of the ASHRAE Global Thermal Comfort Database II](https://doi.org/10.1016/j.buildenv.2018.06.022)。
- [当前 Dryad 元数据 API](https://datadryad.org/api/v2/datasets/doi%3A10.6078%2FD1F671) 返回 versionNumber=9、publicationDate=2022-07-15、license=`https://spdx.org/licenses/CC0-1.0.html`。

当前上游元数据的 CC0 声明与本地历史副本的完整来源链是两个问题。后者尚未逐文件匹配，不用最新记录的元数据替代未核实的历史信息。新增公开数据时继续单独记录来源、版本和使用条件。
