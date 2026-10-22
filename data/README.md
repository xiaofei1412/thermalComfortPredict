# 数据说明

| 路径 | 当前用途 | 状态 |
| --- | --- | --- |
| `raw/ashrae_db2.csv` | 现存 ASHRAE 格式原始表，供重新核对与构建任务 | 保留原文件；准确版本、字段单位及许可待核验 |
| `processed/cleaned_asds4.csv` | 历史脚本处理后的五列表，供复现旧流程 | 保留原文件；不直接作为新路线的最终训练表 |

现存原始表包含冷热感觉、舒适程度、冷热偏好及可接受性等不同字段，后续应分别定义任务。部分温度单位列名存在乱码或重名，需与官方字段字典核对后再读取和转换。

参考来源：

- [ASHRAE Global Thermal Comfort Database II 官方维护仓库](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II)
- [官方数据档案](https://datadryad.org/dataset/doi:10.6078/D1F671)

现存文件不自动视为官方最新版，也不视为已恢复原实验室的完整数据。后续新增数据需记录下载来源、版本、校验值与使用条件；代码许可与第三方数据许可分别处理。
