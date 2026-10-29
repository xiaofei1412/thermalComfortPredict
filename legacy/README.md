# 历史代码归档

最终公开main只保留此说明；原`master`的`preprocess4.py`和`ffnn2.py`可从Git历史`fe646d7`恢复，不作为当前运行入口。历史源内容没有重写；新实现位于`thermal_comfort/`。

旧预处理把冷热感觉命名为热舒适，按温度与标签联合过滤，并使用不一致的数据文件名；旧随机状态与完整实验条件缺失。新流程已逐值复核旧清洗表、重建来源分组与固定种子基线，但没有声称复现缺失状态的原始分数。

[数据清单](../data/README.md) · [新实验入口](../docs/experiments/baseline.md) · [预测基线结果](../reports/baseline/REPORT.md)
