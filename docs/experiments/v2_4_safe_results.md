# V2-4新版入口：已关闭结果的复核

本次三种子完成，已关闭授权/专用副本；有效供暖联合门0/6，按预登记停止，不启动第二配置、留出或RL工作台。原首次预算预扣失败仍独立保存。学习前设置见[v2_4_safe](v2_4_safe.md)，实际结果见[报告](../../reports/v2_4_safe/REPORT.md)。

在项目根目录及已装requirements/sac.txt的独立V2环境执行：

```powershell
outputs/v2-sac/env/Scripts/python.exe -m unittest discover -s tests -v
outputs/v2-sac/env/Scripts/python.exe -m scripts.audit_v2_residual_safe outputs/v2-4-safe/live-registered --output outputs/v2-4-safe/recheck-archive.json
```

第一条包含临时夹具中的安全入口/动作链验收，不连接真实FMU；第二条只读取已关闭的本机归档，重算原生/KPI、12小时预报边界、最终包/日志绑定、全部残差映射及开发回合实际状态上的规则锚。数据量大，可能需要数分钟，物理调用0。不要用训练入口恢复这个已关闭试次。

公开Git含源码、学习前收据、协议/授权关闭链和小型对照证据；完整原生/预报/动作日志、模型/经验池/RNG包在outputs/v2-4-safe，旧模型及FMU仍按原资产步骤准备。新克隆可运行代码与夹具回归，但不能只靠公开摘要重建所有原始轨迹；完整只读复核需要本机归档或未来另验收的私有恢复包。新资产尚未取得V1那样的异地恢复认证。

manifest中的protocol_sha256沿用原configs/v2_comparison；child科学配置、safe工程配置和新入口由sources_sha256补充绑定，method.json说明含义。保存RNG/模型包不等于支持重启真实学习或FMU状态；本次没有恢复、挑中间包或选择最佳种子。
