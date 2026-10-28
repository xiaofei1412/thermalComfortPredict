# V2-4当前关闭态与复核

本次试次因预算字段契约错误，在实际select前停止。**原入口`run_v2_residual`仅作为失败版本留存，不用于启动新训练。** [原学习前方案](v2_4.md)及[实际报告](../../reports/v2_4/REPORT.md)分别保留；前者不是已通过证明。

在项目根目录、安装requirements/sac.txt的独立V2环境执行，只读复核本机故障原件：

```powershell
outputs/v2-sac/env/Scripts/python.exe -m scripts.audit_v2_4_failure outputs/v2-4/live-registered --output outputs/v2-4/recheck-failure.json
outputs/v2-sac/env/Scripts/python.exe -m scripts.check_v2_4_budget --output outputs/v2-4/recheck-budget.json
outputs/v2-sac/env/Scripts/python.exe -m unittest discover -s tests -v
```

第二条只使用临时账本夹具，不连接BOPTEST，不训练，不创建试次/授权。它复现原KeyError，并验证改为`select_max/initialize_max/advance_max`后的预算契约；候选函数在thermal_comfort/rl_budget_contract.py中，没有热改冻结入口。公开报告含关闭授权/计数/故障和审计摘要；本机calls/attempts/日志位于outputs/v2-4，完整旧模型/轨迹不在Git中。

后续必要工程门：新版本入口必须在创建目录或授权前验证完整协议，且用真实安全账本和BudgetTransport＋mock HTTP贯通select/初始化/推进预扣、总量/种子/时钟拒绝、两槽成功/失败派发和句柄关闭。通过后才另登记新试次，原种子/比较门/配置保持；不续本次，不用留出调试，不因预算未消费就自行重启。未获整体资格，不进入留出或工作台。
