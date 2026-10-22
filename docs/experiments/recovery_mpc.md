# 舒适恢复与天气误差实验协议

## 固定范围与比较

继续使用 BOPTEST 0.9.0 `bestest_hydronic_heat_pump`、同一固定 FMU、900秒控制步长和192 m²面积。原实验与工件不覆盖。本阶段只使用单热区供暖，不增加制冷能力，不接入尚未校准的人员偏好模型。

`configs/recovery_mpc.json` 在完整测试前固定。控制器先在 day84 验证回合运行，再在 day98、119、140 各评测7天、预热7天。day98/119 是已看过前阶段结果的回归测试；day140 是本阶段新增日期，仍然只是同一年度、同一FMU，不构成独立楼宇或真实迁移证据。测试输出不用于选参，失败时保留输出并报告。

主比较是**新恢复规则与新MPC组合**：相同初始化、动作限幅18～23.25°C、0.5 K/步限速、恢复/减热保护、日程与天气可用信息；前8步相同规则。与旧规则的比较单独列出，用于判断恢复策略的实际影响。总越界K·h之外，报告过冷/过热积分、越界时长、最大偏差与每日结果，避免用过热改善掩盖过冷恶化。

验收：对每个日期和天气模式分别检查耗电严格降低，且原生30秒积分的温度越界不增加（仅容许1e-6的数值误差）；另列过冷/过热分项及最大偏差。不能只用跨日期汇总宣称成功；中止带18～28°C也不等同于舒适达标。

## 训练与校准

保留原 day14/35/56 训练与 day77 验证、day98/119 测试。在 day63 补采7天训练，day84 补采7天验证；均为预先固定种子、2～4小时量级的分段设定值、18～24°C限幅、0.5 K/步及原激励温度保护。暖工况通过较高设定值和长停机段扩大室内温度/空载覆盖，**不等于补齐炎热室外天气**。

全部训练评测截至day70，验证的预热从day70开始；全部验证评测截至day91，测试预热从day91开始。跨角色的半开时间窗互不重叠；同角色新增回合的预热与旧回合评测重叠，不能把它们当作独立随机样本。不得把 day98/119 的温度、热量或耗电用于拟合、裕量或恢复参数选择。

2R2C、执行器、Ridge残差与支持范围只用训练拟合。残差权重用原验证前3天选择；裕量使用原验证第4天以后与新增验证第16步以后，两者与选择目标时间隔离。空载热量门槛0.05 kW，训练空载平均电功率作为非负待机项；运行项用 `min(Q/0.5,1)`、`Q`、`Q max(20-Tout,0)/20` 非负回归，连续过渡，避免停机时添加固定运行截距。单独用真实区间热量评估电功率映射，也评估包含执行器误差的自由滚动预测。

离线新旧模型使用相同起点（16至回合末减16，步距4）比较15分钟、1/2/4小时误差；起点数量与上一阶段为24小时长预测统一截断的表不同，不能直接跨表比数值。

## 回退与MPC

恢复规则查看未来4小时已知舒适日程；非占用设定值不低于20.5°C，提前加热使用0.3/0.6 K启停滞环。热惯性可能仍导致恢复滞后或过热，必须通过FMU而不是模型内回放判断。MPC保留2小时、多步分块候选、经验误差裕量和模型范围检查；初始化、恢复、过热、OOD、不可行或非法预测均回退。

MPC对给定预报及其偏暖/偏冷包络同时检查温度约束，以最坏预测电费加动作变化惩罚选动作。支持范围检查针对中心预报与预测温度，天气包络边缘可能外推；包络不是置信区间，不保证真实温度安全。模型拟合后重新进行day84/90成对FMU高低档干预，方向门槛匹配当前模型与配置摘要后才允许运行控制。

## 天气误差

三种预先固定情形：官方确定性完美天气；未来预报逐渐偏暖2 K且辐照高估25%；偏冷2 K且辐照低估25%。偏差从当前测量的零开始，2小时线性达到全幅。只改变控制器收到的预报，FMU天气、当前测量、真实热量观察更新、日程、KPI积分始终使用原值。每次决策重新生成预报视图，不能把上次扰动累加到原始数据。

这是**合成天气预报误差压力测试**，不是接入真实气象服务，也不是由历史预报误差拟合的概率分布。两个偏差方向只有一个幅度，不代表所有误差形态。新规则只利用未来日程，其天气模式结果应数值相同；MPC可利用天气，其结果可能不同。

## 重现命令

```powershell
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery collect --output outputs/recovery-calibration
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery fit --output outputs/recovery-model
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery fit --output outputs/recovery-model-repeat
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery response --output outputs/recovery-response
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery run --output outputs/recovery-validation --start-days 84 --weather perfect
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery run --output outputs/recovery-control
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.recovery run --output outputs/recovery-repeat --days 0.25
.\.venv-cpu\Scripts\python.exe scripts/verify_recovery.py
.\.venv-cpu\Scripts\python.exe scripts/report_recovery.py
```

所有输出目录要求不存在。完整轨迹与权重留在忽略的`outputs/`，Git保存紧凑报告、摘要、配置、验证证据和代码。CI运行离线测试；FMU测试在本地完成。
