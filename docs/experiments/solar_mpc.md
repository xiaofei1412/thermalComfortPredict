# 暖天气切分、蓄热消融与提前停热协议

## 范围与固定日期

使用既有BOPTEST 0.9.0固定单热区FMU、192 m²、900秒控制、30秒原生积分；原实验不覆盖。目标是逐日期/天气条件验证相对恢复规则能否节电且不增加操作温度越界，不把主观热感觉/偏好当作同一指标。

- 训练：旧day14/35/56激励、旧day63补采、旧day119激励、新day147激励。**旧day119原为测试，本轮明确退休并改为训练**，不得再称该日期为留出结果。
- 验证：新day168和day189各7天。day168前3天、32步预测终点不超出该段的起点用于模型结构/残差权重选择；day168第4天以后与day189用于经验裕量。模型开发会查看验证结果，经验裕量不作独立概率保证。
- 新测试：day238、273、315各7天，各预热7天；完美、偏暖、偏冷天气模式，各运行恢复规则、最小供暖、提前停热MPC。只根据固定天气文件的外生变量预先核查季节与覆盖，不查看测试控制结果来选日期。

训练最晚评测结束day154，验证最早预热day161；验证最晚结束day196，测试最早预热day231。跨角色以及测试回合之间的预热/评测半开时间窗不重叠。旧训练day56/63存在同角色预热相关，不当作独立随机样本。所有新测试仍来自同一FMU与同一年度，不是跨楼宇或真实Sim-to-Real证据。

暖季可能自然超过上一阶段28°C中止阈值，本轮在观察新温度轨迹之前将仿真中止带固定为18～35°C；用于完整记录炎热天气，不是放宽舒适验收。**官方舒适日程完全不改**，超28°C的原生点另行统计，不能以未触发中止宣称安全。

## 模型与蓄热消融

仅用训练数据重新拟合正参数RC、执行器、残差、非负连续门控电功率；暖天气不只提高室内设定值。三个候选使用相同训练数据与选模起点：

1. 原2R2C＋递归潜在状态观测，作为扩充数据后的简单基线。
2. 同一RC＋过去16步（4小时）操作温度/真实区间供热的窗口最小二乘状态估计。
3. 即时太阳增益进入热区、存储太阳增益进入潜在热容量的正参数网络＋同一窗口估计。

第三种模型仍为两状态有效网络，不能解释为已辨识真实楼板/墙体温度。窗口估计仅用当前及过去观测，没有未来真实温度或供热；弱正则和潜在状态限幅均预先固定。每种候选在验证前段选择0/0.5/1残差权重，再按1/2/6/8小时温度RMSE均值选结构，复杂结构不胜出则采用简单模型。保留全部候选、参数、同数据消融和失败结果。

选模之后，时间留出的验证起点给出逐步绝对温差95%分位数＋0.1 K的经验裕量。离线各模型使用同一157个起点、相同未来动作与外生输入，自由滚动1/4/8/24/32步，不回填真实中间状态。原模型与新模型对照包含训练数据差异；同数据候选之间才用于判断结构改动。

## 控制与最小供暖

共同动作18～23.25°C、0.5 K/步；前8步使用相同恢复规则与初始化。恢复规则保持前阶段4小时日程前瞻和滞环。最小供暖仅在当前温度低于官方下界＋0.3 K（非占用至少19°C）时加热，至下界＋0.6 K后释放，其余选择动作下限；它是可操作的低供暖参照，**不是能耗或过热的数学下界，也未匹配相同舒适水平**。

提前停热MPC使用6小时计划、8小时停热投影，增加实际可执行的逐步下降动作。若最低设定值投影在8小时经验下界以上、且将接近或超过舒适上界，则提前停止新增供暖；否则在候选中优化电费和动作变化，使用经验温度裕量、模型支持范围与原规则回退。另模拟同模型、同初始状态的恢复规则轨迹，候选预测温度/潜在蓄热不得高于该参照，首动作也不高于规则首动作；这是模型内过滤，不是实际闭环的单调或安全证明。日程预测多取4小时以支持规则投影。没有可行计划不意味着真实舒适安全。停热投影可超出训练温度上界，不能将其作为外推准确性证明。

天气误差仍只改策略视图，当前测量、观测更新、真实FMU与KPI不改。本轮沿用同一误差生成函数，随6小时计划从零线性增至±2 K和±25%辐照；与上一阶段2小时全幅不同。该MPC按给定预报规划，三模式是压力测试，**没有声称使用概率天气预报或鲁棒安全证书**。

day168/189各三档独立FMU干预，4小时共同预序列＋8小时动作；预测使用真实限速动作，在6小时核查高低档温差/耗电方向，并要求至少一个情境有非零响应。门槛匹配模型和配置摘要后才运行控制，它不证明任意候选的反事实幅度准确。

## 验收与重现

主对照按每个日期与天气条件报告耗电、过冷/过热积分、越界时长、最大偏差和每日结果；主门槛为严格少耗电且总越界不增（数值容差1e-6），同时检查分项、时长和最大偏差。不把跨日期平均替代逐组验收。另报告相对最小供暖的取舍和零供暖仍过热的轨迹，不能仅据其非零过热断言物理不可避免。

```powershell
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar collect --output outputs/solar-data
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar fit --output outputs/solar-selected
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar fit --output outputs/solar-refit
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar response --output outputs/solar-response
git apply --unidiff-zero reports/solar_mpc/validation_v1_source.patch
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar validation --output outputs/solar-validation
Copy-Item thermal_comfort/solar.py outputs/solar-validation/source_solar_v1.py
git apply -R --unidiff-zero reports/solar_mpc/validation_v1_source.patch
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar validation --output outputs/solar-validation-v2
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar test --output outputs/solar-control
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.solar test --output outputs/solar-repeat --days 0.25
.\.venv-cpu\Scripts\python.exe scripts/verify_solar.py
.\.venv-cpu\Scripts\python.exe scripts/report_solar.py
```

需先具备前阶段`outputs/excitation`、`outputs/recovery-calibration`与`outputs/recovery-model`，以及运行中的本地服务；来源与部署见前阶段协议。输出目录要求不存在。`git apply`四行用于在没有其他修改、没有实验运行的独立副本中重现历史失败；补丁将当前源码还原到初版，保留源文件后必须反向恢复，才运行第二版和测试。该初版源码的摘要与初版运行清单完全匹配，审计会加载它重放旧策略。

开发期模型`outputs/solar-model`、`solar-model-final`与初版验证`solar-validation`保留，正式模型为`solar-selected`。初版验证增加过热，加入规则参照过滤后重新验证为`solar-validation-v2`，再冻结进入新测试；没有用测试结果修改控制。CI验证离线测试，真实FMU集成在本地完成。
