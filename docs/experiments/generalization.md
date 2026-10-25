# 冻结控制器的跨天气与跨建筑验证

## 预先固定的范围

基线为提交`d156645b93cd289081b1c8f0755b65121a9c83b0`的SolarController与`outputs/solar-selected/model.joblib`。不重训、不改规则、动作界限、保护、经验裕量或选模。冻结清单在任何目标控制结果产生前保存源码/配置/模型和FMU摘要；每次运行和审计重新核验。新日期固定day210/259/343，覆盖夏季、过渡季和冬季，不根据目标结果改日期。三域×三日期×三预报条件×两控制器，共54个7天回合；失败或中止同样保留，不能删掉再报成功率。

三个域分别为原192 m²住宅/布鲁塞尔天气、同住宅/丹佛TMY天气、官方48 m² BESTEST Case900办公室/丹佛TMY天气。住宅天气迁移仅替换FMU中的Brussels.mos与weather.csv，并修改案例名称；二进制、模型描述、舒适日程与其余资源逐字节保持。原FMU不覆盖。丹佛文件来自相同BOPTEST 0.9.0提交的bestest_air；这是真实作用于FMU的典型气象年迁移，不是新观测年或随机气候样本。实际天气测量与预报需核对，太阳几何也记录；若读入后位置未改变，必须披露，不能称完整地理迁移。

各域的预热为6天原内置控制＋1天共同22°C设定值预序列，评测前重置策略观测器，正式7天从整日开始。共同预序列用于使办公室夜间15°C默认设定与冻结18°C动作下限兼容，不根据结果调整；不计入评测能耗，保留预序列记录。两策略在相同域/日期的初始化、预序列和前8步必须一致。新时间窗与此前训练、验证和正式测试连同预热不重叠；不同域同日期为配对设计，不当作独立天气样本。

## 接口适配与解释边界

只做语义和单位明确的接口映射。办公室控制`con_oveTSetHea_u`，冷却仍由相同内置策略管理。`zon_reaTRooAir_y`是空气温度，**不是操作温度**；仅作为冻结模型输入的迁移压力测试，结果与住宅分域列示。办公室官方GasPower为`fcu_reaPHea_y`；依据该FMU对应FanCoilUnit_T.mo的`powHea(k=1/eff), eff=0.9`，观测器供热输入取0.9×该读数，冷却作用未纳入原供暖模型。源模型的热泵电功率映射保持原样，明知不适合燃气末端，也不利用目标数据重拟合；因此要单独检查电量预测偏差，不能将源电费目标解释为目标办公室总能源成本优化。

电耗、燃气、两者相加的终端能量分别积分，不将燃气称电耗，不做未经指定的一次能源换算。各域用其原官方温度日程计算过冷/过热K·h、时长和最大偏差。主验收要求每组严格少耗电，同时总越界、冷热分项、时长、最大偏差均不增加；办公室还要求燃气和总终端能量不增加，防止能源转移。零基线电耗的百分比记为空，不能声称节电。

冻结策略原有18～35°C中止带保持；完整原生轨迹另审计。若触发中止，保留部分轨迹与失败原因，不能拿不同长度积分比较。完美/偏暖/偏冷仍为6小时内渐变±2K/±25%预报偏差，仅改控制器视图。规则应跨预报模式数值相同。

## 预测与回退审计

每回合从第16步起每4步取共同起点，自由滚动1/4/24/32步；只用起点及之前真实温度和供热初始化，不回填中间真值。分别报告持久性与冻结RC残差模型的温度RMSE、偏差、经验裕量覆盖，以及累计电量RMSE/偏差。预测条件是记录的未来动作与给定预报，属于事后条件预测，不是未来动作已知的在线闭环能力证明。额外以完美天气视图区分模型与预报误差。只检验冻结裕量，不重新标定。

回退分母同时报告全部步数与去掉共同前8步后的步数，分别计数优化、提前停热、支持范围外、无可行计划和其他失败原因；不能把提前停热当回退，也不能把大量规则回退的成绩全归于模型迁移。实际原生积分对照官方能源/温度KPI；全部正式决策重放，并独立重跑每支首6小时验证前缀，未重复完整7天。

## 执行

先按原部署协议启动BOPTEST，扩展固定版本稀疏检出：

```powershell
git -C outputs/boptest-v0.9.0 sparse-checkout add testcases/bestest_air
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.generalization prepare
wsl -d Ubuntu-24.04 -u root -- docker compose -f /mnt/d/thermalComfortPredict/infra/boptest.compose.yml run --rm -v /mnt/d/thermalComfortPredict/outputs/generalization-freeze/testcases:/generalization:ro provision python3 -m boptest_submit --shared --path /generalization
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.generalization run --output outputs/generalization-control
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.generalization run --output outputs/generalization-repeat --days 0.25
.\.venv-cpu\Scripts\python.exe scripts/verify_generalization.py
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -q
```

冻结与完整原始工件在outputs，轻量结果与验证证据归档reports/generalization。该阶段仅完成零目标训练的仿真迁移评估，没有少样本校准、真实数据或真实闭环结论。
