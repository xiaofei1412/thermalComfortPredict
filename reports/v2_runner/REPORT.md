# V2-3：三算法共同运行器工程验收

2026-10-28。**SAC/PPO/TD3共同运行器及本轮工程门已通过：PPO步数、动作留痕、完整包原子保存和累计预算/时钟均有验证。** 正式训练未启动，真实FMU调用和留出初始化为0；不产生新节电结论。冻结的12周期/47,808训练步、最终检查点与故障即停协议保持。

## 变更与用途

- 新增[共用训练循环](../../thermal_comfort/rl_campaign.py)与[动作记录层](../../thermal_comfort/rl_runner.py)：三算法走相同六日期周期、原503维观测/奖励/保护，继承冻结住宅接口；没有修改父阶段源码或参数。SAC/TD3采样与原SB3的结果和随机消耗顺序相同，日志不额外抽样。
- PPO在reset前检查664步/83 batch/10 epochs，每回合核对actor、epoch、优化器实际step与已消费rollout，防止默认2048步造成超采。记录原高斯建议、库裁剪、缓冲动作/log-prob；物理层另存actor投影、实际/测量设定和回退。
- [跨进程预算账本](../../thermal_comfort/rl_budget.py)在发送前扣attempt，失败不退；文件锁和原子JSON更新防止并发丢计数。固定原process/stage时钟，wall回拨、重新构造账本或原地改预算都不能延长；更新、导出、检查点校验/flush和最终结果写盘后都复核。超时/不完整不能标成功，cleanup仍可执行。
- [完整检查点](../../thermal_comfort/rl_checkpoint.py)先写独占partial，实际重载核对模型/目标网络/优化器/熵/回放，验证RNG、回调窗口/偏移、预算/原时钟、全部回合工件与KPI SHA；最后恢复live随机流，fsync并复查时限后同卷rename发布。只保留最近两完整包、全部清单和失败partial，不用早期包替代缺失最终包。
- [登记/运行入口](../../scripts/run_v2_comparison.py)仍写入`formal_training_authorized=false`；当前seed命令在Client构造或预算修改前拒绝。没有自动续跑入口，也不恢复FMU内部状态。

## 实际验证

完整回归 **294项通过，36.354秒**（原279＋新增15）。完整结果见[工程审计](runner-audit.json)、[完整包随机流联检](bundle-rng-audit.json)和[证据/失败记录](evidence.json)。

| 最终合成验收，种子11 | actor步 | 原生周期虚拟advance | 更新口径 | 动作行数 |
| --- | --- | --- | --- | --- |
| SAC | 3,984 | 4,032 | actor/critic/熵各3,728 optimizer steps | 3,984 |
| PPO | 3,984 | 4,032 | 60 epochs、480 optimizer steps | 3,984 |
| TD3 | 3,984 | 4,032 | critic3,728、actor1,864 optimizer steps | 3,984 |

每算法6次合成select/initialize，共18回合；使用正式网络、超参、664 actor步和六日期顺序。测试账本只缩窄owner为该算法/种子，计数/时间上限不变；每周期后刻意停止，未记作正式完成种子。夹具无楼宇动力学，其KPI由人工信号生成，不能当官方服务成绩、训练数据或新增天气场景。

- PPO单测另连续两个回合，恰好1,328 actor、2复位/1,344虚拟推进、160优化器步，无额外reset。缓冲原动作与日志一致，log-prob有限且保留；最终六回合有1,315个原高斯建议超Box，库裁剪后均有界，未被隐藏。各周期6次人为takeover只为测试，不解释为真实保护率。
- 假BOPTEST客户端验证三算法真实物理适配层：低温规则接管时原21.5°C建议与21°C实际/测量值分别保留；机械限幅/限速和12小时权限继续由既有冻结回归覆盖。
- 3个独立进程共享账本完成60次预留，无计数丢失；单测覆盖全局/单种子上限、失败不退、重复启动/顺序/协议漂移、wall回拨与monotonic更换拒绝。
- 缺RNG/回调/预算、日志前缀篡改、提前开发、校验失败、rename失败、保存耗时超限全部拒绝；失败不发布完整目录，原partial保留。保留最近两包且不删partial/旧清单、拒绝清理SHA不符目录的测试通过。
- 在**6个全新进程**重载三个完整包，Python/NumPy/Torch、环境/模型空间、随机策略/噪声、回放及PPO乱序探针全部逐值相同；无新增梯度或控制。这里只证明完整包的加载/RNG联接，不证明完整学习轨迹或FMU状态无缝续跑。

先前一次夹具导出漏过滤DummyVecEnv添加的终止观测，在SAC首个合成回合后失败；生产导出原已有处理，夹具修复为复用同一函数。失败、源码、部分导出与中间通过版本均保留。最终代码又增加收尾时钟/预算漂移、fsync及rename注入后重验。三次受控验收合计24,568合成actor/24,864虚拟advance（不含回归夹具），全部为工程计算，正式训练/FMUs仍为0，未隐去重跑成本。

## 适用范围与复核

原SAC短预算成绩、RNG历史缺失/3份首训练KPI响应缺失、189严格失败和所有旧保护角色不变。六层父锁、V1规则模型与本地FMU字节SHA核验保持，未升级依赖/启动Docker或更改工作台。CPU/固定版本/一环境/无SDE边界保持；真实服务连接、三worker仿真并发性能、突然断电与真实进程崩溃未在本轮测试。

小型源码/协议/审计在Git，大型合成包、完整动作/预报/观测/账本、失败和源码快照在`outputs/v2-runner/`；未加入V1既有异地备份认证。JSON公开副本统一LF，原登记字节SHA在evidence另列，原内部引用未改。

```powershell
outputs/v2-sac/env/Scripts/python.exe -m unittest discover -s tests -v
outputs/v2-sac/env/Scripts/python.exe -m scripts.verify_rl_runner --output outputs/v2-runner/acceptance-recheck
outputs/v2-sac/env/Scripts/python.exe -m scripts.check_rl_checkpoint verify --archive outputs/v2-runner/acceptance-recheck --output outputs/v2-runner/bundle-rng-recheck
```

后两条须使用全新输出目录；工程循环在本机约71～93秒/算法（不含额外故障/新进程检查），不代表正式FMU训练成本。当前运行器构建与指定工程门完成。下一步明确启动后，按[冻结比较协议](../../docs/experiments/v2_comparison.md)另存授权/源码登记，在原预算内核对副本/FMU资产、首回合共同前缀/原生KPI和真实成本；任何失败停止，不额外预跑、追加配置或消费留出。详细接口和边界见[运行器说明](../../docs/experiments/v2_runner.md)及[路线图](../../ROADMAP_V2.md)。
