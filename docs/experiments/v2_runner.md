# V2-3：共同运行器工程验收

2026-10-28。本轮实现冻结[正式比较协议](v2_comparison.md)的共同运行器，不修改configs/v2_comparison.json、任何父锁或历史结果。正式训练仍未授权，留出/旧保护区/历史238均不进入。假环境的参数更新只作工程测试，不是住宅数据或节电成绩。

## 共用执行链

`scripts.run_v2_comparison register`核对父协议和源码，写入`formal_training_authorized=false`及空预算账本，不开始阶段时钟。`seed`在构造Client、加载模型或修改账本之前拒绝当前未授权登记；本轮没有开启授权的入口。以后明确启动时另行留存授权与源码登记，不改冻结父协议中的false。

`TrainingRunner`统一按六日期运行完整周期，训练回合固定664 actor步，12周期后才允许三个开发回合，不能用早期检查点进入开发；每回合新环境、同一算法模型及累计经验，原8步前缀计分但不进经验。观测、动作投影、奖励、保护、12小时预报、原生评分与worker权限继承冻结SACEnv/CampaignClient；共同适配层只改算法标识、动作留痕和累计预算接入，不改旧源文件。

SAC/TD3采样层保留SB32.9.2的一环境/Box/无SDE随机调用顺序，保存随机起步或策略原建议、探索噪声、噪声后建议、库裁剪和回放动作。PPO保存原高斯动作、库裁剪动作、缓冲动作与log-prob；物理层另存actor投影、实际命令、测量设定和回退原因。日志不额外调用随机策略或噪声。确定性PPO开发动作记录裁剪前均值，且不采样。

PPO在任何reset前检查664/83/10设置；每回合结束核对累计actor、epoch和实际优化器步，确认完整rollout已消费。SAC/TD3也核对实际actor/critic/熵优化器步及有限终止/非截断经验，不能只看_n_updates。动作日志和96步曲线窗口跨回合累计，并保存字节偏移及前缀SHA。

## 预算与时间

`BudgetLedger`为OS文件锁保护的共享JSON账本，临时文件/fsync/replace提交更新。请求经过冻结worker/日期/预报权限检查后，在真正发送前预扣attempt；失败不退，未知advance不重发。单种子/全阶段上限仍为原协议75/675初始化、50,400/453,600实际推进；最多3个同算法worker，顺序SAC→PPO→TD3及种子11/23/37，重复启动拒绝。

种子原时钟从CLI早期进程入口记录，涵盖依赖/模型加载、控制、更新、导出和检查点；阶段原时钟从第一次select开始。wall/monotonic与已记最大elapsed共同限制，回拨wall、重新构造账本或保存检查点均不能重置；协议字典原地改变也拒绝。每步、更新后、导出后、检查点校验/flush后和最终结果写盘后复查时限。结果先写partial，最后核对时间/完整调用量才标记完成；过期或不足步数判失败，完整结果不发布。cleanup stop在故障/超时后仍允许，不解锁新控制动作。

时限是调用前后及边界检查，不承诺中断正在阻塞的系统调用。进程退出/服务丢失的真实故障恢复未在本轮测试；没有自动重启、预算退款或替补种子。

## 完整检查点

仅释放worker、完成六回合周期及所有更新/导出后保存。独占`cycle-NN.partial`写入模型/优化器、SAC/TD3经验池、绑定RNG、回调状态、原累计预算/时钟及完整清单；关联每个已完成回合的原生数据、日程、预报、观测/终止观测、奖励动作与官方KPI导出SHA。缺文件/终止观测/官方KPI、非完整回合或不符计数均拒绝。

在临时目录完成实际加载、权重/目标网络/优化器/熵张量、全部回放数组与终止位的比较，校验源码/协议/运行时和日志前缀。校验加载会重设随机流，结束后最后恢复live模型的RNG，不能污染后续训练。文件fsync、原时钟复查及最终清单写盘后，才在同卷rename为`cycle-NN`。失败partial保留，不冒充完整包；缺最终包不回退旧包。只删除SHA验证过的旧完整目录，保留最新两包、全部清单及失败partial。

`inspect_bundle`与`check_rl_checkpoint`只供读档验收，既不连接FMU，也不授权训练续跑。当前协议无计划进程重启/自动续跑，PPO未消费rollout和FMU内部状态不能靠本包恢复。原子发布的文件/进程错误路径经过注入，未做突然断电/文件系统损坏测试。

## 本轮有限验证与复核

使用正式网络、超参、503维形状、664步回合和六日期顺序，分别对SAC/PPO/TD3执行一个合成周期、种子11；测试账本仅缩窄owner登记为本算法/本种子，计数/时间上限不改。夹具没有楼宇动力学，KPI由合成信号生成。三算法各3,984个合成actor/4,032个虚拟advance，周期结束即停止，不能当正式种子完成。单测另用假BOPTEST客户端核对真实物理适配层的投影/测量/回退链。

注入缺RNG/回调/预算、动作日志篡改、校验失败、rename失败和保存耗时超限；验证没有错误完整包。另以3个全新进程争用账本计数；完整包在6个全新进程中重载并复核随机探针。生产`seed`拒绝测试证明当前没有Client或账本控制调用。完整回归继续覆盖原冻结接口/计分/12小时边界；本轮不查询真实引擎。

已有独立SAC依赖环境时，可在全新输出目录重做工程验收，耗时/磁盘取决于机器，不消耗FMU预算：

```powershell
outputs/v2-sac/env/Scripts/python.exe -m unittest discover -s tests -v
outputs/v2-sac/env/Scripts/python.exe -m scripts.verify_rl_runner --output outputs/v2-runner/acceptance-recheck
outputs/v2-sac/env/Scripts/python.exe -m scripts.check_rl_checkpoint verify --archive outputs/v2-runner/acceptance-recheck --output outputs/v2-runner/bundle-rng-recheck
```

大型合成检查点/明细留在outputs/v2-runner，小型审计在[本轮报告](../../reports/v2_runner/REPORT.md)。正式预算、数据角色、奖励、门限、种子与最终选择不因工程验收改变。下一步明确启动后，在原预算内核对副本/FMU资产与首回合共同前缀、原生KPI和真实成本；失败按既定停止，不追加预跑/日期/配置或启用留出。
