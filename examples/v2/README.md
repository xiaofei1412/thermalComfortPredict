# 已封存开发轨迹的小型回放

来自BOPTEST0.9.0住宅单热区、原固定FMU/TMY；包含全部三开发窗口、V1规则/MPC/minimum_heat及SAC/PPO/TD3/规则残差SAC的所有登记种子，共45条完整7天轨迹。原始指标从30秒数据重算；`replay.json.gz`仅保存673个15分钟边界帧/条，显示数值保留6位小数，门判定使用原全精度指标。

`manifest.json`记录派生包SHA、每条原始summary及native/schedule/steps摘要、12层研究锁锚点。完整原生/预报、模型与学习状态保存在私有归档；该小样例不声称可单独重建原始研究，也不是重新评估或训练。

生成命令`python -m scripts.build_v2_delivery`只读取本机已封存数据，物理调用0；公共复核`python -m scripts.verify_v2_delivery`不需第三方包。演示入口见[DEMO_V2](../../docs/DEMO_V2.md)。

这是软件仿真数值，不是真实人员热舒适数据。第三方生成器/FMUs/天气条件及派生数值例外见[LICENSE_SCOPE](../../LICENSE_SCOPE.md)与[来源说明](../../THIRD_PARTY_NOTICES.md)，不因根MIT而统一重新授权。
