# thermalComfortPredict

面向热舒适与节能的楼宇世界模型及自主控制研究项目。

项目起源于 2023 年在西交利物浦大学参与的智慧楼宇研究，现有资产包括热环境预测脚本与 ASHRAE 格式数据。后续以公开数据和软件仿真为基础，逐步建立热感觉与冷热偏好预测、建筑动态建模、受约束控制及真实数据迁移验证流程。

**当前状态：本轮办公室迁移攻关已结束，负结果已封存。** 最终源结构因果候选通过60/72开发门和2/2响应，长时能源、温度回归及供暖覆盖仍未全过；按预设停止，维持规则，新校准/盲评均为0次。部分离线预测改善不代表控制升级，固定仿真适用边界已记录；人员需求接入、系统学习控制和真实迁移仍未完成。

## 项目入口

- [PROJECT.md：项目总览、里程碑与阶段结果](PROJECT.md)
- [正式预测基线结果](reports/baseline/REPORT.md)
- [安装、运行与评测协议](docs/experiments/baseline.md)
- [单热区仿真与规则控制结果](reports/simulation/REPORT.md)
- [仿真部署与轨迹采集协议](docs/experiments/simulation.md)
- [激励与动态预测结果](reports/dynamics/REPORT.md)
- [动态模型协议与 MPC 预测接口](docs/experiments/dynamics.md)
- [灰箱残差、动作响应与 MPC 对照结果](reports/mpc/REPORT.md)
- [MPC 运行协议与保护边界](docs/experiments/mpc.md)
- [舒适恢复、空载耗电与天气误差结果](reports/recovery_mpc/REPORT.md)
- [恢复MPC与天气压力测试协议](docs/experiments/recovery_mpc.md)
- [暖天气、蓄热消融与提前停热结果](reports/solar_mpc/REPORT.md)
- [新时间切分、最小供暖与复现协议](docs/experiments/solar_mpc.md)
- [冻结方案的跨天气与跨建筑结果](reports/generalization/REPORT.md)
- [零目标训练泛化验证协议](docs/experiments/generalization.md)
- [办公室少样本校准与独立时间留出结果](reports/office_calibration/REPORT.md)
- [办公室语义、末端、能源适配与冻结协议](docs/experiments/office_calibration.md)
- [冷暖季末端、未触碰校准与新气候结果](reports/terminal_dynamics/REPORT.md)
- [开发锁、工况裕量和场景隔离协议](docs/experiments/terminal_dynamics.md)
- [原生末端与联合训练开发结果](reports/native_terminal/REPORT.md)
- [分季节、分能源能力门协议](docs/experiments/native_terminal.md)
- [PI、切换、风量与焓差限定诊断](reports/terminal_diagnostic/REPORT.md)
- [固定单因素消融与停止条件](docs/experiments/terminal_diagnostic.md)
- [办公室最终一轮与负结果封存](reports/office_final_round/REPORT.md)
- [办公室适用边界与结束决定](reports/office_final_round/BOUNDARY.md)
- [最终有限修复停止协议](docs/experiments/office_final_round.md)
- [阶段开发日志约定](docs/development/README.md)
- [数据说明](data/README.md)
- [历史代码说明](legacy/README.md)
- [开发协作约定](AGENTS.md)

## 目录

```text
thermalComfortPredict/
├── README.md
├── PROJECT.md
├── AGENTS.md
├── pyproject.toml          # 固定核心依赖
├── configs/               # 正式实验配置
├── thermal_comfort/       # 预测、仿真采集与动态模型
├── tests/                 # 数据、模型与仿真协议检查
├── scripts/               # 部署、结果验证与报告
├── reports/baseline/      # 提交到 Git 的紧凑结果与证据
├── reports/simulation/    # 仿真指标、轨迹样例与验证证据
├── reports/dynamics/      # 隔离切分、多步误差与保护证据
├── reports/mpc/           # 灰箱、动作干预与短时控制对照
├── reports/recovery_mpc/  # 恢复、停机耗电与天气误差证据
├── reports/solar_mpc/     # 暖天气新切分、蓄热消融与提前停热
├── reports/generalization/ # 冻结模型、跨天气与跨建筑验证
├── reports/office_calibration/ # 目标校准、预测裕量与独立控制对照
├── reports/terminal_dynamics/ # 冷暖季末端、未触碰校准与新气候失败证据
├── reports/native_terminal/ # 原生辨识、联合训练与能力门负结果
├── reports/terminal_diagnostic/ # 开发末端诊断与非部署oracle消融
├── reports/office_final_round/ # 最终有限修复、失败门与办公室边界封存
├── infra/                 # 本地 BOPTEST 部署
├── legacy/                 # 保留原算法，作为历史基线
├── data/
│   ├── README.md
│   ├── raw/                # 现有原始数据
│   └── processed/          # 现有历史处理数据
└── docs/
    ├── experiments/       # 实验协议与数据来源核查
    └── development/       # 各阶段开发日志
```

## 运行基线

按 [安装说明](docs/experiments/baseline.md) 创建环境并安装依赖后，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m thermal_comfort run
.\.venv\Scripts\python.exe -u scripts/verify_baseline.py
```

默认完整工件保存在 `outputs/baseline/`，不提交大型模型或逐行预测。仅重建数据可以运行 `python -m thermal_comfort prepare`。

## 开发方式

本地工作目录为 `D:\thermalComfortPredict`，Git 远端为 [xiaofei1412/thermalComfortPredict](https://github.com/xiaofei1412/thermalComfortPredict)，统一维护 `main` 分支。

可以在 VS Code 中打开项目根目录，通过内置源代码管理查看、提交与同步变更。各阶段完成后，保留开发日志，并在根目录 `PROJECT.md` 中简要更新结果、验证情况与限制。

## 参考基础

- [ASHRAE Global Thermal Comfort Database II](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II)
- [BOPTEST 建筑控制评测平台](https://ibpsa.github.io/project1-boptest/)
- [ThermBuild 数据集](https://fordatis.fraunhofer.de/handle/fordatis/486)

现存数据采用固定本地快照，准确上游发行版尚未逐文件匹配；已确认的来源、单位与许可证据见 [数据来源核查](docs/experiments/data-provenance.md)。仓库不声明覆盖第三方数据的统一许可证。
