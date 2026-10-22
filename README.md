# thermalComfortPredict

面向热舒适与节能的楼宇世界模型及自主控制研究项目。

项目起源于 2023 年在西交利物浦大学参与的智慧楼宇研究，现有资产包括热环境预测脚本与 ASHRAE 格式数据。后续以公开数据和软件仿真为基础，逐步建立热感觉与冷热偏好预测、建筑动态建模、受约束控制及真实数据迁移验证流程。

**当前状态：预测、单热区仿真、灰箱残差模型与短时 MPC 基线链路已贯通。** 已有非负耗电、独立动作干预、经验裕量及规则回退对照；耗电下降伴随温度越界略增，可靠性和人员需求接入仍待完善。

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
