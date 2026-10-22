# thermalComfortPredict

面向热舒适与节能的楼宇世界模型及自主控制研究项目。

项目起源于 2023 年在西交利物浦大学参与的智慧楼宇研究，现有资产包括热环境预测脚本与 ASHRAE 格式数据。后续以公开数据和软件仿真为基础，逐步建立热感觉与冷热偏好预测、建筑动态建模、受约束控制及真实数据迁移验证流程。

**当前状态：里程碑 1 进行中，预测基线、BOPTEST 单热区与规则控制采集已完成。** 已有固定切分的预测结果，以及两场景、两控制器的 14 天仿真对照和轨迹；动态模型与 MPC 待开始。

## 项目入口

- [PROJECT.md：项目总览、里程碑与阶段结果](PROJECT.md)
- [正式预测基线结果](reports/baseline/REPORT.md)
- [安装、运行与评测协议](docs/experiments/baseline.md)
- [单热区仿真与规则控制结果](reports/simulation/REPORT.md)
- [仿真部署与轨迹采集协议](docs/experiments/simulation.md)
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
├── thermal_comfort/       # 预测基线与仿真采集
├── tests/                 # 数据、模型与仿真协议检查
├── scripts/               # 部署、结果验证与报告
├── reports/baseline/      # 提交到 Git 的紧凑结果与证据
├── reports/simulation/    # 仿真指标、轨迹样例与验证证据
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
