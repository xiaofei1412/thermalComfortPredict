# thermalComfortPredict

面向热舒适与节能的楼宇世界模型及自主控制研究项目。

项目起源于 2023 年在西交利物浦大学参与的智慧楼宇研究，现有资产包括热环境预测脚本与 ASHRAE 格式数据。后续以公开数据和软件仿真为基础，逐步建立热感觉与冷热偏好预测、建筑动态建模、受约束控制及真实数据迁移验证流程。

**当前状态：里程碑 1 尚未开始。** 现有算法为历史基线，尚未完成新的复现与评测。公开运行日志用于离线模型验证；真实楼宇闭环控制效果需要另外验证。

## 项目入口

- [PROJECT.md：项目总览、里程碑与阶段结果](PROJECT.md)
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
├── legacy/                 # 保留原算法，作为历史基线
├── data/
│   ├── README.md
│   ├── raw/                # 现有原始数据
│   └── processed/          # 现有历史处理数据
└── docs/
    └── development/        # 后续各阶段开发日志
```

## 开发方式

本地工作目录为 `D:\thermalComfortPredict`，Git 远端为 [xiaofei1412/thermalComfortPredict](https://github.com/xiaofei1412/thermalComfortPredict)，统一维护 `main` 分支。

可以在 VS Code 中打开项目根目录，通过内置源代码管理查看、提交与同步变更。各阶段完成后，保留开发日志，并在根目录 `PROJECT.md` 中简要更新结果、验证情况与限制。

## 参考基础

- [ASHRAE Global Thermal Comfort Database II](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II)
- [BOPTEST 建筑控制评测平台](https://ibpsa.github.io/project1-boptest/)
- [ThermBuild 数据集](https://fordatis.fraunhofer.de/handle/fordatis/486)

现存数据的准确版本、引用及许可将在数据复现阶段核对。仓库暂不声明覆盖第三方数据的统一许可证。
