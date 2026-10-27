# thermalComfortPredict

面向人员需求的楼宇动态建模、预测控制与交互仿真研究项目。

**第二版开发：[同一住宅环境的强化学习与两代比较](ROADMAP_V2.md)。** [三算法共同运行器工程验收](reports/v2_runner/REPORT.md)已完成：PPO步数/动作留痕、完整检查点和累计预算/时钟通过，294项回归及6新进程随机探针通过。本轮只用合成夹具，正式训练/真实FMU/留出0；原SAC首轮无稳定整体优势结论保持。下一步明确启动冻结预算，不追加场景或预跑；工作台/V1交付保持。

项目起源于 2023 年在西交利物浦大学参与的智慧楼宇研究，现有资产包括热环境预测脚本与 ASHRAE 格式数据。后续以公开数据和软件仿真为基础，逐步建立热感觉与冷热偏好预测、建筑动态建模、受约束控制及真实数据迁移验证流程。

**当前状态：本轮办公室迁移攻关已结束，负结果已封存。** 最终源结构因果候选通过60/72开发门和2/2响应，长时能源、温度回归及供暖覆盖仍未全过；按预设停止，维持规则，新校准/盲评均为0次。部分离线预测改善不代表控制升级，固定仿真适用边界已记录；人员主观需求模型闭环、系统学习控制和真实迁移仍未完成。

**可以亲自体验了：[本地房间工作台V0.1](docs/workbench.md)。** 连接真实BOPTEST，支持两位虚拟住户需求的确定性目标映射、冲突裁决、撤销/过期、按需求/手动/演示规则步进、保存回放及同起点规则对照。需求是供暖设定意图，不生成真实热感觉标签；尚未接入MPC或人员预测模型。协议与验证见[B阶段报告](reports/demand-b/REPORT.md)。

[C1预注册](reports/residential_c/REPORT.md)及[C2开发收尾](reports/residential_c2/REPORT.md)已归档：完成464动作需求补采，稳态覆盖与能源门未全过，按约定封存负结果；重拟合/新校准/盲评0次，规则工作台继续可用。

**当前交付候选**：[交付报告](DELIVERY.md) · [十分钟演示](docs/DEMO.md) · [分层复现](docs/REPRODUCE.md) · [发布说明](RELEASE_NOTES.md)。首次离线回放无需BOPTEST、历史CSV或权重，在源码根目录运行：

```powershell
py -3.14 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements/workbench.lock.txt
./.venv/Scripts/python.exe -m scripts.verify_delivery
./scripts/start_workbench.ps1 -PythonPath ./.venv/Scripts/python.exe
```

打开终端打印的URL（默认8765），历史列表选`0a2e84`。新建实时仿真才需要[部署BOPTEST](docs/experiments/simulation.md)。[安装与边界](docs/workbench.md) · [F验收/发布缺口](reports/delivery-f/REPORT.md)。[F02独立完整部署与本机恢复](reports/deployment-f2/REPORT.md)已按Windows/WSL范围通过；自有代码已按所有者确认采用[MIT](LICENSE)，[第三方/历史数据例外](LICENSE_SCOPE.md)独立保留；所有者完成百度网盘上传/重新下载后，独立恢复与只读复核通过，F03关闭，本轮按源码目录＋小样例＋私有灾备范围交付验收结束，见[验收报告](reports/release-final/REPORT.md)。完整研究包不公开，C2/办公室负结果保持。

## 项目入口

- [V1.0收尾计划：阶段、验收、停止条件与持续更新](ROADMAP.md)
- [交付范围、十分钟流程、资产复现清单与发布缺口](docs/delivery-scope.md)
- [交互工作台：启动、操作、保存与回放](docs/workbench.md)
- [第一版简要交付报告：做了什么、原因与后续方向](reports/DELIVERY_V1.md)
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
├── workbench/             # 本地交互房间网页，无前端构建依赖
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
├── reports/workbench/     # 交互工作台验收与能力边界
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
