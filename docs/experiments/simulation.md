# BOPTEST 单热区与规则控制协议

本阶段只完成仿真环境、规则控制和轨迹采集。使用操作温度约束衡量温度越界，不把它称为主观热舒适度或 PMV；不接入前一阶段尚未校准的人员预测模型。

## 固定环境与场景

- 官方 BOPTEST **v0.9.0**，源码提交 `9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9`；worker/web 镜像摘要固定在 `infra/boptest.compose.yml`。
- `bestest_hydronic_heat_pump`：192 m² 单热区住宅、地板辐射供暖、空气源热泵，Brussels 气象。`reaTZon_y` 是操作温度，单位 K，不是空气温度。
- 使用官方 `peak_heat_day`（年内第 16～30 天）和 `typical_heat_day`（第 108～122 天）；官方 scenario 初始化各自包含 **7 天内置控制预热**，随后评测 14 天。时间原点是模型年初，不声称具体公历年份。
- 控制周期 900 s；模型原生输出约 30 s，保留事件点和重复时间点。电价场景 constant；无天气预测噪声。每个控制器重新选择 FMU 并独立初始化。
- 配置固定在 `configs/simulation.json`。规则参数在正式运行前确定，未使用测试结果调参。

## 本地部署

Windows 使用现有 WSL Ubuntu-24.04，WSL 内需要 Docker Engine 与 Compose v2。本次通过 Ubuntu 软件源安装 `docker.io docker-compose-v2`，无需 Docker Desktop 或 GPU。首次构建需要网络和数 GB 磁盘空间。

```powershell
.\scripts\setup_boptest.ps1 up
.\scripts\setup_boptest.ps1 status
.\.venv-cpu\Scripts\python.exe -m thermal_comfort.simulation --days 0.25 --output outputs/simulation-smoke
.\.venv-cpu\Scripts\python.exe -u -m thermal_comfort.simulation --output outputs/simulation
.\.venv-cpu\Scripts\python.exe -u -m thermal_comfort.simulation --days 1 --output outputs/simulation-repeat
.\.venv-cpu\Scripts\python.exe scripts/verify_simulation.py --output outputs/simulation --repeat outputs/simulation-repeat
.\.venv-cpu\Scripts\python.exe scripts/report_simulation.py --output outputs/simulation
# 停止服务；不会删除已导出的 outputs/ 轨迹
.\scripts\setup_boptest.ps1 down
```

Python 环境安装沿用 [预测基线说明](baseline.md)，本阶段无新增 Python 依赖。输出目录必须不存在，避免覆盖此前实验。`up` 会核对上游源码与存储服务源码归档的 SHA-256；模型仅向本机 `127.0.0.1:8080` 开放，Redis/存储服务不映射主机端口。Compose 中的账户均为本地演示专用固定值，不能用于公网部署。

上游旧 MinIO Docker Hub/Quay 镜像与官方二进制归档实际不可用，故从 MinIO/mc 官方 GitHub 固定提交源码构建：`f19c534b9f457773dcd043d977433e1a71525c3b` / `1e78af443bf0443eb177f26421fe081f08d83dc5`。使用固定 Go 构建镜像；依赖由上游 go.mod/go.sum 约束。BOPTEST 和 MinIO 源码保存在被忽略的 outputs/，不将第三方代码或镜像复制进 Git。BOPTEST 保留其上游许可，MinIO/mc 为 AGPL-3.0；部署或分发衍生服务时应阅读各自源仓库许可证。本项目未声称覆盖这些组件的统一许可。

服务存储是临时的，重建后重新上传固定 FMU；需要保留的实验均已导出到 outputs/。上传辅助容器沿用上游 Dockerfile，其系统包及 HTTP 依赖未全部锁定；正式仿真 worker/web 和模型已经固定。源码固定不意味着整个辅助构建链逐字节可重建。

## 控制器与信息边界

| 控制器 | 每步动作 | 可用信息 |
| --- | --- | --- |
| embedded | `oveTSet_activate=0`，沿用模型原有设定温度与底层 PI | 官方内置逻辑 |
| rule | `oveTSet_activate=1`；有人时 `LowerSetp + 0.5 K`，无人时 `LowerSetp + 3 K`，裁剪到 API 允许范围 | 当前时刻舒适温度下界与占用 |

只覆盖 `oveTSet_u`，保留原模型的热泵、风机、水泵底层控制。导出全年程外生变量是为对齐离线数据与校验 KPI；规则决策只访问当前行，不读取未来天气或未来占用。两种上层策略有不同启发式，属于控制基线对照，不是隔离单一因素的算法优劣证明。

## 工件与单位

每个场景/控制器目录包含：

- `api_metadata.json`：模型输入、输出、预测变量及原始单位/范围。
- `steps.jsonl`：逐步原始 `before → action → after`，每步写盘；失败时保留已收到的响应。状态变更请求超时不重试，以免动作被执行两次；finally 释放 testid。
- `native.csv.gz`：原生时序，保留 K、W、s 及实际覆盖动作；数据完整性通过摘要检查。
- `exogenous.csv.gz`：30 s 外生采样；气温 K、太阳辐射 W/m²、价格 EUR/kWh、排放 kgCO₂/kWh、温度界限 K、占用人数。非天气时序按官方零阶保持对齐，不能在线性插值中提前引入日程变化。
- `transitions.csv.gz`：900 s 转移表；起止操作温度 °C、起始外温/辐射/占用/舒适区间、请求与实际应用设定温度 °C、覆盖标志、区间 kWh 与 K·h。embedded 的请求设定温度留空；实际生效值仍保存。区间开始边界的动作可能属于上一周期，只在边界之后核对本次动作。
- `api_kpis.json`、`summary.json`：官方指标、独立重算、轨迹 SHA-256、步数、时间范围、实际配置及耗时；根目录 `manifest.json` 记录服务与 Python/NumPy/Pandas/代码版本。

能耗是热泵 + 风机 + 水泵三项功率在原生时间网格的梯形积分，WS/3,600,000 转 kWh；不包含模型之外设备。KPI 能耗/成本/排放除以 192 m²。BOPTEST v0.9.0 内部能量换算常数为 `2.77778e-7`，比精确 SI 系数高约 8e-7；独立同时保存 SI 结果与官方兼容值。温度越界为上下界之外偏差的积分（K·h/zone），包含无人时段较宽温度界限，不等同仅有人时段的主观舒适度。

## 验证与使用限制

自动核对服务版本、温度/功率单位、控制步长、时间连续性、实际应用动作、原生采样边界、有限值及区间总能耗；独立重算能耗、温度越界、成本、排放并与 API KPI 对照（rtol/atol 各 1e-6）。离线单元测试覆盖单位换算、零阶保持、动作对齐、错误响应及超时不重试。真实 FMU 冒烟、正式四回合及重复验证的结果见 [报告](../../reports/simulation/REPORT.md)。CI 跑离线测试，不在共享云服务上启动长仿真。

这些轨迹来自两个场景下的确定性闭环控制，可用于下一阶段动态模型的管线开发；动作激励覆盖有限，不足以单独证明模型可泛化到任意控制动作。后续必须单独设计安全激励采集，并按独立回合/时间和场景划分训练验证测试，防止重叠滑动窗口泄漏。这里没有 MPC、强化学习或真实楼宇闭环结果。

## 官方依据

- [BOPTEST 入门与部署](https://ibpsa.github.io/project1-boptest/docs-userguide/getting_started.html)
- [REST API](https://ibpsa.github.io/project1-boptest/docs-userguide/api.html)
- [单热区热泵案例](https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/index.html)
- [固定版本源码](https://github.com/ibpsa/project1-boptest/tree/v0.9.0)
- [MinIO 固定源码](https://github.com/minio/minio/tree/f19c534b9f457773dcd043d977433e1a71525c3b)、[mc 固定源码](https://github.com/minio/mc/tree/1e78af443bf0443eb177f26421fe081f08d83dc5)
