# 复现入口与资产边界

本轮不追加模型实验。命令在源码根目录执行；支持源码目录运行，未提供独立wheel。Windows/Python3.14为本机验收目标，Linux轻量流程由CI检查。

| 层次 | 所需条件 | 可以复核的内容 |
| --- | --- | --- |
| 离线交付 | 源码候选、轻量依赖 | 内置真实轨迹、需求/动作、原生积分、官方KPI、C2封存门判定 |
| 实时规则 | 上述条件＋BOPTEST0.9.0/WSL Docker | 手工需求→规则动作→真实仿真响应；不涉及在线学习模型 |
| C2完整只读重算 | 原研究依赖＋本机证据包和原模型 | 原生数据、全部分数/响应/门重算；采集与拟合入口已封存 |
| 历史完整研究 | 原数据、各阶段配置/模型/冻结链及相应环境 | 原协议分阶段重建；不是启动网页即可全部复现 |

## 离线：推荐第一入口

```powershell
py -3.14 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements/workbench.lock.txt
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m scripts.verify_delivery
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_delivery_release.py -v
./scripts/start_workbench.ps1 -PythonPath ./.venv/Scripts/python.exe
```

运行器只需NumPy/Pandas及其已列明的传递依赖，不需要Torch、旧CSV或模型权重。样例解压后与B原始会话逐字节一致；验证输出应为5动作/155区间点、KPI最大差约3.47e-18、`c2_stop_decision_reproduced=true`。这是从已保存指标重建停止判定，不是重新训练C2模型。

轻量清单固定版本；不承诺所有平台wheel逐字节相同或无网络安装。源码候选zip排除`data/raw`、`data/processed`、历史权重、FMU和`outputs`；Git原仓库的既有CSV未被删除，来源链缺口仍保留。

## 实时规则（不是本轮新实验）

首次部署按[原仿真协议](experiments/simulation.md)安装WSL/Docker，再用`./scripts/setup_boptest.ps1 -Action up`。核心源码/FMU/镜像和storage构建源有固定摘要，provision的完整构建依赖仍未达到本轮干净容器验收条件。当前只检查既有部署状态/配置，不重建或中断正在运行的服务。

已有引擎在线时按[演示步骤](DEMO.md)新建原day63住宅，正常结束释放worker；不能多个客户端同时操作。离线回放不要求引擎在线。停止带18～35°C是工程中止条件，不是舒适/安全认证。

## 本地原始研究证据

`outputs/delivery-f/private-research-evidence.zip`及并列`.manifest.json`复制了本机研究目录与历史CSV，逐文件读取校验。它不提交GitHub，是同一磁盘的副本，**不是异地备份，也未确认完整数据公开分发权限**。

C2只读重算需原研究依赖；在独立副本恢复原路径后运行：

```powershell
# 环境安装见原基线协议；不要在轻量环境中直接运行完整研究测试。
.venv-cpu/Scripts/python.exe -m thermal_comfort.residential_c2 verify
.venv-cpu/Scripts/python.exe -m thermal_comfort.office_final_round check
```

归档内部的路径和SHA对应原始字节与原运行平台，恢复时不要改JSON换行或重命名冻结文件；原始Windows相对路径需在对应平台恢复。完整历史数据/权重包尚无经过验收的远程获取入口，不能宣称第三方已经能够一键重跑全部研究。

仓库维护者的完整回归仍使用原研究环境和历史CSV：`python -m unittest discover -s tests -v`。不要把这一命令写成轻量源码包已支持的承诺。

## 打包与缺口

维护者在Git工作区审查并提交/暂存交付文件后，运行`python -m scripts.package_delivery source --output outputs/delivery-f/thermal-rule-workbench-rc1.zip`。脚本取已审查的Git索引字节，避免Windows换行差异或未审查工作区修改混入，并从zip读回每个文件核对；输出须为新文件名。包外`.manifest.json`列出全部文件摘要和当前Git基础提交。私有证据包用`private`模式，禁止作为源码候选上传。最终用户运行已解压的工作台不需要Git。

发布条件见[来源说明](../THIRD_PARTY_NOTICES.md)和[F核查](../reports/delivery-f/REPORT.md)。当前不创建正式GitHub Release、不替所有者选择代码许可证，住宅与办公室封存不重启，校准/盲评窗口不消耗。
