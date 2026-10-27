# 干净部署与验收（Windows / WSL amd64）

2026-10-27。[本次实测](../reports/deployment-f2/REPORT.md)覆盖新源码目录、新venv、独立容器组及辅助镜像无缓存构建；共享现有Windows/WSL、Docker守护进程和固定核心镜像层，未验证全新物理机或其他平台。无需新增模型实验。

## 前提与最小实时工作台

已安装Windows x64 CPython3.14、Git、PowerShell、WSL发行版`Ubuntu-24.04`及其中的Docker Engine/Compose；可访问PyPI、GitHub/GHCR、Docker及Go依赖源。首次安装/下载耗时不计入十分钟演示。源码从仓库克隆或源码候选zip解压，命令均在该**独立源码根目录**执行。

```powershell
py -3.14 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements/workbench.lock.txt
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m scripts.verify_delivery
# ProjectName和Port均应使用未占用值，避免碰到原服务。
./scripts/setup_boptest.ps1 -Action up -ProjectName thermal-accept-f2 -Port 8081 -NoCache
./.venv/Scripts/python.exe -m scripts.serve_workbench --port 8770 --engine-url http://127.0.0.1:8081
```

浏览器打开<http://127.0.0.1:8770>。`serve_workbench`是引擎地址适配入口，复用原后端/页面，保留冻结研究源码；只接受显式端口的本机HTTP引擎。原`start_workbench.ps1`默认引擎8080的行为保持。每组引擎只有一个worker，禁止多个客户端同时控制同一组。

脚本从原来源校验BOPTEST提交/FMU、MinIO/mc源码摘要和干净Git树；storage与provision独立构建。provision固定Python基础镜像摘要和6个带SHA的运行包，`--require-hashes`安装。`.dockerignore`限制其构建上下文，仅允许Dockerfile/依赖输入，不将本机档案发送到Docker。核心Redis/worker/web均固定镜像摘要；`-NoCache`只禁用构建层缓存，不会抹除已有基础镜像。

默认仍是`thermal-boptest`/8080；`up/status/down`必须始终传相同的ProjectName、Port及Distribution（若不是默认）。参数只作用于指定Compose项目，不清理其他项目、镜像或卷。

```powershell
./scripts/setup_boptest.ps1 -Action status -ProjectName thermal-accept-f2 -Port 8081
# 先在网页“结束并保存会话”，再在工作台终端Ctrl+C；最后停止自己的引擎。
./scripts/setup_boptest.ps1 -Action down -ProjectName thermal-accept-f2 -Port 8081
# 再次up会重新provision；历史会话仍保存在源码目录outputs/workbench/sessions。
./scripts/setup_boptest.ps1 -Action up -ProjectName thermal-accept-f2 -Port 8081
```

这是本地开发配置，固定凭据仅用于隔离仿真；不是公网/多人生产服务。

## 一次可选的工程烟测

先使用空闲的新工作台，不能接管已有会话。下述命令**会**新建已使用的day63、执行3个15分钟动作并正常结束；它不是离线验证，不是新研究日期或能效评测。只在部署验收需要时执行一次，不作为每次启动的自动任务。

```powershell
./.venv/Scripts/python.exe -m scripts.verify_deployment_live --url http://127.0.0.1:8770 --output outputs/deployment-smoke.json
```

预期：未裁决冲突409且不推进；确认A后动作执行/请求过期；恢复基准、保持请求；3动作/93区间点、官方四项KPI核对、会话保存、释放worker。随后在网页历史列表选择该会话并拖到末步，检查HISTORY、3/3、导出和禁用实时控制。

## 完整研究运行依赖与旧证据恢复

Windows amd64 / CPython3.14 CPU使用30包SHA清单，其他平台不套用该平台锁：

```powershell
py -3.14 -m venv .venv-research
./.venv-research/Scripts/python.exe -m pip install --require-hashes -r requirements/research-windows-py314.lock.txt
./.venv-research/Scripts/python.exe -m pip check
```

如需离线安装，可事先在相同平台用`pip download --only-binary=:all: --require-hashes -r requirements/research-windows-py314.lock.txt -d outputs/research-wheels`保存包，再用`pip install --no-index --find-links outputs/research-wheels --require-hashes -r ...`。本轮已实际在第二个空venv从wheel目录安装，全部30包位于新venv；不是仅导出旧环境清单。provision的Linux CPython3.11锁只用于它自己的容器。

源码候选不含历史CSV/权重/完整轨迹。持有可信的私有证据zip与manifest后，先核对整包SHA，在独立目录恢复原相对路径，逐文件核对大小/SHA；不要覆盖现有不同内容的历史，不要转换换行。原始Windows相对路径按原平台使用。恢复后可做以下**只读**复核，不执行训练/采集入口：

```powershell
./.venv-research/Scripts/python.exe -m unittest discover -s tests -v
./.venv-research/Scripts/python.exe -m thermal_comfort.residential_c2 verify
./.venv-research/Scripts/python.exe -m thermal_comfort.office_final_round check
```

完整历史重新训练仍需各阶段协议/数据用途和冻结链，不是本轮验收目标。私有包目前仍仅本机保存，异地恢复尚未开展，见[待决清单](release-decisions.md)。
