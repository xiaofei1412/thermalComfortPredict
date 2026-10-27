# 来源与发布条件

所有者已确认自有代码可授权并选择[MIT](LICENSE)，适用范围及例外见[LICENSE_SCOPE](LICENSE_SCOPE.md)。第三方代码、数据、生成器、依赖和研究工件不由根MIT统一授权；公开交付以审查过的源码包为准。

| 资产 | 可确认来源/条件 | 本次处理 |
| --- | --- | --- |
| BOPTEST | IBPSA及贡献者，固定提交`9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9`；[原license.md](https://github.com/ibpsa/project1-boptest/blob/9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9/license.md)说明修改版三条款BSD并有附加段落，service另有说明 | 依据本机固定Git对象核查；源码候选不打包其FMU、服务镜像或完整天气源；部署时从原来源获取 |
| 内置回放 | 本项目B阶段真实BOPTEST运行`0a2e84c3cb66`，只有数值轨迹/元数据；SHA与原代码来源见[样例清单](examples/workbench/manifest.json) | 无真实人员标签；不声称该数据许可等同本仓库代码许可或替代生成器的第三方条件 |
| 历史ASHRAE格式CSV | 既有107,584行本地快照；确切发行链尚未逐文件匹配，见[原核查](docs/experiments/data-provenance.md) | 原文件不改、不追加到源码候选zip；上游当前许可元数据不能替代本地历史链确认 |
| Python依赖 | [轻量版本清单](requirements/workbench.lock.txt)、[Windows CPU完整SHA锁](requirements/research-windows-py314.lock.txt)、[Linux provision SHA锁](requirements/provision-linux-amd64-py311.lock.txt)及原`pyproject.toml`，各包保留其自身许可 | 源码包不携带依赖二进制；通过包索引安装，未授予这些依赖统一许可 |
| 本地研究证据包 | 已有本机数据、模型、失败版本和元数据的复制 | 所有者指定百度网盘私有加密灾备；准备归档不等于真实远程下载验收，仍不公开完整研究包 |

本次不修改上游许可、不替换历史CSV、不发布Docker镜像或公开完整研究档案。历史CSV链仍未核实，排除于审查过的源码包；原Git/历史副本保留，不把GitHub自动源码包当作本项目审查发行包。

F02部署验收保持；自有代码MIT决定已落实，历史CSV/第三方范围独立保留。百度网盘私有归档/恢复见[交接](docs/BACKUP.md)，远程恢复未通过前F03仍不关闭。
