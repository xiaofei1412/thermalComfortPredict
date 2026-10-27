# 来源与发布条件

本仓库尚未由所有者确定根代码许可证，本次不代选MIT/BSD等许可，也不宣称统一开源授权。交付候选提供源码与证据供核查，正式再分发范围仍待确认。

| 资产 | 可确认来源/条件 | 本次处理 |
| --- | --- | --- |
| BOPTEST | IBPSA及贡献者，固定提交`9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9`；[原license.md](https://github.com/ibpsa/project1-boptest/blob/9b1610bf7a108826bb3d22c72bffd2d71d7bb0a9/license.md)说明修改版三条款BSD并有附加段落，service另有说明 | 依据本机固定Git对象核查；源码候选不打包其FMU、服务镜像或完整天气源；部署时从原来源获取 |
| 内置回放 | 本项目B阶段真实BOPTEST运行`0a2e84c3cb66`，只有数值轨迹/元数据；SHA与原代码来源见[样例清单](examples/workbench/manifest.json) | 无真实人员标签；不声称该数据许可等同本仓库代码许可或替代生成器的第三方条件 |
| 历史ASHRAE格式CSV | 既有107,584行本地快照；确切发行链尚未逐文件匹配，见[原核查](docs/experiments/data-provenance.md) | 原文件不改、不追加到源码候选zip；上游当前许可元数据不能替代本地历史链确认 |
| Python依赖 | [轻量版本清单](requirements/workbench.lock.txt)及原`pyproject.toml`，各包保留其自身许可 | 源码包不携带依赖二进制；通过包索引安装，未授予这些依赖统一许可 |
| 本地研究证据包 | 已有本机数据、模型、失败版本和元数据的复制 | 仅本机保存、未上传GitHub，不代表已完成异地备份或取得公开分发权 |

本次不修改上游许可、不替换历史CSV、不发布Docker镜像或上传完整研究档案。正式发行前仍需确认根代码许可及历史数据处理范围；源码候选的工程验收不替代这些确认。
