# 第二版复现层次与归档

本轮不追加模型实验。公开演示、门判定复核、原生数据重算和重新训练是不同层次；本交付只执行前三类中的已声明范围。

| 层次 | 条件 | 能核对什么 |
| --- | --- | --- |
| 公开源码与小演示 | Python3.14标准库 | 45条封存开发指标/原门、全部方法与种子、演示字节和12层研究锁；不等于重算全部原始数据 |
| 原生完整只读 | 私有V2证据、固定FMU/旧模型、精确CPU依赖 | V2-2基线、九个正式RL候选和三个残差候选的原生/KPI、权限/动作、最终包及规则锚 |
| 从头再学习 | 另登记试次、原引擎与预算授权 | 本次不执行；没有恢复旧试次、选择最佳种子或消耗留出 |
| 真实闭环 | 实际观测、执行硬件与新协议 | 本项目没有该证据，不由离线复核替代 |

公开入口：

```powershell
py -3.14 -m scripts.verify_v2_delivery
py -3.14 -m scripts.serve_v2_demo --port 8772
```

演示压缩样例不到1MB，包含45×673边界帧，原始指标仍为全精度。`scripts.build_v2_delivery`仅从本机封存数据派生展示，不训练、不连FMU；修改派生显示不能追溯改变原报告或温度门。更早V1历史留出成绩不能直接与不同协议RL相减，必须用V2-2共同基线。

## 固定环境与独立恢复

Windows amd64 / Python3.14原V2环境的34项实际包版本和wheel SHA保存在`requirements/v2-windows-py314.lock.txt`。私有归档包含这些wheel；无需把venv或editable项目路径打包。Python解释器本身不在包内，需已有受支持解释器。其他平台公开标准库回放可用，完整数值复核以验收平台为准，不承诺跨版本逐字节重训一致。

使用[备份说明](BACKUP_V2.md)从可信收据恢复到新目录。以下命令在恢复后的`restored-source`根目录执行，解释器路径指向新建独立环境：

```powershell
py -3.14 -m venv .venv-v2
./.venv-v2/Scripts/python.exe -m pip install --no-index --find-links wheels --require-hashes -r requirements/v2-windows-py314.lock.txt
./.venv-v2/Scripts/python.exe -m pip check
./.venv-v2/Scripts/python.exe -m scripts.verify_v2_delivery
./.venv-v2/Scripts/python.exe -m unittest discover -s tests -p "test_v2*.py" -v
New-Item -ItemType Directory -Path outputs/v2-delivery/verification -Force
./.venv-v2/Scripts/python.exe -m scripts.verify_v2_baselines outputs/v2-2/live-registered --output outputs/v2-delivery/verification/baselines.json
./.venv-v2/Scripts/python.exe -m scripts.audit_v2_safe_formal outputs/v2-safe-formal/live-registered --output outputs/v2-delivery/verification/main-audit.json
./.venv-v2/Scripts/python.exe -m scripts.audit_v2_residual_safe outputs/v2-4-safe/live-registered --output outputs/v2-delivery/verification/residual-audit.json
```

它们只读取已关闭档案、写新核验输出，不需要开启Docker；完整预报/原生数据量较大，可能需要多分钟。V2专项目录不带历史ASHRAE CSV，因此不把源码zip单独运行全仓库研究测试当成承诺；V1完整研究恢复另按原私有包。恢复出的BOPTEST文件只够只读资产核验，不是完整Git克隆，不能直接拿来setup或宣称引擎已重建。

最终模型、优化器、经验池、随机状态和日志清单都保留，首个SAC工程链路的缺口/续跑限制也保留；“包可读取”不等于支持真实FMU续跑。全部训练入口保持关闭，本轮公开命令没有网络/控制调用。

## 源码候选与证据范围

审查的源码zip来自Git索引，排除CSV、历史脚本、完整模型/FMUs、outputs与密钥。完整私有档案包含V2-0～4的闭合试次、原失败、元数据/日志及住宅冻结模型；不包含环境目录或运行中FMU状态。源码候选在最终恢复收据提交前固定，因此最终验收记录由Git另行保存，避免自引用SHA。

原始数据恢复、依赖安装、源码导入位置和完整只读重算各自留证；本机副本不构成异地恢复。最新结果与缺口以[交付验收](../reports/v2_delivery/REPORT.md)为准。
