# 内置真实仿真回放

`0a2e84c3cb66.json.gz`是B阶段原始会话JSON的无损gzip压缩，解压后的SHA-256记录在`manifest.json`。不是合成室温、模拟截图或新盲评；原会话内的源码/页面摘要保持原执行版本，不改写成F版本。

- BOPTEST0.9.0 `bestest_hydronic_heat_pump`、day63、7天预热、5个15分钟动作、155个含重复边界的30秒点。
- 包含更暖/更凉冲突、撤销、精确过期、不可行26°C、显式裁决和回基准；所有角色是虚拟控制意图，不是实际人体观察。
- 只收录数值轨迹、已用小段外生数据和来源元数据，不收录FMU、完整天气源文件、ASHRAE记录或训练权重。生成器许可与代码许可分别见[第三方说明](../../THIRD_PARTY_NOTICES.md)。

启动器会用`scripts.install_demo`验证压缩及原始摘要后安装至`outputs/workbench/sessions/`，已有不同内容的同ID历史不会被覆盖。历史选择`0a2e84`即可回放；不会连接或推进仿真引擎。`python -m scripts.verify_delivery`重算原生KPI及C2封存门判定。
