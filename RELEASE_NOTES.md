# 源码交付候选 rc2 · 2026-10-27

- 独立源码/新完整环境198项回归；固定30项Windows CPU依赖及6项Linux provision依赖SHA，辅助镜像无缓存构建通过。
- 独立引擎/网络、3步真实规则操作、网页回放和隔离down/up通过，原服务保留；这是部署验收，无新模型实验。
- 本机档案4,164文件恢复/校验，C2与办公室只读复核通过；F02按实测Windows/WSL范围关闭。
- F01代码许可/历史CSV链与F03异地保存仍待[所有者决定](docs/release-decisions.md)，不创建正式Release。rc2不覆盖rc1，研究版本与封存条件不变。

见[部署指南](docs/DEPLOYMENT.md)及[验收报告](reports/deployment-f2/REPORT.md)。

---

# 源码交付候选 rc1 · 2026-10-27

- 交付规则工作台、真实压缩回放样例、十分钟演示及分层复现说明。
- 轻量Python依赖与离线CI独立于Torch/历史CSV/权重；可重算原生轨迹和C2停止判定。
- 保留历史住宅有限成功、办公室与C2负结果，不追加模型、仿真回合或新校准/盲评。
- 制作本机私有证据副本及逐文件SHA清单；未上传原始研究档案。

这是源码候选，不是已统一授权的正式V1.0发行；代码许可/CSV准确发行链、完整新容器构建及异地资产获取仍有缺口。wheel不支持，按源码目录运行。当前研究包版本仍是`pyproject.toml`中的0.1.0，候选名不改变旧模型/协议版本。

入口：[交付报告](DELIVERY.md) · [演示](docs/DEMO.md) · [复现](docs/REPRODUCE.md) · [核查结果](reports/delivery-f/REPORT.md)。
