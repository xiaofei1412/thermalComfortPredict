# V2共享预算与失败片段：离线修复

本轮只修工程，不运行研究训练、真实FMU、开发评估或留出。原正式试次保持关闭，八层旧源锁不改。结果见[报告](../../reports/v2_io_repair/REPORT.md)。

## 修复边界

- 新增`thermal_comfort.rl_budget_safe.BudgetLedger`。公开读取、预扣、启动和结束共用同一侧车OS互斥锁；读取在锁内解析并关闭JSON句柄，只返回内存对象。已经持锁的变更调用私有读取，避免二次取锁。
- 新增`read_snapshot/write_snapshot`，可变监控JSON也按相同生命周期处理。写入沿用临时文件、flush、fsync、关闭、replace；不重试文件替换或仿真请求，不退还已预扣失败调用。AST回归核对原三项预算变更逻辑，仅内部读取入口不同。
- Windows使用非阻塞字节锁，5毫秒间隔、最多10秒等待；POSIX使用非阻塞flock。仅互斥获取可等待，超时传播为失败；只在成功取得锁后释放，句柄总在上下文退出时关闭。首次并发测试使用旧`LK_LOCK`遇到errno36，失败证据保留，未修改历史锁实现。
- 新增`rl_failure_export`，失败回合只保存已有动作、原生数据、预报和观测。没有完整`score`也可导出，不进入完整首门、不虚构终止观测/KPI/回合耗时。片段不可比较或恢复，成本仍以原累计账本为准；完整回合首门继承原实现。
- 新入口`scripts.run_v2_safe`统一派发/监控/worker的安全账本和日志句柄管理。登记目录必须全新；默认授权为false，无授权命令。原目录缺少新版本标识即拒绝，拒绝前不创建侧车。新登记核对八父锁和新源摘要。

## 离线复核

完整依赖环境运行：

```powershell
./outputs/v2-sac/env/Scripts/python.exe -m unittest discover -s tests -v
```

纯标准库、全新输出目录进行Windows进程回归：

```powershell
py -3.14 -m scripts.verify_v2_io_repair verify --output outputs/my-new-io-regression
```

脚本及其子进程禁止socket连接，三写者、两读者和一个监控进程同时工作；核对计数、单调快照、原时钟和监控读写重叠。另注入一次传输失败，以及仅对自建夹具持锁进程执行退出，核对等待者能继续、未提交临时文件未进入账本。数字均为文件夹具计数，不是FMU交互预算。独立Windows CI无须安装模型或仿真依赖，完整片段/入口测试由原完整环境CI执行。

运行期查看必须经`BudgetLedger.read`或`read_snapshot`；对可变JSON直接`Get-Content/read_text`不属于协作协议。外部不遵守互斥的程序、断电、所有杀毒/文件系统情形未得到保证。离线通过不等于正式真实并发训练通过。后续另行授权并登记全新试次，保留本轮和旧试次成本，不复用旧运行目录；不得直接启用历史入口或留出。
