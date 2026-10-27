# 百度网盘私有备份与恢复

2026-10-27：所有者选择已有百度网盘会员，用于私有灾备。自有代码采用MIT，范围见[LICENSE_SCOPE](../LICENSE_SCOPE.md)；完整研究档案只作私有保存，不生成公开分享链接。备份不训练模型、不启动仿真、不消耗保护窗口。

## 本轮交付及权限边界

- 公开源码：GitHub `main`与审查过的rc3源码zip，排除原CSV、历史脚本、完整权重、FMU、`outputs/`和凭据。
- 私有归档：原研究证据zip及逐文件清单、rc3源码及清单、部署/安装/运行证据和可选wheel。密码不在上传目录或源码包内。
- 加密：标准7z AES-256并加密文件名，随机密钥仅留在本机独立目录，设当前Windows用户访问权限。请另外保存密钥到你自己的可靠离线位置；仅留同一台电脑，电脑损坏后仍可能无法恢复。
- 保留“封存研究版”和最近两次交付版；本轮不自动删除旧版本、不创建定时任务或公开下载入口。

百度网盘站点访问被浏览器安全策略阻止，工具明确禁止改用其他入口绕过。因此自动上传/下载不可执行；本地归档已准备后，只有网盘传输和远程来源确认需要所有者手动完成。远程下载验收前F03不关闭。

## 手动传输（文件准备完成后）

1. 本机上传目录：`D:\thermalComfortPredict\outputs\release-final\upload`。只上传其中的加密`.7z`、收据和说明；**不要上传`key`目录或密码文件**。
2. 在自己的百度网盘创建私有目录`thermalComfortPredict/2026-10-27`，上传上述文件；不点击“分享”。
3. 上传完成后，从网盘重新下载`.7z`到`D:\thermalComfortPredict\outputs\release-final\download`。本地复制或复制上传缓存不算远程下载。
4. 将实际下载路径告知维护者，由其运行下述校验并保存远程来源/日期。网页“上传完成”或文件名相同不能替代SHA和恢复校验。

收据包含加密包SHA、字节数与内部清单SHA，无密码。公共核对副本在`reports/release-final/backup-receipt.json`（归档构建后生成）；应使用本机/Git中的可信收据，不只信网盘一起下载的可变文件。

## 恢复命令

可复用本轮`outputs/release-final/backup-venv`；其他电脑需要新环境，安装可选的9项备份依赖。它们不加入工作台/研究环境。

```powershell
py -3.14 -m venv .venv-backup
./.venv-backup/Scripts/python.exe -m pip install -r requirements/backup.lock.txt
./.venv-backup/Scripts/python.exe -m pip check
./.venv-backup/Scripts/python.exe -m scripts.restore_private_backup --archive outputs/release-final/download/thermalComfortPredict-private-20261027.7z --receipt reports/release-final/backup-receipt.json --key-file outputs/release-final/key/RECOVERY_KEY.txt --destination outputs/release-final/remote-restore
```

目标目录必须为全新目录；失败时保留已解密的诊断文件，不自动清理用户文件。脚本先核对加密包，再核对内部文件和两个zip的逐文件摘要；拒绝路径越界和覆盖不同的历史内容。恢复后完整新研究环境应在恢复源码目录执行：

```powershell
# 先切换到outputs/release-final/remote-restore/restored-source；解释器使用原新建完整研究环境的绝对路径。
D:/thermalComfortPredict/outputs/deployment-f2/acceptance-venv/Scripts/python.exe -m scripts.verify_delivery
D:/thermalComfortPredict/outputs/deployment-f2/acceptance-venv/Scripts/python.exe -m thermal_comfort.residential_c2 verify
D:/thermalComfortPredict/outputs/deployment-f2/acceptance-venv/Scripts/python.exe -m thermal_comfort.office_final_round check
```

仅当确实从网盘重新下载且摘要、解密、文件恢复和上述只读核验均通过，才记录异地恢复通过。私有灾备不等于第三方已能公开获取全部历史研究；公开交付保持源码＋小样例范围。
