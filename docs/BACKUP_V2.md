# V2私有备份：本机与异地验收分开

沿用所有者已选的百度网盘会员、私有加密归档和密钥单独保存方案；不另买存储、不重新询问MIT、不生成公开分享。V1已验收的r2不覆盖本次新增RL资产。

本次私有研究主ZIP含8,046个文件，压缩约13GB；所有V2阶段的正式结果、工程故障、模型/回放/RNG/日志及住宅冻结模型保留，排除env/venv、缓存和凭据。另含审查源码、固定住宅FMU/上游许可及34个精确版本wheel。复核所需依赖不等于完整可运行引擎或Python解释器。

## 制作和本机恢复

`scripts.package_v2_evidence`逐文件复制、读回校验主档案；`scripts.build_v2_backup`按已审查源码ZIP构建7z AES-256并加密文件名。密钥只放本机独立key目录并限制当前用户访问，不写入源码、收据、上传目录或聊天。大ZIP用同卷临时硬链接进入加密载荷，不代表异地副本；恢复仍会从加密字节解密到全新目录。

发布目录：`outputs/v2-delivery/upload`，只包含`.7z`、`backup-receipt.json`及上传说明。可信收据另存`reports/v2_delivery/backup-receipt.json`并提交Git。密钥位于`outputs/v2-delivery/key/RECOVERY_KEY_V2.txt`；请另存到自己可靠的离线位置，同机密钥不能抵御整机丢失。

```powershell
# 可选备份环境沿用V1已安装py7zr的outputs/release-final/backup-venv；不加到演示运行时。
outputs/release-final/backup-venv/Scripts/python.exe -m scripts.restore_v2_backup --archive outputs/v2-delivery/upload/thermalComfortPredict-V2-private-20261029.7z --receipt reports/v2_delivery/backup-receipt.json --key-file outputs/v2-delivery/key/RECOVERY_KEY_V2.txt --destination outputs/v2-delivery/local-restore
```

目标必须不存在；不覆盖用户历史或旧包。脚本先验加密包和内层清单，再逐文件恢复源码、研究证据、上游与wheel，原路径/字节保留。随后在独立环境、恢复源码目录中按[REPRODUCE_V2](REPRODUCE_V2.md)只读核验。本机恢复通过与否由验收记录说明，不由制作收据的prepared状态推断。

## V2-R01：远程回下载验收

现有[V1备份记录](BACKUP.md)显示网盘网页曾被工具安全策略阻止，传输由所有者完成；本轮延续人工传输流程，不换入口绕过限制。

1. 把`outputs/v2-delivery/upload`的三个文件上传到自己的百度网盘私有目录，例如`thermalComfortPredict/V2/2026-10-29`；不要上传key，不创建公开链接。
2. 上传后真正从网盘重新下载`.7z`到`outputs/v2-delivery/download`，不能用本地复制或上传缓存替代。
3. 记录实际下载来源/日期，使用Git/本机可信收据与独立密钥，在全新目录再次解密恢复并只读核验。不要只相信一起下载、可能被替换的收据。

只有该步骤实际完成，才关闭V2-R01。当前准备、本机恢复或V1认证均不能自动关闭它。大档案、密钥和原数据不上传GitHub；保留封存研究版及最近两次交付版，不自动删旧版本，也没有创建定时备份。
