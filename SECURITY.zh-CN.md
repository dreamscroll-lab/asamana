# 安全

[English](SECURITY.md) | 简体中文

## 部署模型

Asamana 面向**单用户自托管**，API **没有鉴权**：

- 能访问 API 的人都可以建世界、运行世界、删除世界、做导演干预等所有操作，用的是部署者配置的 LLM key，消耗部署者的 Token 预算。
- Docker 部署默认只监听本机（`127.0.0.1:8080`）。如果要开放给别人，请在前面加您自己的访问控制（反向代理鉴权、VPN 等）。
- **开发者工具**（`ASAMANA_DEV_TOOLS=true` / `web.dev_tools_enabled`）会起子进程、用任意 prompt 调用都会消耗 LLM Token 预算、向源码目录写文件。该功能建议只在自己的开发机上打开，不要在别人能访问的网络环境下打开。
- API key 只放在环境变量或 `deploy/.env`（已被 gitignore）里，配置文件只写变量名。

## 报告漏洞

请您不要公开提 issue。请发邮件到 **[finley@dreamscroll.net](mailto:finley@dreamscroll.net)**，写明受影响的版本（commit）、复现步骤与影响。我们会尽快回复，并在修复发布后致谢（如您愿意）。