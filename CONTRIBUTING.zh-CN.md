# 参与贡献

[English](CONTRIBUTING.md) | 简体中文

感谢你愿意参与 Asamana!

## 提 issue

- **Bug**：写清现象、复现步骤、使用的配置与部署方式，附上相关日志（注意抹掉 API key）。
- **功能建议**：先说想解决什么问题，再说设想的做法。
- **安全问题**：不要公开提 issue，按 [SECURITY.zh-CN.md](SECURITY.zh-CN.md) 私下报告。

## 提 PR

1. 搭好开发环境，见 [docs/development.zh-CN.md](docs/development.zh-CN.md)。
2. 动手前读 [`CLAUDE.md`](CLAUDE.md)——设计原则、分层边界、错误处理、日志、prompt 写法等约定都在那里，评审以它为准。
3. 改动和测试放在同一个 PR 里；测试用 `config/config.test.yaml` 的 Mock / 内存实现，保持快速、确定、不依赖网络。
4. 提交前本地通过：
   ```bash
   python -m pytest tests -q
   ruff check .
   cd frontend && npm run build    # 动了前端时
   ```
5. commit message 用一行祈使句，带子系统前缀，如 `engine: add clock skeleton`、`agent: fix need competition`。
6. PR 描述写清：目的、影响的子系统、测试证据。

## 许可

提交贡献即表示你同意它以 [Apache License 2.0](LICENSE) 发布。
