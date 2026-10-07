# 开发

[English](development.md) | 简体中文

## 环境

- Python 3.11+
- Node 22+（前端）

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # 运行时依赖 + pytest + ruff
cd frontend && npm install
```

下文的 `python` 都指虚拟环境里的解释器（`.venv/bin/python`）。

## 本地运行

前后端分开起，前端热更新：

```bash
# 终端 1：后端（API + WebSocket + 引擎，单进程）
ASAMANA_DEV_TOOLS=true python main.py web                      # 默认配置，需要 key
CONFIG=config/config.test.yaml python main.py web              # Mock LLM，不需要 key，只适合看界面

# 终端 2：前端开发服务器，自动把 /api 与 /ws 代理到 :7860
cd frontend && npm run dev                                     # http://localhost:5173
```

要让开发服务器代理到已经用 `deploy/asamana.sh` 跑起来的那套服务（nginx 在 8080），设 `ASAMANA_BACKEND=http://127.0.0.1:8080` 再 `npm run dev`。

也可以让后端直接托管构建好的前端：`cd frontend && npm run build`，再 `python main.py web`，打开 [http://127.0.0.1:7860/](http://127.0.0.1:7860/)。这只是本地便利，正式部署用两个镜像，见 [部署](deployment.zh-CN.md)。

用 Docker 开发见 [部署 · 开发者工具](deployment.zh-CN.md#开发者工具)。

## 命令行

```bash
python main.py build "<主题>"    # 构建一个世界
python main.py list               # 列出世界
python main.py web [--host HOST] [--port PORT]   # 启动后端
```

确认初始化、运行控制（运行 / 暂停 / 继续 / 停止）、重置、实时观察与回放都在 Web 界面里完成。命令行构建的世界会注册进世界列表，前端刷新即可看到。

## 测试与检查

```bash
python -m pytest tests -q            # 全部后端测试
python -m pytest tests/unit -q       # 单元测试
ruff check .                         # lint
cd frontend && npm run build         # 前端：类型检查 + vitest + 构建
```

测试使用 `config/config.test.yaml`（Mock LLM、纯内存存储），不需要 key、不访问网络。CI 在 Python 3.11 与 3.13 上跑同样的检查，并构建两个镜像。

## 开发者工具

开发者工具默认关闭：它们会起子进程、用任意 prompt 调用会消耗 LLM Token 预算，不建议出现在公开部署里。本地用环境变量打开：

```bash
ASAMANA_DEV_TOOLS=true python main.py web
```

打开后侧栏底部出现两个页面的入口（前端按 `/api/deployment` 的 `dev_tools` 决定显示与否；关闭时直接访问这两个地址只会看到一句未开启的说明）：

- **`#/dev` 开发者工具**
  - **Trace** —— 每一次 LLM 调用的完整记录与汇总，见 [可观测性与审查](observability.zh-CN.md)。
  - **Audit** —— 世界叙事质量审查。
  - **Debug** —— 用当前代码单独调试某个认知阶段、prompt 回放（编辑一次真实调用的文本后重放）、记忆召回调试。
  - **Director** —— 导演层的组装 → 调用 → 校验。
- **`#/lab` 地图工作台** —— 单独查看地图（可缩放）、逐帧查看角色美术、用固定场景检查渲染、导入新地图。地图的规范见 [`worlds/templates/README.md`](../worlds/templates/README.zh-CN.md)。

## 代码约定

项目的设计原则、分层边界、错误处理、日志、prompt 写法等约定都在 [`CLAUDE.md`](../CLAUDE.md) 里，它是这些规则的唯一来源，提交改动前请先阅读。贡献流程见 [CONTRIBUTING.md](../CONTRIBUTING.zh-CN.md)。