# 部署

[English](deployment.md) | 简体中文

前后端是**两个独立的镜像**：

| 镜像 | 构建自 | 提供 |
|---|---|---|
| **backend** | `deploy/backend.Dockerfile`（Context = 仓库根） | Python API + WebSocket + 叙事引擎，端口 7860 |
| **frontend** | `deploy/frontend.Dockerfile`（Context = `frontend/`） | nginx 托管构建好的前端（端口 80），并把 `/api`、`/ws` 反代到后端 |

两者构建期互不依赖，可以分别部署、分别升级。`deploy/docker-compose.yml` 把它们编排在一起。

| 文件 | 用途 |
|---|---|
| `backend.Dockerfile` / `.dockerignore` | 后端镜像 |
| `frontend.Dockerfile` / `.dockerignore` | 前端镜像（nginx） |
| `docker-compose.yml` | 编排前后端两个服务 |
| `.env.example` | 环境变量模板（API key、配置选择） |
| `asamana.sh` | compose 的常用操作脚本 |

## 前置条件

Docker 20+，带 Compose v2 与 BuildKit（Docker 23+ 默认开启）。

## 快速开始

```bash
./deploy/asamana.sh start
```

第一次运行时还没有 `deploy/.env`，脚本会先进入 `setup` 引导：是否现在配置 API Key（是则选择配置、输入所需的 key，输入不回显）、是否打开开发者工具，然后写出 `deploy/.env`（权限 600，只有你能读）并启动。打开 <http://localhost:8080/>。

不在终端里运行（比如 CI）时没法交互，改为手动 `cp deploy/.env.example deploy/.env` 并填写。

- 前端只监听本机 `127.0.0.1:8080`。API **没有鉴权**，能访问这个端口的人就能用您的 key 建世界、跑世界、删世界等所有操作。如果要开放给别人，请在前面加上您自己的访问控制（反向代理鉴权、VPN 等）。
- 后端默认不对宿主机暴露，只在 compose 网络内以 `http://backend:7860` 被前端访问。需要直接访问 API 时，取消 `docker-compose.yml` 里 `backend` 下 `ports:` 的注释。
- 没有 key 时服务也能启动：已有世界可以浏览、回放，建世界、运行等会调用模型的操作会被禁止。区别详见 README「有 key 与无 key 的区别」。
- key 要么全配、要么全不配。如果只进行部分配置，后端拒绝启动并列出缺少的变量。
- `GET /api/templates` 返回空列表时表示没有世界地图，此时创建不了新世界。

## 选择配置

后端读哪份配置由 `deploy/.env` 里的 `ASAMANA_CONFIG` 决定，默认 `config/config.yaml`（阿里云百炼）。最简单的修改方式是运行 `./deploy/asamana.sh keys`（重新选厂商并填 key）；也可以直接编辑 `deploy/.env`:

```bash
# deploy/.env
ASAMANA_CONFIG=config/presets/deepseek.yaml
DEEPSEEK_API_KEY=sk-...
EMBEDDING_API_KEY=sk-...
```

各厂商配置文件预设需要哪些 key 写在预设文件的开头。配置的写法见 [配置](configuration.zh-CN.md)。

`config/` 打包在镜像里，改配置后需要 `./deploy/asamana.sh rebuild backend`。只有 `config/content.yaml`（首页推荐的叙事主题）是挂载进去的，改完刷新页面即生效。

## 日常操作

`deploy/asamana.sh <命令>`，可以在任意目录运行：

| 命令 | 作用 |
|---|---|
| `setup` | 引导选择配置、填写 key（可跳过）、打开或关闭开发者工具，写出 `deploy/.env` |
| `keys` | 补上或更换模型厂商与 key，只改写 `.env` 里与模型相关的行；后端在运行时自动重建它使新 key 生效 |
| `start` | 后台启动（镜像不存在时自动构建；没有 `.env` 时先跑 `setup`） |
| `stop` | 停止容器（保留，可快速再启动） |
| `restart [svc]` | 重启全部或某个服务（`backend` / `frontend`） |
| `rebuild [svc]` | 重新构建镜像并启动——改了代码或 `config/` 之后用 |
| `down` | 停止并删除容器与网络（`data/` 保留） |
| `reset` | 回到刚克隆时的状态：删除容器、网络、本地构建的镜像、`data/` 与 `deploy/.env`（含 API Key）。不可恢复，需输入 `reset` 确认 |
| `logs [svc]` | 跟踪日志 |
| `status` | 查看服务状态 |

不同改动的生效方式：

| 改了什么 | 怎么生效 |
|---|---|
| 前后端代码 | `./deploy/asamana.sh rebuild`（可只重建一个服务） |
| `config/*.yaml` | `./deploy/asamana.sh rebuild backend` |
| `deploy/.env`（含重新 `setup`） | `./deploy/asamana.sh start`（用新环境变量重建容器；`restart` 不会读新的 `.env`） |
| 用 `keys` 改了 key | 后端在运行时自动生效；否则 `./deploy/asamana.sh start` |
| `config/content.yaml` | 刷新页面 |

## 开发者工具

在 `setup` 里选择打开，或在 `deploy/.env` 里写 `ASAMANA_DEV_TOOLS=true`，然后 `./deploy/asamana.sh start`。

- 后端挂载开发者工具（`#/dev`）与地图工作台（`#/lab`）的路由。这些工具会起子进程、用任意 prompt 调用 LLM、消耗 Token 预算，**请不要在别人能访问的机器上打开**。
- `tuning/scenarios` 与 `worlds/templates` 始终以可写方式挂载进容器，开发者工具保存的调优场景、导入的地图会直接写回仓库，git 看得见、重建也不丢。工具关闭时没有任何东西会写这两个目录。

## 数据持久化

后端的运行期状态（世界、快照、向量、trace、日志）都在仓库根目录的 `data/` 下，挂载为容器内的 `/app/data`。`stop` / `restart` / `down`（不带 `-v`）都不会丢数据。要清空全部状态，运行 `./deploy/asamana.sh reset`。

全新安装第一次启动时，会把 `examples/` 里的示例世界复制进 `data/`，并在 `data/` 里留下标记文件 `.examples_seeded`。之后不会再导入：删掉示例世界，甚至手动删掉 `data/worlds.json`，示例世界都不会再出现。只有清空整个 `data/`（比如 `reset`）之后再启动，才会重新导入。

停止或重启后端时，正在运行的世界会先把当前这一步走完再退出，所以 `stop` / `restart` / `rebuild` 可能要等上最多 3 分钟（`stop_grace_period`）。超过这个时间 Docker 会强制结束进程，这一步会在下次恢复时重跑。

## 单独部署前后端（不推荐）

**后端**——任何容器平台：

```bash
docker build -f deploy/backend.Dockerfile -t asamana-backend .
docker run -p 127.0.0.1:7860:7860 --env-file deploy/.env -v "$PWD/data:/app/data" asamana-backend
```

**前端**——两种方式：

1. **nginx 镜像（反代到后端）**，运行时指定后端地址：
   ```bash
   docker build -f deploy/frontend.Dockerfile -t asamana-frontend frontend/
   docker run -p 127.0.0.1:8080:80 -e BACKEND_URL=https://api.example.com asamana-frontend
   ```
2. **静态托管 / CDN（无反代）**：把 API 地址编译进前端，后端通过 CORS 放行该来源：
   ```bash
   cd frontend && VITE_API_BASE=https://api.example.com npm run build
   # 把 frontend/dist/ 部署到任意静态托管
   # 后端设置 ASAMANA_CORS_ORIGINS=https://app.example.com
   ```

`VITE_API_BASE` 为空（默认）时前端使用同源相对路径，配合 nginx 反代；设为完整 URL 时用于 CDN 场景，WebSocket 地址会自动从它推出。

## 环境变量

| 变量 | 作用于 | 说明 |
|---|---|---|
| `LLM_API_KEY` | backend | 没写 `api_key_env` 的 LLM provider 读它（默认的 `config/config.yaml` 就是这样） |
| `EMBEDDING_API_KEY` | backend | embedding 的 key（独立的，可以是另一家厂商、另一个账号） |
| 各厂商 key | backend | LLM provider 用 `api_key_env` 指名了自己的变量时读那个变量。`config/presets/` 下的预设都指名了（如 `DEEPSEEK_API_KEY`），文件开头写明了需要哪些 |
| `ASAMANA_CONFIG` | compose | 后端加载哪份配置文件，默认 `config/config.yaml` |
| `ASAMANA_DEV_TOOLS` | backend | `true` 时打开开发者工具，见上文；默认关 |
| `ASAMANA_CORS_ORIGINS` | backend | 允许的浏览器来源（逗号分隔），默认 `*`；CDN 场景需要 |
| `BACKEND_URL` | frontend(nginx) | 反代的后端地址，默认 `http://backend:7860` |
| `VITE_API_BASE` | frontend（构建参数） | 把远端 API 地址编译进前端；为空 = 相对路径 |

## 端口

| 端口 | 服务 |
|---|---|
| `8080` | 前端（nginx），应用入口 |
| `7860` | 后端（API + WebSocket）；compose 下默认只在内部网络 |
| `5173` | Vite 开发服务器（仅本地开发） |
