# Asamana

[English](README.md) | 简体中文

[![CI](https://github.com/dreamscroll-lab/asamana/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dreamscroll-lab/asamana/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

<p align="center"><img src="docs/images/cover.jpg" alt="Asamana" width="1024"></p>

<!-- TODO 演示地址：视频发布后把下一行换成 ▶️ **演示视频**：[在 Bilibili 观看](https://www.bilibili.com/video/BV...) -->
▶️ **演示视频**：即将在 Bilibili 发布

**Asamana 是一个 AI 驱动的涌现叙事引擎。**

简单地说：你给出一个主题，Asamana 会构建一群有各自认知人设的 AI 角色放到一个虚拟空间里，让他们在这个空间里围绕主题自主地展开叙事与戏剧化演绎。故事没有任何预设情节，没有剧本，完全由角色共同叙事而产生。

> **项目状态**：早期迭代中。接口、配置格式与存档格式都可能在版本之间变化，暂不保证向后兼容。变更见 [CHANGELOG.md](CHANGELOG.md)。

> **语言**：目前 prompt 与生成的叙事（对白、记忆、情节文本）都是中文。

## 项目哲学：AI as a human

把角色当作人，让角色拥有完整的人类认知。

## 项目特点

### 1. 角色拥有完整的人类认知系统

世界中的角色拥有完整的人类认知，他们有性格，价值观，情绪，目标，需求，关系，记忆，也有思考，决策和行动能力。

### 2. 世界围绕单一主题展开叙事

区别于让角色在空间里自由、发散地涌现，Asamana 让所有角色在有限的空间里**围绕同一个主题**展开叙事。这让故事更有焦点、有记忆点、有观赏性，也更能勾起人的好奇心。就像人们更喜欢看电影和电视剧一样，却没有人想看一部只记录每天起床、吃饭、上学、上班、睡觉反复循环枯燥无味的日常生活纪录片。

### 3. 以 Agent Loop 理念设计，用系统编排技术实现

目前主流 Agent 大多是 Loop 模式，在这种模式下，上下文会随着轮次不断累积变大，既抬高成本，又稀释 LLM 对关键信息的注意力。Asamana 以 Loop 的理念设计角色，但用以下三件套实现：

- **分阶段编排**：放弃对话循环，把认知拆成 `感知 → 动机 → 决策 → 行动 → 反馈` 五个阶段，由代码确定性地编排。每个阶段都是独立的 LLM 调用，可单独测试与调优，也不会无脑累积上下文。
- **类型化输出约束**：阶段之间用确定的类型化输出（JSON）衔接，而不是互相传递自由文本。这样可以提高准确率，也可以避免阶段间的幻觉被级联放大。
- **overlap 上下文管理**：为保证角色的一致性、持续性和连贯性，把人设、情绪、方向（需求与目标）、关系与处境、近期记忆作为 overlap 重复注入各阶段的上下文中。

## 技术架构

<p align="center"><img src="docs/images/arch.png" alt="Asamana 架构" width="1024"></p>

- **后端**（Python）：单进程同时提供 REST API、WebSocket 推送，并把叙事引擎作为后台任务运行。严格分层：`interaction → world / engine / agent → core ← providers`，业务逻辑只依赖 `core/` 的接口，LLM、embedding、向量存储、快照等基础设施都是可插拔的。
- **前端**（React + Vite + TypeScript + Phaser）：世界管理、实时观察、回放、2D 地图。前后端只通过 API 交互，引擎不关心渲染方式。
- **LLM 接入** ：任何 OpenAI 兼容 endpoint 都能直接接入，不需要改代码，同时支持不同场景可以路由到不同的模型。
- **部署** ：前后端各一个镜像，Docker Compose 一键启动。

```
interaction/   API(REST + WebSocket)、回放、世界管理
world/         世界构建
engine/        运行时：时钟、调度、环境、消息、事件、行动执行
agent/         角色认知：人格、记忆、需求、目标、关系、决策
providers/     可插拔基础设施：LLM、Embedding、向量存储、快照
core/          接口、DI 容器、日志
config/        配置（config.yaml 为默认，presets/ 为各 LLM 厂商预设配置）
worlds/        世界地图（Tiled 地图 + 角色美术）
examples/      示例世界（首次启动时自动导入）
frontend/      React 前端
deploy/        Docker 部署
tuning/        离线评审与调优等开发者工具
```

## 快速开始

推荐在 macOS 上运行。需要 Docker（含 Compose v2），安装方法见[部署文档](docs/deployment.zh-CN.md#安装-docker)。完整的功能需要配置必要的 LLM key。没有配置 LLM key 也可以正常启动体验，见下文。

```bash
git clone https://github.com/dreamscroll-lab/asamana.git && cd asamana
./deploy/asamana.sh start
```

第一次运行会进入引导：先选择确认是否现在配置 API Key（选是则接着选模型厂商、输入 key），再选择是否打开开发者工具，生成 `deploy/.env` 后构建镜像并启动。之后浏览器打开 <http://localhost:8080/>，输入一个主题创建世界，审阅后确认，然后运行并正式进入叙事。

在中国大陆，引导里选择使用国内镜像源可以加快安装速度，详见[部署文档](docs/deployment.zh-CN.md#中国大陆的镜像源)。

第一次启动时会自动导入两个已经跑完的示例世界（玄武门之变、谁是内鬼），不配 key 也可以直接浏览和回放。

### 有 key 与无 key 的区别

key 要么全配、要么全不配，只配一部分时后端拒绝启动并列出缺了哪些。

| | 有 key | 无 key |
|---|---|---|
| 浏览、回放已有世界（地图、角色、关系、叙事流） | ✅ | ✅ |
| 只读的开发者工具（LLM 调用追踪、审查报告、地图工作台） | ✅ | ✅ |
| 创建世界 | ✅ | ❌ |
| 运行、推一步、重置、导演台 | ✅ | ❌ |
| 会调用模型的开发者工具（prompt 重放、导演控制台、审查/调优运行、记忆召回） | ✅ | ❌ |

随时补上或更换 key（也可以用来换模型厂商）：

```bash
./deploy/asamana.sh keys
```

如果后端正在运行，会自动用新 key 重建后端容器，刷新页面即可。要从头重新引导，运行 `./deploy/asamana.sh setup`。

> **安全提示**：API 没有鉴权，能访问端口的人都能用你的 key 建世界、跑世界等所有操作。默认只监听本机（`127.0.0.1:8080`），如果需要开放到公网，请您先认真评估。详见 [SECURITY.zh-CN.md](SECURITY.zh-CN.md)。

更多文档：

- [部署](docs/deployment.zh-CN.md) —— Docker、独立部署前后端、环境变量、数据持久化
- [配置](docs/configuration.zh-CN.md) —— 接入 LLM 厂商、按场景路由模型、embedding
- [开发](docs/development.zh-CN.md) —— 本地运行、命令行、测试、开发者工具
- [可观测性与审查](docs/observability.zh-CN.md) —— LLM 调用追踪、叙事质量审查

## 提示和建议

### 模型选择

同一个世界建议只用一类模型（也可以混用），建议：

| 目标 | 推荐 | 预设 |
|---|---|---|
| 测试、调试、追求速度与性价比 | DeepSeek | `config/presets/deepseek.yaml` |
| 叙事速度与质量平衡 | Qwen | `config/config.yaml`（默认） |
| 追求叙事质量 | GLM | `config/presets/zhipu.yaml`（智谱官方）或 `config/presets/dashscope_glm.yaml`（经阿里云百炼） |

Kimi 与 MiniMax 不推荐：Kimi 成本较高、叙事偏抽象（偶尔有点冷幽默）；MiniMax 的 JSON 遵循率不稳定，叙事质量一般。

阿里云百炼（DashScope）上基本可以选到国内头部厂商的开源模型，也能直连部分其他厂商的模型。通过百炼直接接入通常会更方便统一管理，但是叙事质量，速度，并发，RPM，TPM等是否有差异，目前并没有验证过。

世界初始化时推荐使用比较大的模型，因为好的剧本得有一个好的开始；运行时可以使用小一些，快一些的模型，这样可以控制成本，也会有一个比较好的体验。

### Token 消耗

以下是本地 31 次实际运行的统计（每次 30–35 步，6–8 个角色，flash 档模型）：

| 规模 | LLM 调用 | 输入 token | 输出 token |
|---|---|---|---|
| 建世界（一次性） | 约 10 次 | 约 2.5–3.5 万 | 约 0.5–1 万 |
| 运行 35 步 · 6–8 个角色 | 约 900–1,400 次 | 约 240–390 万 | 约 15–30 万 |
| **合计（一个 35 步的世界）** | **约 900–1,400 次** | **约 245–395 万** | **约 15–30 万** |

一个 35 步的世界总共大约消耗 **250–420 万 token**。消耗以输入为主，并随角色数、剧情密度和所选模型上下浮动。各阶段的 prompt 按“不变的在前、易变的在后”编排，提高前缀缓存命中率，因此支持前缀缓存的 LLM 厂商实际计费会更低。

> **使用前请先确认自己的 Token 预算。** 世界运行时每一步都会为每个角色发起多次 LLM 调用，消耗随步数线性累积。运行页的步数框默认 50 步；**留空只跑一步**，没有“一直跑下去”这一选项，想长跑就填一个大数，随时可暂停或停止（主要是出于预算安全方便考虑）。建议：
>
> - 开跑前按所选模型的单价和要跑的步数估一下费用，确认账户余额或额度够用；
> - 在厂商控制台设置用量告警或消费上限；
> - 第一次使用先用 flash 档模型、少量步数试跑，看过效果再跑长剧情。

### 生成内容

世界里的人物、对白与情节全部由所接入的 LLM 实时生成，不代表项目作者的观点，也可能出现不准确或不适宜的内容。请遵守所用模型厂商的使用政策，并自行判断是否公开传播生成的内容。

### 开发者工具

本地开发时打开开发者工具（见 [开发文档](docs/development.zh-CN.md#开发者工具)），在 `#/dev` 页面：

- **Trace** —— 查看每一次 LLM 调用的完整记录：prompt、响应、token、耗时、所属阶段。
- **Audit** —— 用 LLM 评审一个跑完的世界的叙事质量。主要关注 `mams`、`sams`、`init` 这几个 scope。其中 `mams` 审查的是整体叙事的质量，`sams` 审查的是每个角色认知的质量，`init` 主要看的是世界初始化。

## 致谢

Asamana 受益于以下工作：

- **AgentSociety**（清华大学 FIB Lab）—— Piao et al., [*AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society*](https://arxiv.org/abs/2502.08691), arXiv 2025.
  - **需求驱动理念**：以马斯洛需求层次为基础的需求作为动机，驱动角色围绕最迫切的需求制定目标与行动。
  - **双记忆流架构**：客观事件流与主观感受流分开记录、相互绑定，即对应 Asamana 的 `FACTUAL`（世界客观上发生了什么）与 `EXPERIENTIAL`（角色对客观发生的事情产生的主观理解）两条记忆流。
- **Generative Agents**（Stanford）—— Park et al., [*Generative Agents: Interactive Simulacra of Human Behavior*](https://doi.org/10.1145/3586183.3606763), UIST 2023.
  - **Recency、Importance、Relevance 三维检索机制**：按时近性、重要性、相关性加权召回记忆。
  - **反思机制**：把近期经历凝聚成更高层的洞察（insight），并指回作为依据的记忆。

## 参与贡献

欢迎您提 issue 和 PR，请先阅读 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。

## 引用

如果 Asamana 对您的研究有帮助，引用信息见 [CITATION.cff](CITATION.cff)（GitHub 仓库页右侧的 “Cite this repository” 可直接导出 BibTeX）。

## 联系

finley@dreamscroll.net (finley)

## 许可证

[Apache License 2.0](LICENSE)。仓库内的地图与角色美术资产（`worlds/templates/`）由 AI 与人共同创作，同样以 Apache-2.0 发布。
