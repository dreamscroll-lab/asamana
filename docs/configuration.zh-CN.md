# 配置

[English](configuration.md) | 简体中文

## 配置文件

| 文件 | 用途 |
|---|---|
| `config/config.yaml` | 默认配置：阿里云百炼（DashScope） |
| `config/presets/deepseek.yaml` | DeepSeek |
| `config/presets/zhipu.yaml` | 智谱（bigmodel 国内站），GLM |
| `config/presets/kimi.yaml` | Kimi（月之暗面） |
| `config/presets/minimax.yaml` | MiniMax（国内站） |
| `config/presets/dashscope_glm.yaml` | 阿里云百炼上的智谱 GLM（需要 `DASHSCOPE_WORKSPACE_ID`） |
| `config/config.example.yaml` | 带完整注释的参考样例：每个字段的含义与可选值 |
| `config/config.test.yaml` | 本地测试用：Mock LLM、纯内存存储，不需要 key |
| `config/content.yaml` | 预设首页推荐叙事主题 |

用环境变量 `CONFIG` 指定配置文件，默认 `config/config.yaml`：

```bash
CONFIG=config/presets/deepseek.yaml python main.py web
```

Docker 部署时用 `deploy/.env` 里的 `ASAMANA_CONFIG`，见 [部署](deployment.zh-CN.md#选择配置)。

所有预设的 embedding 都走阿里云百炼原生端点（只有它返回混合检索要用的 sparse 向量），所以无论选哪家 LLM，都需要一个百炼的 `EMBEDDING_API_KEY`。

## 环境变量占位符

配置里的字符串可以引用环境变量：

- `${VAR}` —— 必须设置，缺失时启动即报错。
- `${VAR:-default}` —— 未设置或为空时取 `default`。

## 接入 LLM

LLM 配置分两层，各写各的差异：

```yaml
llm:
  providers:               # 每家 endpoint 在哪、用谁的 key、默认用哪个模型、怎么调
    dashscope:
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      model: deepseek-v4-flash
      params:
        enable_thinking: false       # endpoint 自有配置参数，只对这家有效

  default_provider: dashscope        # 默认的 provider

  scenes:                            # 只需要写有特殊配置的 scenes，没写的走默认配置
    world_building:
      provider: dashscope
      model: deepseek-v4-pro         # 省略则用该 provider 自己声明的 model
      params:
        enable_search: true
```

绝大多数 scene 不需要写。scene 条目**是完整的**：写了就要写 `provider`，这样一条记录自己读得懂用的是哪家 endpoint，更加直接易懂。`model` 省略则取 provider 声明的，所以换 provider 时模型跟着换，不会把上一家的模型名带到新 endpoint 上。

同时**混用 provider** 时，一个变量装不下多个 key，每一个 provider 都要用 `api_key_env` 显式指定自己的变量（漏写会在启动时报错）：

```yaml
  providers:
    kimi:
      base_url: https://api.moonshot.cn/v1
      model: kimi-k2.6
      api_key_env: MOONSHOT_API_KEY
    zhipu:
      base_url: https://open.bigmodel.cn/api/paas/v4
      model: glm-5.3
      api_key_env: ZHIPU_API_KEY
```

常见厂商的 `base_url`：

| provider 名（自取） | `base_url` |
|---|---|
| `dashscope`（阿里云百炼） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `deepseek` | `https://api.deepseek.com/v1` |
| `kimi`（Moonshot） | `https://api.moonshot.cn/v1` |
| `minimax` | `https://api.minimaxi.com/v1`（国际站 `api.minimax.io`，key 与站点绑定） |
| `zhipu`（BigModel） | `https://open.bigmodel.cn/api/paas/v4` |

provider 名指的是**端点账号**，不是模型家族：`dashscope` 同时提供 DeepSeek / Kimi / GLM 的模型，所以 `provider: dashscope` + `model: deepseek-v4-flash` 是合法的配置。同一家开两个账号（独立计费 / 受限子 key）就写两条声明，即同样的 `base_url`、不同的 `api_key_env`。

## Embedding

embedding 是独立的，endpoint、key、模型都与 LLM 无关：

```yaml
embedding:
  provider: openai_compat          # 任意 OpenAI 兼容 /embeddings（只有 dense）
  params:
    model: text-embedding-3-large
    dimension: 3072
    base_url: https://api.openai.com/v1
```

凭证读 `EMBEDDING_API_KEY`。默认的 `dashscope` provider 走百炼**原生**端点，因为只有它返回学习型 sparse 向量（混合检索要用）。LLM 与 embedding 恰好是同一个账号时，两个变量填同一个值即可。

换 `dimension` 前要先清掉 `data/vectors/`，维度不匹配时写入会直接报错。

## Web

```yaml
web:
  host: 127.0.0.1
  port: 7860
  cors_origins: ["*"]
  dev_tools_enabled: ${ASAMANA_DEV_TOOLS:-false}
```

命令行 `--host` / `--port` 与环境变量 `ASAMANA_CORS_ORIGINS`（逗号分隔）可以覆盖它们。`dev_tools_enabled` 默认关，见 [开发者工具](development.zh-CN.md#开发者工具)。
