# Configuration

English | [简体中文](configuration.zh-CN.md)

## Config files

| File | Purpose |
|---|---|
| `config/config.yaml` | Default config: Alibaba Cloud Model Studio (DashScope) |
| `config/presets/deepseek.yaml` | DeepSeek |
| `config/presets/zhipu.yaml` | Zhipu GLM (BigModel, China site) |
| `config/presets/kimi.yaml` | Kimi (Moonshot AI) |
| `config/presets/minimax.yaml` | MiniMax (China site) |
| `config/presets/dashscope_glm.yaml` | Zhipu GLM served through DashScope (requires `DASHSCOPE_WORKSPACE_ID`) |
| `config/config.example.yaml` | Annotated reference for all fields and accepted values |
| `config/config.test.yaml` | For local testing: mock LLM, in-memory storage, no keys needed |
| `config/content.yaml` | Suggested story themes shown on the home page |

Select a config file with the `CONFIG` environment variable (default: `config/config.yaml`):

```bash
CONFIG=config/presets/deepseek.yaml python main.py web
```

With Docker, set `ASAMANA_CONFIG` in `deploy/.env` instead; see [Deployment](deployment.md#choosing-a-config).

Every preset uses the native DashScope endpoint for embeddings to obtain the sparse vectors required for hybrid retrieval. You therefore need a DashScope key in `EMBEDDING_API_KEY`, regardless of your LLM provider.

## Environment variable placeholders

Strings in a config file can reference environment variables:

- `${VAR}` — required; startup fails if it is missing.
- `${VAR:-default}` — falls back to `default` when unset or empty.

## Connecting an LLM

LLM configuration has two layers: provider defaults and scene-specific overrides.

```yaml
llm:
  providers:               # endpoints, API key variables, default models and request parameters
    dashscope:
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      model: deepseek-v4-flash
      params:
        enable_thinking: false       # endpoint-specific parameter; applies to this provider only

  default_provider: dashscope        # the default provider

  scenes:                            # list only the scenes that need special settings; the rest use the defaults
    world_building:
      provider: dashscope
      model: deepseek-v4-pro         # omit to use the model declared by the provider
      params:
        enable_search: true
```

Most scenes use the defaults and need no entry. Each scene entry must specify its `provider`. If `model` is omitted, it uses that provider's default model, so switching providers does not carry a model name from the previous provider to the new endpoint.

When using **multiple providers**, each must specify its API key environment variable through `api_key_env`; omitting it causes a startup error:

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

Common provider endpoints:

| Provider name (your choice) | `base_url` |
|---|---|
| `dashscope` (Alibaba Cloud Model Studio) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `deepseek` | `https://api.deepseek.com/v1` |
| `kimi` (Moonshot) | `https://api.moonshot.cn/v1` |
| `minimax` | `https://api.minimaxi.com/v1` (international site: `api.minimax.io`; keys are tied to the site) |
| `zhipu` (BigModel) | `https://open.bigmodel.cn/api/paas/v4` |

A provider name identifies an **endpoint account**, not a model family. DashScope serves DeepSeek, Kimi and GLM models, so `provider: dashscope` with `model: deepseek-v4-flash` is a valid combination. To use two accounts with the same vendor (separate billing, or a restricted sub-key), declare two providers with the same `base_url` and different `api_key_env` values.

## Embeddings

Embedding configuration is independent of LLM configuration, with its own endpoint, key and model.

```yaml
embedding:
  provider: openai_compat          # any OpenAI-compatible /embeddings endpoint (dense only)
  params:
    model: text-embedding-3-large
    dimension: 3072
    base_url: https://api.openai.com/v1
```

The key is read from `EMBEDDING_API_KEY`. The default `dashscope` provider uses the **native** endpoint to obtain learned sparse vectors for hybrid retrieval. If your LLM and embedding services use the same account and key, set both API key variables to that value.

Clear `data/vectors/` before changing `dimension`; vector writes fail if dimensions do not match.

## Web

```yaml
web:
  host: 127.0.0.1
  port: 7860
  cors_origins: ["*"]
  dev_tools_enabled: ${ASAMANA_DEV_TOOLS:-false}
```

The `--host` and `--port` flags override `host` and `port`. `ASAMANA_CORS_ORIGINS` overrides `cors_origins` with a comma-separated list. `dev_tools_enabled` is off by default; see [Developer tools](development.md#developer-tools).
