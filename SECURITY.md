# Security

English | [简体中文](SECURITY.zh-CN.md)

## Deployment model

Asamana is built for **single-user self-hosting**, and its API has **no authentication**:

- Anyone who can reach the API can perform all operations, including creating, running and deleting worlds and using the director console. Model calls use the deployer's API keys and token budget.
- The Docker deployment listens only on localhost (`127.0.0.1:8080`) by default. Before making it accessible to others, add access controls such as a reverse proxy with authentication or a VPN.
- The **developer tools** (`ASAMANA_DEV_TOOLS=true` / `web.dev_tools_enabled`) spawn subprocesses, send arbitrary prompts to the LLM (using your token budget), and write files to the source directory. Enable them only in a private development environment.
- API keys are stored in environment variables or `deploy/.env` (excluded from version control); configuration files reference the variable names.

## Reporting a vulnerability

Please don't open a public issue. Email **[finley@dreamscroll.net](mailto:finley@dreamscroll.net)** with the affected version or commit, reproduction steps and impact. We'll respond as soon as possible and, with your permission, credit you when the fix is released.
