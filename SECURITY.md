# Security

English | [简体中文](SECURITY.zh-CN.md)

## Deployment model

Asamana is built for **single-user self-hosting**, and its API has **no authentication**:

- Anyone who can reach the API can perform all operations, including creating, running and deleting worlds and using the director console. Model calls use the deployer's API keys and token budget.
- The Docker deployment listens only on localhost (`127.0.0.1:8080`) by default. If you want to open it to others, put your own access control in front of it (an authenticating reverse proxy, a VPN, etc.).
- The **developer tools** (`ASAMANA_DEV_TOOLS=true` / `web.dev_tools_enabled`) spawn subprocesses, send arbitrary prompts to the LLM (spending your token budget), and write files into the source directory. Only enable them on your own development machine, never on a network others can reach.
- API keys are stored in environment variables or `deploy/.env` (excluded from version control); configuration files reference the variable names.

## Reporting a vulnerability

Please don't open a public issue. Email **[finley@dreamscroll.net](mailto:finley@dreamscroll.net)** with the affected version (commit), steps to reproduce and the impact. We'll respond as soon as we can and, if you'd like, credit you once the fix is released.
