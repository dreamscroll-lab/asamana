# Asamana Codex Agents

Each `.toml` file here is a Codex custom agent. `.claude/agents/` holds the same roster for Claude Code; keep the two in sync.

Each agent's instructions follow the same structure:

- `Mission`
- `Owns`
- `Responsibilities`
- `Decision Standard`
- `Do Not`
- `Collaboration Rules`

Agents are organized by module and responsibility, not by individual files, and don't pin a model. Repository-wide rules live in `CLAUDE.md` (`AGENTS.md` is a symlink to it); the code is the source of truth.
