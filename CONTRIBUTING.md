# Contributing

English | [简体中文](CONTRIBUTING.zh-CN.md)

Thanks for your interest in contributing to Asamana!

## Issues

- **Bugs**: describe the problem, steps to reproduce it, and your configuration and deployment method. Include relevant logs with API keys redacted.
- **Feature requests**: start with the problem you want to solve, then describe how you imagine solving it.
- **Security issues**: don't open a public issue; report them privately as described in [SECURITY.md](SECURITY.md).

## Pull requests

1. Set up your development environment; see [docs/development.md](docs/development.md).
2. Read [`CLAUDE.md`](CLAUDE.md) before making changes. It defines the design principles, layer boundaries, error handling, logging, prompt-writing conventions and review criteria.
3. Include implementation changes and their tests in the same PR. Tests use the mock and in-memory providers configured in `config/config.test.yaml`; keep them fast, deterministic and offline.
4. Make sure these pass locally before you submit:
   ```bash
   python -m pytest tests -q
   ruff check .
   cd frontend && npm run build    # if you touched the frontend
   ```
5. Write commit messages as a single imperative line with a subsystem prefix, e.g. `engine: add clock skeleton` or `agent: fix need competition`.
6. In the PR description, explain the purpose, affected subsystems and validation results.

## License

By contributing, you agree that your contribution is released under the [Apache License 2.0](LICENSE).
