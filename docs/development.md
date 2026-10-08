# Development

English | [简体中文](development.zh-CN.md)

## Requirements

- Python 3.11+
- Node 22+ (frontend)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # runtime dependencies + pytest + ruff
cd frontend && npm install
```

The commands below use the virtual environment's Python interpreter. Activate it with `source .venv/bin/activate`, or replace `python` with `.venv/bin/python` when running commands from the repository root.

## Running locally

Run the backend and frontend separately to get hot reload on the frontend:

```bash
# Terminal 1: backend (API + WebSocket + engine in one process)
ASAMANA_DEV_TOOLS=true python main.py web                      # default config; needs keys
CONFIG=config/config.test.yaml python main.py web              # mock LLM, no keys; suitable for UI testing

# Terminal 2: frontend dev server; proxies /api and /ws to :7860
cd frontend && npm run dev                                     # http://localhost:5173
```

To point the dev server at a stack already running via `deploy/asamana.sh` (nginx on 8080), set `ASAMANA_BACKEND=http://127.0.0.1:8080` before `npm run dev`.

The backend can also serve the built frontend: run `cd frontend && npm run build`, return to the repository root, run `python main.py web`, and open [http://127.0.0.1:7860/](http://127.0.0.1:7860/). For deployment, use the two images described in [Deployment](deployment.md).

For Docker-based development, see [Developer tools in the deployment guide](deployment.md#developer-tools).

## Command line

```bash
python main.py build "<theme>"    # build a world
python main.py list               # list worlds
python main.py web [--host HOST] [--port PORT]   # start the backend
```

Use the web UI to confirm newly built worlds, control runs (run / pause / resume / stop), reset worlds, observe live activity and replay past runs. Worlds built from the command line appear in the world list after refreshing the page.

## Tests and checks

```bash
python -m pytest tests -q            # all backend tests
python -m pytest tests/unit -q       # unit tests
ruff check .                         # lint
cd frontend && npm run build         # frontend: type check + vitest + build
```

Tests use `config/config.test.yaml` (mock LLM, in-memory storage), so they need no keys and no network access. CI runs the same checks on Python 3.11 and 3.13 and builds both images.

## Developer tools

Developer tools are disabled by default. They spawn subprocesses and can send arbitrary prompts using your LLM token budget, so enable them only in a private development environment:

```bash
ASAMANA_DEV_TOOLS=true python main.py web
```

When enabled, links to two pages appear at the bottom of the sidebar, based on the `dev_tools` field from `/api/deployment`. When disabled, visiting either page directly shows a notice that the tools are unavailable.

- **`#/dev` — developer tools**
  - **Trace** — a full record and summary of every LLM call; see [Observability and audit](observability.md).
  - **Audit** — narrative quality audit for a world.
  - **Debug** — run a single cognition stage against the current code, edit and replay a recorded prompt, and debug memory recall.
  - **Director** — the director layer's assemble → call → validate pipeline.
- **`#/lab` — map workbench** — preview and zoom into maps, inspect character animation frames, test rendering with predefined scenes, and import new maps. See the [map specification](../worlds/templates/README.md).

## Code conventions

The project's design principles, layer boundaries, error handling, logging and prompt-writing conventions all live in [`CLAUDE.md`](../CLAUDE.md), the single source for these rules. Please read it before submitting changes. For the contribution workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md).
