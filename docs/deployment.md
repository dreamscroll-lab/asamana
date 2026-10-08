# Deployment

English | [简体中文](deployment.zh-CN.md)

The frontend and backend ship as **two separate images**:

| Image | Built from | Provides |
|---|---|---|
| **backend** | `deploy/backend.Dockerfile` (context: repo root) | Python API + WebSocket + narrative engine, on port 7860 |
| **frontend** | `deploy/frontend.Dockerfile` (context: `frontend/`) | nginx serving the built frontend (port 80) and reverse-proxying `/api` and `/ws` to the backend |

Neither depends on the other at build time, so you can deploy and upgrade them independently. `deploy/docker-compose.yml` runs them together.

| File | Purpose |
|---|---|
| `backend.Dockerfile` / `.dockerignore` | Backend image |
| `frontend.Dockerfile` / `.dockerignore` | Frontend image (nginx) |
| `docker-compose.yml` | Orchestrates the two services |
| `.env.example` | Environment variable template (API keys, config selection) |
| `asamana.sh` | Helper script for common Docker Compose operations |
| `install-docker.sh` | Installs Docker and Compose v2 on Linux |

## Prerequisites

Docker 20+ with Compose v2 and BuildKit (enabled by default since Docker 23). `asamana.sh` checks for these every time it runs. If Docker or Compose v2 is missing, it tells you how to install it on your system. If Docker is installed but not running, it starts Docker Desktop (or OrbStack) on macOS and waits until it's ready; on other systems it prints the command to start it.

We recommend running Asamana on macOS. `asamana.sh` is a bash script and supports Linux, macOS and WSL 2. On Windows, run it in WSL 2 (Git Bash is untested), or skip it: copy `deploy/.env.example` to `deploy/.env`, fill it in, and run `docker compose -f deploy/docker-compose.yml up -d --build`.

### Installing Docker

| System | How |
|---|---|
| Linux (including WSL 2) | Run `./deploy/install-docker.sh`. It uses Docker's official install script to install Docker Engine and the Compose plugin, starts the service and enables it at boot, and adds your user to the `docker` group. Every step runs with sudo, and the script lists the steps and asks for confirmation first. On derivatives the official script doesn't recognize (Alibaba Cloud Linux, Rocky Linux, Linux Mint…), it installs from Docker's repository for the distribution they're based on; if it can't tell what that is, it points you to the [manual instructions](https://docs.docker.com/engine/install/) instead |
| macOS | Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/), or run `brew install --cask docker` |
| Windows | Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) (WSL 2 backend), then run `asamana.sh` in WSL 2 or Git Bash |

Joining the `docker` group takes effect after you log in again. Note that members of the `docker` group effectively have root access.

## Quick start

```bash
./deploy/asamana.sh start
```

If `deploy/.env` is missing, the script starts the `setup` wizard. API key configuration is optional: if enabled, you choose a config and enter its required keys (input is hidden). You can also enable developer tools. The wizard writes `deploy/.env` with permissions `600` (owner read/write only) and starts the services. Open <http://localhost:8080/>.

In a non-interactive environment such as CI, the wizard cannot prompt for input. Copy `deploy/.env.example` to `deploy/.env` and configure it before starting the services.

- The frontend binds to `127.0.0.1:8080`. The API has **no authentication**: anyone who can reach this port can perform all API operations using your configured keys. Add access controls, such as an authenticating reverse proxy or VPN, before exposing it to others.
- The backend is not exposed to the host by default; the frontend reaches it through the Compose network at `http://backend:7860`. To access the API directly, uncomment `ports:` under `backend` in `docker-compose.yml`.
- The service starts without keys too: existing worlds can be browsed and replayed, while anything that calls a model, such as creating or running a world, is disabled. See "With and without keys" in the README for details.
- Configure all required API keys or leave them all unset. With a partial set, the backend refuses to start and lists the missing variables.
- If `GET /api/templates` returns an empty list, there are no world maps and no new worlds can be created.

## Choosing a config

`ASAMANA_CONFIG` in `deploy/.env` selects the backend config; the default is `config/config.yaml` (Alibaba Cloud Model Studio). Run `./deploy/asamana.sh keys` to choose a provider and enter its keys, or edit `deploy/.env` directly:

```bash
# deploy/.env
ASAMANA_CONFIG=config/presets/deepseek.yaml
DEEPSEEK_API_KEY=sk-...
EMBEDDING_API_KEY=sk-...
```

The keys each vendor preset needs are listed at the top of the preset file. See [Configuration](configuration.md) for how to write a config.

`config/` is included in the image, so run `./deploy/asamana.sh rebuild backend` after changing it. The exception is `config/content.yaml` (suggested themes on the home page), which is mounted from the host; edit it and refresh the page.

## Day-to-day operations

Run `deploy/asamana.sh <command>` from the repository root, or use the script's absolute path from another directory:

| Command | What it does |
|---|---|
| `setup` | Wizard: pick a config, enter keys (optional), turn developer tools on or off; writes `deploy/.env` |
| `keys` | Add or change the model vendor and keys, rewriting only the model-related lines in `.env`; a running backend is rebuilt automatically to pick up the new keys |
| `start` | Start in the background (builds images if missing; runs `setup` first if there's no `.env`) |
| `stop` | Stop containers without removing them |
| `restart [svc]` | Restart everything or one service (`backend` / `frontend`) |
| `rebuild [svc]` | Rebuild images and start — use after changing code or `config/` |
| `down` | Stop and remove containers and the network (`data/` is kept) |
| `reset` | Remove containers, the network, locally built images, `data/` and `deploy/.env` (including API keys). Irreversible; type `reset` to confirm |
| `logs [svc]` | Follow logs |
| `status` | Show service status |

Applying changes:

| What changed | How to apply it |
|---|---|
| Frontend or backend code | `./deploy/asamana.sh rebuild` (you can rebuild just one service) |
| `config/*.yaml` | `./deploy/asamana.sh rebuild backend` |
| `deploy/.env` (including a fresh `setup`) | `./deploy/asamana.sh start` (recreates containers with the new environment; `restart` doesn't read the new `.env`) |
| Keys changed with `keys` | Applied automatically if the backend is running; otherwise `./deploy/asamana.sh start` |
| `config/content.yaml` | Refresh the page |

## Developer tools

Enable them in `setup`, or set `ASAMANA_DEV_TOOLS=true` in `deploy/.env`, then run `./deploy/asamana.sh start`.

- The backend enables the API routes used by the developer tools (`#/dev`) and map workbench (`#/lab`). These tools spawn subprocesses and send arbitrary prompts to the LLM using your token budget. **Enable them only in a private development environment.**
- `tuning/scenarios` and `worlds/templates` are always mounted read-write. Scenarios saved and maps imported through the developer tools are written to the host repository, where Git can track them and they survive rebuilds. These tools do not write to either directory when disabled.

## Data persistence

All backend runtime state (worlds, snapshots, vectors, traces, logs) is stored under `data/` in the repository root, mounted at `/app/data` in the container. `stop`, `restart` and `down` (without `-v`) preserve this data. To delete all runtime state, run `./deploy/asamana.sh reset`.

On the first start, example worlds are copied from `examples/` to `data/`, and `data/.examples_seeded` records that the import is complete. Deleted examples stay deleted, even if you remove `data/worlds.json`. Clearing all of `data/` (for example, with `reset`) allows the next start to import them again.

When the backend stops or restarts, a running world finishes its current step before the backend exits. Operations such as `stop`, `restart` and `rebuild` can therefore take up to 3 minutes (`stop_grace_period`). After that, Docker terminates the process; the interrupted step is re-run when the world is restored.

## Deploying the frontend and backend separately (not recommended)

**Backend** — on any container platform:

```bash
docker build -f deploy/backend.Dockerfile -t asamana-backend .
docker run -p 127.0.0.1:7860:7860 --env-file deploy/.env -v "$PWD/data:/app/data" asamana-backend
```

**Frontend** — two options:

1. **nginx image (reverse-proxying to the backend)**, with the backend address set at runtime:
   ```bash
   docker build -f deploy/frontend.Dockerfile -t asamana-frontend frontend/
   docker run -p 127.0.0.1:8080:80 -e BACKEND_URL=https://api.example.com asamana-frontend
   ```
2. **Static hosting / CDN (no reverse proxy)**: compile the API address into the frontend and allow that origin on the backend via CORS:
   ```bash
   cd frontend && VITE_API_BASE=https://api.example.com npm run build
   # Deploy frontend/dist/ to any static host
   # On the backend, set ASAMANA_CORS_ORIGINS=https://app.example.com
   ```

With `VITE_API_BASE` empty (the default), the frontend uses same-origin relative paths behind the nginx proxy. Set it to a full URL for the CDN setup; the WebSocket address is derived from it automatically.

## Environment variables

| Variable | Used by | Description |
|---|---|---|
| `LLM_API_KEY` | backend | Read by LLM providers that don't set `api_key_env` (as in the default `config/config.yaml`) |
| `EMBEDDING_API_KEY` | backend | The embedding key (separate; can be a different vendor or account) |
| Vendor keys | backend | An LLM provider that names its own variable via `api_key_env` reads that variable. Every preset under `config/presets/` names one (e.g. `DEEPSEEK_API_KEY`) and lists the keys it needs at the top of the file |
| `ASAMANA_CONFIG` | Docker Compose | Which config file the backend loads; defaults to `config/config.yaml` |
| `ASAMANA_DEV_TOOLS` | backend | `true` enables the developer tools (see above); off by default |
| `ASAMANA_CORS_ORIGINS` | backend | Allowed browser origins (comma-separated), default `*`; needed for the CDN setup |
| `BACKEND_URL` | frontend (nginx) | Backend address to proxy to; defaults to `http://backend:7860` |
| `VITE_API_BASE` | frontend (build arg) | Compiles a remote API address into the frontend; empty means relative paths |

## Ports

| Port | Service |
|---|---|
| `8080` | Frontend (nginx), the app's entry point |
| `7860` | Backend (API + WebSocket); internal network only under Docker Compose by default |
| `5173` | Vite dev server (local development only) |
