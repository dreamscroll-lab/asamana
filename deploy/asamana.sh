#!/usr/bin/env bash
#
# Asamana deployment manager — wraps docker compose for the two-service
# (backend + frontend) stack. Runnable from anywhere; it cd's into deploy/ so
# the compose file's relative paths (.env, ../data) resolve correctly.
#
# Usage: ./asamana.sh <command> [service]
#   setup           write deploy/.env: model vendor and API keys (skippable), developer tools,
#                   build mirrors for mainland China (optional)
#   keys            add or change the model vendor and API keys, then apply them to a running backend
#   start           start containers in the background (runs setup first if there is no .env)
#   stop            stop containers (keep them; fast restart)
#   restart [svc]   restart all services, or one (backend|frontend)
#   rebuild [svc]   rebuild image(s) and start (use after code changes)
#   down            stop and remove containers + network (data/ is kept)
#   reset           back to a fresh checkout: remove containers, images, data/ and .env
#   logs [svc]      follow logs (all services, or one)
#   status          show service status
#
set -euo pipefail

# Resolve the script's own path before cd, so usage() can read it afterwards.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"

usage() {
  # Print the header comment block (skip the shebang, stop at the first code line).
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$SELF"
  exit "${1:-0}"
}

# macos | linux | wsl | windows (Git Bash / MSYS) | other
host_os() {
  case "$(uname -s)" in
    Darwin) echo macos ;;
    Linux) grep -qi microsoft /proc/version 2>/dev/null && echo wsl || echo linux ;;
    MINGW*|MSYS*|CYGWIN*) echo windows ;;
    *) echo other ;;
  esac
}

docker_install_hint() {
  case "$(host_os)" in
    linux)
      echo "Install it with: ./deploy/install-docker.sh   (add --mirror for the Aliyun mirror in mainland China)" ;;
    macos)
      echo "Install Docker Desktop: https://docs.docker.com/desktop/setup/install/mac-install/"
      echo "  or with Homebrew:     brew install --cask docker" ;;
    wsl)
      echo "Install Docker Desktop on Windows and enable WSL integration for this distro:"
      echo "  https://docs.docker.com/desktop/features/wsl/"
      echo "  or install Docker inside WSL: ./deploy/install-docker.sh" ;;
    windows)
      echo "Install Docker Desktop (WSL 2 backend): https://docs.docker.com/desktop/setup/install/windows-install/" ;;
    *)
      echo "See https://docs.docker.com/engine/install/" ;;
  esac
}

# Bring the daemon up where that needs no privileges (a desktop app on macOS); elsewhere say how.
start_docker() {
  local app=""
  if [[ "$(host_os)" == macos ]]; then
    if [[ -d /Applications/Docker.app ]]; then app=Docker
    elif [[ -d /Applications/OrbStack.app ]]; then app=OrbStack
    fi
  fi
  if [[ -z "$app" ]]; then
    echo "error: Docker is installed but its daemon is not running." >&2
    case "$(host_os)" in
      linux) echo "Start it with: sudo systemctl start docker" >&2 ;;
      wsl) echo "Start Docker Desktop on Windows, or inside WSL: sudo service docker start" >&2 ;;
      windows) echo "Start Docker Desktop and wait until it reports that it is running." >&2 ;;
      macos) echo "Start your Docker runtime (e.g. colima start)." >&2 ;;
    esac
    return 1
  fi
  printf 'Starting %s' "$app"
  open -a "$app"
  local waited=0
  until docker info >/dev/null 2>&1; do
    if (( waited >= 120 )); then
      echo
      echo "error: $app did not come up within 120s; start it by hand and retry." >&2
      return 1
    fi
    sleep 2; waited=$((waited + 2)); printf '.'
  done
  echo " ready."
}

# Fail early, and say what to do, when Docker or Compose v2 isn't usable.
require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: Docker is not installed." >&2
    docker_install_hint >&2
    exit 1
  fi
  local info
  if ! info=$(docker info 2>&1); then
    if grep -qi "permission denied" <<<"$info"; then
      echo "error: this user may not talk to the Docker daemon." >&2
      echo "Add it to the docker group, then log out and back in: sudo usermod -aG docker \"\$USER\"" >&2
      exit 1
    fi
    start_docker || exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "error: Docker Compose v2 (the \`docker compose\` plugin) is not available." >&2
    docker_install_hint >&2
    exit 1
  fi
}

# The configs on offer, default first. Paths are relative to the repo root — that is
# what ASAMANA_CONFIG holds and what the backend resolves inside the image.
list_configs() {
  echo "config/config.yaml"
  for f in ../config/presets/*.yaml; do
    [[ -e "$f" ]] && echo "${f#../}"
  done
}

# A config's menu label is the first line of its header comment, so keep that line a short name.
config_label() {
  head -1 "../$1" | sed -e 's/^# *//' -e 's/。.*//' -e 's/配置$//'
}

# The key variables a config reads, one per line. Read from the config itself rather than
# kept in a table here, so a new preset needs no edit to this script. The defaults mirror
# config/models.py (LLM_API_KEY) and core/interfaces/embedding.py (EMBEDDING_API_KEY).
llm_key_vars() {
  local vars
  vars=$(awk '/^[a-z_]+:/ {s=$1} s=="llm:" && /^[[:space:]]*api_key_env:/ {print $2}' "../$1" | sort -u)
  echo "${vars:-LLM_API_KEY}"
}
embedding_key_var() {
  local var
  var=$(awk '/^[a-z_]+:/ {s=$1} s=="embedding:" && /^[[:space:]]*api_key_env:/ {print $2}' "../$1" | head -1)
  echo "${var:-EMBEDDING_API_KEY}"
}
# Non-secret values a config requires through ${VAR} with no default (e.g. a workspace id inside
# a base_url). The backend refuses to start without them, so setup must ask.
config_vars() {
  { grep -o '\${[A-Z0-9_]*}' "../$1" || true; } | sed -e 's/^\${//' -e 's/}$//' | sort -u
}
embedding_provider() {
  awk '/^[a-z_]+:/ {s=$1} s=="embedding:" && /^[[:space:]]*provider:/ {print $2; exit}' "../$1"
}

# Prompt for one key without echoing it; insists on a non-empty value.
read_key() {
  local var="$1" what="$2" value
  while true; do
    read -r -s -p "$var ($what): " value
    echo >&2
    [[ -n "$value" ]] && break
    echo "A key is required." >&2
  done
  printf '%s' "$value"
}

# Written into a keyless .env so the file says why it holds no keys; `keys` drops it.
KEYLESS_LINE="# No model API keys yet: run ./asamana.sh keys to add them."

# Build-time mirrors offered by setup; written to .env and passed to the image builds.
CN_PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
CN_NPM_REGISTRY="https://registry.npmmirror.com"

# What a keyless backend can and cannot do. Printed whenever one is set up or started, so the
# disabled buttons in the browser come as no surprise.
keyless_notice() {
  cat <<'EOF'

No model API keys are configured.
  Available: browsing and replaying existing worlds (map, characters, relations, narrative),
             and the developer tools that only read (traces, audit reports, map workbench).
  Disabled:  building a world, running, stepping, directing or resetting one, and the
             developer tools that call a model (prompt replay, audit/stage runs, recall).
Add keys any time with: ./asamana.sh keys
EOF
}

# Does .env hold model keys? The backend accepts only all or none, so the config's first LLM
# key variable answers for the rest.
has_model_keys() {
  local config var
  config=$(sed -n 's/^ASAMANA_CONFIG=//p' .env | tail -1)
  var=$(llm_key_vars "${config:-config/config.yaml}" | head -1)
  grep -qE "^${var}=.+" .env
}

# Every variable the model part of .env can hold, across all configs. `keys` drops these before
# writing the new vendor's, so a previous vendor's key doesn't linger.
model_var_names() {
  local c
  echo ASAMANA_CONFIG
  while IFS= read -r c; do
    llm_key_vars "$c"
    embedding_key_var "$c"
    config_vars "$c"
  done < <(list_configs)
}

# Fills MODEL_LINES with the model part of .env: the vendor config, its keys, and any value that
# config requires. A global rather than stdout, since the prompts print there too.
MODEL_LINES=()
ask_model_lines() {
  local configs=() c i=1
  while IFS= read -r c; do configs+=("$c"); done < <(list_configs)

  echo "Which model vendor should Asamana use?"
  for c in "${configs[@]}"; do
    printf '  %d) %s  (%s)\n' "$i" "$(config_label "$c")" "$c"
    i=$((i + 1))
  done
  local choice
  while true; do
    read -r -p "Choose [1-${#configs[@]}, default 1]: " choice
    choice="${choice:-1}"
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#configs[@]} )); then
      break
    fi
    echo "Enter a number between 1 and ${#configs[@]}."
  done
  local config="${configs[$((choice - 1))]}"

  MODEL_LINES=("ASAMANA_CONFIG=$config")
  local var
  echo
  echo "API keys are stored in deploy/.env only (gitignored, readable by you alone)."
  for var in $(llm_key_vars "$config"); do
    MODEL_LINES+=("$var=$(read_key "$var" "LLM")")
  done
  # Asked for even when it is the same account: the script cannot tell, and an LLM
  # vendor's key sent to the embedding endpoint fails only once a world is built.
  var=$(embedding_key_var "$config")
  MODEL_LINES+=("$var=$(read_key "$var" "embedding: $(embedding_provider "$config")")")
  local value
  for var in $(config_vars "$config"); do
    value=""
    while [[ -z "$value" ]]; do
      read -r -p "$var (required by $config): " value
    done
    MODEL_LINES+=("$var=$value")
  done
}

require_tty() {
  if [[ ! -t 0 ]]; then
    echo "error: $1 is interactive. Without a terminal, copy .env.example to .env and fill it in." >&2
    exit 1
  fi
}

setup() {
  require_tty setup
  if [[ -f .env ]]; then
    local answer
    read -r -p "deploy/.env already exists. Overwrite it? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { echo "Kept the existing .env."; return 0; }
  fi

  local lines=() now
  echo "Asamana runs on an LLM and an embedding model, called with your own API keys."
  echo "Without keys you can still start it and look around, but not build or run a world."
  read -r -p "Configure API keys now? [Y/n] " now
  if [[ "$now" =~ ^[Nn]$ ]]; then
    lines=("$KEYLESS_LINE")
  else
    echo
    ask_model_lines
    lines=("${MODEL_LINES[@]}")
  fi

  local dev
  echo
  read -r -p "Enable developer tools (trace, audit, prompt replay)? Only on a machine nobody else can reach. [y/N] " dev
  if [[ "$dev" =~ ^[Yy]$ ]]; then
    lines+=("ASAMANA_DEV_TOOLS=true")
  fi

  local cn hub
  echo
  read -r -p "Build through mirrors in mainland China (pip: Tsinghua, npm: npmmirror)? [y/N] " cn
  if [[ "$cn" =~ ^[Yy]$ ]]; then
    lines+=("PIP_INDEX_URL=$CN_PIP_INDEX" "NPM_REGISTRY=$CN_NPM_REGISTRY")
    echo "Base images still come from Docker Hub. If it is unreachable, enter a Docker Hub mirror"
    read -r -p "host (e.g. docker.m.daocloud.io), or leave empty: " hub
    if [[ -n "$hub" ]]; then
      lines+=("IMAGE_REGISTRY=${hub%/}/library/")
    fi
  fi

  (umask 077 && printf '%s\n' "${lines[@]}" > .env)
  echo
  if has_model_keys; then
    echo "Wrote deploy/.env. Run ./asamana.sh setup again to start over, or ./asamana.sh keys to change the vendor or keys."
  else
    echo "Wrote deploy/.env without model API keys."
    keyless_notice
  fi
}

# Replaces only the model part of .env; developer tools and anything hand-added stay.
keys() {
  require_tty keys
  ask_model_lines

  local names kept=() line
  names=$(model_var_names | sort -u)
  if [[ -f .env ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ "$line" == "$KEYLESS_LINE" ]] && continue
      if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)= ]] && grep -qx "${BASH_REMATCH[1]}" <<<"$names"; then
        continue
      fi
      kept+=("$line")
    done < .env
  fi
  (umask 077 && printf '%s\n' "${MODEL_LINES[@]}" "${kept[@]+"${kept[@]}"}" > .env)
  echo
  echo "Wrote the model keys to deploy/.env."

  # `up`, not `restart`: a restarted container keeps the environment it was created with, so
  # only recreating it reads the new .env.
  if docker info >/dev/null 2>&1 && [[ -n "$(docker compose ps -q --status running backend 2>/dev/null)" ]]; then
    docker compose up -d backend
    echo "Backend restarted with the new keys. Reload the page."
  else
    echo "Run ./asamana.sh start to use them."
  fi
}

# Irreversible, so it asks for the word itself rather than a y/N a stray Enter could answer.
reset() {
  if [[ ! -t 0 ]]; then
    echo "error: reset is interactive and cannot run without a terminal." >&2
    exit 1
  fi
  local data
  data="$(cd .. && pwd)/data"
  echo "This permanently deletes:"
  echo "  - the Asamana containers, network and locally built images"
  echo "  - every world, snapshot, vector, trace and log under $data"
  echo "  - deploy/.env, including your API keys"
  local answer
  read -r -p "Type 'reset' to confirm: " answer
  [[ "$answer" == "reset" ]] || { echo "Aborted; nothing was deleted."; exit 1; }

  docker compose down --rmi local --volumes --remove-orphans
  if ! rm -rf "$data"; then
    # On Linux the container writes data/ as root, which the host user cannot delete.
    echo "error: could not delete $data — remove it with: sudo rm -rf \"$data\"" >&2
    exit 1
  fi
  rm -f .env
  echo "Reset complete. Run ./asamana.sh start to set up again."
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  setup)
    setup
    ;;
  keys)
    keys
    ;;
  start)
    require_docker
    [[ -f .env ]] || setup
    docker compose up -d "$@"
    echo "Asamana is starting at http://localhost:8080/"
    has_model_keys || keyless_notice
    ;;
  stop)
    require_docker
    docker compose stop "$@"
    ;;
  restart)
    require_docker
    docker compose restart "$@"
    ;;
  rebuild)
    require_docker
    [[ -f .env ]] || setup
    docker compose up -d --build "$@"
    has_model_keys || keyless_notice
    ;;
  reset)
    require_docker
    reset
    ;;
  down)
    require_docker
    docker compose down "$@"
    ;;
  logs)
    require_docker
    docker compose logs -f "$@"
    ;;
  status|ps)
    require_docker
    docker compose ps "$@"
    ;;
  help|-h|--help)
    usage 0
    ;;
  *)
    echo "error: unknown command '$cmd'" >&2
    usage 1
    ;;
esac
