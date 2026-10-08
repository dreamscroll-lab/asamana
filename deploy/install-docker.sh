#!/usr/bin/env bash
#
# Install Docker Engine and the Compose v2 plugin on Linux (including WSL 2). macOS and
# Windows use Docker Desktop instead; see docs/deployment.md.
#
# Distributions Docker supports directly use its official install script. Derivatives it
# doesn't recognize (Alibaba Cloud Linux, Rocky, Linux Mint…) get Docker's package repository
# for the distribution they are based on.
#
# Usage: ./install-docker.sh [--mirror] [--yes]
#   --mirror   download packages from the Aliyun mirror (for mainland China)
#   --yes      don't ask for confirmation
#
# Every step runs with sudo, and nothing runs before you confirm.
#
set -euo pipefail

INSTALL_SCRIPT_URL="https://get.docker.com"
# The distributions Docker's install script accepts, matched on ID alone: it rejects any other
# ID, even one whose ID_LIKE names a supported base.
SCRIPT_DISTROS="ubuntu debian raspbian centos rhel fedora sles"
MANUAL_DOCS="https://docs.docker.com/engine/install/"
PACKAGES="docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin"

usage() {
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
  exit "${1:-0}"
}

mirror=false
assume_yes=false
for arg in "$@"; do
  case "$arg" in
    --mirror) mirror=true ;;
    --yes|-y) assume_yes=true ;;
    -h|--help) usage 0 ;;
    *) echo "error: unknown option '$arg'" >&2; usage 1 ;;
  esac
done

if [[ "$(uname -s)" != Linux ]]; then
  echo "This script is for Linux. On macOS and Windows, install Docker Desktop:" >&2
  echo "  https://docs.docker.com/desktop/" >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Docker and Compose v2 are already installed:"
  docker --version
  docker compose version
  exit 0
fi

if $mirror; then
  repo_base="https://mirrors.aliyun.com/docker-ce"
else
  repo_base="https://download.docker.com"
fi

has_word() { [[ " $1 " == *" $2 "* ]]; }

ID="" ID_LIKE="" PRETTY_NAME="" PLATFORM_ID="" VERSION_ID="" VERSION_CODENAME="" UBUNTU_CODENAME=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
fi

# script: Docker's install script. rpm / apt: Docker's repository for the base distribution.
method=""
if has_word "$SCRIPT_DISTROS" "$ID"; then
  method=script
elif has_word "$ID_LIKE" rhel || has_word "$ID_LIKE" centos; then
  method=rpm
  # Docker's CentOS repository is laid out by RHEL major version. A derivative's own version
  # can differ (Alibaba Cloud Linux 3 is RHEL 8), so take it from PLATFORM_ID ("platform:al8").
  el_version="${PLATFORM_ID##*[!0-9]}"
  [[ -n "$el_version" ]] || el_version="${VERSION_ID%%.*}"
  if command -v dnf >/dev/null 2>&1; then pkg=dnf; else pkg=yum; fi
elif has_word "$ID_LIKE" ubuntu || has_word "$ID_LIKE" debian; then
  method=apt
  if has_word "$ID_LIKE" ubuntu; then
    apt_distro=ubuntu
    codename="${UBUNTU_CODENAME:-}"
  else
    apt_distro=debian
    codename="${VERSION_CODENAME:-}"
  fi
fi
if [[ -z "$method" ]] || { [[ "$method" == rpm ]] && [[ -z "$el_version" ]]; } \
  || { [[ "$method" == apt ]] && [[ -z "$codename" ]]; }; then
  echo "error: no supported way to install Docker on ${PRETTY_NAME:-this distribution}." >&2
  echo "Install Docker Engine and the Compose plugin by hand: $MANUAL_DOCS" >&2
  exit 1
fi

for tool in curl sudo; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: '$tool' is required." >&2
    exit 1
  fi
done

# Under `sudo ./install-docker.sh`, $USER is root; the group change is for whoever ran sudo.
target_user="${SUDO_USER:-$USER}"

has_systemd=false
[[ -d /run/systemd/system ]] && has_systemd=true

source_note=""
$mirror && source_note=" from the Aliyun mirror"
echo "This will, using sudo:"
case "$method" in
  script)
    echo "  1. download Docker's install script from $INSTALL_SCRIPT_URL and run it to install"
    echo "     Docker Engine and the Compose plugin$source_note" ;;
  rpm)
    echo "  1. add Docker's CentOS $el_version repository$source_note and install Docker Engine"
    echo "     and the Compose plugin with $pkg (${PRETTY_NAME:-$ID} is based on RHEL $el_version)" ;;
  apt)
    echo "  1. add Docker's $apt_distro ($codename) repository$source_note and install Docker Engine"
    echo "     and the Compose plugin with apt (${PRETTY_NAME:-$ID} is based on $apt_distro $codename)" ;;
esac
if $has_systemd; then
  echo "  2. start the Docker service and enable it at boot"
else
  echo "  2. start the Docker service"
fi
echo "  3. add $target_user to the docker group, so Docker works without sudo"
echo "     (members of that group effectively have root access)"
if ! $assume_yes; then
  if [[ ! -t 0 ]]; then
    echo "error: no terminal to confirm in; rerun with --yes." >&2
    exit 1
  fi
  read -r -p "Continue? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Aborted; nothing was changed."; exit 1; }
fi

download_failed() {
  echo "error: could not download $1." >&2
  if $mirror; then
    echo "Follow the Aliyun mirror's instructions instead: https://developer.aliyun.com/mirror/docker-ce" >&2
  else
    echo "Install by hand instead: $MANUAL_DOCS (in mainland China, retry with --mirror)" >&2
  fi
  exit 1
}

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
case "$method" in
  script)
    curl -fsSL "$INSTALL_SCRIPT_URL" -o "$tmp" || download_failed "$INSTALL_SCRIPT_URL"
    if $mirror; then
      sudo sh "$tmp" --mirror Aliyun
    else
      sudo sh "$tmp"
    fi
    ;;
  rpm)
    curl -fsSL "$repo_base/linux/centos/docker-ce.repo" -o "$tmp" \
      || download_failed "$repo_base/linux/centos/docker-ce.repo"
    # Pin the RHEL version and the chosen host: the repo file uses the system's $releasever
    # and points at download.docker.com.
    sed -e "s/\$releasever/$el_version/g" -e "s#https://download.docker.com#$repo_base#g" "$tmp" \
      | sudo tee /etc/yum.repos.d/docker-ce.repo >/dev/null
    sudo "$pkg" -y install $PACKAGES
    ;;
  apt)
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL "$repo_base/linux/$apt_distro/gpg" -o /etc/apt/keyrings/docker.asc \
      || download_failed "$repo_base/linux/$apt_distro/gpg"
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] $repo_base/linux/$apt_distro $codename stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y $PACKAGES
    ;;
esac

docker_started=true
if $has_systemd; then
  sudo systemctl enable --now docker
elif command -v service >/dev/null 2>&1; then
  sudo service docker start
else
  docker_started=false
fi

sudo usermod -aG docker "$target_user"

echo
docker --version
sudo docker compose version
echo
if ! $docker_started; then
  echo "This system has neither systemd nor \`service\`: start the daemon yourself (sudo dockerd)."
fi
echo "Docker is installed. Log out and back in (or run: newgrp docker) so the group change"
echo "takes effect, then run ./deploy/asamana.sh start"
if $mirror; then
  echo
  echo "Images are still pulled from Docker Hub. If that is slow, see 中国大陆的镜像源"
  echo "in docs/deployment.zh-CN.md."
fi
