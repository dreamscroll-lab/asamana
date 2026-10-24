"""Deployment key writes secure both newly created and existing environment files."""

from __future__ import annotations

from pathlib import Path
import stat
import subprocess

import pytest


@pytest.mark.parametrize("command", ["setup", "keys"])
@pytest.mark.parametrize("existing", [False, True])
def test_key_writes_make_env_owner_only(tmp_path: Path, command: str, existing: bool) -> None:
    source = Path("deploy/asamana.sh").read_text(encoding="utf-8")
    functions = source[source.index("setup() {") : source.index("# Irreversible,")]
    env_file = tmp_path / ".env"
    if existing:
        env_file.write_text("EXTRA=retained\n", encoding="utf-8")
        env_file.chmod(0o644)
    shell = """\
set -euo pipefail
KEYLESS_LINE='# No keys'
require_tty() { return 0; }
ask_model_lines() { MODEL_LINES=('LLM_API_KEY=test-secret'); }
has_model_keys() { return 0; }
model_var_names() { echo LLM_API_KEY; }
docker() { return 1; }
""" + functions + "\n" + command
    answers = ("y\n" if existing else "") + "y\nn\nn\n"
    subprocess.run(
        ["bash", "-c", shell], cwd=tmp_path, input=answers,
        text=True, capture_output=True, check=True,
    )
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert "LLM_API_KEY=test-secret" in env_file.read_text(encoding="utf-8")
    if command == "keys" and existing:
        assert "EXTRA=retained" in env_file.read_text(encoding="utf-8")
