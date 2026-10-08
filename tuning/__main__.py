"""``python -m tuning`` entrypoint."""

from __future__ import annotations

import sys

from tuning.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
