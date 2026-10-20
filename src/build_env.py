"""Read KEY=VALUE lines from a gitignored .env next to config.py (no extra package)."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_simple() -> None:
    f = Path(__file__).resolve().parent.parent / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))
