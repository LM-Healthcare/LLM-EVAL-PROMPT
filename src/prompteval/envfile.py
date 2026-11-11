"""Minimal .env loader (no extra dependency).

docker compose reads `.env` automatically, but a native run (conda/venv) does
not: this module gives the CLI the same behaviour. Variables already set in
the environment with a non-empty value are never overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    # unquoted: drop an inline comment introduced by " #"
    for marker in (" #", "\t#"):
        if marker in raw:
            raw = raw.split(marker, 1)[0]
    return raw.strip()


def parse_env_file(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        if key.isidentifier():
            values[key] = _parse_value(raw)
    return values


def find_env_file(config_path: str | Path | None = None) -> Path | None:
    """First `.env` found in: the current directory, the repository holding the
    config file (configs/<file>.yaml -> repository root), the package's repository."""
    candidates = [Path.cwd() / ".env"]
    if config_path:
        cfg = Path(config_path).resolve()
        candidates += [cfg.parent / ".env", cfg.parent.parent / ".env"]
    candidates.append(Path(__file__).resolve().parents[2] / ".env")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_env_file(config_path: str | Path | None = None) -> tuple[Path | None, list[str]]:
    """Load the first `.env` found; return its path and the names of the variables set."""
    path = find_env_file(config_path)
    if path is None:
        return None, []
    loaded = []
    for key, value in parse_env_file(path).items():
        if not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return path, loaded
