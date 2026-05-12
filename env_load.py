"""Load .env-style files from several common locations (later files override earlier ones)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def candidate_env_paths(project_base: Path) -> List[Path]:
    paths: List[Path] = []
    raw = (os.getenv("DOTENV_PATH") or "").strip()
    if raw:
        paths.append(Path(raw).expanduser())
    paths.append(project_base.parent / "rag" / ".env")
    paths.append(project_base.parent / ".env")
    paths.append(project_base / ".env")
    return paths


def format_env_search_list(project_base: Path) -> str:
    return "; ".join(f"{p} ({'exists' if p.exists() else 'missing'})" for p in candidate_env_paths(project_base))


def load_env_file(env_path: Path) -> bool:
    if not env_path.exists():
        return False
    text = env_path.read_text(encoding="utf-8-sig")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
            if "=" not in line:
                continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value
    return True


def load_dotenv_for_project(project_base: Path) -> List[Path]:
    """Apply env files in order; later files override earlier keys. Returns paths that were read."""
    loaded: List[Path] = []
    for path in candidate_env_paths(project_base):
        if load_env_file(path):
            loaded.append(path)
    return loaded
