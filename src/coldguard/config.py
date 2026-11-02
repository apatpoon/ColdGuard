from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config, config_path.parent.parent


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def require_keys(config: dict[str, Any], section: str, keys: list[str]) -> None:
    if section not in config or not isinstance(config[section], dict):
        raise ValueError(f"Missing configuration section: {section}")
    missing = [key for key in keys if key not in config[section]]
    if missing:
        raise ValueError(f"Missing keys in section {section}: {missing}")

