from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if overrides:
        config = recursive_update(config, overrides)
    return config


def recursive_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = recursive_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def require_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in config or not isinstance(config[key], Mapping):
        raise KeyError(f"Missing config section: {key}")
    return config[key]
