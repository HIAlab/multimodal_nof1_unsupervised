from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


CONFIG_ENV_VAR = "MULTIMODAL_CONFIG_PATH"
DEFAULT_CONFIG_NAME = "pilot_parameters.yaml"
ALIASES = {
    "pilot": "pilot_parameters.yaml",
    "pilot_parameters": "pilot_parameters.yaml",
    "pilot_parameters.yaml": "pilot_parameters.yaml",
    "sim": "sim_parameters.yaml",
    "simulation": "sim_parameters.yaml",
    "sim_parameters": "sim_parameters.yaml",
    "sim_parameters.yaml": "sim_parameters.yaml",
}


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_path(config: str | None = None) -> Path:
    candidate = config or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_NAME
    candidate = ALIASES.get(candidate, candidate)
    path = Path(candidate)
    if not path.is_absolute():
        path = workflow_root() / path
    return path.resolve()


def set_active_config(config: str | None = None) -> Path:
    path = resolve_config_path(config)
    os.environ[CONFIG_ENV_VAR] = str(path)
    return path


def load_config(config: str | None = None) -> dict:
    config_path = set_active_config(config)
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_CONFIG_PATH"] = str(config_path)
    data["_WORKFLOW_ROOT"] = str(workflow_root())
    return data


def resolve_path(value: str | None, config: dict | None = None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    base = workflow_root()
    if config and config.get("_CONFIG_PATH"):
        base = Path(config["_CONFIG_PATH"]).parent
    return str((base / path).resolve())


def parse_range_like(value):
    if isinstance(value, str) and value.startswith("range(") and value.endswith(")"):
        args = [int(x.strip()) for x in value[6:-1].split(",") if x.strip()]
        return list(range(*args))
    return value


def ensure_sys_path(path_value: str | None, config: dict | None = None) -> None:
    if not path_value:
        return
    resolved = resolve_path(path_value, config)
    if resolved and Path(resolved).exists() and resolved not in sys.path:
        sys.path.append(resolved)
