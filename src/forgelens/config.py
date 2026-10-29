"""Config: packaged YAML defaults < optional user YAML < environment < CLI flags."""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "FORGELENS_"


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _env_overrides(defaults: dict, environ: dict[str, str]) -> dict:
    """FORGELENS_REDACT_PAD_PX=10 -> {"redact": {"pad_px": 10}}. Values parsed as YAML."""
    out: dict[str, Any] = {}
    for section, body in defaults.items():
        keys = body.keys() if isinstance(body, dict) else [None]
        for key in keys:
            name = ENV_PREFIX + (f"{section}_{key}" if key else section).upper()
            if name in environ:
                value = yaml.safe_load(environ[name])
                if key:
                    out.setdefault(section, {})[key] = value
                else:
                    out[section] = value
    return out


def load_config(
    user_file: str | Path | None = None,
    cli: dict | None = None,
    environ: dict[str, str] | None = None,
) -> dict:
    text = resources.files("forgelens").joinpath("default_config.yaml").read_text()
    cfg = yaml.safe_load(text)
    defaults = copy.deepcopy(cfg)
    if user_file:
        _deep_update(cfg, yaml.safe_load(Path(user_file).read_text()) or {})
    _deep_update(cfg, _env_overrides(defaults, dict(os.environ if environ is None else environ)))
    if cli:
        _deep_update(cfg, {k: v for k, v in cli.items() if v is not None})
    return cfg
