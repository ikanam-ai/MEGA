"""Загрузка config/*.yaml и переменных окружения из .env."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"

load_dotenv(ROOT_DIR / ".env")


@cache
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def models_config() -> dict[str, Any]:
    return load_yaml("models.yaml")


def prompts_config() -> dict[str, Any]:
    return load_yaml("prompts.yaml")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Переменная окружения {name} не задана. Проверьте .env (см. .env.example).")
    return value
