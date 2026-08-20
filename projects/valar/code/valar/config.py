from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if "experiment" not in cfg:
        raise ValueError(f"Missing 'experiment' key in {path}")
    return cfg


def load_run_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if "annotation_run" not in cfg:
        raise ValueError(f"Missing 'annotation_run' key in {path}")
    return cfg


def load_model_registry(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if "model_registry" not in cfg:
        raise ValueError(f"Missing 'model_registry' key in {path}")
    return cfg
