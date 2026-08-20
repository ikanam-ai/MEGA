from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


TAPE_SUBSETS = [
    "chegeka",
    "multiq",
    "per_ethics",
    "ru_openbook",
    "ru_worldtree",
    "sit_ethics",
    "winograd",
]


def iter_items(local_path: str | Path, subset: str, split: str = "test") -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the 'annotation' extras: `poetry install --with annotation`"
        ) from exc

    ds = load_dataset(str(local_path), subset, split=split, trust_remote_code=True)
    for idx, row in enumerate(ds):
        yield {
            "item_id": f"tape_{subset}_{split}_{idx:06d}",
            "source_dataset": "tape",
            "source_subset": subset,
            "source_split": split,
            "text": _extract_text(row, subset),
            "raw": row,
        }


def _extract_text(row: dict[str, Any], subset: str) -> str:
    return str(row.get("inputs", row.get("question", "")))
