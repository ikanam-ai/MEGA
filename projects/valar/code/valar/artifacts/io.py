from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write dictionaries as UTF-8 JSON Lines, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write dictionaries as UTF-8 CSV using the union of fields in order."""
    materialized = [dict(row) for row in rows]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not materialized:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(dict.fromkeys(key for row in materialized for key in row))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
