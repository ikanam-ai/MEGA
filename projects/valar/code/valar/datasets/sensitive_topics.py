from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator


def iter_items(
    local_path: str | Path,
    version: str = "Version1",
    subset: str = "appropriateness",
) -> Iterator[dict[str, Any]]:
    root = Path(local_path) / version / subset
    csv_files = list(root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {root}")

    for csv_file in csv_files:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text = row.get("text", row.get("message", row.get("utterance", "")))
                yield {
                    "item_id": f"rst_{version}_{subset}_{csv_file.stem}_{idx:06d}",
                    "source_dataset": "russian_sensitive_inappropriate_topics",
                    "source_version": version,
                    "source_subset": subset,
                    "text": text.strip(),
                    "label": row.get("label", row.get("category", None)),
                    "raw": dict(row),
                }
