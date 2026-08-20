from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from valar.annotators.simple_parser import parse_batch
from valar.annotators.valuellama import measure_perceptions_async
from valar.artifacts.io import write_jsonl, write_csv
from valar.scoring.annotation import aggregate, to_row
from valar.value_space.schwartz import SCHWARTZ_10
from valar import env


def load_texts(path: str, text_col: str, limit: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            text = row.get(text_col, "").strip()
            if not text:
                continue
            rows.append({
                "item_id": f"{Path(path).stem}_{i:07d}",
                "source_file": Path(path).name,
                "text": text,
                **{k: v for k, v in row.items() if k != text_col},
            })
    return rows


async def annotate_items(
    items: list[dict],
    values: list[str],
    model: str,
    api_base_url: str,
    concurrency: int,
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    texts = [it["text"] for it in items]

    all_perceptions = parse_batch(texts)
    flat_perceptions = [p for ps in all_perceptions for p in ps]

    print(f"[valar] {len(items)} items → {len(flat_perceptions)} perceptions")
    print(f"[valar] values={len(values)}  concurrency={concurrency}")
    print(f"[valar] API calls ≈ {len(flat_perceptions) * len(values)} (relevance) + up to same (valence)")

    if dry_run:
        print("\n[valar] DRY RUN — no API calls made. First 3 perceptions:")
        for p in flat_perceptions[:3]:
            print(f"  {repr(p[:100])}")
        return [], []

    t0 = time.time()

    perception_results = await measure_perceptions_async(
        perceptions=flat_perceptions,
        values=values,
        model=model,
        api_base_url=api_base_url,
        concurrency=concurrency,
    )

    elapsed = time.time() - t0
    print(f"[valar] Done in {elapsed:.1f}s  ({elapsed / len(flat_perceptions):.2f}s/perception)")

    generation_rows, score_rows = [], []
    perc_index = 0

    for item, item_perceptions in zip(items, all_perceptions):
        n = len(item_perceptions)
        pr_slice = perception_results[perc_index : perc_index + n]
        perc_index += n

        profile = aggregate(pr_slice, values)
        n_relevant = sum(len(pr.relevant_values) for pr in pr_slice)
        parse_ok = all(len(pr.relevant_values) >= 0 for pr in pr_slice)

        gen_row = {
            **item,
            "perceptions": item_perceptions,
            "perception_results": [
                {
                    "perception": pr.perception,
                    "relevant_values": pr.relevant_values,
                    "relevances": pr.relevances,
                    "valences": pr.valences,
                }
                for pr in pr_slice
            ],
            "profile": profile,
        }
        generation_rows.append(gen_row)

        extra = {k: v for k, v in item.items() if k not in ("item_id", "text", "source_file")}
        score_rows.append(to_row(
            item_id=item["item_id"],
            source_dataset=item["source_file"],
            text=item["text"],
            profile=profile,
            perceptions=item_perceptions,
            parse_ok=parse_ok,
            n_relevant_total=n_relevant,
            extra=extra,
        ))

    return generation_rows, score_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="CSV with texts")
    parser.add_argument("--text-column", default="text", help="Column name for text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-base-url", default=None, help="vLLM base URL (or VALAR_VALUELLAMA_API_BASE_URL)")
    parser.add_argument("--model-name", default=None, help="Served model name (or VALAR_SERVED_MODEL_NAME)")
    parser.add_argument("--limit", type=int, default=0, help="Max items (0 = all)")
    parser.add_argument("--concurrency", type=int, default=64, help="Parallel API calls")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_base = args.api_base_url or env.valuellama_api_base_url()
    model_name = args.model_name or env.served_model_name()

    print(f"[valar] api={api_base}  model={model_name}")
    print(f"[valar] input={args.input_file}  limit={args.limit or 'all'}")

    items = load_texts(args.input_file, args.text_column, args.limit)
    print(f"[valar] Loaded {len(items)} items")

    gen_rows, score_rows = asyncio.run(annotate_items(
        items=items,
        values=SCHWARTZ_10,
        model=model_name,
        api_base_url=api_base,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    ))

    if args.dry_run:
        return

    out = Path(args.output_dir)
    (out / "generation").mkdir(parents=True, exist_ok=True)
    (out / "scoring").mkdir(parents=True, exist_ok=True)

    write_jsonl(gen_rows, out / "generation" / "annotation_rows.jsonl")
    write_csv(score_rows, out / "scoring" / "value_scores.csv")

    summary = {
        "input_file": args.input_file,
        "n_items": len(items),
        "n_scored": len(score_rows),
        "model": model_name,
        "api_base": api_base,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "generation" / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[valar] Results → {out}")
    print(f"  generation/annotation_rows.jsonl  ({len(gen_rows)} rows)")
    print(f"  scoring/value_scores.csv          ({len(score_rows)} rows)")


if __name__ == "__main__":
    main()
