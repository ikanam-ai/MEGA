from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from valar.annotators.valuellama import ProgressTracker

from valar.annotators.simple_parser import parse_batch
from valar.annotators.valuellama import measure_perceptions_async
from valar.artifacts.io import write_jsonl
from valar.scoring.annotation import aggregate, top1_value
from valar.value_space.schwartz import SCHWARTZ_10
from valar import env


def load_item_bank(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def make_run_id(bank_path: Path) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{bank_path.stem}_{ts}"


def write_value_scores_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_value_scores_by_topic(score_rows: list[dict], gen_rows: list[dict], path: Path) -> None:
    topic_buckets: dict[str, list[dict]] = defaultdict(list)

    for sr, gr in zip(score_rows, gen_rows):
        topics = gr.get("meta", {}).get("active_topics", [])
        if not topics:
            topic_buckets["__none__"].append(sr)
        for t in topics:
            topic_buckets[t].append(sr)

    agg_rows = []
    for topic, items in sorted(topic_buckets.items()):
        row: dict = {"topic": topic, "n_items": len(items)}
        for v in SCHWARTZ_10:
            col = f"score_{v.lower().replace('-','_')}"
            vals = [it[col] for it in items if it.get(col) is not None]
            row[f"mean_{col}"] = round(sum(vals) / len(vals), 4) if vals else None
            row[f"coverage_{col}"] = round(len(vals) / len(items), 3)
        agg_rows.append(row)

    if agg_rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)


async def annotate_items(
    items: list[dict],
    model: str,
    api_base_url: str,
    concurrency: int,
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    texts = [it["item_text"] for it in items]

    all_perceptions = parse_batch(texts)
    flat_perceptions = [p for ps in all_perceptions for p in ps]
    n_calls_est = len(flat_perceptions) * len(SCHWARTZ_10)

    n_calls_rel = len(flat_perceptions) * len(SCHWARTZ_10)
    eta_min = n_calls_rel * 1.2 / 37 / 60

    print(f"\n[valar] {len(items):,} items → {len(flat_perceptions):,} perceptions")
    print(f"[valar] Phase 1 calls (relevance): {n_calls_rel:,}")
    print(f"[valar] Phase 2 calls (valence):   ≈ {n_calls_rel//5:,}  (varies by hit rate)")
    print(f"[valar] Concurrency: {concurrency}  |  ETA ≈ {eta_min:.0f} min")
    print()

    if dry_run:
        print("[valar] DRY RUN — first 3 perceptions:")
        for p in flat_perceptions[:3]:
            print(f"  {repr(p[:120])}")
        return [], []

    t0 = time.time()

    perception_results = await measure_perceptions_async(
        perceptions=flat_perceptions,
        values=SCHWARTZ_10,
        model=model,
        api_base_url=api_base_url,
        concurrency=concurrency,
    )

    elapsed = time.time() - t0
    n_total_calls = n_calls_rel + sum(
        len(pr.relevant_values) for pr in perception_results
    )
    print(f"\n[valar] Annotation complete: {elapsed:.0f}s total  "
          f"({n_total_calls:,} API calls  |  "
          f"{n_total_calls/elapsed:.0f} calls/s actual)")

    generation_rows, score_rows = [], []
    perc_idx = 0

    for item, item_percs in zip(items, all_perceptions):
        n = len(item_percs)
        pr_slice = perception_results[perc_idx : perc_idx + n]
        perc_idx += n

        profile = aggregate(pr_slice, SCHWARTZ_10)
        n_relevant = sum(len(pr.relevant_values) for pr in pr_slice)

        gen_row = {
            **item,
            "n_perceptions": n,
            "n_relevant_total": n_relevant,
            "perceptions": item_percs,
            "perception_results": [
                {
                    "perception":      pr.perception,
                    "relevant_values": pr.relevant_values,
                    "relevances":      pr.relevances,
                    "valences":        pr.valences,
                }
                for pr in pr_slice
            ],
            "value_profile": profile,
        }
        generation_rows.append(gen_row)

        score_row: dict = {
            "item_id":          item["item_id"],
            "dataset":          item.get("dataset", ""),
            "subset":           item.get("subset", ""),
            "split":            item.get("split", ""),
            "item_text_preview": item["item_text"][:100],
            "n_perceptions":    n,
            "n_relevant":       n_relevant,
            "top1_value":       top1_value(profile),
        }
        for v in SCHWARTZ_10:
            score_row[f"score_{v.lower().replace('-','_')}"] = profile.get(v)
        meta = item.get("meta", {})
        for key in ("inappropriate","human_labeled","toxic_auto","active_topics",
                    "per_virtue","per_moral","per_law","sit_virtue","sit_moral","sit_law",
                    "source"):
            if key in meta:
                val = meta[key]
                score_row[f"meta_{key}"] = json.dumps(val) if isinstance(val, list) else val
        score_rows.append(score_row)

    n_with_any   = sum(1 for r in score_rows if r.get("top1_value"))
    n_no_value   = len(score_rows) - n_with_any
    top1_dist: dict[str, int] = {}
    for r in score_rows:
        v = r.get("top1_value")
        if v:
            top1_dist[v] = top1_dist.get(v, 0) + 1

    print(f"\n[valar] Items with ≥1 value label:  {n_with_any:,} / {len(score_rows):,}"
          f"  ({100*n_with_any/len(score_rows):.1f}%)")
    print(f"[valar] Items with no value signal:  {n_no_value:,}"
          f"  ({100*n_no_value/len(score_rows):.1f}%)")
    print("[valar] top1_value distribution:")
    for v, n in sorted(top1_dist.items(), key=lambda kv: -kv[1]):
        bar = "█" * (n * 30 // max(top1_dist.values()))
        print(f"    {v:20}  {n:5,}  {bar}")

    return generation_rows, score_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate item bank with ValueLlama")
    parser.add_argument("--item-bank",    required=True,  help="Path to item bank JSONL")
    parser.add_argument("--results-root", default="data/item_banks", help="Output root dir")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--model-name",   default=None)
    parser.add_argument("--concurrency",  type=int, default=64)
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    api_base   = args.api_base_url or env.valuellama_api_base_url()
    model_name = args.model_name   or env.served_model_name()
    bank_path  = Path(args.item_bank)
    run_id     = make_run_id(bank_path)
    out_dir    = Path(args.results_root) / run_id

    print(f"[valar] run_id:  {run_id}")
    print(f"[valar] bank:    {bank_path}")
    print(f"[valar] api:     {api_base}")
    print(f"[valar] model:   {model_name}")
    print(f"[valar] concurr: {args.concurrency}")

    items = load_item_bank(bank_path)
    print(f"[valar] Loaded {len(items):,} items from bank")

    gen_rows, score_rows = asyncio.run(annotate_items(
        items        = items,
        model        = model_name,
        api_base_url = api_base,
        concurrency  = args.concurrency,
        dry_run      = args.dry_run,
    ))

    if args.dry_run:
        print("[valar] Dry run complete — no files written.")
        return

    gen_dir   = out_dir / "generation"
    score_dir = out_dir / "scoring"
    gen_dir.mkdir(parents=True,   exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(gen_rows, gen_dir / "annotation_rows.jsonl")

    n_any_value = sum(1 for r in score_rows if r.get("top1_value"))
    summary = {
        "run_id":         run_id,
        "item_bank":      str(bank_path),
        "n_items":        len(items),
        "n_annotated":    len(gen_rows),
        "n_with_value":   n_any_value,
        "coverage_pct":   round(100 * n_any_value / len(gen_rows), 1) if gen_rows else 0,
        "model":          model_name,
        "api_base":       api_base,
        "concurrency":    args.concurrency,
        "finished_at":    datetime.now(timezone.utc).isoformat(),
    }
    (gen_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_value_scores_csv(score_rows, score_dir / "value_scores.csv")

    if any("meta_active_topics" in r for r in score_rows):
        write_value_scores_by_topic(score_rows, gen_rows, score_dir / "value_scores_by_topic.csv")

    manifest = {
        "run_id":       run_id,
        "item_bank":    str(bank_path),
        "model":        model_name,
        "api_base":     api_base,
        "concurrency":  args.concurrency,
        "values":       SCHWARTZ_10,
        "started_at":   summary["finished_at"],
        "artifacts": {
            "annotation_rows":  str(gen_dir  / "annotation_rows.jsonl"),
            "summary":          str(gen_dir  / "summary.json"),
            "value_scores":     str(score_dir / "value_scores.csv"),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[valar] Results → {out_dir}")
    print(f"  generation/annotation_rows.jsonl  ({len(gen_rows):,} rows)")
    print(f"  scoring/value_scores.csv          ({len(score_rows):,} rows)")
    print(f"  coverage: {summary['coverage_pct']}% items received at least one value label")


if __name__ == "__main__":
    main()
