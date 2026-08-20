from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from valar.value_space.schwartz import SCHWARTZ_10


def load_gpt_labels(banks_root: Path, gpt_file: Path | None) -> dict[str, dict]:
    if gpt_file is None:
        candidates = sorted((banks_root / "gpt_labels").glob("gpt_annotation_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No gpt_annotation*.jsonl found in {banks_root}/gpt_labels/\n"
                "Run scripts/annotate/run_gpt_annotation.py first."
            )
        gpt_file = candidates[-1]

    print(f"[gold] GPT labels: {gpt_file}")
    index: dict[str, dict] = {}
    with open(gpt_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                index[row["item_id"]] = row
    return index


def load_meta_index(banks_root: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for jsonl_path in sorted(banks_root.glob("*/generation/annotation_rows.jsonl")):
        run_dir   = jsonl_path.parent.parent.name
        bank_stem = re.sub(r"_\d{8}T\d{6}Z$", "", run_dir)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                index[row["item_id"]] = {
                    "dataset":   row.get("dataset", ""),
                    "subset":    row.get("subset", ""),
                    "split":     row.get("split", ""),
                    "meta":      row.get("meta", {}),
                    "bank_stem": bank_stem,
                }
    return index


def select_top_n(
    candidates: list[dict],
    n: int,
    max_per_bank: int,
    seen_texts: set[str] | None = None,
) -> list[dict]:
    bank_counts: dict[str, int] = defaultdict(int)
    selected: list[dict] = []
    for item in sorted(candidates, key=lambda x: x["valuellama_score"] or 0, reverse=True):
        bank = item.get("bank_stem", "unknown")
        if bank_counts[bank] >= max_per_bank:
            continue
        text = item.get("item_text", "")
        if seen_texts is not None and text in seen_texts:
            continue
        selected.append(item)
        bank_counts[bank] += 1
        if seen_texts is not None:
            seen_texts.add(text)
        if len(selected) == n:
            break
    return selected


def build_gold(
    gpt_labels: dict[str, dict],
    meta_index: dict[str, dict],
    n_per_value: int,
    max_per_bank: int,
) -> tuple[list[dict], dict]:

    tier1: dict[str, list[dict]] = defaultdict(list)
    tier2: dict[str, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()

    for item_id, gpt_row in gpt_labels.items():
        vl_value  = gpt_row.get("valuellama_top1", "")
        gpt_label = gpt_row.get("gpt_label", "")
        score     = gpt_row.get("valuellama_score")

        if not vl_value or score is None:
            continue
        if gpt_label in ("parse_error", "api_error", "error"):
            continue

        meta_info = meta_index.get(item_id, {})
        candidate = {
            "item_id":          item_id,
            "item_text":        gpt_row["item_text"],
            "gold_value":       vl_value,
            "valuellama_score": score,
            "gpt_label":        gpt_label,
            "dataset":          meta_info.get("dataset", ""),
            "subset":           meta_info.get("subset", ""),
            "split":            meta_info.get("split", ""),
            "bank_stem":        meta_info.get("bank_stem", ""),
            "meta":             meta_info.get("meta", {}),
        }

        if gpt_label == vl_value:
            tier1[vl_value].append(candidate)
        elif gpt_label == "none":
            tier2[vl_value].append(candidate)

    gold_items: list[dict] = []
    stats: dict = {}

    for v in SCHWARTZ_10:
        t1_pool = tier1.get(v, [])
        t2_pool = tier2.get(v, [])

        selected = select_top_n(t1_pool, n_per_value, max_per_bank, seen_texts)
        n_t1 = len(selected)

        if len(selected) < n_per_value:
            gap  = n_per_value - len(selected)
            fill = select_top_n(t2_pool, gap, max_per_bank, seen_texts)
            for item in fill:
                item["tier"] = 2
            selected += fill

        for item in selected:
            item.setdefault("tier", 1)

        scores = [x["valuellama_score"] for x in selected if x["valuellama_score"] is not None]
        bank_dist: dict[str, int] = defaultdict(int)
        for it in selected:
            bank_dist[it.get("bank_stem", "unknown")] += 1

        stats[v] = {
            "n_selected":  len(selected),
            "tier1":       n_t1,
            "tier2":       len(selected) - n_t1,
            "tier1_pool":  len(t1_pool),
            "tier2_pool":  len(t2_pool),
            "score_min":   round(min(scores), 4) if scores else None,
            "score_max":   round(max(scores), 4) if scores else None,
            "score_mean":  round(sum(scores) / len(scores), 4) if scores else None,
            "bank_dist":   dict(bank_dist),
        }

        gold_items.extend(selected)

    return gold_items, stats


def print_report(gold_items: list[dict], stats: dict) -> None:
    print(f"\n{'Ценность':22} {'N':>4}  {'T1':>4}  {'T2':>4}  {'T1 pool':>9}  "
          f"{'score min':>10}  {'score max':>10}  Банки")
    print("  " + "─" * 110)

    total_t1 = total_t2 = 0
    for v in SCHWARTZ_10:
        s = stats[v]
        banks = "  ".join(f"{b}:{n}" for b, n in sorted(s["bank_dist"].items(), key=lambda x: -x[1]))
        print(f"  {v:20} {s['n_selected']:4}  {s['tier1']:4}  {s['tier2']:4}  "
              f"{s['tier1_pool']:9,}  "
              f"{s['score_min']:10.4f}  {s['score_max']:10.4f}  {banks}")
        total_t1 += s["tier1"]
        total_t2 += s["tier2"]

    print("  " + "─" * 110)
    n = len(gold_items)
    print(f"  {'ИТОГО':20} {n:4}  {total_t1:4}  {total_t2:4}")
    print(f"\n  Tier 1 (оба согласились): {total_t1}/{n}  ({100*total_t1/n:.1f}%)")
    if total_t2:
        print(f"  Tier 2 (только VL, GPT=none): {total_t2}/{n}  ({100*total_t2/n:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Schwartz-10 gold bank (1,000 items)")
    parser.add_argument("--banks-root",   default="data/item_banks")
    parser.add_argument("--gpt-file",     default=None,
                        help="GPT annotation JSONL (auto-detects most recent if omitted)")
    parser.add_argument("--output-dir",   default="data/item_banks/gold")
    parser.add_argument("--n-per-value",  type=int, default=100,
                        help="Items per Schwartz value (default: 100)")
    parser.add_argument("--max-per-bank", type=int, default=60,
                        help="Max items from one source bank per value (default: 60)")
    args = parser.parse_args()

    banks_root = Path(args.banks_root)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gold] Loading GPT labels ...")
    gpt_labels = load_gpt_labels(
        banks_root,
        Path(args.gpt_file) if args.gpt_file else None,
    )
    print(f"[gold] {len(gpt_labels):,} GPT-labeled items")

    print(f"[gold] Loading item metadata from annotation runs ...")
    meta_index = load_meta_index(banks_root)
    print(f"[gold] {len(meta_index):,} items in meta index")

    print(f"\n[gold] Selecting {args.n_per_value} items per value "
          f"(max {args.max_per_bank} per bank) ...")
    gold_items, stats = build_gold(
        gpt_labels=gpt_labels,
        meta_index=meta_index,
        n_per_value=args.n_per_value,
        max_per_bank=args.max_per_bank,
    )


    out_jsonl = out_dir / "schwartz10_ru_1000.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in gold_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "n_total":      len(gold_items),
        "n_per_value":  args.n_per_value,
        "max_per_bank": args.max_per_bank,
        "built_at":     datetime.now(timezone.utc).isoformat(),
        "values":       stats,
    }
    out_summary = out_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print_report(gold_items, stats)

    print(f"\n[gold] {out_jsonl}  ({len(gold_items):,} items)")
    print(f"[gold] {out_summary}")


if __name__ == "__main__":
    main()
