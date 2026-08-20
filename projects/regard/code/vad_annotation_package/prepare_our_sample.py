#!/usr/bin/env python3
"""
Prepare a stratified sample from our CIS experiment for human VAD annotation.

Output: human_annotation_sample.csv — ready for prepare_our_data.py

Design:
  - 500 items total: 250 ru-group + 250 intl-group
  - ~62-63 items per model (within group, balanced)
  - Stratified by target category (proportional to category size in our 500-target bank)
  - Only evaluative_stance prompt
  - Only generations where at least 1 judge has target_coverage >= 0.5
  - LM VAD = mean across valid judges
  - Each item annotated by 3 annotators (no separate overlap set needed)
  - Reproducible: seed=20260627
"""

import json
import math
import random
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATIONS_PATH = ROOT / "data" / "generations" / "generations_raw.jsonl"
SCORES_PATH = ROOT / "data" / "scores" / "judge_scores_raw.jsonl"
TARGETS_PATH = ROOT / "data" / "targets" / "aist_cis_targets_final.csv"
OUT_PATH = Path(__file__).resolve().parent / "human_annotation_sample.csv"

SEED = 20260627
TOTAL_PER_GROUP = 150

RU_MODELS = ["yandexgpt", "gigachat", "tpro", "avibe"]
INTL_MODELS = ["glm", "gemma2_27b", "qwen25_14b", "ministral_14b"]

MODEL_LABELS = {
    "yandexgpt": "YandexGPT-5-Lite-8B",
    "gigachat": "GigaChat3.1-10B",
    "tpro": "T-pro-it-2.1",
    "avibe": "AVIBE",
    "glm": "GLM-4.7-Flash",
    "gemma2_27b": "Gemma-4-26B",
    "qwen25_14b": "Qwen2.5-14B",
    "ministral_14b": "Ministral-3-14B",
}


def load_targets():
    import csv as _csv
    targets = {}
    with open(TARGETS_PATH, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            targets[row["target_id"]] = {
                "target_name": row.get("prompt_target_ru") or row["target_name"],
                "category": row["category"],
                "target_family": row["target_family"],
                "sensitivity_label": row.get("sensitivity_label", "standard"),
                "country": row.get("country", ""),
            }
    return targets


def load_scored_generations(targets):
    scores_by_gen = defaultdict(list)
    with open(SCORES_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            scores_by_gen[r["generation_id"]].append(r)

    records = []
    with open(GENERATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            if g["prompt_id"] != "evaluative_stance":
                continue
            sc = scores_by_gen.get(g["generation_id"], [])
            valid_sc = [s for s in sc if s.get("target_coverage", 0) >= 0.5]
            if len(valid_sc) < 1:
                continue
            tid = g["target_id"]
            if tid not in targets:
                continue
            lm_valence = sum(s["lm_valence"] for s in valid_sc) / len(valid_sc)
            lm_arousal = sum(s["lm_arousal"] for s in valid_sc) / len(valid_sc)
            lm_dominance = sum(s["lm_dominance"] for s in valid_sc) / len(valid_sc)
            lm_confidence = sum(s["lm_confidence"] for s in valid_sc) / len(valid_sc)
            records.append({
                "generation_id": g["generation_id"],
                "model_id": g["model_id"],
                "target_id": tid,
                "target_name": targets[tid]["target_name"],
                "target_family": targets[tid]["target_family"],
                "category": targets[tid]["category"],
                "sensitivity_label": targets[tid]["sensitivity_label"],
                "country": targets[tid]["country"],
                "response_text": g["response_text"],
                "lm_valence": lm_valence,
                "lm_arousal": lm_arousal,
                "lm_dominance": lm_dominance,
                "lm_confidence": lm_confidence,
            })
    return records


def stratified_sample(pool, n, strat_key, seed):
    """Sample n items from pool, stratified by strat_key (proportional)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in pool:
        groups[item[strat_key]].append(item)

    total = len(pool)
    quotas = {}
    allocated = 0
    keys = sorted(groups.keys())
    for k in keys:
        q = math.floor(len(groups[k]) / total * n)
        quotas[k] = q
        allocated += q
    # distribute remainder by largest fractional part
    remainders = sorted(
        keys,
        key=lambda k: (len(groups[k]) / total * n) - quotas[k],
        reverse=True,
    )
    for k in remainders[:n - allocated]:
        quotas[k] += 1

    sampled = []
    for k in keys:
        shuffled = list(groups[k])
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:quotas[k]])
    return sampled


def sample_group(records, model_list, total, seed):
    """Sample `total` items from models in model_list, balanced across models and stratified by category."""
    rng = random.Random(seed)
    per_model = total // len(model_list)
    remainder = total % len(model_list)

    result = []
    for i, model_id in enumerate(model_list):
        pool = [r for r in records if r["model_id"] == model_id]
        n = per_model + (1 if i < remainder else 0)
        sampled = stratified_sample(pool, n, "category", seed + i)
        result.extend(sampled)
    return result


def main():
    rng = random.Random(SEED)

    print("Loading targets...")
    targets = load_targets()

    print("Loading scored generations...")
    records = load_scored_generations(targets)
    print(f"  Valid generations: {len(records)}")

    print("Sampling ru-group...")
    ru_sample = sample_group(records, RU_MODELS, TOTAL_PER_GROUP, SEED)
    print(f"  ru-group: {len(ru_sample)}")

    print("Sampling intl-group...")
    intl_sample = sample_group(records, INTL_MODELS, TOTAL_PER_GROUP, SEED + 100)
    print(f"  intl-group: {len(intl_sample)}")

    # Build final rows — no overlap copies, each item appears once
    combined = ru_sample + intl_sample
    all_items = []
    for item in combined:
        row = dict(item)
        row["model_label"] = MODEL_LABELS.get(item["model_id"], item["model_id"])
        row["group"] = "ru" if item["model_id"] in RU_MODELS else "intl"
        row["item_id"] = f"td_{item['generation_id'][:16]}"
        all_items.append(row)

    rng.shuffle(all_items)

    fieldnames = [
        "item_id", "generation_id", "model_id", "model_label", "group",
        "target_id", "target_name", "target_family", "category",
        "sensitivity_label", "country",
        "response_text",
        "lm_valence", "lm_arousal", "lm_dominance", "lm_confidence",
    ]

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_items)

    print(f"\nWritten: {OUT_PATH}")
    print(f"Total rows: {len(all_items)}  (each annotated by 3 annotators)")
    print()

    from collections import Counter
    groups = Counter(r["group"] for r in all_items)
    models = Counter(r["model_id"] for r in all_items)
    cats = Counter(r["category"] for r in all_items)
    print("Group balance:")
    for k, v in sorted(groups.items()): print(f"  {k}: {v}")
    print("Per model:")
    for k, v in sorted(models.items()): print(f"  {k}: {v}")
    print("Per category:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
