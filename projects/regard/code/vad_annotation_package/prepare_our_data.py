#!/usr/bin/env python3
"""
Convert human_annotation_sample.csv into annotation items, assignments, and an
annotator-account template.

Supports multiple annotators with overlap items assigned to two annotators each.

Usage:
    python prepare_our_data.py [--annotators 3] [--out data]
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "human_annotation_sample.csv"

SEED = 20260627

EMOTIONS = {
    "admiration":     {"ru": "восхищение",      "v": 0.969, "a": 0.583, "d": 0.726},
    "amusement":      {"ru": "веселье",          "v": 0.929, "a": 0.837, "d": 0.803},
    "anger":          {"ru": "злость",           "v": 0.167, "a": 0.865, "d": 0.657},
    "annoyance":      {"ru": "раздражение",      "v": 0.167, "a": 0.718, "d": 0.342},
    "approval":       {"ru": "одобрение",        "v": 0.854, "a": 0.460, "d": 0.889},
    "caring":         {"ru": "забота",           "v": 0.635, "a": 0.469, "d": 0.500},
    "confusion":      {"ru": "замешательство",   "v": 0.255, "a": 0.667, "d": 0.277},
    "curiosity":      {"ru": "любопытство",      "v": 0.750, "a": 0.755, "d": 0.463},
    "desire":         {"ru": "желание",          "v": 0.896, "a": 0.692, "d": 0.647},
    "disappointment": {"ru": "разочарование",    "v": 0.115, "a": 0.490, "d": 0.336},
    "disapproval":    {"ru": "неодобрение",      "v": 0.085, "a": 0.551, "d": 0.367},
    "disgust":        {"ru": "отвращение",       "v": 0.052, "a": 0.775, "d": 0.317},
    "embarrassment":  {"ru": "смущение",         "v": 0.143, "a": 0.685, "d": 0.226},
    "excitement":     {"ru": "возбуждение",      "v": 0.896, "a": 0.684, "d": 0.731},
    "fear":           {"ru": "страх",            "v": 0.073, "a": 0.840, "d": 0.293},
    "gratitude":      {"ru": "благодарность",    "v": 0.885, "a": 0.441, "d": 0.610},
    "grief":          {"ru": "горе",             "v": 0.070, "a": 0.640, "d": 0.474},
    "joy":            {"ru": "радость",          "v": 0.980, "a": 0.824, "d": 0.794},
    "love":           {"ru": "любовь",           "v": 1.000, "a": 0.519, "d": 0.673},
    "nervousness":    {"ru": "нервозность",      "v": 0.163, "a": 0.915, "d": 0.241},
    "optimism":       {"ru": "оптимизм",         "v": 0.949, "a": 0.565, "d": 0.814},
    "pride":          {"ru": "гордость",         "v": 0.729, "a": 0.634, "d": 0.848},
    "realization":    {"ru": "осознание",        "v": 0.554, "a": 0.510, "d": 0.836},
    "relief":         {"ru": "облегчение",       "v": 0.844, "a": 0.278, "d": 0.481},
    "remorse":        {"ru": "раскаяние",        "v": 0.103, "a": 0.673, "d": 0.377},
    "sadness":        {"ru": "грусть",           "v": 0.052, "a": 0.288, "d": 0.164},
    "surprise":       {"ru": "удивление",        "v": 0.875, "a": 0.875, "d": 0.562},
    "neutral":        {"ru": "нейтрально",       "v": 0.469, "a": 0.184, "d": 0.357},
}


def nearest_emotions(v, a, d, top_n=5):
    def dist(e):
        return math.sqrt((e["v"]-v)**2 + (e["a"]-a)**2 + (e["d"]-d)**2)
    ranked = sorted(EMOTIONS.items(), key=lambda kv: dist(kv[1]))
    return [name for name, _ in ranked[:top_n]]


def row_to_item(row):
    v = float(row["lm_valence"])
    a = float(row["lm_arousal"])
    d = float(row["lm_dominance"])
    ne = nearest_emotions(v, a, d)
    return {
        "source_row_count": 2,
        "n_lm_judges": 2,
        "target_id": row["target_id"],
        "target": row["target_name"],
        "target_name": row["target_name"],
        "target_label": row["target_name"],
        "target_family": row["target_family"],
        "item_metadata": {
            "category": row["category"],
            "sensitivity_label": row["sensitivity_label"],
            "country": row["country"],
            "group": row["group"],
            "generation_id": row["generation_id"],
        },
        "model_id": row["model_id"],
        "model_label": row["model_label"],
        "prompt_id": "evaluative_stance",
        "language": "ru",
        "generation_id": row["generation_id"],
        "run_id": None,
        "response_text": row["response_text"],
        "response_word_len": len(row["response_text"].split()),
        "response_char_len": len(row["response_text"]),
        "finish_reason": "stop",
        "lm_valence": v,
        "lm_arousal": a,
        "lm_dominance": d,
        "lm_confidence": float(row["lm_confidence"]),
        "target_coverage": 1.0,
        "scoring_modes": ["evaluative_stance"],
        "scorer_models": ["qwen36_35b", "gpt4o_mini"],
        "rationales": [],
        "evidence_spans": [],
        "vad_bin": (
            "valence_high" if v >= 0.65 else
            "valence_low" if v <= 0.35 else
            "valence_mid"
        ),
        "source_format": "our_experiment",
        "item_id": row["item_id"],
        "sample_rank": None,
        "nearest_emotions": ne,
        "default_emotion": ne[0] if ne else "neutral",
    }


def assign_items(items, annotators, annotations_per_item=3):
    """
    Assign each item to exactly `annotations_per_item` different annotators.
    Uses round-robin so load is balanced evenly.
    """
    rng = random.Random(SEED)
    n = len(annotators)
    shuffled = list(items)
    rng.shuffle(shuffled)

    per_ann = defaultdict(list)
    for i, item in enumerate(shuffled):
        for k in range(annotations_per_item):
            ann = annotators[(i + k) % n]
            per_ann[ann].append(item["item_id"])

    assignments = []
    for ann in annotators:
        item_ids = list(per_ann[ann])
        rng.shuffle(item_ids)
        for order, iid in enumerate(item_ids, start=1):
            assignments.append({"username": ann, "item_id": iid, "order": order})

    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", type=int, default=3,
                        help="Number of annotators (default: 3)")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotators = [f"ann{i+1:02d}" for i in range(args.annotators)]

    with open(SAMPLE_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    items = [row_to_item(row) for row in rows]
    assignments = assign_items(items, annotators)

    # annotation_items.jsonl
    with (out_dir / "annotation_items.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # assignments.csv
    with (out_dir / "assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["username", "item_id", "order"])
        writer.writeheader()
        writer.writerows(assignments)

    # users.example.csv: copy to users.csv and replace every password before use.
    with (out_dir / "users.example.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["username", "password", "display_name"])
        writer.writeheader()
        for ann in annotators:
            writer.writerow({"username": ann, "password": "CHANGE_ME",
                             "display_name": f"Annotator {ann}"})

    # emotions
    with (out_dir / "emotions_vad_goemotions_nrc.json").open("w", encoding="utf-8") as f:
        json.dump(EMOTIONS, f, ensure_ascii=False, indent=2)

    # report
    ann_counts = Counter(a["username"] for a in assignments)
    report = {
        "total_items": len(items),
        "annotations_per_item": 3,
        "annotators": annotators,
        "items_per_annotator": dict(ann_counts),
        "models": sorted({it["model_id"] for it in items}),
        "categories": dict(Counter(it["item_metadata"]["category"] for it in items)),
        "groups": dict(Counter(it["item_metadata"]["group"] for it in items)),
    }
    (out_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nData written to: {out_dir}/")


if __name__ == "__main__":
    main()
