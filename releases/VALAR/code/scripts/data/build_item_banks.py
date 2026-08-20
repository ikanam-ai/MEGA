from __future__ import annotations

import csv
import json
import random
from pathlib import Path

random.seed(42)

DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"
BANKS_OUT     = Path(__file__).resolve().parents[2] / "data" / "candidate_pool"


def write_bank(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} items → {path.relative_to(BANKS_OUT.parent.parent.parent)}")


def write_readme(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_tape_ethics() -> None:
    TAPE_RAW = DATASETS_ROOT / "tape" / "dummy" / "raw"
    out_dir  = BANKS_OUT / "tape_ethics"
    total = 0

    for subset_name in ["per_ethics", "sit_ethics"]:
        label_prefix = "per" if subset_name == "per_ethics" else "sit"
        label_keys   = [f"{label_prefix}_{k}" for k in ("virtue","moral","law","justice","util")]

        for split in ("train", "test"):
            src = TAPE_RAW / subset_name / f"{split}.jsonl"
            if not src.exists():
                continue

            rows = []
            with open(src, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    raw = json.loads(line)
                    text = raw.get("text", "").strip()
                    if not text:
                        continue

                    item_id = f"tape_{subset_name}_{split}_{idx:06d}"
                    meta = {k: raw.get(k) for k in label_keys}
                    meta["source"] = raw.get("source")

                    rows.append({
                        "item_id":        item_id,
                        "task_family":    "value_annotation",
                        "item_text":      text,
                        "dataset":        "tape",
                        "subset":         subset_name,
                        "split":          split,
                        "source_dataset": f"RussianNLP/tape:{subset_name}:{split}",
                        "meta":           meta,
                    })

            write_bank(rows, out_dir / f"tape_{subset_name}_{split}.jsonl")
            total += len(rows)

    write_readme(out_dir / "SOURCE_README.md", f"""\
# TAPE ethics item bank

Source: RussianNLP/tape (HuggingFace, local copy)
Subsets: per_ethics, sit_ethics
Splits: train (labeled), test (all meta labels = -1)

Label fields in meta:
  per/sit_virtue, per/sit_moral, per/sit_law, per/sit_justice, per/sit_util
  Values: 1 (positive), 0 (negative), -1 (unlabeled/test split)

Texts are 3rd-person Russian news articles about ethical situations.
Length: ~17–1016 words, median ~138 words.

Total items: {total:,}
""")
    print(f"tape_ethics: {total:,} total items")


def build_v2_sensitive_topics() -> None:
    src     = DATASETS_ROOT / "russian_sensitive_inappropriate_topics" / "Version2" / "sensitive_topics" / "sensitive_topics.csv"
    out_dir = BANKS_OUT / "v2_sensitive_topics"

    TOPIC_COLS = [
        "offline_crime","online_crime","drugs","gambling","pornography",
        "prostitution","slavery","suicide","terrorism","weapons",
        "body_shaming","health_shaming","politics","racism",
        "religion","sexual_minorities","sexism","social_injustice",
    ]

    rows = []
    with open(src, newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            text = row.get("text", "").strip()
            if not text:
                continue

            topics_active = [c for c in TOPIC_COLS if row.get(c,"0").strip() in ("1","1.0")]
            meta = {c: float(row.get(c, 0) or 0) for c in TOPIC_COLS}
            meta["active_topics"] = topics_active

            rows.append({
                "item_id":        f"v2_st_{idx:07d}",
                "task_family":    "value_annotation",
                "item_text":      text,
                "dataset":        "russian_sensitive_inappropriate_topics",
                "subset":         "sensitive_topics",
                "split":          "full",
                "source_dataset": "s-nlp/inappropriate-sensitive-topics:Version2:sensitive_topics",
                "meta":           meta,
            })

    write_bank(rows, out_dir / "v2_sensitive_topics_full.jsonl")
    write_readme(out_dir / "SOURCE_README.md", f"""\
# V2 Sensitive Topics item bank

Source: github.com/s-nlp/inappropriate-sensitive-topics, Version2/sensitive_topics
Total items: {len(rows):,}

Label fields in meta (float 0.0–1.0, multi-label):
  offline_crime, online_crime, drugs, gambling, pornography, prostitution,
  slavery, suicide, terrorism, weapons, body_shaming, health_shaming,
  politics, racism, religion, sexual_minorities, sexism, social_injustice

Active topics (threshold = 1.0) stored in meta.active_topics list.

Text type: Russian social media comments and posts.
Length: mostly short (median ~84 chars).
""")
    print(f"v2_sensitive_topics: {len(rows):,} items")


def build_v3_inap_high(n_neutral_sample: int = 5_000) -> None:
    src    = DATASETS_ROOT / "russian_sensitive_inappropriate_topics" / "Version3" / "Inappapropriate_messages.csv"
    st_src = DATASETS_ROOT / "russian_sensitive_inappropriate_topics" / "Version3" / "sensitive_topics.csv"
    out_dir = BANKS_OUT / "v3_inap"

    st_texts: set[str] = set()
    with open(st_src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st_texts.add(row.get("text","").strip())

    high_rows, low_rows = [], []
    with open(src, newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            text = row.get("text","").strip()
            if not text or text in st_texts:
                continue
            conf = float(row.get("inappropriate", 0))
            item = {
                "item_id":        f"v3_inap_{idx:07d}",
                "task_family":    "value_annotation",
                "item_text":      text,
                "dataset":        "russian_sensitive_inappropriate_topics",
                "subset":         "inap_messages",
                "split":          "full",
                "source_dataset": "s-nlp/inappropriate-sensitive-topics:Version3:Inappapropriate_messages",
                "meta": {
                    "inappropriate_confidence": conf,
                    "tier": "inappropriate" if conf >= 0.9 else "appropriate",
                },
            }
            if conf >= 0.9:
                high_rows.append(item)
            else:
                low_rows.append(item)

    write_bank(high_rows, out_dir / "v3_inap_high_confidence.jsonl")

    neutral_sample = random.sample(low_rows, min(n_neutral_sample, len(low_rows)))
    write_bank(neutral_sample, out_dir / "v3_inap_neutral_sample.jsonl")

    write_readme(out_dir / "SOURCE_README.md", f"""\
# V3 Inappropriate Messages item banks

Source: s-nlp/inappropriate-sensitive-topics:Version3:Inappapropriate_messages
Excludes: texts already in v3/v2 sensitive_topics (no duplication across banks)

The V3 file has a strict BIMODAL confidence distribution (0.1–0.9 zone is empty —
authors already filtered out ambiguous cases before publishing the CSV):
  ≥ 0.9  → clearly inappropriate (value-laden: Security violations, Power abuse, etc.)
  ≤ 0.1  → clearly appropriate/neutral (negative baseline for value annotation)

Files:
  v3_inap_high_confidence.jsonl  → {len(high_rows):,} inappropriate texts (confidence ≥ 0.9)
  v3_inap_neutral_sample.jsonl   → {len(neutral_sample):,} neutral texts (confidence ≤ 0.1, seed=42 sample)

meta fields per item:
  inappropriate_confidence: float (≥0.9 or ≤0.1 only)
  tier: "inappropriate" | "appropriate"

Note: Krippendorff's alpha = 0.65 on the original annotation collection.
""")
    print(f"v3_inap_high_confidence: {len(high_rows):,} items")
    print(f"v3_inap_neutral_sample:  {len(neutral_sample):,} items")


if __name__ == "__main__":
    print("Building VALAR item banks...\n")
    build_tape_ethics()
    build_v2_sensitive_topics()
    build_v3_inap_high()
    print("\nDone. Item banks in:", BANKS_OUT)
