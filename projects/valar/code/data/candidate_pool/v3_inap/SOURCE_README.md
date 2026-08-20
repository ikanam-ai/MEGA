# V3 Inappropriate Messages item banks

Source: s-nlp/inappropriate-sensitive-topics:Version3:Inappapropriate_messages
Excludes: texts already in v3/v2 sensitive_topics (no duplication across banks)

The V3 file has a strict BIMODAL confidence distribution (0.1–0.9 zone is empty —
authors already filtered out ambiguous cases before publishing the CSV):
  ≥ 0.9  → clearly inappropriate (value-laden: Security violations, Power abuse, etc.)
  ≤ 0.1  → clearly appropriate/neutral (negative baseline for value annotation)

Files:
  v3_inap_high_confidence.jsonl  → 28,931 inappropriate texts (confidence ≥ 0.9)
  v3_inap_neutral_sample.jsonl   → 5,000 neutral texts (confidence ≤ 0.1, seed=42 sample)

meta fields per item:
  inappropriate_confidence: float (≥0.9 or ≤0.1 only)
  tier: "inappropriate" | "appropriate"

Note: Krippendorff's alpha = 0.65 on the original annotation collection.
