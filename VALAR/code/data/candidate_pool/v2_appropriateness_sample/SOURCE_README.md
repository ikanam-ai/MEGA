# V2 Appropriateness sample item bank

Source: s-nlp/inappropriate-sensitive-topics:Version2:appropriateness
Excludes: texts already present in v2_sensitive_topics (avoid duplication)
Stratified sample: 50% appropriate (meta.inappropriate=0), 30% inappropriate (=1), 20% unlabeled
Total items: 10,000  (seed=42)

Label fields in meta:
  inappropriate (0.0/1.0/None), human_labeled (bool), toxic_auto (float 0–1)
  + all 18 topic flags (same as v2_sensitive_topics)
  active_topics: list of topic labels with score = 1.0
