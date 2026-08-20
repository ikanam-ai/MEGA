# data/item_banks/

## Структура

```
item_banks/
  <bank_name>_<timestamp>/    ← вывод аннотации ValueLlama по каждому candidate_pool банку
    run_manifest.json
    generation/
      annotation_rows.jsonl   ← полные оценки: p_rel, p_val, value_profile
      summary.json
    scoring/
      value_scores.csv        ← плоская таблица (item_id + score_* + top1_value + meta_*)
      value_scores_by_topic.csv
  gold/                       ← ФИНАЛЬНЫЙ ITEM BANK (создаётся после фильтрации)
    schwartz10_ru_1000.jsonl  ← 1 000 текстов, 100 × 10 ценностей, ручная проверка
    SOURCE_README.md
```

## Флоу

1. `data/candidate_pool/` → разметка ValueLlama → `data/item_banks/<bank>_<timestamp>/`
2. Фильтрация по `score_*`, балансировка (100/ценность), ручная проверка
3. Итог → `data/item_banks/gold/`
4. Эксперименты с разными моделями → `results/`
