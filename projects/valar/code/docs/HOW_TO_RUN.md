# Как запустить аннотацию ValueLlama

## Общая схема

```
data/candidate_pool/                   ← тексты-кандидаты (70 649 текстов)
        ↓  annotate with ValueLlama
data/item_banks/<bank>_<timestamp>/    ← размеченные тексты (generation + scoring)
        ↓  filter by score, balance 100/value, human review
data/item_banks/gold/                  ← финальный item bank (1 000 текстов, 100 × 10 ценностей)
        ↓  run experiments (разные LLM-модели)
results/<experiment_run_id>/           ← результаты экспериментов по распознаванию ценностей
```

---

## Подготовка (один раз)

```bash
cd /path/to/VALAR
set -a; source .env; set +a
```

---

## Команды для разметки candidate_pool (по порядку)

### 1. TAPE per_ethics test — ~8 мин

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/tape_ethics/tape_per_ethics_test.jsonl \
  --concurrency 100
```

### 2. TAPE sit_ethics test — ~7 мин

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/tape_ethics/tape_sit_ethics_test.jsonl \
  --concurrency 100
```

### 3. TAPE train (оба) — ~2 мин суммарно

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/tape_ethics/tape_per_ethics_train.jsonl \
  --concurrency 100

PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/tape_ethics/tape_sit_ethics_train.jsonl \
  --concurrency 100
```

### 4. V3 inap high-confidence — ~156 мин

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/v3_inap/v3_inap_high_confidence.jsonl \
  --concurrency 100
```

### 5. V3 neutral sample — ~27 мин

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/v3_inap/v3_inap_neutral_sample.jsonl \
  --concurrency 100
```

### 6. V2/V3 Sensitive Topics — ~180 мин

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/v2_sensitive_topics/v2_sensitive_topics_full.jsonl \
  --concurrency 100
```

---

## Что получается после каждого запуска

```
data/item_banks/
  <bank_name>_<timestamp>/
    run_manifest.json              ← параметры, провенанс
    generation/
      annotation_rows.jsonl        ← полный вывод: перцепции, p_rel, p_val, profile
      summary.json
    scoring/
      value_scores.csv             ← плоская таблица: item_id + score_* + top1_value + meta_*
      value_scores_by_topic.csv    ← агрегат по meta.active_topics
```

---

## Dry-run

```bash
PYTHONPATH=. python3 scripts/annotate/run_annotation.py \
  --item-bank data/candidate_pool/tape_ethics/tape_per_ethics_test.jsonl \
  --dry-run
```

---

## GPT cross-annotation (шаг 2 — согласованность разметчиков)

Берёт тексты, которым ValueLlama присвоил top1_value (~43K), и спрашивает GPT-4.1-mini
ту же задачу: выбрать одну ценность Шварца или "none". Результат идёт в `data/item_banks/gpt_labels/`.

### Dry-run (проверить промпты, без API-вызовов)

```bash
PYTHONPATH=. python3 scripts/annotate/run_gpt_annotation.py --dry-run --limit 5
```

### Тест на 50 текстах (~2 мин)

```bash
PYTHONPATH=. python3 scripts/annotate/run_gpt_annotation.py --limit 50
```

### Полный прогон — ~43K текстов (~30–40 мин)

```bash
PYTHONPATH=. python3 scripts/annotate/run_gpt_annotation.py
```

### Возобновление после обрыва

```bash
PYTHONPATH=. python3 scripts/annotate/run_gpt_annotation.py \
  --output data/item_banks/gpt_labels/gpt_annotation_<timestamp>.jsonl
```

Скрипт сам определит уже готовые item_id и пропустит их.

### Вывод

```
data/item_banks/gpt_labels/
  gpt_annotation_<timestamp>.jsonl   ← построчно, пишется инкрементально
```

Каждая строка:
```json
{
  "item_id": "v2_st_0000042",
  "item_text": "...",
  "valuellama_top1": "Security",
  "valuellama_score": 0.97,
  "gpt_label": "Security",
  "agreed": true,
  "gpt_raw": "Security"
}
```

---

## Сводная таблица ETA

| Bank | Текстов | ETA |
|---|---|---|
| tape_per_ethics_test | 1 466 | ~8 мин |
| tape_sit_ethics_test | 1 436 | ~7 мин |
| tape_per_ethics_train | 259 | ~1 мин |
| tape_sit_ethics_train | 254 | ~1 мин |
| v3_inap_high_confidence | 28 931 | ~156 мин |
| v3_inap_neutral_sample | 5 000 | ~27 мин |
| v2_sensitive_topics_full | 33 303 | ~180 мин |
| **ИТОГО** | **70 649** | **~6.5 ч** |
