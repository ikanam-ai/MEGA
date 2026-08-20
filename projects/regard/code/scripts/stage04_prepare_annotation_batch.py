"""Этап 4 — подготовка батча для человеческой разметки.

Вход:  data/generations/generations_raw.jsonl
Выход: data/annotation/annotation_batch.jsonl   (500 таргетов x 2 ответа = 1000 единиц)

На каждый таргет случайно выбирается 1 ответ от локальной модели и 1 от
международной (роль — из config/models.yaml), затем единица назначается 3
аннотаторам из пула. Seed фиксирован — повторный запуск даёт тот же батч
(воспроизводимость, см. project_status.md, раздел "Схема человеческой разметки").

Запуск:
    uv run python -m scripts.stage04_prepare_annotation_batch
"""
from __future__ import annotations

import random

from scripts.common.config import ROOT_DIR
from scripts.common.io_utils import read_jsonl, write_jsonl

GENERATIONS_PATH = ROOT_DIR / "data" / "generations" / "generations_raw.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "annotation" / "annotation_batch.jsonl"

SEED = 20260621
ANNOTATORS_PER_ITEM = 3
ANNOTATOR_POOL = [f"annotator{i}" for i in range(1, 9)]  # 5-10 человек, см. project_status.md


def build_batch(generations: list[dict], rng: random.Random) -> list[dict]:
    by_target: dict[str, dict[str, list[dict]]] = {}
    for g in generations:
        by_target.setdefault(g["target_id"], {"local": [], "international": []})
        by_target[g["target_id"]][g["model_role"]].append(g)

    batch: list[dict] = []
    skipped: list[str] = []
    for idx, (target_id, by_role) in enumerate(sorted(by_target.items()), start=1):
        if not by_role["local"] or not by_role["international"]:
            skipped.append(target_id)
            continue

        local_gen = rng.choice(by_role["local"])
        intl_gen = rng.choice(by_role["international"])
        annotators = rng.sample(ANNOTATOR_POOL, ANNOTATORS_PER_ITEM)

        batch.append({
            "annotation_item_id": f"ann_{idx:04d}",
            "target_id": target_id,
            "target_name": local_gen["target_name"],
            "local_model_id": local_gen["model_id"],
            "local_response_text": local_gen["response_text"],
            "intl_model_id": intl_gen["model_id"],
            "intl_response_text": intl_gen["response_text"],
            "assigned_annotators": annotators,
        })

    if skipped:
        print(f"Пропущено {len(skipped)} таргетов без полной пары local+international: {skipped[:5]}...")

    return batch


def main() -> None:
    generations = list(read_jsonl(GENERATIONS_PATH))
    if not generations:
        raise FileNotFoundError(f"{GENERATIONS_PATH} пуст или не существует — выполните Этап 2.")

    rng = random.Random(SEED)
    batch = build_batch(generations, rng)
    write_jsonl(OUTPUT_PATH, batch)
    print(f"Записано {len(batch)} единиц разметки в {OUTPUT_PATH} (план: 500).")


if __name__ == "__main__":
    main()
