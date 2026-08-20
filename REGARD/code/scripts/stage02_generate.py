"""Generate REGARD responses for 19 models and 500 post-Soviet targets.

Вход:  data/targets/aist_cis_targets_final.csv  (target_id, target_name, prompt_target_ru, category, ...)
Выход: data/generations/generations_raw.jsonl (19 x 500 x 3 = 28,500 rows)

Идемпотентно: при повторном запуске уже сгенерированные пары (target_id, model_id)
пропускаются — можно прерывать и продолжать без потери прогресса/дублей.

Параллелизация — по таргетам внутри одной модели (ThreadPoolExecutor), не между
моделями: каждая модель обрабатывается своим пулом из --workers потоков, модели
идут друг за другом. Запись в выходной файл защищена локом — без него параллельные
append() из разных потоков могут чередовать байты одной JSON-строки.

Запуск:
    poetry run python -m scripts.stage02_generate
    poetry run python -m scripts.stage02_generate --models yandexgpt,gigachat
"""

from __future__ import annotations

import argparse
import datetime as dt
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scripts.common.config import ROOT_DIR, models_config, prompts_config
from scripts.common.io_utils import append_jsonl, load_done_keys, read_csv
from scripts.providers.client import ProviderError, call_model

TARGETS_PATH = ROOT_DIR / "data" / "targets" / "aist_cis_targets_final.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "generations" / "generations_raw.jsonl"

DEFAULT_PROMPT_KEY = "main"
TEMPERATURE = 0.7
MAX_TOKENS = 512
DEFAULT_WORKERS = 4

_write_lock = threading.Lock()


def _done_key(record: dict) -> str:
    return f"{record['target_id']}::{record['model_id']}::{record['prompt_id']}"


def _resolve_prompt(prompt_key: str) -> tuple[str, str]:
    """Возвращает (prompt_id, template) по ключу из config/prompts.yaml.
    prompt_key="main" -- основной промпт; иначе ищется в generation.robustness."""
    generation_cfg = prompts_config()["generation"]
    if prompt_key == DEFAULT_PROMPT_KEY:
        entry = generation_cfg["main"]
    else:
        entry = generation_cfg["robustness"][prompt_key]
    return entry["id"], entry["template"]


def _generate_one(
    target: dict, model_id: str, model_role: str, prompt_id: str, prompt_template: str, run_id: str
) -> dict | None:
    prompt_text = prompt_template.format(target=target["prompt_target_ru"])
    try:
        result = call_model(model_id, user=prompt_text, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    except ProviderError as e:
        print(f"[ОШИБКА] {target['target_id']} / {model_id}: {e}")
        return None

    return {
        "target_id": target["target_id"],
        "target_name": target["target_name"],
        "prompt_target_ru": target["prompt_target_ru"],
        "target_family": target.get("category", ""),
        "model_id": model_id,
        "model_role": model_role,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "response_text": result.text,
        "response_word_len": len(result.text.split()),
        "finish_reason": result.finish_reason,
        "generation_id": str(uuid.uuid4()),
        "run_id": run_id,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "language": "ru",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="Список model_id через запятую (по умолчанию — все)")
    parser.add_argument(
        "--prompt-id",
        default=DEFAULT_PROMPT_KEY,
        help="Ключ промпта: main (по умолчанию) или один из generation.robustness "
        "в config/prompts.yaml (neutral_descriptive, evaluative_paraphrase)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Параллельных запросов на одну модель (по умолчанию {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    targets = list(read_csv(TARGETS_PATH))
    if not targets:
        raise FileNotFoundError(f"{TARGETS_PATH} пуст или не существует — выполните Этап 1.")

    generators = models_config()["generators"]
    model_ids = args.models.split(",") if args.models else list(generators.keys())
    unknown = sorted(set(model_ids) - set(generators))
    if unknown:
        raise SystemExit(f"Unknown generator model_id: {unknown}")

    prompt_id, prompt_template = _resolve_prompt(args.prompt_id)
    run_id = f"gen-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"

    for model_id in model_ids:
        model_role = generators[model_id]["role"]
        done = load_done_keys(OUTPUT_PATH, _done_key)
        pending = [t for t in targets if f"{t['target_id']}::{model_id}::{prompt_id}" not in done]
        print(
            f"[{model_id} / {prompt_id}] к выполнению: {len(pending)} из {len(targets)} "
            f"(пропущено уже готовых: {len(targets) - len(pending)}), workers={args.workers}"
        )

        if not pending:
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_generate_one, target, model_id, model_role, prompt_id, prompt_template, run_id)
                for target in pending
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"generate[{model_id}]"):
                record = future.result()
                if record is not None:
                    with _write_lock:
                        append_jsonl(OUTPUT_PATH, record)


if __name__ == "__main__":
    main()
