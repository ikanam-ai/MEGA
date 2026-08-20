from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from json_repair import repair_json
from tqdm import tqdm

from scripts.common.config import ROOT_DIR, models_config, prompts_config
from scripts.common.io_utils import append_jsonl, load_done_keys, read_jsonl
from scripts.providers.client import ProviderError, call_model

GENERATIONS_PATH = ROOT_DIR / "data" / "generations" / "generations_raw.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "scores" / "judge_scores_raw.jsonl"

JUDGE_CONTRACT_ID = "td_v1_contract_ru"
PRIMARY_JUDGE = "qwen36_35b"
DEFAULT_WORKERS = 4
TEMPERATURE = 0.0
MAX_TOKENS = 800
HEARTBEAT_SECONDS = 15
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_write_lock = threading.Lock()
_inflight_lock = threading.Lock()
_inflight_started: dict[int, float] = {}  # thread id -> время начала текущего call_model


def _done_key(record: dict) -> str:
    return f"{record['generation_id']}::{record['judge_model']}"


def _extract_json(text: str) -> dict:
    """Judge иногда оборачивает JSON в markdown-код или добавляет пояснение —
    вырезаем первый похожий на JSON блок; json_repair восстанавливает
    нестрогий JSON (комментарии, markdown внутри строк, незакавыченные
    значения) надёжнее точечных regex-патчей."""
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"В ответе judge не найден JSON-блок: {text[:200]}")
    block = match.group(0)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return json.loads(repair_json(block))


def _score_one(gen: dict, judge_id: str, system_prompt: str, user_template: str, pbar: tqdm) -> None:
    """Пишет результат в OUTPUT_PATH сама, не возвращая его наружу -- не зависим
    от as_completed()/future.result() в главном потоке (см. инцидент: futures
    помечались done(), но as_completed() не отдавал их главному потоку)."""
    user_prompt = user_template.format(target=gen["target_name"], response_text=gen["response_text"])
    started = time.monotonic()
    tid = threading.get_ident()
    with _inflight_lock:
        _inflight_started[tid] = started
    try:
        result = call_model(
            judge_id, system=system_prompt, user=user_prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        )
        elapsed = time.monotonic() - started
        if result.finish_reason not in ("stop", "end_turn"):
            print(
                f"[WARN] {gen['generation_id']} / {judge_id}: finish_reason={result.finish_reason} "
                f"({elapsed:.1f}s) -- скорее всего обрезано по max_tokens"
            )
        parsed = _extract_json(result.text)
    except ProviderError as e:
        elapsed = time.monotonic() - started
        print(f"[ОШИБКА] {gen['generation_id']} / {judge_id} ({elapsed:.1f}s): {e}")
        pbar.update(1)
        return
    except (ValueError, json.JSONDecodeError) as e:
        elapsed = time.monotonic() - started
        print(
            f"[ОШИБКА] {gen['generation_id']} / {judge_id} ({elapsed:.1f}s): {e}\n"
            f"--- RAW RESPONSE ---\n{result.text}\n--- END RAW ---"
        )
        pbar.update(1)
        return
    finally:
        with _inflight_lock:
            _inflight_started.pop(tid, None)

    record = {
        "generation_id": gen["generation_id"],
        "target_id": gen["target_id"],
        "model_id": gen["model_id"],
        "prompt_id": gen["prompt_id"],
        "prompt_text": gen["prompt_text"],
        "judge_model": judge_id,
        "lm_valence": parsed["lm_valence"],
        "lm_arousal": parsed["lm_arousal"],
        "lm_dominance": parsed["lm_dominance"],
        "lm_confidence": parsed["lm_confidence"],
        "target_coverage": parsed["target_coverage"],
        "knowledge_specificity": parsed["knowledge_specificity"],
        "refusal_flag": parsed["refusal_flag"],
        "generic_answer_flag": parsed["generic_answer_flag"],
        "unknown_flag": parsed["unknown_flag"],
        "hallucination_or_mismatch_flag": parsed["hallucination_or_mismatch_flag"],
        "valid_for_vad_analysis": parsed["valid_for_vad_analysis"],
        "evidence_spans": parsed["evidence_spans"],
        "rationale": parsed["rationale"],
        "scoring_contract": JUDGE_CONTRACT_ID,
    }
    with _write_lock:
        append_jsonl(OUTPUT_PATH, record)
    pbar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges",
        default=PRIMARY_JUDGE,
        help=f"Comma-separated judge IDs (default: {PRIMARY_JUDGE}, the 19-model primary judge)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optionally score only generations from these comma-separated generator IDs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Параллельных запросов (по умолчанию {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    generations = list(read_jsonl(GENERATIONS_PATH))
    if not generations:
        raise FileNotFoundError(f"{GENERATIONS_PATH} пуст или не существует — выполните Этап 2.")

    requested_models = set(args.models.split(",")) if args.models else None
    if requested_models:
        known_models = set(models_config()["generators"])
        unknown_models = sorted(requested_models - known_models)
        if unknown_models:
            raise SystemExit(f"Unknown generator model_id: {unknown_models}")
        generations = [row for row in generations if row["model_id"] in requested_models]

    judges_cfg = models_config()["judges"]
    judge_ids = args.judges.split(",")
    unknown_judges = sorted(set(judge_ids) - set(judges_cfg))
    if unknown_judges:
        raise SystemExit(f"Unknown judge model_id: {unknown_judges}")

    contract = prompts_config()["judge"][JUDGE_CONTRACT_ID]
    system_prompt = contract["system"]
    user_template = contract["user_template"]

    done = load_done_keys(OUTPUT_PATH, _done_key)
    tasks = [
        (gen, judge_id)
        for gen in generations
        for judge_id in judge_ids
        if f"{gen['generation_id']}::{judge_id}" not in done
    ]
    print(
        f"К выполнению: {len(tasks)} из {len(generations) * len(judge_ids)} "
        f"(пропущено готовых: {len(done)}), workers={args.workers}."
    )

    if not tasks:
        return

    stop_heartbeat = threading.Event()
    futures: list = []

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            now = time.monotonic()
            with _inflight_lock:
                ages = sorted(now - t for t in _inflight_started.values())
            done_count = sum(1 for f in futures if f.done())
            if ages:
                print(
                    f"[HEARTBEAT] в полёте: {len(ages)} запросов, "
                    f"возраст min={ages[0]:.0f}s max={ages[-1]:.0f}s "
                    f"median={ages[len(ages) // 2]:.0f}s; futures.done()={done_count}/{len(futures)}"
                )
            else:
                print(f"[HEARTBEAT] в полёте: 0 запросов (между задачами); futures.done()={done_count}/{len(futures)}")

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    pbar = tqdm(total=len(tasks), desc="score")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures.extend(
                executor.submit(_score_one, gen, judge_id, system_prompt, user_template, pbar)
                for gen, judge_id in tasks
            )
            # executor.__exit__ дожидается завершения всех submit() через
            # shutdown(wait=True) -- не нужен явный сбор результатов в цикле.
    finally:
        pbar.close()
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
