"""Run the VALAR Schwartz-10 evaluation against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).parent
GOLD_BANK = ROOT / "data" / "schwartz10_ru_1000.jsonl"
L0_BANK = ROOT / "data" / "l0_h0_100.jsonl"

SCHWARTZ_10 = [
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
]
LABEL_SET = ", ".join(SCHWARTZ_10)

_ALIASES: dict[str, str] = {}
for _value in SCHWARTZ_10:
    for _alias in (
        _value,
        _value.lower(),
        _value.replace("-", " "),
        _value.replace("-", ""),
        _value.replace("-", "_"),
    ):
        _ALIASES[re.sub(r"[^a-z0-9]+", "", _alias.lower())] = _value
_ALIASES.update(
    {
        "selfdirectionvalue": "Self-Direction",
        "stim": "Stimulation",
        "achievementvalue": "Achievement",
        "powerdominance": "Power",
        "safetysecurity": "Security",
        "obedience": "Conformity",
        "religiontradition": "Tradition",
        "benevolent": "Benevolence",
        "universalist": "Universalism",
    }
)

DEFAULT_SYSTEM_PROMPT = (
    "Вы — ассистент по классификации ценностей. "
    "Анализируйте тексты строго по схеме Шварца. "
    "Всегда возвращайте только JSON."
)


def canonicalize(raw: Any) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())
    return _ALIASES.get(key)


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.I | re.S).strip()
    match = re.search(r"\{.*\}", text, re.S)
    candidate = match.group() if match else None
    if candidate is None:
        truncated = re.search(r"\{.*", text, re.S)
        candidate = truncated.group() + "}" if truncated else None
    if candidate is None:
        return None
    for attempt in (candidate, re.sub(r"[\x00-\x1f\x7f]", " ", candidate)):
        try:
            parsed = json.loads(attempt)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def gold_prompt(item_text: str) -> str:
    return (
        "Определите, какую базовую ценность Шварца наиболее явно выражает следующий текст.\n\n"
        f"Выберите три наиболее подходящие ценности из этого набора:\n{LABEL_SET}\n\n"
        f"Текст:\n{item_text}\n\n"
        "Верните только один корректный JSON-объект. Не используйте markdown и не добавляйте "
        "пояснений вне JSON. Значения top1, top2, top3 должны быть скопированы точно из набора "
        "выше. Обоснование — одна строка, не более 25 слов.\n\n"
        "Обязательная схема:\n"
        '{"top1":"точная метка","top2":"точная метка",'
        '"top3":"точная метка","rationale":"краткое обоснование"}'
    )


def build_gold_messages(
    item_text: str,
    system_prompt: str | None = None,
    no_system_role: bool = False,
) -> list[dict[str, str]]:
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    user = gold_prompt(item_text)
    if no_system_role:
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    disable_thinking: bool,
) -> tuple[str, dict]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"].get("content") or "", body.get("usage") or {}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_model_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_") or "model"


def _stable_id(item: dict) -> str:
    return str(item.get("task_id") or item.get("item_id") or item.get("scenario_id"))


def _run_bank(
    *,
    items: list[dict],
    messages_for: Callable[[dict], list[dict[str, str]]],
    output_path: Path,
    bank_name: str,
    args: argparse.Namespace,
) -> list[dict]:
    previous = load_jsonl(output_path) if output_path.exists() else []
    done = {row["item_id"] for row in previous if row.get("api_ok")}
    todo = [item for item in items if _stable_id(item) not in done]
    print(f"[{bank_name}] pending={len(todo)} already_done={len(done)}")

    def work(item: dict) -> dict:
        started = time.monotonic()
        content = ""
        usage: dict = {}
        error: str | None = None
        for attempt in range(2):
            try:
                content, usage = call_api(
                    base_url=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages_for(item),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_sec=args.timeout_sec,
                    disable_thinking=args.disable_thinking,
                )
                error = None
                break
            except Exception as exc:  # preserve the failed item for resumability
                error = repr(exc)
                if attempt == 0:
                    time.sleep(1)
        row = {
            "item_id": _stable_id(item),
            "model": args.model,
            "content": content,
            "usage": usage,
            "latency": round(time.monotonic() - started, 4),
            "api_ok": error is None,
        }
        if error:
            row["error"] = error
        for field in (
            "task_family",
            "gold_value",
            "gold_value_a",
            "gold_value_b",
            "gold_pair_unordered",
            "candidate_value",
            "answer",
            "item_text",
            "scenario_text",
            "source_dataset",
        ):
            if field in item:
                row[field] = item[field]
        return row

    fresh: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = [pool.submit(work, item) for item in todo]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            fresh.append(row)
            status = "OK" if row["api_ok"] else "ERR"
            print(f"[{bank_name}][{status}] {index}/{len(todo)} id={row['item_id']}")

    combined = previous + fresh
    write_jsonl(output_path, combined)
    return combined


def _parse_gold_row(row: dict) -> dict:
    parsed = parse_json(row.get("content", "")) or {}
    top = [canonicalize(parsed.get(f"top{i}")) for i in range(1, 4)]
    return {**row, "top1": top[0], "top2": top[1], "top3": top[2], "parse_ok": top[0] is not None}


def score_gold(rows: list[dict]) -> tuple[dict, list[dict]]:
    scored = [_parse_gold_row(row) for row in rows]
    counts = {value: {"top1": 0, "top3": 0, "total": 0} for value in SCHWARTZ_10}
    for row in scored:
        gold = row.get("gold_value")
        if gold not in counts:
            continue
        counts[gold]["total"] += 1
        counts[gold]["top1"] += int(row["top1"] == gold)
        counts[gold]["top3"] += int(gold in (row["top1"], row["top2"], row["top3"]))
    recall1 = {value: count["top1"] / count["total"] if count["total"] else 0.0 for value, count in counts.items()}
    recall3 = {value: count["top3"] / count["total"] if count["total"] else 0.0 for value, count in counts.items()}
    n = len(scored)
    summary = {
        "n_items": n,
        "parse_rate": round(sum(row["parse_ok"] for row in scored) / max(n, 1), 4),
        "micro_acc1": round(sum(count["top1"] for count in counts.values()) / max(n, 1), 4),
        "macro_acc1": round(sum(recall1.values()) / len(SCHWARTZ_10), 4),
        "macro_acc3": round(sum(recall3.values()) / len(SCHWARTZ_10), 4),
        "recall1_per_value": {key: round(value, 4) for key, value in recall1.items()},
        "recall3_per_value": {key: round(value, 4) for key, value in recall3.items()},
    }
    return summary, scored


def _parse_l0_row(row: dict) -> dict:
    family = row.get("task_family", "")
    parsed = parse_json(row.get("content", "")) or {}
    result = {**row, "parse_ok": False, "hits1": None, "hits3": None, "pair_match": None, "relevance_correct": None}
    if family == "h0_item_to_value":
        top = [canonicalize(parsed.get(f"top{i}")) for i in range(1, 4)]
        result.update(
            parse_ok=top[0] is not None,
            hits1=int(top[0] == row.get("gold_value")),
            hits3=int(row.get("gold_value") in top),
        )
    elif family == "h0_conflict_recognition":
        values = [canonicalize(parsed.get("value_a")), canonicalize(parsed.get("value_b"))]
        valid = all(values)
        predicted = "|".join(sorted(values)) if valid else ""
        gold = "|".join(sorted(row.get("gold_pair_unordered", "").split("|")))
        result.update(parse_ok=valid, pair_match=int(predicted == gold))
    elif family == "h0_contextual_relevance":
        answer = str(parsed.get("answer", "")).strip().lower()
        valid = answer in {"yes", "no"}
        result.update(parse_ok=valid, relevance_correct=int(answer == str(row.get("answer", "")).lower()))
    return result


def score_l0(rows: list[dict]) -> tuple[dict, list[dict]]:
    scored = [_parse_l0_row(row) for row in rows]

    def mean(family: str, field: str) -> float:
        values = [row[field] for row in scored if row.get("task_family") == family and row.get(field) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "n_items": len(scored),
        "h0_hits_at_1": mean("h0_item_to_value", "hits1"),
        "h0_hits_at_3": mean("h0_item_to_value", "hits3"),
        "h0_pair_match": mean("h0_conflict_recognition", "pair_match"),
        "h0_relv_acc": mean("h0_contextual_relevance", "relevance_correct"),
    }, scored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VALAR Schwartz-10 evaluation")
    parser.add_argument("--model", required=True, help="Model name served by the endpoint")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base, usually ending in /v1")
    parser.add_argument("--api-key", default=os.environ.get("VALAR_API_KEY", "dummy"))
    parser.add_argument("--parallelism", type=int, default=int(os.environ.get("VALAR_PARALLELISM", "16")))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--run-dir", type=Path, help="Resume an existing run directory")
    parser.add_argument("--smoke", action="store_true", help="Run five items from each bank")
    parser.add_argument("--system-prompt")
    parser.add_argument("--no-system-role", action="store_true")
    parser.add_argument(
        "--thinking",
        choices=("auto", "always", "never"),
        default="auto",
        help="Send chat_template_kwargs={enable_thinking:false}; auto omits it for Mistral models",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mistral = any(token in args.model.lower() for token in ("mistral", "ministral"))
    args.disable_thinking = args.thinking == "always" or (args.thinking == "auto" and not mistral)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or args.results_dir / f"{_safe_model_id(args.model)}_{timestamp}"
    gold_items = load_jsonl(GOLD_BANK)
    l0_items = load_jsonl(L0_BANK)
    if args.smoke:
        gold_items, l0_items = gold_items[:5], l0_items[:5]

    print(f"model={args.model} gold={len(gold_items)} l0={len(l0_items)} run_dir={run_dir}")
    gold_rows = _run_bank(
        items=gold_items,
        messages_for=lambda item: build_gold_messages(item["item_text"], args.system_prompt, args.no_system_role),
        output_path=run_dir / "gold1000" / "generation_rows.jsonl",
        bank_name="gold1000",
        args=args,
    )
    l0_rows = _run_bank(
        items=l0_items,
        messages_for=lambda item: [{"role": "user", "content": item["prompt_text"]}],
        output_path=run_dir / "l0_h0" / "generation_rows.jsonl",
        bank_name="l0_h0",
        args=args,
    )

    gold_summary, gold_scored = score_gold(gold_rows)
    l0_summary, l0_scored = score_l0(l0_rows)
    write_jsonl(run_dir / "gold1000" / "scored_rows.jsonl", gold_scored)
    write_jsonl(run_dir / "l0_h0" / "scored_rows.jsonl", l0_scored)
    write_json(run_dir / "gold1000" / "scores.json", gold_summary)
    write_json(run_dir / "l0_h0" / "scores.json", l0_summary)
    write_json(
        run_dir / "combined_summary.json",
        {"model": args.model, "run_timestamp_utc": timestamp, "gold1000": gold_summary, "l0_h0": l0_summary},
    )
    print(f"gold Acc@1={gold_summary['micro_acc1']:.3f} Acc@3={gold_summary['macro_acc3']:.3f}")
    print(f"L0 hits@1={l0_summary['h0_hits_at_1']:.3f} pair={l0_summary['h0_pair_match']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
