from __future__ import annotations

import argparse
import asyncio
import csv as _csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from valar.value_space.schwartz import SCHWARTZ_10


SYSTEM_PROMPT = (
    "Ты эксперт по теории базовых ценностей Шварца. "
    "Твоя задача — по тексту определить, какую из 10 базовых ценностей он выражает наиболее явно.\n\n"
    "Ценности и их описания:\n"
    "• Self-Direction  — независимость мышления и действий, свобода выбора, творчество, исследование\n"
    "• Stimulation     — стремление к новизне, острые ощущения, риск, захватывающая жизнь\n"
    "• Hedonism        — наслаждение жизнью, чувственное удовольствие, удовлетворение желаний\n"
    "• Achievement     — личный успех через демонстрацию компетентности, амбиции, достижения\n"
    "• Power           — власть, контроль над людьми и ресурсами, социальный статус и престиж\n"
    "• Security        — безопасность, стабильность, порядок, защита от угроз\n"
    "• Conformity      — соблюдение норм и правил, сдержанность, уважение обязательств перед другими\n"
    "• Tradition       — уважение к обычаям и традициям своей культуры или религии\n"
    "• Benevolence     — забота о благополучии близких людей, помощь своей группе\n"
    "• Universalism    — справедливость, равенство, толерантность, защита всех людей и природы\n\n"
    "Если текст не выражает ни одной из этих ценностей — ответь: none"
)

USER_TEMPLATE = (
    "Определи одну наиболее выраженную ценность в тексте.\n\n"
    "Текст:\n"
    "«{text}»\n\n"
    "Ответь ровно одним словом из списка (без пояснений):\n"
    "Self-Direction | Stimulation | Hedonism | Achievement | Power | "
    "Security | Conformity | Tradition | Benevolence | Universalism | none"
)

VALID_LABELS = set(SCHWARTZ_10) | {"none"}


@dataclass
class ProgressTracker:
    total: int
    label: str = ""
    print_every: int = field(init=False)
    _done: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)
    _start: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self.print_every = max(10, self.total // 200)

    def tick(self, error: bool = False) -> None:
        self._done += 1
        if error:
            self._errors += 1
        if self._done % self.print_every == 0 or self._done == self.total:
            self._render()

    def _render(self) -> None:
        elapsed = time.monotonic() - self._start
        rate = self._done / elapsed if elapsed > 0 else 0
        remain = (self.total - self._done) / rate if rate > 0 else float("inf")
        pct = 100 * self._done / self.total
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        eta = f"{remain:.0f}s" if remain < 60 else (
              f"{remain/60:.1f}m" if remain < 3600 else f"{remain/3600:.1f}h")
        err_str = f"  err={self._errors}" if self._errors else ""
        if self._done == self.total:
            elapsed_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
            print(f"\r  [{self.label}] {bar} 100.0%  "
                  f"{self.total:,}/{self.total:,}  "
                  f"{rate:.1f} calls/s  done in {elapsed_str}{err_str}")
        else:
            print(f"\r  [{self.label}] {bar} {pct:5.1f}%  "
                  f"{self._done:,}/{self.total:,}  "
                  f"{rate:.1f} calls/s  ETA {eta}{err_str}  ",
                  end="", flush=True)


def _parse_label(raw: str) -> str:
    cleaned = raw.strip().rstrip(".,;:!?\"'")
    if cleaned in VALID_LABELS:
        return cleaned
    lower = cleaned.lower()
    for label in VALID_LABELS:
        if label.lower() == lower:
            return label
    for label in sorted(VALID_LABELS, key=len, reverse=True):
        if label.lower() in lower:
            return label
    return "parse_error"


async def _call_gpt(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    item: dict,
) -> dict:
    text = item["item_text"][:1500]
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
                ],
                max_tokens=20,
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"\n  [gpt] API error on {item['item_id']}: {exc}")
            return {
                "item_id":          item["item_id"],
                "item_text":        item["item_text"],
                "valuellama_top1":  item["valuellama_top1"],
                "valuellama_score": item["valuellama_score"],
                "gpt_label":        "api_error",
                "agreed":           False,
                "gpt_raw":          str(exc)[:300],
            }

    gpt_label = _parse_label(raw)
    return {
        "item_id":          item["item_id"],
        "item_text":        item["item_text"],
        "valuellama_top1":  item["valuellama_top1"],
        "valuellama_score": item["valuellama_score"],
        "gpt_label":        gpt_label,
        "agreed":           gpt_label == item["valuellama_top1"],
        "gpt_raw":          raw.strip(),
    }


def load_labeled_items(banks_root: Path) -> list[dict]:
    text_index: dict[str, str] = {}
    for jsonl_path in sorted(banks_root.glob("*/generation/annotation_rows.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text_index[row["item_id"]] = row["item_text"]

    items: list[dict] = []
    seen: set[str] = set()

    for csv_path in sorted(banks_root.glob("*/scoring/value_scores.csv")):
        with open(csv_path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                top1 = row.get("top1_value", "").strip()
                if not top1:
                    continue
                item_id = row["item_id"]
                if item_id in seen:
                    continue
                seen.add(item_id)

                col = f"score_{top1.lower().replace('-', '_')}"
                score_str = row.get(col, "")
                score = float(score_str) if score_str else None

                full_text = text_index.get(item_id, row.get("item_text_preview", ""))

                items.append({
                    "item_id":          item_id,
                    "item_text":        full_text,
                    "valuellama_top1":  top1,
                    "valuellama_score": score,
                })

    return items


def _is_api_error(row: dict) -> bool:
    if row.get("gpt_label") == "api_error":
        return True
    if row.get("gpt_label") == "parse_error" and row.get("gpt_raw") == "error":
        return True
    return False


def clean_api_errors(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    keep, removed = [], 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                if _is_api_error(json.loads(line)):
                    removed += 1
                    continue
            except Exception:
                pass
            keep.append(line)
    if removed:
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(keep)
    return removed


def load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                    if not _is_api_error(row):
                        done.add(row["item_id"])
                except Exception:
                    pass
    return done


async def run_async(
    items: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    concurrency: int,
    output_path: Path,
) -> list[dict]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    tracker = ProgressTracker(total=len(items), label="GPT labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _call_write(item: dict) -> dict:
        row = await _call_gpt(client, sem, model, item)
        if row["gpt_label"] != "api_error":
            async with lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tracker.tick(error=row["gpt_label"] in ("parse_error", "api_error"))
        return row

    results = await asyncio.gather(*[_call_write(it) for it in items])
    await client.close()
    return list(results)


def print_summary(results: list[dict], elapsed: float) -> None:
    n = len(results)
    if n == 0:
        return
    agreed     = sum(1 for r in results if r.get("agreed"))
    no_value   = sum(1 for r in results if r.get("gpt_label") == "none")
    api_errors = sum(1 for r in results if r.get("gpt_label") == "api_error")
    parse_errs = sum(1 for r in results if r.get("gpt_label") == "parse_error")

    print(f"\n[gpt] Done: {n:,} items in {elapsed:.0f}s  ({n/elapsed:.1f} items/s)")
    print(f"[gpt] Agreement (ValueLlama == GPT): {agreed:,}/{n:,}  ({100*agreed/n:.1f}%)")
    print(f"[gpt] GPT said 'none':               {no_value:,}  ({100*no_value/n:.1f}%)")
    print(f"[gpt] API errors (will retry):       {api_errors:,}")
    print(f"[gpt] Parse errors:                  {parse_errs:,}")

    per_value: dict[str, dict] = defaultdict(lambda: {"total": 0, "agreed": 0})
    for r in results:
        v = r["valuellama_top1"]
        per_value[v]["total"] += 1
        if r.get("agreed"):
            per_value[v]["agreed"] += 1

    max_total = max((d["total"] for d in per_value.values()), default=1)
    print("\n[gpt] Agreement by value (ValueLlama top1):")
    for v in SCHWARTZ_10:
        d = per_value.get(v)
        if not d or d["total"] == 0:
            continue
        pct = 100 * d["agreed"] / d["total"]
        bar = "█" * int(pct / 5)
        print(f"  {v:20} {d['agreed']:5,}/{d['total']:5,}  {pct:5.1f}%  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-annotate ValueLlama items with GPT Schwartz-10 classification"
    )
    parser.add_argument("--banks-root",  default="data/item_banks",
                        help="Root dir with ValueLlama annotation outputs")
    parser.add_argument("--output",      default=None,
                        help="Output JSONL path (auto-timestamped if omitted; "
                             "pass existing file to resume)")
    parser.add_argument("--api-key",     default=None,
                        help="GPT API key (or env: VALAR_GPT_API_KEY)")
    parser.add_argument("--base-url",    default=None,
                        help="API base URL (or env: VALAR_GPT_BASE_URL)")
    parser.add_argument("--model",       default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max parallel requests (default: 20)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Process only first N items (for testing)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show plan and sample prompts, no API calls")
    args = parser.parse_args()

    api_key  = args.api_key  or os.environ.get("VALAR_GPT_API_KEY",  "")
    base_url = args.base_url or os.environ.get("VALAR_GPT_BASE_URL",
                                               "https://api.aitunnel.ru/v1/")

    if not api_key and not args.dry_run:
        raise SystemExit(
            "[gpt] Error: provide --api-key or set VALAR_GPT_API_KEY in environment"
        )

    banks_root = Path(args.banks_root)

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = banks_root / "gpt_labels" / f"gpt_annotation_{ts}.jsonl"

    print(f"[gpt] Loading ValueLlama-labeled items from {banks_root} ...")
    all_items = load_labeled_items(banks_root)
    print(f"[gpt] Found {len(all_items):,} items with top1_value")

    removed = clean_api_errors(output_path)
    if removed:
        print(f"[gpt] Cleaned {removed:,} api_error rows from output file (will retry)")

    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"[gpt] Resuming — {len(done_ids):,} items already done, skipping")
    items = [it for it in all_items if it["item_id"] not in done_ids]

    if args.limit:
        items = items[: args.limit]

    eta_min = len(items) / max(args.concurrency, 1) * 0.8 / 60
    print(f"[gpt] Items to annotate: {len(items):,}")
    print(f"[gpt] Model:             {args.model}")
    print(f"[gpt] Base URL:          {base_url}")
    print(f"[gpt] Concurrency:       {args.concurrency}")
    print(f"[gpt] ETA:               ≈{eta_min:.0f} min")
    print(f"[gpt] Output:            {output_path}")
    print()

    if args.dry_run:
        print("[gpt] DRY RUN — sample prompts for first 2 items:")
        for item in items[:2]:
            print(f"\n  item_id:         {item['item_id']}")
            print(f"  valuellama_top1: {item['valuellama_top1']}"
                  f"  (score={item['valuellama_score']})")
            print(f"  text preview:    {item['item_text'][:120]!r}")
            print(f"\n  [system] {SYSTEM_PROMPT[:120]}...")
            print(f"\n  [user]   {USER_TEMPLATE.format(text=item['item_text'][:80])}")
        print("\n[gpt] Dry run complete — no API calls made.")
        return

    t0 = time.time()
    results = asyncio.run(run_async(
        items=items,
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        concurrency=args.concurrency,
        output_path=output_path,
    ))
    print_summary(results, time.time() - t0)
    print(f"\n[gpt] Output → {output_path}")


if __name__ == "__main__":
    main()
