from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from valar.value_space.schwartz import SCHWARTZ_10


def build_prompt(text: str) -> list[dict[str, str]]:
    """Build the documented Schwartz-10 score prompt for one text."""
    labels = ", ".join(SCHWARTZ_10)
    return [
        {
            "role": "system",
            "content": "Score value expression using the Schwartz-10 taxonomy.",
        },
        {
            "role": "user",
            "content": (
                f"Text: {text}\n\n"
                f"Values: {labels}\n\n"
                "Return one JSON object mapping every value to a score from 0 to 1."
            ),
        },
    ]


def parse_output(raw: str) -> dict[str, object]:
    """Parse a complete JSON score map and clamp numeric scores to [0, 1]."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"parse_ok": False, "scores": None}

    if not isinstance(payload, dict) or any(value not in payload for value in SCHWARTZ_10):
        return {"parse_ok": False, "scores": None}

    scores: dict[str, float] = {}
    try:
        for value in SCHWARTZ_10:
            scores[value] = min(1.0, max(0.0, float(payload[value])))
    except (TypeError, ValueError):
        return {"parse_ok": False, "scores": None}

    return {"parse_ok": True, "scores": scores}


RELEVANCE_TEMPLATE = (
    '[Task] Given a sentence and a value, determine whether the sentence is '
    'relevant to the value. If the sentence is relevant to the value, output '
    '"yes", otherwise output "no".\n'
    'Sentence: {sentence}\n'
    'Value: {value}\n'
    'Output:\n'
)

VALENCE_TEMPLATE = (
    '[Task] Given a sentence and a value, determine whether the sentence '
    'supports or opposes the value. If the sentence supports the value, output '
    '"support". If the sentence opposes the value, output "oppose". If you need '
    'more context to make a decision, output "either".\n'
    'Sentence: {sentence}\n'
    'Value: {value}\n'
    'Output:\n'
)

RELEVANCE_THRESHOLD = 0.5


def _extract_prob(top_logprobs: dict[str, float], keyword: str) -> float:
    for tok, logp in top_logprobs.items():
        if tok.strip().lower() == keyword.lower():
            return math.exp(logp)
    return 1e-9


def _parse_relevance(top_logprobs: dict[str, float]) -> list[float]:
    p_yes = _extract_prob(top_logprobs, 'yes')
    p_no  = _extract_prob(top_logprobs, 'no')
    total = p_yes + p_no
    if total < 1e-15:
        return [0.5, 0.5]
    return [p_yes / total, p_no / total]


def _parse_valence(top_logprobs: dict[str, float]) -> list[float]:
    p_s = _extract_prob(top_logprobs, 'support')
    p_o = _extract_prob(top_logprobs, 'oppose')
    p_e = _extract_prob(top_logprobs, 'either')
    total = p_s + p_o + p_e
    if total < 1e-15:
        return [1/3, 1/3, 1/3]
    return [p_s / total, p_o / total, p_e / total]


def get_score(valence_vec: list[float]) -> float | None:
    p_s, p_o, p_e = valence_vec
    if p_e > p_s and p_e > p_o:
        return None
    return p_s - p_o


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
        rate    = self._done / elapsed if elapsed > 0 else 0
        remain  = (self.total - self._done) / rate if rate > 0 else float("inf")
        pct     = 100 * self._done / self.total
        filled  = int(pct / 5)
        bar     = "█" * filled + "░" * (20 - filled)

        if remain < 60:
            eta = f"{remain:.0f}s"
        elif remain < 3600:
            eta = f"{remain/60:.1f}m"
        else:
            eta = f"{remain/3600:.1f}h"

        err_str = f"  err={self._errors}" if self._errors else ""
        line = (
            f"\r  [{self.label}] {bar} {pct:5.1f}%  "
            f"{self._done:,}/{self.total:,}  "
            f"{rate:.0f} calls/s  ETA {eta}{err_str}  "
        )
        if self._done == self.total:
            elapsed_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
            print(f"\r  [{self.label}] {bar} 100.0%  "
                  f"{self.total:,}/{self.total:,}  "
                  f"{rate:.0f} calls/s  done in {elapsed_str}{err_str}")
        else:
            print(line, end="", flush=True)


async def _tracked(coro, tracker: ProgressTracker):
    try:
        result = await coro
        tracker.tick(error=False)
        return result
    except Exception:
        tracker.tick(error=True)
        raise


async def _call_completion(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
) -> dict[str, float] | None:
    async with semaphore:
        try:
            resp = await client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                logprobs=20,
                temperature=0,
            )
            return resp.choices[0].logprobs.top_logprobs[0]
        except Exception as exc:
            print(f"  [valar] API error: {exc}")
            return None


async def _get_relevance_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    perception: str,
    value: str,
) -> list[float]:
    prompt = RELEVANCE_TEMPLATE.format(sentence=perception, value=value)
    top_logprobs = await _call_completion(client, semaphore, model, prompt)
    if top_logprobs is None:
        return [0.5, 0.5]
    return _parse_relevance(top_logprobs)


async def _get_valence_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    perception: str,
    value: str,
) -> list[float]:
    prompt = VALENCE_TEMPLATE.format(sentence=perception, value=value)
    top_logprobs = await _call_completion(client, semaphore, model, prompt)
    if top_logprobs is None:
        return [1/3, 1/3, 1/3]
    return _parse_valence(top_logprobs)


@dataclass
class PerceptionResult:
    perception: str
    relevant_values: list[str]
    relevances: list[list[float]]
    valences: list[list[float]]


async def measure_perceptions_async(
    perceptions: list[str],
    values: list[str],
    model: str,
    api_base_url: str,
    api_key: str | None = None,
    concurrency: int = 64,
) -> list[PerceptionResult]:
    from valar import env as _env
    resolved_key = api_key or _env.valuellama_api_key()
    client = AsyncOpenAI(base_url=api_base_url, api_key=resolved_key)
    sem = asyncio.Semaphore(concurrency)

    n_p, n_v = len(perceptions), len(values)
    n_rel_calls = n_p * n_v

    rel_tracker = ProgressTracker(total=n_rel_calls, label="Phase 1/2  Relevance")
    rel_tasks = [
        _tracked(_get_relevance_async(client, sem, model, p, v), rel_tracker)
        for p in perceptions
        for v in values
    ]
    rel_flat = await asyncio.gather(*rel_tasks)

    rel_matrix = [
        [rel_flat[i * n_v + j] for j in range(n_v)]
        for i in range(n_p)
    ]

    relevant_pairs: list[tuple[str, str]] = []
    pair_map: dict[tuple[str, str], list[float]] = {}

    for i, p in enumerate(perceptions):
        for j, v in enumerate(values):
            if rel_matrix[i][j][0] > RELEVANCE_THRESHOLD:
                relevant_pairs.append((p, v))

    if relevant_pairs:
        val_tracker = ProgressTracker(total=len(relevant_pairs), label="Phase 2/2  Valence ")
        val_tasks = [
            _tracked(_get_valence_async(client, sem, model, p, v), val_tracker)
            for p, v in relevant_pairs
        ]
        val_results = await asyncio.gather(*val_tasks)
        for (p, v), vec in zip(relevant_pairs, val_results):
            pair_map[(p, v)] = vec

    results: list[PerceptionResult] = []
    for i, p in enumerate(perceptions):
        rel_values, rel_vecs, val_vecs = [], [], []
        for j, v in enumerate(values):
            if rel_matrix[i][j][0] > RELEVANCE_THRESHOLD:
                rel_values.append(v)
                rel_vecs.append(rel_matrix[i][j])
                val_vecs.append(pair_map.get((p, v), [1/3, 1/3, 1/3]))
        results.append(PerceptionResult(
            perception=p,
            relevant_values=rel_values,
            relevances=rel_vecs,
            valences=val_vecs,
        ))

    await client.close()
    return results
