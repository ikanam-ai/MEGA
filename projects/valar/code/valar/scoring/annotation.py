from __future__ import annotations

from valar.annotators.valuellama import PerceptionResult, get_score
from valar.value_space.schwartz import SCHWARTZ_10


def aggregate(
    perception_results: list[PerceptionResult],
    values: list[str] = SCHWARTZ_10,
) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = {v: [] for v in values}

    for pr in perception_results:
        for value, valence_vec in zip(pr.relevant_values, pr.valences):
            score = get_score(valence_vec)
            if score is not None:
                buckets[value].append(score)

    return {
        v: (sum(scores) / len(scores)) if scores else None
        for v, scores in buckets.items()
    }


def top1_value(profile: dict[str, float | None]) -> str | None:
    candidates = {v: s for v, s in profile.items() if s is not None}
    if not candidates:
        return None
    return max(candidates, key=lambda v: candidates[v])


def to_row(
    item_id: str,
    source_dataset: str,
    text: str,
    profile: dict[str, float | None],
    perceptions: list[str],
    parse_ok: bool,
    n_relevant_total: int,
    extra: dict | None = None,
) -> dict:
    row: dict = {
        "item_id": item_id,
        "source_dataset": source_dataset,
        "text_preview": text[:120],
        "n_perceptions": len(perceptions),
        "n_relevant_total": n_relevant_total,
        "parse_ok": parse_ok,
        "top1_value": top1_value(profile),
    }
    for v in SCHWARTZ_10:
        row[f"score_{v.lower().replace('-', '_')}"] = profile.get(v)
    if extra:
        row.update(extra)
    return row
