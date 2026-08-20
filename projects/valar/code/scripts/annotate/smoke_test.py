from __future__ import annotations

import argparse
import asyncio

from valar.annotators.valuellama import (
    measure_perceptions_async,
    get_score,
    RELEVANCE_THRESHOLD,
)
from valar.value_space.schwartz import SCHWARTZ_10
from valar import env


SMOKE_TEXTS = [
    "Thinking up new ideas and being creative is important to me.",
    "It is important to me to be rich and have expensive things.",
    "I believe everyone should have equal opportunities in life.",
    "I try hard to do what my religion requires.",
    "Для меня важно самому принимать решения и быть независимым.",
    "Я стремлюсь к богатству и власти над другими.",
    "Все люди заслуживают равных прав и возможностей.",
    "Важно соблюдать традиции и обычаи, которые передаются из поколения в поколение.",
    "Читаю. Драки, грабежи, ДТП.",
    "Начальники ДК и нынешнего места работы?",
]


async def run_smoke(api_base: str, model_name: str) -> None:
    print(f"\n[smoke] API: {api_base}  model: {model_name}")
    print(f"[smoke] Testing {len(SMOKE_TEXTS)} perceptions × {len(SCHWARTZ_10)} values\n")

    results = await measure_perceptions_async(
        perceptions=SMOKE_TEXTS,
        values=SCHWARTZ_10,
        model=model_name,
        api_base_url=api_base,
        concurrency=16,
    )

    for text, pr in zip(SMOKE_TEXTS, results):
        print(f"TEXT: {repr(text[:80])}")
        if not pr.relevant_values:
            print("  → no relevant values found")
        else:
            for v, rel_vec, val_vec in zip(pr.relevant_values, pr.relevances, pr.valences):
                score = get_score(val_vec)
                if score is None:
                    label, score_str = "either", "  None"
                elif score > 0:
                    label, score_str = "support", f"{score:+.3f}"
                else:
                    label, score_str = "oppose", f"{score:+.3f}"
                p_yes = rel_vec[0]
                print(f"  {v:20} p_rel={p_yes:.3f}  score={score_str}  → {label}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    api_base = args.api_base_url or env.valuellama_api_base_url()
    model_name = args.model_name or env.served_model_name()

    asyncio.run(run_smoke(api_base, model_name))


if __name__ == "__main__":
    main()
