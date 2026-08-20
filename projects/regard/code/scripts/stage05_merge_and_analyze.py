"""Этап 5 — слияние всех источников и расчёт метрик RQ1-RQ4.

Вход:
    data/generations/generations_raw.jsonl
    data/scores/judge_scores_raw.jsonl
Выход:
    data/processed/merged_dataset.parquet        (широкая таблица для notebooks/analysis.ipynb)
    data/processed/agreement_stats.csv           (judge-vs-judge по V/A/D)

TODO: judge-vs-human (вторая половина RQ4) пока не реализован — нужен
data/annotation/annotation_results_raw.csv (появится после сбора форм у аннотаторов,
см. Этап 4). Путь зарезервирован в ANNOTATION_RESULTS_PATH, join с merged_dataset
дописать сюда, когда разметка будет собрана.

Запуск:
    uv run python -m scripts.stage05_merge_and_analyze
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.common.config import ROOT_DIR

MIN_TARGET_COVERAGE = 0.5
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260627

GENERATIONS_PATH = ROOT_DIR / "data" / "generations" / "generations_raw.jsonl"
SCORES_PATH = ROOT_DIR / "data" / "scores" / "judge_scores_raw.jsonl"
ANNOTATION_RESULTS_PATH = ROOT_DIR / "data" / "annotation" / "annotation_results_raw.csv"

MERGED_OUTPUT = ROOT_DIR / "data" / "processed" / "merged_dataset.parquet"
AGREEMENT_OUTPUT = ROOT_DIR / "data" / "processed" / "agreement_stats.csv"

VAD_AXES = ["valence", "arousal", "dominance"]


def load_generations() -> pd.DataFrame:
    return pd.read_json(GENERATIONS_PATH, lines=True)


def load_scores() -> pd.DataFrame:
    return pd.read_json(SCORES_PATH, lines=True)


def build_merged_dataset() -> pd.DataFrame:
    generations = load_generations()
    scores = load_scores()

    # каждая генерация скорится 2 judge — pivot в широкий формат:
    # lm_valence_qwen36_35b, lm_valence_gpt4o_mini, ...
    wide_scores = scores.pivot_table(
        index="generation_id",
        columns="judge_model",
        values=["lm_valence", "lm_arousal", "lm_dominance", "lm_confidence", "target_coverage"],
    )
    wide_scores.columns = [f"{metric}__{judge}" for metric, judge in wide_scores.columns]
    wide_scores = wide_scores.reset_index()

    return generations.merge(wide_scores, on="generation_id", how="left")


def rq1_rq2_vad_distance(
    merged: pd.DataFrame,
    judge_model: str,
    min_coverage: float = MIN_TARGET_COVERAGE,
    prompt_id: str = "evaluative_stance",
) -> pd.DataFrame:
    """VAD-euclidean distance и направленная дельта (local - international) по
    каждой оси, по таргету. Агрегируется по target_family (RQ2) и в среднем по
    всем таргетам (RQ1).

    Фильтр по prompt_id обязателен: с появлением robustness-промптов
    (neutral_descriptive, evaluative_paraphrase) в одном generations_raw.jsonl
    может быть несколько генераций одной модели на один таргет — без фильтра
    groupby(["target_id", ..., "model_role"]).mean() молча усреднит их вместе
    и смешает основной эффект с prompt-robustness-вариантами.

    Строки с target_coverage < min_coverage отфильтрованы до агрегации — иначе
    низкая дельта может означать "судья не смог привязать ответ к таргету", не
    "модели реально оценивают объект одинаково" (knowledge-gap confound, см.
    reviewer_faq.md, раздел 9)."""
    merged = merged[merged["prompt_id"] == prompt_id]
    coverage_col = f"target_coverage__{judge_model}"
    cols = [f"lm_{axis}__{judge_model}" for axis in VAD_AXES]

    covered = merged[merged[coverage_col] >= min_coverage] if coverage_col in merged.columns else merged
    dropped = len(merged) - len(covered)
    if dropped:
        print(f"  [{judge_model}] отфильтровано {dropped} строк с target_coverage < {min_coverage}")

    by_role = (
        covered.groupby(["target_id", "target_family", "model_role"])[cols]
        .mean()
        .reset_index()
    )
    pivoted = by_role.pivot_table(index=["target_id", "target_family"], columns="model_role", values=cols)

    rows = []
    for (target_id, family), row in pivoted.iterrows():
        try:
            local_vec = [row[(col, "local")] for col in cols]
            intl_vec = [row[(col, "international")] for col in cols]
        except KeyError:
            continue
        delta = {f"delta_{axis}": local_vec[i] - intl_vec[i] for i, axis in enumerate(VAD_AXES)}
        dist = sum((a - b) ** 2 for a, b in zip(local_vec, intl_vec)) ** 0.5
        rows.append({"target_id": target_id, "target_family": family, "vad_distance": dist, **delta})

    return pd.DataFrame(rows)


def rq_prompt_robustness(
    merged: pd.DataFrame,
    judge_model: str,
    prompt_ids: list[str],
    min_coverage: float = MIN_TARGET_COVERAGE,
) -> pd.DataFrame:
    """Сравнение направленной дельты (local - international) по осям VAD между
    несколькими промптами (main + robustness-варианты, см. Limitations:
    "Results are tied to a single, deliberately evaluative prompt"). Один ряд
    на (prompt_id, axis) с bootstrap CI -- позволяет увидеть, меняется ли знак
    или значимость эффекта при смене формулировки промпта."""
    rows = []
    for prompt_id in prompt_ids:
        dist_df = rq1_rq2_vad_distance(merged, judge_model, min_coverage=min_coverage, prompt_id=prompt_id)
        for axis in VAD_AXES:
            mean, lo, hi = bootstrap_ci(dist_df[f"delta_{axis}"])
            rows.append({
                "prompt_id": prompt_id, "axis": axis, "n_targets": len(dist_df),
                "mean_delta": mean, "ci_low": lo, "ci_high": hi,
            })
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap по targets: (mean, ci_low, ci_high). Метод зафиксирован в
    project_status.md ("Design-решения, зафиксированные при подготовке reviewer
    FAQ") — bootstrap CI по targets, не mixed-effects модель."""
    rng = np.random.default_rng(seed)
    arr = values.dropna().to_numpy()
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iterations)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def rq3_refusal_rate(
    merged: pd.DataFrame,
    judge_model: str,
    neutral_threshold: float = 0.15,
    prompt_id: str = "evaluative_stance",
) -> pd.DataFrame:
    """Доля нейтральных/низко-уверенных ответов (proxy для refusal/уклончивости) по модели.

    Valence на шкале [0, 1] с нейтральной точкой в 0.5. Фильтр по prompt_id —
    см. docstring rq1_rq2_vad_distance."""
    merged = merged[merged["prompt_id"] == prompt_id].copy()
    valence_col = f"lm_valence__{judge_model}"
    confidence_col = f"lm_confidence__{judge_model}"
    merged["is_neutral_or_evasive"] = ((merged[valence_col] - 0.5).abs() < neutral_threshold) | (
        merged[confidence_col] < 0.5
    )
    return merged.groupby(["model_id", "model_role"])["is_neutral_or_evasive"].mean().reset_index(
        name="neutral_evasive_rate"
    )


def rq4_judge_agreement(merged: pd.DataFrame, judge_a: str, judge_b: str) -> pd.DataFrame:
    """Корреляция judge-vs-judge по осям V/A/D (Pearson)."""
    rows = []
    for axis in VAD_AXES:
        col_a, col_b = f"lm_{axis}__{judge_a}", f"lm_{axis}__{judge_b}"
        valid = merged[[col_a, col_b]].dropna()
        corr = valid[col_a].corr(valid[col_b]) if len(valid) > 1 else float("nan")
        rows.append({"comparison": f"{judge_a}_vs_{judge_b}", "axis": axis, "pearson_r": corr, "n": len(valid)})
    return pd.DataFrame(rows)


def main() -> None:
    merged = build_merged_dataset()
    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_OUTPUT, index=False)
    print(f"Слитый датасет: {len(merged)} строк -> {MERGED_OUTPUT}")

    judge_ids = sorted({c.split("__")[1] for c in merged.columns if c.startswith("lm_valence__")})
    if len(judge_ids) >= 1:
        dist_df = rq1_rq2_vad_distance(merged, judge_ids[0])
        print(f"RQ1/RQ2 VAD-distance ({judge_ids[0]}): топ-5 семейств по расхождению:")
        print(dist_df.groupby("target_family")["vad_distance"].mean().sort_values(ascending=False).head())

        print(f"\nRQ1 направленная дельта (local - international), bootstrap CI по {len(dist_df)} targets:")
        for axis in VAD_AXES:
            mean, lo, hi = bootstrap_ci(dist_df[f"delta_{axis}"])
            print(f"  delta_{axis}: mean={mean:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")

        refusal_df = rq3_refusal_rate(merged, judge_ids[0])
        print("\nRQ3 доля нейтральных/уклончивых ответов по модели:")
        print(refusal_df)

        prompt_ids_present = sorted(merged["prompt_id"].dropna().unique().tolist())
        if len(prompt_ids_present) > 1:
            robustness_df = rq_prompt_robustness(merged, judge_ids[0], prompt_ids_present)
            print(f"\nPrompt robustness ({judge_ids[0]}): направленная дельта по каждому промпту:")
            print(robustness_df)

    if len(judge_ids) == 2:
        agreement_df = rq4_judge_agreement(merged, judge_ids[0], judge_ids[1])
        agreement_df.to_csv(AGREEMENT_OUTPUT, index=False)
        print(f"\nRQ4 judge-vs-judge agreement -> {AGREEMENT_OUTPUT}")
        print(agreement_df)
    else:
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] найдено {len(judge_ids)} judge-моделей в данных, ожидалось 2 — "
              "RQ4 judge-vs-judge не посчитан.")


if __name__ == "__main__":
    main()
