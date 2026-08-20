"""Reproduce the REGARD 19-model profile, clustering, and judge checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

SEED = 20260713
AXES = ["valence", "arousal", "dominance"]
SCORE_COLUMNS = [f"lm_{axis}" for axis in AXES]
PRIMARY_JUDGE = "qwen36_35b"
MAIN_PROMPT = "evaluative_stance"
EXPECTED_MODELS = {
    "avibe",
    "gemma2_27b",
    "gigachat",
    "glm",
    "granite",
    "granite3_8b",
    "llama3_8b",
    "ministral_14b",
    "ministral_8b",
    "mistral_7b",
    "mistral_nemo",
    "phi3_medium",
    "qwen25_14b",
    "qwen25_7b",
    "qwen25_coder_14b",
    "qwen36_27b",
    "solar_10b",
    "tpro",
    "yandexgpt",
}
DISPLAY_NAMES = {
    "avibe": "AVIBE",
    "gemma2_27b": "Gemma-4-26B",
    "gigachat": "GigaChat",
    "glm": "GLM-4.7",
    "granite": "Granite-4.1",
    "granite3_8b": "Granite3.3-8B",
    "llama3_8b": "Llama-3-8B",
    "ministral_14b": "Ministral-14B",
    "ministral_8b": "Ministral-8B",
    "mistral_7b": "Mistral-7B",
    "mistral_nemo": "Mistral-Nemo",
    "phi3_medium": "Phi-3-Medium",
    "qwen25_14b": "Qwen2.5-14B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen25_coder_14b": "Qwen2.5-Coder-14B",
    "qwen36_27b": "Qwen3.6-27B",
    "solar_10b": "SOLAR-10.7B",
    "tpro": "T-pro-it-2.1",
    "yandexgpt": "YandexGPT",
}
CLUSTER_NAMES = {1: "C1 Evasive", 2: "C2 Moderate", 3: "C3 Expressive"}
CLUSTER_COLORS = {1: "#3b82c4", 2: "#3ca66b", 3: "#c83e8b"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("REGARD_DATA_ROOT", repo)),
        help="Directory containing data/generations and data/scores, or generations and scores",
    )
    parser.add_argument("--output-dir", type=Path, default=repo / "analysis" / "output")
    return parser.parse_args(argv)


def _locate(root: Path, relative: str) -> Path:
    candidates = [root / "data" / relative, root / relative]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {relative}; checked: {', '.join(map(str, candidates))}")


def load_data(root: Path, repository_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generations = pd.read_json(_locate(root, "generations/generations_raw.jsonl"), lines=True)
    scores = pd.read_json(_locate(root, "scores/judge_scores_raw.jsonl"), lines=True)
    try:
        target_path = _locate(root, "targets/aist_cis_targets_final.csv")
    except FileNotFoundError:
        target_path = repository_root / "data" / "targets" / "aist_cis_targets_final.csv"
    targets = pd.read_csv(target_path)
    return generations, scores, targets


def validate_panel(generations: pd.DataFrame, scores: pd.DataFrame) -> None:
    models = set(generations["model_id"].unique())
    missing = sorted(EXPECTED_MODELS - models)
    extra = sorted(models - EXPECTED_MODELS)
    if missing or extra:
        raise ValueError(f"Expected the fixed 19-model panel; missing={missing}, extra={extra}")
    duplicate_generations = generations.duplicated(["target_id", "model_id", "prompt_id"]).sum()
    if duplicate_generations:
        raise ValueError(f"Found {duplicate_generations} duplicate model-target-prompt generations")
    primary_models = set(scores.loc[scores["judge_model"] == PRIMARY_JUDGE, "model_id"].unique())
    if primary_models != EXPECTED_MODELS:
        raise ValueError(
            f"Primary judge does not cover the fixed panel: missing={sorted(EXPECTED_MODELS - primary_models)}"
        )


def build_primary_table(
    generations: pd.DataFrame,
    scores: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    primary = scores[
        (scores["judge_model"] == PRIMARY_JUDGE)
        & (scores["prompt_id"] == MAIN_PROMPT)
        & (scores["target_coverage"] >= 0.5)
    ].copy()
    generation_columns = [
        "generation_id",
        "target_id",
        "model_id",
        "model_role",
        "prompt_id",
        "response_word_len",
    ]
    merged = primary.merge(
        generations[generation_columns],
        on=["generation_id", "target_id", "model_id", "prompt_id"],
        how="inner",
        validate="one_to_one",
    )
    return merged.merge(
        targets[["target_id", "category", "country"]],
        on="target_id",
        how="left",
        validate="many_to_one",
    )


def model_profiles(primary: pd.DataFrame) -> pd.DataFrame:
    profiles = (
        primary.groupby(["model_id", "model_role"], as_index=False)
        .agg(
            valence=("lm_valence", "mean"),
            arousal=("lm_arousal", "mean"),
            dominance=("lm_dominance", "mean"),
            generic_rate=("generic_answer_flag", "mean"),
            refusal_rate=("refusal_flag", "mean"),
            mismatch_rate=("hallucination_or_mismatch_flag", "mean"),
            mean_words=("response_word_len", "mean"),
            n_scored=("generation_id", "size"),
        )
        .sort_values("arousal")
        .reset_index(drop=True)
    )
    profiles.insert(1, "display_name", profiles["model_id"].map(DISPLAY_NAMES))
    return profiles


def assign_clusters(profiles: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = profiles[["valence", "arousal", "dominance", "generic_rate"]].to_numpy(float)
    tree = linkage(features, method="ward", metric="euclidean")
    raw = fcluster(tree, t=3, criterion="maxclust")
    means = pd.DataFrame({"raw": raw, "arousal": profiles["arousal"]}).groupby("raw")["arousal"].mean()
    relabel = {raw_id: index + 1 for index, raw_id in enumerate(means.sort_values().index)}
    output = profiles.copy()
    output["cluster"] = [relabel[value] for value in raw]
    output["cluster_name"] = output["cluster"].map(CLUSTER_NAMES)
    return output, tree


def correlation_table(profiles: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("arousal", "generic_rate"),
        ("arousal", "dominance"),
        ("arousal", "valence"),
        ("generic_rate", "dominance"),
        ("generic_rate", "valence"),
    ]
    records = []
    rng = np.random.default_rng(SEED)
    for left, right in pairs:
        coefficient, p_value = stats.pearsonr(profiles[left], profiles[right])
        bootstrap = []
        values = profiles[[left, right]].to_numpy(float)
        for _ in range(10_000):
            sample = values[rng.integers(0, len(values), len(values))]
            if np.std(sample[:, 0]) and np.std(sample[:, 1]):
                bootstrap.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        records.append(
            {
                "variable_x": left,
                "variable_y": right,
                "pearson_r": coefficient,
                "p_value": p_value,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "n_models": len(profiles),
            }
        )
    return pd.DataFrame(records)


def grouped_vad(primary: pd.DataFrame, profiles: pd.DataFrame, field: str) -> pd.DataFrame:
    cluster_map = profiles.set_index("model_id")["cluster"]
    table = primary.assign(cluster=primary["model_id"].map(cluster_map))
    grouped = table.groupby([field, "cluster"], as_index=False)[SCORE_COLUMNS].mean()
    return grouped.rename(columns={f"lm_{axis}": axis for axis in AXES})


def judge_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    # The paper reports agreement over all 12,000 doubly scored generations;
    # coverage filtering is reserved for the primary 19-model profile.
    qwen = scores[scores["judge_model"] == PRIMARY_JUDGE].set_index("generation_id")
    gpt = scores[scores["judge_model"] == "gpt4o_mini"].set_index("generation_id")
    paired = qwen.join(gpt, how="inner", lsuffix="_qwen", rsuffix="_gpt")
    records = []
    for axis in AXES:
        left, right = paired[f"lm_{axis}_qwen"], paired[f"lm_{axis}_gpt"]
        coefficient, p_value = stats.pearsonr(left, right)
        records.append(
            {
                "axis": axis,
                "pearson_r": coefficient,
                "p_value": p_value,
                "mae": (left - right).abs().mean(),
                "n_pairs": len(paired),
            }
        )
    return pd.DataFrame(records)


def non_generic_cluster_gap(primary: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    cluster_map = profiles.set_index("model_id")["cluster"]
    table = primary[~primary["generic_answer_flag"].astype(bool)].copy()
    table["cluster"] = table["model_id"].map(cluster_map)
    means = table.groupby("cluster")["lm_arousal"].mean()
    return {
        "comparison": "C1_minus_C3",
        "mean_arousal_gap": means[1] - means[3],
        "n_targets": int(primary["target_id"].nunique()),
        "n_non_generic_responses_c1": int((table["cluster"] == 1).sum()),
        "n_non_generic_responses_c3": int((table["cluster"] == 3).sum()),
    }


def save_figures(profiles: pd.DataFrame, tree: np.ndarray, category: pd.DataFrame, output: Path) -> None:
    ordered = profiles.sort_values("arousal")
    colors = ordered["cluster"].map(CLUSTER_COLORS)

    figure, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    for index, axis in enumerate(AXES):
        axes[index].barh(ordered["display_name"], ordered[axis], color=colors)
        axes[index].set_title(axis.capitalize())
        axes[index].set_xlim(0, 1)
        axes[index].set_xlabel("Mean score")
        if index:
            axes[index].tick_params(axis="y", labelleft=False)
    figure.tight_layout()
    figure.savefig(output / "fig_permodel_vad.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5.5))
    dendrogram(tree, labels=profiles["display_name"].tolist(), leaf_rotation=45, leaf_font_size=8, ax=axis)
    axis.set_ylabel("Ward distance")
    figure.tight_layout()
    figure.savefig(output / "fig_cluster_dendrogram_19.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for cluster, group in profiles.groupby("cluster"):
        axis.scatter(
            group["valence"],
            group["arousal"],
            s=100 + 900 * group["generic_rate"],
            color=CLUSTER_COLORS[cluster],
            label=CLUSTER_NAMES[cluster],
            alpha=0.8,
        )
        for row in group.itertuples():
            axis.annotate(
                row.display_name, (row.valence, row.arousal), fontsize=7, xytext=(3, 3), textcoords="offset points"
            )
    axis.set(xlabel="Mean valence", ylabel="Mean arousal")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "fig_cluster_scatter_19.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(ordered["display_name"], ordered["arousal"], color=colors)
    axes[1].barh(ordered["display_name"], ordered["generic_rate"], color=colors)
    axes[0].set_xlabel("Mean arousal")
    axes[1].set_xlabel("Generic-answer rate")
    axes[1].tick_params(axis="y", labelleft=False)
    figure.tight_layout()
    figure.savefig(output / "fig_cluster_bars_19.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    categories = sorted(category["category"].dropna().unique())
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    width = 0.25
    x = np.arange(len(categories))
    for row_index, axis_name in enumerate(AXES):
        panel = axes[row_index, 0]
        gaps = axes[row_index, 1]
        pivot = category.pivot(index="category", columns="cluster", values=axis_name).reindex(categories)
        for cluster in (1, 2, 3):
            panel.bar(
                x + (cluster - 2) * width, pivot[cluster], width, color=CLUSTER_COLORS[cluster], label=f"C{cluster}"
            )
        gaps.barh(categories, pivot[3] - pivot[1], color="#777777")
        gaps.axvline(0, color="black", linewidth=0.7)
        panel.set_ylabel(axis_name.capitalize())
        panel.set_xticks(x, categories, rotation=35, ha="right")
        gaps.set_xlabel("C3 - C1")
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output / "fig_category_vad.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(data_root: Path, output: Path) -> dict:
    repository_root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    generations, scores, targets = load_data(data_root.resolve(), repository_root)
    validate_panel(generations, scores)
    primary = build_primary_table(generations, scores, targets)
    profiles, tree = assign_clusters(model_profiles(primary))
    correlations = correlation_table(profiles)
    category = grouped_vad(primary, profiles, "category")
    country = grouped_vad(primary, profiles, "country")
    agreement = judge_agreement(scores)
    non_generic = non_generic_cluster_gap(primary, profiles)

    profiles.to_csv(output / "per_model_main.csv", index=False)
    profiles[["model_id", "display_name", "cluster", "cluster_name"]].to_csv(
        output / "cluster_assignments.csv", index=False
    )
    correlations.to_csv(output / "correlations.csv", index=False)
    category.to_csv(output / "category_vad.csv", index=False)
    country.to_csv(output / "country_vad.csv", index=False)
    agreement.to_csv(output / "judge_agreement.csv", index=False)
    save_figures(profiles, tree, category, output)

    summary = {
        "n_targets": int(generations["target_id"].nunique()),
        "n_models": int(generations["model_id"].nunique()),
        "n_generations": len(generations),
        "n_scores": len(scores),
        "primary_judge": PRIMARY_JUDGE,
        "primary_main_rows_after_coverage_filter": len(primary),
        "non_generic_arousal_gap": non_generic,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args.data_root, args.output_dir.resolve())
    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
