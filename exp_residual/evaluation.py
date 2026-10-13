"""
评估侧：秩相关指标、逐层分析流程、各类报告与 CSV 的保存。
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    LABELS,
    OUTPUT_DIR,
    POOLING_METHODS,
    SAMPLES,
    TARGET_MATCH,
    layer_output_dir,
)
from modeling import cosine_matrix, exp_pool, last_token_pool, mean_pool
from plotting import (
    plot_all_layers,
    plot_matrix,
    plot_similarity_curves,
    plot_target_comparison,
)


# ============================================================
# 9. Rank-correlation metrics
# ============================================================

# We use NumPy/Pandas implementations so scipy is not required.


def average_ranks(values):
    """
    Convert values to ranks.
    Ties receive their average rank.
    """
    return pd.Series(np.asarray(values)).rank(
        method="average"
    ).to_numpy(dtype=float)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x, y):
    return safe_pearson(
        average_ranks(x),
        average_ranks(y),
    )


def kendall_tau_b(x, y):
    """
    Kendall tau-b, with ties handled explicitly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i, j in combinations(range(len(x)), 2):
        dx = x[i] - x[j]
        dy = y[i] - y[j]

        if dx == 0 and dy == 0:
            continue
        elif dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1

    denominator = np.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )

    if denominator == 0:
        return np.nan

    return float(
        (concordant - discordant) / denominator
    )


def ranking_accuracy(target_values, model_values):
    """
    Among sample-pair comparisons where the human ratings differ,
    calculate how often the model gets the ordering right.

    Correct ordering = 1
    Reversed ordering = 0
    Model tie = 0.5
    """
    correct = 0.0
    total = 0

    for i, j in combinations(range(len(target_values)), 2):
        human_diff = target_values[i] - target_values[j]

        if human_diff == 0:
            continue

        model_diff = model_values[i] - model_values[j]
        total += 1

        if human_diff * model_diff > 0:
            correct += 1.0
        elif model_diff == 0:
            correct += 0.5

    return correct / total if total else np.nan


PAIR_INDICES = list(combinations(range(5), 2))

TARGET_PAIRS = np.array([
    TARGET_MATCH[i, j] for i, j in PAIR_INDICES
], dtype=float)


def evaluate_one_matrix(model_matrix):
    """
    Evaluate one model cosine matrix against the human target.

    The diagonal is excluded. Each unique pair is used once.
    """
    if isinstance(model_matrix, pd.DataFrame):
        model_matrix = model_matrix.loc[LABELS, LABELS].values

    model_matrix = np.asarray(model_matrix, dtype=float)

    if model_matrix.shape != (5, 5):
        raise ValueError(
            f"Model matrix must be 5x5; got {model_matrix.shape}"
        )

    if not np.isfinite(model_matrix).all():
        raise ValueError("Model matrix contains NaN or infinity")

    model_pairs = np.array([
        model_matrix[i, j] for i, j in PAIR_INDICES
    ], dtype=float)

    return {
        "spearman_rho": spearman_rho(TARGET_PAIRS, model_pairs),
        "kendall_tau_b": kendall_tau_b(TARGET_PAIRS, model_pairs),
        "ranking_accuracy": ranking_accuracy(
            TARGET_PAIRS, model_pairs
        ),
        "pearson_r": safe_pearson(TARGET_PAIRS, model_pairs),
    }


# ============================================================
# 11. Analyze every layer and preserve similarity matrices
# ============================================================

def analyze_representation(
    hidden_by_layer,
    representation,
    labels,
    attention_mask,
    output_dir,
    records,
    summaries,
    similarity_matrices,
    evaluation_rows,
):
    """
    Returns and stores every layer's cosine similarity matrix.

    similarity_matrices keys:
        (representation, pooling, layer_id)

    Values:
        5x5 pandas DataFrame
    """
    matrices_by_pooling = {
        method: {} for method in POOLING_METHODS
    }

    for layer_id in sorted(hidden_by_layer.keys()):
        hidden = hidden_by_layer[layer_id]

        for pooling in POOLING_METHODS:
            if pooling == "mean":
                vectors = mean_pool(hidden, attention_mask)
            elif pooling == "last":
                vectors = last_token_pool(hidden, attention_mask)
            elif pooling == "exp":
                vectors = exp_pool(hidden, attention_mask)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            matrix = cosine_matrix(vectors, labels)

            key = (representation, pooling, int(layer_id))
            similarity_matrices[key] = matrix
            matrices_by_pooling[pooling][layer_id] = matrix

            # 逐层小图统一放进 ./outputs/{representation}_{pooling}/
            layer_dir = layer_output_dir(representation, pooling)
            prefix = f"layer_{layer_id:02d}"

            # Detailed per-layer CSV and heatmap
            matrix.to_csv(layer_dir / f"{prefix}_cosine.csv")

            plot_matrix(
                matrix,
                f"{representation} | layer {layer_id} | {pooling}",
                layer_dir / f"{prefix}_heatmap.png",
            )

            # Human target evaluation
            metrics = evaluate_one_matrix(matrix)

            evaluation_rows.append({
                "representation": representation,
                "pooling": pooling,
                "layer": int(layer_id),
                **metrics,
            })

            plot_target_comparison(
                matrix,
                representation,
                pooling,
                int(layer_id),
                layer_dir / f"{prefix}_target_comparison.png",
            )

            # Record each unique sample pair
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    left = labels[i]
                    right = labels[j]

                    records.append({
                        "representation": representation,
                        "layer": int(layer_id),
                        "pooling": pooling,
                        "left": left,
                        "right": right,
                        "cosine": float(matrix.loc[left, right]),
                        "human_target": float(TARGET_MATCH[i, j]),
                    })

            # Preserve the original A-E comparison summaries
            if all(x in labels for x in ["A", "C", "D", "E"]):
                ae = float(matrix.loc["A", "E"])
                ac = float(matrix.loc["A", "C"])
                ad = float(matrix.loc["A", "D"])
                ed = float(matrix.loc["E", "D"])

                summaries.append({
                    "representation": representation,
                    "layer": int(layer_id),
                    "pooling": pooling,
                    "cosine_AE": ae,
                    "cosine_AC": ac,
                    "cosine_AD": ad,
                    "cosine_ED": ed,
                    "AE_minus_AC": ae - ac,
                    "AE_minus_AD": ae - ad,
                    "AC_minus_AD": ac - ad,
                })

    # 大图（全层总览、相似度曲线）放在 ./outputs 顶层。
    for pooling in POOLING_METHODS:
        plot_all_layers(
            matrices_by_pooling[pooling],
            output_dir / f"{representation}_all_layers_{pooling}.png",
            f"{representation}: all layers ({pooling} pooling)",
        )

        plot_similarity_curves(
            records,
            output_dir / (
                f"{representation}_similarity_curves_{pooling}.png"
            ),
            representation,
            pooling,
        )

    return matrices_by_pooling


# ============================================================
# 12. Global evaluation report
# ============================================================

def save_evaluation_reports(evaluation_rows):
    if not evaluation_rows:
        raise RuntimeError("No layer evaluation results were produced.")

    results = pd.DataFrame(evaluation_rows).sort_values(
        ["representation", "pooling", "layer"]
    )

    # Detailed metrics for every layer
    results.to_csv(
        OUTPUT_DIR / "human_match_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Overall comparison curves: all representations and pooling methods
    for metric in [
        "spearman_rho",
        "kendall_tau_b",
        "ranking_accuracy",
        "pearson_r",
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))

        for (representation, pooling), group in results.groupby(
            ["representation", "pooling"]
        ):
            group = group.sort_values("layer")

            ax.plot(
                group["layer"],
                group[metric],
                marker="o",
                markersize=3,
                label=f"{representation} / {pooling}",
            )

        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric)
        ax.set_title(f"Human target agreement: {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / f"overview_{metric}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Best layer for each representation/pooling combination
    valid = results.dropna(subset=["spearman_rho"])

    if valid.empty:
        print(
            "Warning: all Spearman results are undefined. "
            "Check whether the target matrix has variation."
        )
        best = pd.DataFrame()
    else:
        best_indices = valid.groupby(
            ["representation", "pooling"]
        )["spearman_rho"].idxmax()

        best = valid.loc[best_indices].sort_values(
            ["representation", "pooling"]
        )

        best.to_csv(
            OUTPUT_DIR / "best_matching_layers.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n========== Best matching layers ==========")

    if not best.empty:
        print(best.to_string(index=False))

    print("\n========== All layer evaluations ==========")
    print(results.to_string(index=False))

    print("\nEvaluation files saved:")
    print(f"  {OUTPUT_DIR}/human_match_evaluation.csv")
    print(f"  {OUTPUT_DIR}/best_matching_layers.csv")
    print(f"  {OUTPUT_DIR}/overview_spearman_rho.png")
    print(f"  {OUTPUT_DIR}/overview_kendall_tau_b.png")
    print(f"  {OUTPUT_DIR}/overview_ranking_accuracy.png")
    print(f"  {OUTPUT_DIR}/overview_pearson_r.png")

    return results


# ============================================================
# 12b. Intermediate artifacts
# ============================================================

def save_pairwise_similarities(records):
    """Save all unique pairwise cosine similarities."""
    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "all_pairwise_similarities.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_relationship_reports(summaries):
    """Save original relationship diagnostics and print the top margins."""
    if not summaries:
        return

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUTPUT_DIR / "target_relationships.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for representation in summary_df["representation"].unique():
        for pooling in POOLING_METHODS:
            subset = summary_df[
                (summary_df["representation"] == representation)
                & (summary_df["pooling"] == pooling)
            ]

            if subset.empty:
                continue

            print(
                f"\n{representation} | {pooling}: "
                "largest AE-minus-AC margins"
            )
            print(
                subset.sort_values(
                    "AE_minus_AC",
                    ascending=False,
                ).head(10).to_string(index=False)
            )

            print(
                f"\n{representation} | {pooling}: "
                "largest AE-minus-AD margins"
            )
            print(
                subset.sort_values(
                    "AE_minus_AD",
                    ascending=False,
                ).head(10).to_string(index=False)
            )


def save_sample_inputs():
    """Save sample text and human target for reproducibility."""
    with open(
        OUTPUT_DIR / "samples.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(SAMPLES, file, ensure_ascii=False, indent=2)

    pd.DataFrame(
        TARGET_MATCH,
        index=LABELS,
        columns=LABELS,
    ).to_csv(
        OUTPUT_DIR / "human_target_matrix.csv",
        encoding="utf-8-sig",
    )


def save_all_reports(records, summaries, evaluation_rows):
    """按原有顺序落盘所有结果。"""
    save_pairwise_similarities(records)
    save_relationship_reports(summaries)
    save_sample_inputs()
    return save_evaluation_reports(evaluation_rows)
