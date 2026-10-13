"""
cosine 矩阵、结构指标、落盘与出图。

结构指标沿用 exp_residual 里定下来的那几个，方便直接对照：

    split  = mean(AE, BC) - mean(AB, AC, BE, CE)    正 = {A,E}/{B,C} 各自抱团
    sil    = {A,E} vs {B,C} 的 2x2 silhouette       正 = 结构对
    isoD   = mean(不含D的6对) - mean(含D的4对)      正 = D 被甩开
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LABELS, OUTPUT_DIR


def cosine_matrix(vectors, labels=LABELS):
    """向量已 L2 归一化，直接内积。"""
    vectors = np.asarray(vectors, dtype=np.float64)

    matrix = np.clip(vectors @ vectors.T, -1.0, 1.0)

    return pd.DataFrame(matrix, index=labels, columns=labels)


def structure_metrics(matrix):
    """把 5x5 结构压成几个可比的标量。"""

    def cos(a, b):
        return float(matrix.loc[a, b])

    def dist(a, b):
        return 1.0 - cos(a, b)

    pairs = {
        f"{a}{b}": cos(a, b)
        for a, b in combinations(LABELS, 2)
    }

    no_d = ["AB", "AC", "AE", "BC", "BE", "CE"]
    with_d = ["AD", "BD", "CD", "DE"]

    # {A,E} vs {B,C} 的 2x2 silhouette
    scores = []
    for member, own in [("A", "E"), ("E", "A"), ("B", "C"), ("C", "B")]:
        inner = dist(member, own)
        outer = np.mean([
            dist(member, other)
            for other in ["A", "B", "C", "E"]
            if other != member and other != own
        ])
        denom = max(inner, outer)
        scores.append((outer - inner) / denom if denom > 0 else 0.0)

    ae, bc, ac = cos("A", "E"), cos("B", "C"), cos("A", "C")

    return {
        "cos_AE": ae,
        "cos_BC": bc,
        "cos_AC": ac,
        "split": float(
            np.mean([ae, bc])
            - np.mean([cos("A", "B"), ac, cos("B", "E"), cos("C", "E")])
        ),
        "silhouette_2x2": float(np.mean(scores)),
        "isoD": float(
            np.mean([pairs[p] for p in no_d])
            - np.mean([pairs[p] for p in with_d])
        ),
        "AE_minus_AC": ae - ac,
        "BC_minus_AC": bc - ac,
    }


def plot_matrix(matrix, title, output_path, vmin=-1, vmax=1):
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix.values, cmap="coolwarm", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(
                j, i, f"{matrix.iloc[i, j]:.3f}",
                ha="center", va="center", fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_matrix_csv(matrix, output_path):
    matrix.to_csv(output_path, encoding="utf-8-sig")


def plot_focus_pairs(matrices_by_variant, output_path):
    """把几个变体下 FOCUS_PAIRS 的余弦并排画出来。"""
    from config import FOCUS_PAIRS

    variants = list(matrices_by_variant)
    labels = [f"{a}-{b}" for a, b in FOCUS_PAIRS]

    x = np.arange(len(labels))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for offset, variant in enumerate(variants):
        matrix = matrices_by_variant[variant]
        values = [float(matrix.loc[a, b]) for a, b in FOCUS_PAIRS]

        bars = ax.bar(
            x + offset * width, values, width, label=variant,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8)

    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Focus pairs")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
