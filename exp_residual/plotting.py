"""
所有 matplotlib 绘图函数：单层热力图、全层总览、
相似度曲线、人工目标矩阵对比。
"""

import matplotlib.pyplot as plt
import pandas as pd

from config import LABELS, TARGET_MATCH


# ============================================================
# 8. Plotting functions
# ============================================================

def plot_matrix(matrix, title, output_path, vmin=-1, vmax=1):
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        matrix.values,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Sample")

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(
                j,
                i,
                f"{matrix.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all_layers(matrices, output_path, title):
    if not matrices:
        return

    layer_ids = sorted(matrices.keys())
    columns = 5
    rows = (len(layer_ids) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3.0, rows * 2.8),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for i, layer_id in enumerate(layer_ids):
        ax = axes.flat[i]
        ax.axis("on")

        matrix = matrices[layer_id]

        ax.imshow(
            matrix.values,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        ax.set_title(f"Layer {layer_id}", fontsize=10)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)

        for r in range(len(matrix.index)):
            for c in range(len(matrix.columns)):
                ax.text(
                    c,
                    r,
                    f"{matrix.iloc[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_curves(records, output_path, representation, pooling):
    fig, ax = plt.subplots(figsize=(11, 6))

    pairs = [
        ("A", "E"),
        ("A", "C"),
        ("B", "C"),
        ("B", "E"),
        ("C", "E"),
        ("A", "B")
    ]

    for left, right in pairs:
        rows = [
            r for r in records
            if r["representation"] == representation
            and r["pooling"] == pooling
            and {r["left"], r["right"]} == {left, right}
        ]

        if not rows:
            continue

        rows.sort(key=lambda r: r["layer"])

        ax.plot(
            [r["layer"] for r in rows],
            [r["cosine"] for r in rows],
            marker="o",
            markersize=3,
            label=f"{left}-{right}",
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"{representation} | {pooling} pooling")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_target_comparison(
    model_matrix,
    representation,
    pooling,
    layer,
    output_path,
):
    """
    Save a detailed comparison of the human target and model matrix.
    """
    if isinstance(model_matrix, pd.DataFrame):
        model_matrix = model_matrix.loc[LABELS, LABELS].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    im0 = axes[0].imshow(
        TARGET_MATCH,
        cmap="viridis",
        vmin=1,
        vmax=3,
    )

    im1 = axes[1].imshow(
        model_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    axes[0].set_title("Human target (1-3)")
    axes[1].set_title("Model cosine similarity")

    for ax in axes:
        ax.set_xticks(range(5))
        ax.set_xticklabels(LABELS)
        ax.set_yticks(range(5))
        ax.set_yticklabels(LABELS)

    for i in range(5):
        for j in range(5):
            axes[0].text(
                j, i, f"{TARGET_MATCH[i, j]:.0f}",
                ha="center", va="center", fontsize=9,
            )
            axes[1].text(
                j, i, f"{model_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
            )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{representation} | {pooling} | Layer {layer}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
