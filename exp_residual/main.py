"""
残差流 / block 净更新表征分析入口。

流程：
    1. 编码 5 个样本文本
    2. 抓取每层的 block 净更新，以及完整 residual stream 作对照
    3. 逐层、逐 pooling 计算 cosine 相似度矩阵
    4. 与人工目标矩阵对比并按指标排序，保存图与 CSV

模块划分：
    config.py      常量、样本、人工目标矩阵
    modeling.py    设备、模型加载、pooling、抓取更新、cosine 矩阵
    evaluation.py  秩相关指标、逐层分析、报告落盘
    plotting.py    所有绘图函数
"""

import pandas as pd

from config import (
    LABELS,
    MAX_LENGTH,
    OUTPUT_DIR,
    POOLING_METHODS,
    SAMPLES,
    SAVE_RESIDUAL_STREAM,
    TARGET_MATCH,
    ensure_output_dirs,
)
from modeling import extract_block_updates, get_device, load_model
from evaluation import analyze_representation, save_all_reports


# ============================================================
# 13. Pipeline steps
# ============================================================

def print_intro():
    print("Samples:")
    for label, text in SAMPLES.items():
        print(f"  {label}: {text}")

    print("\nHuman target matrix:")
    print(pd.DataFrame(
        TARGET_MATCH,
        index=LABELS,
        columns=LABELS,
    ))


def encode_samples(tokenizer, device):
    """把样本编码成一个 batch，并打印每个样本的 token 数。"""
    batch = tokenizer(
        list(SAMPLES.values()),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    batch = {key: value.to(device) for key, value in batch.items()}

    print("\nToken counts:")
    for label, length in zip(
        LABELS,
        batch["attention_mask"].sum(dim=1).tolist(),
    ):
        print(f"  {label}: {length}")

    return batch


def collect_representations(model, batch):
    """
    返回 (每层 block 净更新, 每层 residual stream)，
    两者都以 layer id 为键。
    """
    print("\nExtracting block updates and residual stream...")

    updates, hidden_states = extract_block_updates(model, batch)

    print(f"Captured {len(updates)} block updates.")
    print(f"Captured {len(hidden_states)} residual-stream states.")

    residual_by_layer = {
        layer_id: hidden.detach().float().cpu()
        for layer_id, hidden in enumerate(hidden_states)
    }

    return updates, residual_by_layer


def print_output_summary():
    print(f"\nFinished. All results: {OUTPUT_DIR.resolve()}")

    print(f"\n{OUTPUT_DIR}/  （大图 + 汇总表）")
    for representation in ["update", "residual"]:
        if representation == "residual" and not SAVE_RESIDUAL_STREAM:
            continue
        for pooling in POOLING_METHODS:
            print(f"  {representation}_all_layers_{pooling}.png")
            print(f"  {representation}_similarity_curves_{pooling}.png")

    print("  overview_spearman_rho.png")
    print("  overview_kendall_tau_b.png")
    print("  overview_ranking_accuracy.png")
    print("  overview_pearson_r.png")
    print("  human_match_evaluation.csv")
    print("  best_matching_layers.csv")
    print("  all_pairwise_similarities.csv")
    print("  target_relationships.csv")
    print("  human_target_matrix.csv")
    print("  samples.json")

    print(f"\n{OUTPUT_DIR}/<representation>_<pooling>/  （逐层小图）")
    for representation in ["update", "residual"]:
        if representation == "residual" and not SAVE_RESIDUAL_STREAM:
            continue
        for pooling in POOLING_METHODS:
            print(f"  {representation}_{pooling}/")
    print("  layer_NN_cosine.csv")
    print("  layer_NN_heatmap.png")
    print("  layer_NN_target_comparison.png")


def main():
    ensure_output_dirs()
    print_intro()

    device = get_device()
    tokenizer, model = load_model(device)

    batch = encode_samples(tokenizer, device)
    attention_mask = batch["attention_mask"].cpu()

    updates, residual_by_layer = collect_representations(model, batch)

    # All layer-level cosine matrices are retained here.
    similarity_matrices = {}

    all_records = []
    all_summaries = []
    evaluation_rows = []

    # 1. Per-block net updates
    # 2. Full residual stream as a control
    representations = [("update", updates)]
    if SAVE_RESIDUAL_STREAM:
        representations.append(("residual", residual_by_layer))

    for representation, hidden_by_layer in representations:
        analyze_representation(
            hidden_by_layer=hidden_by_layer,
            representation=representation,
            labels=LABELS,
            attention_mask=attention_mask,
            output_dir=OUTPUT_DIR,
            records=all_records,
            summaries=all_summaries,
            similarity_matrices=similarity_matrices,
            evaluation_rows=evaluation_rows,
        )

    del updates, residual_by_layer

    # Evaluate all layers and save overview figures in ./.
    save_all_reports(all_records, all_summaries, evaluation_rows)

    print_output_summary()


if __name__ == "__main__":
    main()
