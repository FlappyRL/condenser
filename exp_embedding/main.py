"""
用 Qwen3-Embedding-0.6B 编码 5 段用户自述，看最后 embedding 的相似度结构。

背景（exp_residual 的结论）：
    base model 的 residual / block update 里，稳定是 AC > AE——
    句式主导：A/C 共享「学习X，研究如何Y」的整句模板，E 只共享实词。
    3 种 pooling、29 层里符号一致。

这里换成一个专门训过的 embedding 模型，看它给的是不是
    AE 高、BC 高、AC 低、D 孤立

跑两个变体：
    plain     对称相似度，不加 instruction（官方对对称任务的做法）
    instruct  5 段都加上同一个 instruction，看对结构有没有影响
"""

import pandas as pd

from analysis import (
    cosine_matrix,
    plot_focus_pairs,
    plot_matrix,
    save_matrix_csv,
    structure_metrics,
)
from config import (
    EXPECTED_CLUSTERS,
    LABELS,
    OUTPUT_DIR,
    SAMPLES,
    USER_INSTRUCTION,
    ensure_output_dirs,
)
from modeling import encode, get_device, load_model


def print_intro():
    print("Samples:")
    for label, text in SAMPLES.items():
        print(f"  {label}: {text}")

    print("\n期望分区:", EXPECTED_CLUSTERS)


def print_token_counts(tokenizer):
    print("\nToken counts:")
    for label, text in SAMPLES.items():
        length = len(tokenizer(text)["input_ids"])
        print(f"  {label}: {length}")


def print_matrix(title, matrix):
    print(f"\n========== {title} ==========")
    print(matrix.round(3).to_string())


def print_metrics(rows):
    frame = pd.DataFrame(rows).set_index("variant")

    print("\n========== 结构指标 ==========")
    print(frame.round(3).to_string())

    print(
        "\n判据：split / silhouette_2x2 / isoD 越正越好；"
        "AE_minus_AC 和 BC_minus_AC 都应为正"
    )

    return frame


def main():
    ensure_output_dirs()
    print_intro()

    device = get_device()
    tokenizer, model = load_model(device)

    print_token_counts(tokenizer)

    texts = list(SAMPLES.values())

    matrices_by_variant = {}
    rows = []

    variants = [
        ("plain", None),
        ("instruct", USER_INSTRUCTION),
    ]

    for name, instruction in variants:
        print(f"\n编码中（{name}）...")

        vectors = encode(
            tokenizer, model, texts, device, instruction=instruction,
        )

        matrix = cosine_matrix(vectors, LABELS)
        matrices_by_variant[name] = matrix

        print_matrix(f"{name} | cosine", matrix)

        save_matrix_csv(
            matrix, OUTPUT_DIR / f"embedding_cosine_{name}.csv",
        )
        plot_matrix(
            matrix,
            f"Qwen3-Embedding-0.6B | {name}",
            OUTPUT_DIR / f"embedding_cosine_{name}.png",
        )

        rows.append({
            "variant": name,
            "dim": int(vectors.shape[1]),
            **structure_metrics(matrix),
        })

    print_metrics(rows)

    plot_focus_pairs(
        matrices_by_variant,
        OUTPUT_DIR / "focus_pairs.png",
    )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "structure_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nFinished. All results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
