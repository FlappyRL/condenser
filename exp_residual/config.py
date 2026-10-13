"""
全局配置：模型、路径、样本、人工目标矩阵。

其它模块都从这里读取常量，避免硬编码散落各处。
"""

from pathlib import Path

import numpy as np


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_DIR = Path("./models/Qwen3-0.6B-Base")

# 所有结果都放在这里：
#   顶层  = 大图（全层总览、相似度曲线、指标总览）与各类 CSV
#   子目录 = 逐层小图，按 representation_pooling 分类
OUTPUT_DIR = Path("./outputs/")

MAX_LENGTH = 4096

# pooling 方法：
#   mean = 平均所有有效 token
#   last = 取最后一个有效 token
#   exp  = 指数加权，越靠后的 token 权重越大，权重和为 1
POOLING_METHODS = ["mean", "last", "exp"]

# exp pooling 的集中程度：w_t ∝ exp(alpha * t / (T - 1))
#   0   -> 权重拉平，退化成 mean
#   越大 -> 越集中在句尾，趋于 last
EXP_POOL_ALPHA = 5.0

# 是否分析完整 residual stream
SAVE_RESIDUAL_STREAM = True

# 固定使用五个样本，不再要求手动输入
SAMPLES = {
    "A": "学习强化学习和无人机的自主系统，研究如何让无人机在复杂环境中自主决策和学习",
    "B": "学习历史和哲学，研究人类社会的发展和思想的演变",
    "C": "学习AI的公共政策和社会影响,研究如何制定政策来管理人工智能技术的发展",
    "D": "对生物学比较感兴趣，喜欢去野外收集一些昆虫标本，研究它们的生活习性和生态环境",
    "E": "最近在打一个无人机+RL的竞赛",
}

LABELS = list(SAMPLES.keys())


def ensure_output_dirs():
    """创建输出根目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def layer_output_dir(representation, pooling):
    """
    逐层小图的分类目录，例如 ./outputs/residual_mean。

    每个 (representation, pooling) 组合一个文件夹，
    里面放该组合下每一层的 cosine CSV、热力图和目标对比图。
    """
    path = OUTPUT_DIR / f"{representation}_{pooling}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 2. Human target matrix
# ============================================================

# 人工匹配等级：
# 1 = 弱匹配
# 2 = 中等匹配
# 3 = 强匹配
#
# 行列顺序必须与 LABELS 一致，即 A, B, C, D, E。
#
# 以下是示例目标矩阵，请按你的真实人工判断修改。
# 对角线固定为 3，矩阵必须对称。

TARGET_MATCH = np.array([
    [3, 1, 2, 1, 3],
    [1, 3, 2, 1, 1],
    [2, 2, 3, 1, 2],
    [1, 1, 1, 3, 1],
    [3, 1, 2, 1, 3],
], dtype=float)


def validate_target(target):
    target = np.asarray(target, dtype=float)

    if target.shape != (5, 5):
        raise ValueError(
            f"TARGET_MATCH 必须为 5x5，当前形状为 {target.shape}"
        )

    if not np.isfinite(target).all():
        raise ValueError("TARGET_MATCH 存在 NaN 或无穷值")

    if not np.allclose(target, target.T):
        raise ValueError("TARGET_MATCH 必须是对称矩阵")

    if not np.allclose(np.diag(target), 3):
        raise ValueError("TARGET_MATCH 对角线必须全部为 3")

    off_diagonal = target[~np.eye(5, dtype=bool)]

    if not np.isin(off_diagonal, [1, 2, 3]).all():
        raise ValueError("人工匹配等级只能是 1、2、3")

    return target


TARGET_MATCH = validate_target(TARGET_MATCH)
