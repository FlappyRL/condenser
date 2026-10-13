"""降维。

PCA 直接用 numpy 的 SVD 手写,不引入 scikit-learn —— 对 18×1024 这种量级,
SVD 就是精确解,多装一个依赖对「方便移植」是净损失。以后想换 t-SNE / UMAP,
只要保持同样的入参出参,替换这一个函数就行。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pca(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """主成分分析。

    参数
    ----
    X : (n_samples, n_features) 的特征矩阵,通常是嵌入向量。
    n_components : 保留几个主成分。

    返回
    ----
    coords : (n_samples, k) 的投影坐标,可直接当 3D 点云的 xyz。
    explained_variance_ratio : (k,) 每个主成分解释的方差占比,加起来就是
        前 k 个主成分保留了多少信息量。

    实现说明:对中心化后的矩阵做经济型 SVD(Xc = U S Vt),主成分得分就是
    U[:, :k] * S[:k]。样本数远小于维度时(这里是 18 << 1024),
    这样算比先求 1024×1024 的协方差矩阵再特征分解快得多,数值也更稳。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X 必须是二维矩阵,收到 shape={X.shape}")

    n_samples, n_features = X.shape
    if n_samples == 0:
        raise ValueError("X 里一条样本都没有")

    # 可保留的主成分上限:min(样本数, 特征数)
    max_k = min(n_samples, n_features)
    k = min(int(n_components), max_k)
    if k < n_components:
        logger.warning("只保留得下 %d 个主成分(请求 %d),已自动降到 %d",
                       max_k, n_components, k)

    # 中心化:去均值是 PCA 的定义,漏了它就是纯 SVD 而不是 PCA
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean

    U, S, _Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = U[:, :k] * S[:k]

    # --- 符号约定 ---------------------------------------------------------
    # SVD 的奇异向量方向本身是任意的:同一份数据两次运行,某一列可能整体
    # 翻转,图看起来就像镜像过。这里约定每列绝对值最大的那个分量为正,
    # 让同一份数据永远画出同一个朝向,方便对比不同批次的结果。
    for j in range(k):
        column = coords[:, j]
        if column.size and column[np.argmax(np.abs(column))] < 0:
            coords[:, j] = -column

    # 方差 = S^2 / (n-1);用比值表示占比,所以 n-1 约得掉
    variance = S**2
    total = variance.sum()
    ratio = variance[:k] / total if total > 0 else np.zeros(k)

    logger.info("PCA 完成:%s -> %s,前 %d 个主成分解释方差 %s(累计 %.1f%%)",
                X.shape, coords.shape, k,
                np.round(ratio, 4).tolist(), ratio.sum() * 100)

    return coords, ratio


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只要平面的快捷入口,给不需要 3D 的场合用。"""
    return pca(X, n_components=2)
