"""全局配置:路径、模型 ID、超参数。

本模块**零第三方依赖**,只用到标准库,因此可以被任何其他模块安全导入,
包括被单独复制到别的项目里。所有可调参数集中在这里,改行为不用翻其他文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATA_FILE = DATA_DIR / "raw.txt"
DEFAULT_EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"
DEFAULT_PLOT_STEM = "pca3d"


# --------------------------------------------------------------------------
# 超参数
# --------------------------------------------------------------------------


@dataclass
class Config:
    """一次运行的全部可调参数。

    device / dtype 留 None 表示自动探测(见 model_loader.resolve_device /
    resolve_dtype):有 Intel Arc 就走 xpu,有 N 卡走 cuda,否则 cpu。
    """

    # --- 模型 ---
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str | None = None
    dtype: str | None = None
    trust_remote_code: bool = False

    # --- 编码 ---
    # 数据是约 100 字的人设短句,512 token 有大量余量。
    # 显式设上限可以避免长文本拖慢批处理。
    batch_size: int = 16
    max_length: int = 512
    normalize: bool = True

    # --- 降维 ---
    n_components: int = 3

    # --- 可视化 ---
    dark: bool = True  # 3D 点云默认深色底,点更跳
    point_size: float = 5.0
    point_opacity: float = 0.85

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1,收到 {self.batch_size}")
        if self.max_length < 1:
            raise ValueError(f"max_length 必须 >= 1,收到 {self.max_length}")
        if self.n_components < 2:
            raise ValueError(f"n_components 至少为 2 才能出点云,收到 {self.n_components}")
