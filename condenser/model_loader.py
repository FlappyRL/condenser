"""模型加载 helper —— 本项目**唯一**与 HuggingFace 打交道的地方。

设计约束(为了让主文件能独立移植):
    * 只依赖 torch / transformers / numpy,不 import 本项目除 config 外的任何模块;
    * 只负责「把模型和 tokenizer 拉起来」,不关心数据从哪来、图怎么画;
    * 设备与精度自动决策,调用方无脑用即可。

想把这个算法搬到别的项目,复制 config.py + model_loader.py + main.py 三个文件就够。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from config import Config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 设备 / 精度
# --------------------------------------------------------------------------


def resolve_device(prefer: str | None = None) -> str:
    """挑一个能用的设备,优先级:显式指定 > xpu > cuda > cpu。

    本机是 Intel Arc,走 xpu;torch.xpu 在非 XPU 构建里可能整个属性都不存在,
    所以要先 hasattr 再问 is_available。
    """
    if prefer:
        return prefer
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str, prefer: str | torch.dtype | None = None) -> torch.dtype:
    """挑精度。

    CPU 上 float16 很多算子没有加速实现,反而更慢甚至报错,所以一律 float32;
    GPU(xpu / cuda)上 half 能省一半显存并且更快。
    """
    if prefer is not None:
        return getattr(torch, prefer) if isinstance(prefer, str) else prefer
    return torch.float32 if device == "cpu" else torch.float16


# --------------------------------------------------------------------------
# 池化 —— Qwen3-Embedding 的模型契约,不能换
# --------------------------------------------------------------------------


def last_token_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """取每条序列**最后一个非 padding token** 的隐状态作为句向量。

    Qwen3-Embedding 系列用的是 last-token pooling(不是 mean pooling),
    照抄模型卡实现。左 padding 时最后一个位置天然就是真实末 token,
    右 padding 时得自己按长度去取 —— 两种都兼容,不依赖 tokenizer 的 padding 方向。
    """
    # 整批都是左 padding 的话(每行末位掩码都为 1),直接切最后一列
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_idx, lengths]


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------


@dataclass
class Embedder:
    """一个已经 `.eval()` 好、搬上设备的模型 + 配套 tokenizer。

    刻意保持成一个「哑」容器:这里不写编码循环,编码算法在 main.embed_texts 里,
    这样加载和算法是两件可以分开搬走的事。
    """

    model: Any
    tokenizer: Any
    device: str
    dtype: torch.dtype
    model_id: str

    @property
    def hidden_size(self) -> int | None:
        return getattr(self.model.config, "hidden_size", None)


def load_embedder(cfg: Config | None = None, device: str | None = None) -> Embedder:
    """按配置加载 Qwen3-Embedding。

    首次运行会从 HuggingFace 下载权重(0.6B 约 1.2GB)到
    ~/.cache/huggingface/hub,之后走本地缓存。
    """
    cfg = cfg or Config()
    device = resolve_device(device if device is not None else cfg.device)
    dtype = resolve_dtype(device, cfg.dtype)

    logger.info("加载 %s (device=%s, dtype=%s)", cfg.model_id, device, dtype)

    # padding_side='left' 是 Qwen3-Embedding 的推荐设置:左 padding 下
    # 「最后一个位置」正好是每条序列的真实末 token,池化逻辑最省事。
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        padding_side="left",
        trust_remote_code=cfg.trust_remote_code,
    )

    # transformers v5 起 `torch_dtype` 已废弃(v5 里传它会 warning 并忽略),
    # 统一用 `dtype`。
    model = AutoModel.from_pretrained(
        cfg.model_id,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.to(device)
    model.eval()

    return Embedder(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_id=cfg.model_id,
    )


def describe_embedder(embedder: Embedder) -> str:
    """给日志用的一句话摘要。"""
    n_params = sum(p.numel() for p in embedder.model.parameters())
    return (
        f"{embedder.model_id} | device={embedder.device} dtype={embedder.dtype} "
        f"| 参数量={n_params / 1e6:.1f}M | hidden={embedder.hidden_size}"
    )
