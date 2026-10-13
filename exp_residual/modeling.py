"""
模型侧的工具：设备、加载、定位 Transformer block、pooling、
抓取每层净更新、cosine 相似度矩阵。
"""

import pandas as pd
import torch

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

from config import EXP_POOL_ALPHA, MODEL_DIR, MODEL_ID


# ============================================================
# 3. Device and model
# ============================================================

def get_device():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU unavailable. Please check your PyTorch XPU setup."
        )

    print("Using Intel XPU:", torch.xpu.get_device_name(0))
    return torch.device("xpu")


def load_model(device):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR.resolve()),
    )

    model_path = str(MODEL_DIR.resolve())

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    return tokenizer, model


# ============================================================
# 4. Locate Transformer blocks
# ============================================================

def get_transformer_blocks(model):
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError(
            "Cannot locate Transformer blocks. "
            "Inspect model structure and update get_transformer_blocks()."
        )

    print(f"Found {len(layers)} Transformer blocks.")
    return layers


# ============================================================
# 5. Pooling
# ============================================================

def mean_pool(hidden, attention_mask):
    """
    Average token vectors, excluding padding.

    hidden: [batch, seq_len, hidden_dim]
    attention_mask: [batch, seq_len]
    """
    mask = attention_mask.to(
        device=hidden.device,
        dtype=hidden.dtype,
    ).unsqueeze(-1)

    return (
        (hidden * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    )


def last_token_pool(hidden, attention_mask):
    """
    Select the last valid token vector for each sample.
    Assumes right padding.
    """
    indices = (
        attention_mask.sum(dim=1).long().clamp_min(1) - 1
    ).to(hidden.device)

    batch_indices = torch.arange(
        hidden.shape[0],
        device=hidden.device,
    )

    return hidden[batch_indices, indices]


def exp_pool(hidden, attention_mask, alpha=EXP_POOL_ALPHA):
    """
    指数加权平均：越靠后的 token 权重越大，权重和为 1。

        w_t ∝ exp(alpha * t / (T - 1))

    位置先按句长归一化到 [0, 1]，让短句和长句的"后段"权重分配一致，
    否则同一个 alpha 在 15 字和 37 字的句子上集中程度相差很多。

    alpha = 0 时权重拉平，退化成 mean_pool；
    alpha 越大越集中在句尾，趋于 last_token_pool。

    hidden: [batch, seq_len, hidden_dim]
    attention_mask: [batch, seq_len]
    """
    seq_len = hidden.shape[1]

    mask = attention_mask.to(hidden.device)

    positions = torch.arange(
        seq_len,
        device=hidden.device,
        dtype=torch.float32,
    )

    lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)

    frac = positions.unsqueeze(0) / (lengths - 1).clamp_min(1)

    # 右 padding 落在句尾、原始权重最大，必须在 softmax 之前 mask 掉，
    # 否则 padding 会吃掉大部分归一化权重。
    logits = (alpha * frac).masked_fill(
        mask == 0,
        float("-inf"),
    )

    weights = torch.softmax(logits, dim=1)

    return (hidden * weights.unsqueeze(-1)).sum(dim=1)


# ============================================================
# 6. Extract per-block net updates
# ============================================================

def extract_block_updates(model, batch):
    """
    For every Transformer block l:

        update_l = block_output_l - block_input_l

    This is the net change across the entire block, not an
    attention-only or MLP-only output.

    Returns:
        updates:
            dict[layer_idx, CPU float32 tensor]
            each tensor has shape [batch, seq_len, hidden_dim]

        hidden_states:
            complete residual-stream states from the model
    """
    updates = {}
    handles = []

    layers = get_transformer_blocks(model)

    def make_hook(layer_idx):
        def hook(module, module_inputs, module_output):
            x = module_inputs[0]

            if isinstance(module_output, tuple):
                y = module_output[0]
            elif hasattr(module_output, "last_hidden_state"):
                y = module_output.last_hidden_state
            else:
                y = module_output

            delta = y.detach().float() - x.detach().float()
            updates[layer_idx] = delta.cpu()

        return hook

    for i, layer in enumerate(layers):
        handles.append(
            layer.register_forward_hook(make_hook(i))
        )

    try:
        with torch.inference_mode():
            outputs = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if len(updates) != len(layers):
        raise RuntimeError(
            f"Expected {len(layers)} block updates, "
            f"but captured {len(updates)}."
        )

    return updates, outputs.hidden_states


# ============================================================
# 7. Cosine similarity
# ============================================================

def cosine_matrix(vectors, labels):
    vectors = vectors.float()

    normalized = torch.nn.functional.normalize(
        vectors,
        p=2,
        dim=-1,
        eps=1e-12,
    )

    matrix = normalized @ normalized.T
    matrix = matrix.clamp(-1.0, 1.0)

    return pd.DataFrame(
        matrix.detach().cpu().numpy(),
        index=labels,
        columns=labels,
    )
