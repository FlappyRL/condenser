"""
Qwen3-Embedding 的加载与编码。

官方用法要点：
  1. tokenizer 必须 padding_side="left"，池化取最后一个 token
  2. 对称任务（文本 vs 文本）默认不加 instruction
  3. 非对称任务才对 query 侧加 "Instruct: {task}\\nQuery: {text}"
"""

import torch

from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from config import MAX_LENGTH, MODEL_DIR, MODEL_ID


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Using Intel XPU:", torch.xpu.get_device_name(0))
        return torch.device("xpu")

    print("XPU 不可用，回落到 CPU")
    return torch.device("cpu")


def load_model(device):
    """权重下到 ./models/，不用 HF 默认 cache。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR.resolve()),
    )

    model_path = str(MODEL_DIR.resolve())

    # 左 padding，last_token_pool 才能直接取 -1
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    return tokenizer, model


def last_token_pool(last_hidden_states, attention_mask):
    """
    左 padding 下最后一个 token 就是 -1；
    右 padding 才需要按各自长度取。
    """
    left_padding = bool(
        (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    )

    if left_padding:
        return last_hidden_states[:, -1]

    lengths = attention_mask.sum(dim=1) - 1

    indices = torch.arange(
        last_hidden_states.shape[0],
        device=last_hidden_states.device,
    )

    return last_hidden_states[indices, lengths]


def encode(tokenizer, model, texts, device, instruction=None, batch_size=8):
    """
    编码一组文本，返回 L2 归一化后的向量 [n, dim]。

    instruction 不为空时，按官方格式给每段文本加
    "Instruct: {instruction}\\nQuery: {text}"——对称比较时 5 段一视同仁。
    """
    if instruction:
        texts = [
            f"Instruct: {instruction}\nQuery: {text}"
            for text in texts
        ]

    vectors = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]

        batch = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.inference_mode():
            outputs = model(**batch)

        pooled = last_token_pool(
            outputs.last_hidden_state,
            batch["attention_mask"],
        )

        pooled = torch.nn.functional.normalize(
            pooled.float(),
            p=2,
            dim=-1,
            eps=1e-12,
        )

        vectors.append(pooled.cpu())

    return torch.cat(vectors, dim=0)
