"""
exp_embedding 配置：模型、样本、期望结构、输出路径。

模型权重一律用 snapshot_download(local_dir=...) 落在 ./models/，
不使用 HF 默认 cache。
"""

from pathlib import Path


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_DIR = Path("./models/Qwen3-Embedding-0.6B")

OUTPUT_DIR = Path("./outputs/")

MAX_LENGTH = 8192

# 与 exp_residual 完全一致的 5 段用户自述
SAMPLES = {
    "A": "学习强化学习和无人机的自主系统，研究如何让无人机在复杂环境中自主决策和学习",
    "B": "学习历史和哲学，研究人类社会的发展和思想的演变",
    "C": "学习AI的公共政策和社会影响,研究如何制定政策来管理人工智能技术的发展",
    "D": "对生物学比较感兴趣，喜欢去野外收集一些兔子，研究它们的生活习性和生态环境",
    "E": "最近在打一个无人机+RL的竞赛",
}

LABELS = list(SAMPLES.keys())

# 人工期望的分区：{A,E} 技术向，{B,C} 人文社科向，{D} 自己一边
EXPECTED_CLUSTERS = [("A", "E"), ("B", "C"), ("D",)]

# 盯的三对，前两对期望高、第三对期望低
FOCUS_PAIRS = [("A", "E"), ("B", "C"), ("A", "C")]

# 非对称用法下的任务描述。对称比较时要么整批都加，要么都不加。
USER_INSTRUCTION = (
    "Given a description of a person's interests and background, "
    "retrieve other people with similar interests"
)


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
