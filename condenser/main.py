"""主文件 / 主函数。

核心是 embed_texts():一堆文本进,(n, dim) 的归一化向量出。
它**只依赖 config 和 model_loader**,不碰数据解析、不碰画图。

    移植说明:把 config.py + model_loader.py + main.py 三个文件复制走就能用,
    其余模块都可以不要。

数据集解析(dataset)、降维(projection)、可视化(visualization)只在 main()
函数体内部 import,所以在文件顶部写 `from main import embed_texts` 不会把
它们连带拉起来 —— 独立性是靠这个延迟导入保证的,不是靠嘴说。

命令行
------
    python main.py                     # 全流程:解析 -> 编码 -> PCA -> 出图
    python main.py --viz-only          # 复用已存的 npz 重画,不加载模型(秒出)
    python main.py --backend matplotlib
    python main.py --highlight BV1qLp76vEA7   # 只高亮一组,其余压暗
    python main.py --label BV1Y8HS6HE94=经济学科普
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_DATA_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_PLOT_STEM,
    OUTPUT_DIR,
    Config,
)

logger = logging.getLogger("condenser")

# 进程内缓存:同一个 (模型, 设备) 只加载一次。
# 这样连续调 embed_texts 不会反复读 1.2GB 权重,但也不引入任何全局状态管理。
# 类型标注写成字符串,免得为了一个注解在模块顶层 import model_loader。
_EMBEDDER_CACHE: dict[tuple[str, str], "model_loader.Embedder"] = {}


# ==========================================================================
# 核心:文本 -> 向量
# ==========================================================================


def embed_texts(
    texts,
    *,
    embedder: "model_loader.Embedder | None" = None,
    model=None,
    tokenizer=None,
    cfg: Config | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    normalize: bool | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """把一批文本编码成向量矩阵。

    参数
    ----
    texts : 可迭代的字符串。
    embedder : 已加载好的 model_loader.Embedder。不传就按 cfg 自动加载(并缓存)。
    model / tokenizer : 想直接塞 HuggingFace 原生对象时用,会自动包成 Embedder。
    normalize : 是否做 L2 归一化(默认跟随 cfg,即 True)。归一化后
        两个向量的点积就是余弦相似度,做聚类/检索请保持开启。

    返回
    ----
    (n, dim) 的 float32 数组,第 i 行对应 texts[i]。

    说明
    ----
    Qwen3-Embedding 用最后一个非 padding token 的隐状态做句向量,这里是文档侧
    编码,**不加** `Instruct: ...` 前缀(那个前缀只对查询侧有意义,对聚类这种
    「一视同仁地编码所有样本」的场景加了反而引入额外变量)。
    """
    # torch / model_loader 在这里才导入:算法本身当然需要它们,但
    # `import main` 不该为此付出加载 2GB torch 的代价(比如 --viz-only 只想画个图)。
    import torch
    import torch.nn.functional as F

    import model_loader

    cfg = cfg or Config()
    texts = list(texts)

    if not texts:
        dim = embedder.hidden_size if embedder is not None else 0
        logger.warning("embed_texts 收到空输入,返回 shape=(0, %d) 的空矩阵", dim or 0)
        return np.zeros((0, int(dim or 0)), dtype=np.float32)

    embedder = _resolve_embedder(embedder, model, tokenizer, cfg, device)

    batch_size = int(batch_size or cfg.batch_size)
    max_length = int(max_length or cfg.max_length)
    normalize = cfg.normalize if normalize is None else normalize

    batches = range(0, len(texts), batch_size)
    if show_progress:
        batches = _progress(batches, len(texts), batch_size)

    chunks: list[np.ndarray] = []
    for start in batches:
        batch = texts[start : start + batch_size]
        encoded = embedder.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(embedder.device)

        with torch.inference_mode():
            output = embedder.model(**encoded)

        pooled = model_loader.last_token_pool(
            output.last_hidden_state, encoded["attention_mask"]
        )
        # 先升到 float32 再归一化,顺序不能反:fp16 下算模长误差约 2e-4,
        # 返回的向量就不是严格单位长度了。升到 fp32 后误差降到 1e-7 量级,
        # 下游可以放心把点积当余弦相似度用。
        pooled = pooled.to(torch.float32)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        chunks.append(pooled.cpu().numpy())

    vectors = np.concatenate(chunks, axis=0)
    logger.info("编码完成:%d 条 -> shape=%s", len(texts), vectors.shape)
    return vectors


def _resolve_embedder(embedder, model, tokenizer, cfg, device):
    """把各种传参形式统一成一个 Embedder。"""
    import model_loader

    if embedder is not None:
        return embedder

    # 允许直接传 HuggingFace 原生对象,方便把这个函数挪到已有代码里
    if model is not None and tokenizer is not None:
        dev = model_loader.resolve_device(device or cfg.device)
        return model_loader.Embedder(
            model=model,
            tokenizer=tokenizer,
            device=dev,
            dtype=model_loader.resolve_dtype(dev, cfg.dtype),
            model_id=getattr(model.config, "_name_or_path", "custom"),
        )

    dev = model_loader.resolve_device(device or cfg.device)
    # 缓存键带上 dtype:同一个模型用不同精度加载出来是两个不同的东西
    key = (cfg.model_id, dev, str(model_loader.resolve_dtype(dev, cfg.dtype)))
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = model_loader.load_embedder(cfg, device=dev)
        logger.info("模型已就绪:%s", model_loader.describe_embedder(_EMBEDDER_CACHE[key]))
    return _EMBEDDER_CACHE[key]


def _progress(iterable, n_items: int, batch_size: int):
    """tqdm 进度条;没装 tqdm 就原样返回,不影响功能。"""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        total=(n_items + batch_size - 1) // batch_size,
        desc="编码",
        unit="batch",
    )


# ==========================================================================
# 产物读写:把编码结果落盘,重画时就不用再跑模型
# ==========================================================================


def save_artifacts(
    path: str | Path,
    vectors: np.ndarray,
    coords: np.ndarray,
    explained: np.ndarray,
    bvids,
    texts,
) -> Path:
    """存成一个 npz:向量、PCA 坐标、方差占比、BVID、原文。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=vectors.astype(np.float32),
        coords=coords.astype(np.float32),
        explained=np.asarray(explained, dtype=np.float32),
        bvid=np.array(list(bvids), dtype=object),
        text=np.array(list(texts), dtype=object),
    )
    logger.info("已写出 %s", path)
    return path


def load_artifacts(path: str | Path) -> dict:
    """读回 save_artifacts 存的 npz。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。先跑一次 `python main.py` 生成嵌入,再用 --viz-only。"
        )
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


# ==========================================================================
# 编排
# ==========================================================================


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="condenser",
        description="用 Qwen3-Embedding 编码用户画像文本,降维后画成 3D 点云。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE,
                        help="原始数据(raw.txt)")
    parser.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS_FILE,
                        help="嵌入产物 npz 的存放路径")
    parser.add_argument("--viz-only", action="store_true",
                        help="不加载模型,复用已存的 npz 直接出图")
    parser.add_argument("--backend", choices=["auto", "plotly", "matplotlib"],
                        default="auto", help="可视化后端")
    parser.add_argument("--device", default=None, help="xpu / cuda / cpu,默认自动探测")
    parser.add_argument("--dtype", default=None, help="float16 / float32,默认按设备决定")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--highlight", default=None,
                        help="只高亮这个 BVID,其余压暗(8 色难分辨时的主要读图手段)")
    parser.add_argument("--label", action="append", default=[], metavar="BVID=名称",
                        help="给图例换可读名字,可重复,如 --label BV1Y8HS6HE94=经济学科普")
    parser.add_argument("--light", action="store_true", help="用浅色主题(默认深色)")
    parser.add_argument("--show", action="store_true",
                        help="额外弹出窗口 / 浏览器预览")
    parser.add_argument("--no-table", action="store_true", help="不导出坐标 CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def parse_labels(pairs) -> dict[str, str]:
    """把 ['BVID=名字', ...] 解析成 dict。"""
    labels: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--label 需要 BVID=名称 的格式,收到 {item!r}")
        key, _, value = item.partition("=")
        labels[key.strip()] = value.strip()
    return labels


def main(argv=None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    # --- 延迟导入 ---------------------------------------------------------
    # 放在函数体里而不是文件顶部,是为了让 `from main import embed_texts`
    # 保持轻量:不 import 本文件就不会拉起解析和画图那一坨。
    import dataset
    import projection
    import visualization

    cfg = Config(device=args.device, dtype=args.dtype)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.max_length:
        cfg.max_length = args.max_length
    cfg.dark = not args.light

    labels = parse_labels(args.label)

    # --- 1. 拿到向量和坐标 ------------------------------------------------
    if args.viz_only:
        logger.info("--viz-only:直接复用 %s,不加载模型", args.out)
        bundle = load_artifacts(args.out)
        vectors = bundle["vectors"]
        coords = bundle["coords"]
        explained = bundle["explained"]
        bvids = [str(b) for b in bundle["bvid"]]
        texts = [str(t) for t in bundle["text"]]
        logger.info("读回 %d 条样本、%d 个 BVID", len(bvids), len(set(bvids)))
    else:
        records = dataset.load_records(args.data)
        texts = dataset.texts_of(records)
        bvids = dataset.bvids_of(records)

        vectors = embed_texts(texts, cfg=cfg)
        coords, explained = projection.pca(vectors, n_components=cfg.n_components)

        save_artifacts(args.out, vectors, coords, explained, bvids, texts)

    # --- 2. 出图 ----------------------------------------------------------
    # 先问清楚到底用哪个后端,再据此定后缀 —— 不能让两边各自猜一次
    backend = visualization.resolve_backend(args.backend)
    suffix = ".html" if backend == "plotly" else ".png"
    plot_path = OUTPUT_DIR / f"{DEFAULT_PLOT_STEM}{suffix}"

    written = visualization.plot_pointcloud_3d(
        coords,
        bvids,
        texts,
        backend=backend,
        out_path=plot_path,
        labels=labels or None,
        dark=cfg.dark,
        point_size=cfg.point_size,
        point_opacity=cfg.point_opacity,
        highlight=args.highlight,
        explained=explained,
        show=args.show,
    )

    if not args.no_table:
        visualization.write_coords_table(
            OUTPUT_DIR / f"{DEFAULT_PLOT_STEM}_coords.csv", coords, bvids, texts
        )

    # --- 3. 汇总 ----------------------------------------------------------
    _print_summary(vectors, bvids, explained, plot_path if written else None, labels)
    return 0


def _print_summary(vectors, bvids, explained, plot_path, labels) -> None:
    groups = list(dict.fromkeys(bvids))
    print()
    print("=" * 62)
    print(f"  样本数      : {vectors.shape[0]}")
    print(f"  向量维度    : {vectors.shape[1]}")
    print(f"  BVID 分组数 : {len(groups)}")
    print(f"  主成分方差  : {', '.join(f'PC{i + 1} {r * 100:.1f}%' for i, r in enumerate(explained))}"
          f"  (累计 {explained.sum() * 100:.1f}%)")
    print("-" * 62)
    for bvid in groups:
        count = sum(1 for b in bvids if b == bvid)
        name = f"  {labels[bvid]}" if labels and bvid in labels else ""
        print(f"    {bvid}  ×{count}{name}")
    print("-" * 62)
    if plot_path:
        print(f"  点云        : {plot_path}")
    print(f"  坐标 CSV    : {OUTPUT_DIR / f'{DEFAULT_PLOT_STEM}_coords.csv'}")
    print("=" * 62)
    print()
    print("提示:8 种颜色在散点里彼此难以分辨,靠图例点选隔离单组、")
    print("      或 --highlight <BVID> 压暗其余组来看聚类。形状是更可靠的身份线索。")


def _setup_logging(verbose: bool) -> None:
    # Windows 控制台默认不是 UTF-8,中文日志容易变成乱码或直接抛
    # UnicodeEncodeError;这里强制转一手,顺手把不可编码的字符替换掉。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    raise SystemExit(main())
