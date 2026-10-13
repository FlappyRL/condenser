# CLAUDE.md

## 项目在干什么

用 **Qwen3-0.6B-Base** 分析 5 段样本文本的表征

流程：hook 抓每层 block 净更新 `block_output - block_input`，同时取完整 residual stream 作对照
→ 每层 × 每 pooling（mean / last）算 5×5 cosine 矩阵
→ 与人工匹配等级矩阵 `TARGET_MATCH`（1/2/3）逐对比，算 Spearman ρ / Kendall τ-b / 排序准确率 / Pearson r
→ 出图 + CSV，并挑出每个组合的最佳层。

## 环境与运行

```bash
conda activate transformer        # C:\Users\atomicbool\anaconda3\envs\transformer
python main.py
```

- torch 2.14.1+xpu，**必须**有 Intel XPU：`get_device()` 拿不到 XPU 直接 raise。
- 模型已缓存在 `./models/Qwen3-0.6B-Base`，`load_model` 走 `local_files_only=True`，不会联网（除非缓存缺失）。

## 模块划分（单向依赖，无循环）

| 文件 | 职责 |
|---|---|
| `config.py` | 常量、`SAMPLES`、`TARGET_MATCH` + `validate_target`、`ensure_output_dirs`、`layer_output_dir` |
| `modeling.py` | XPU/模型加载、定位 block、`mean_pool`/`last_token_pool`、`extract_block_updates`、`cosine_matrix` |
| `evaluation.py` | 秩相关指标、`evaluate_one_matrix`、`analyze_representation`（主循环）、报告落盘 |
| `plotting.py` | 4 个纯绘图函数，只依赖 config |
| `main.py` | 只做编排：`main` + `print_intro` / `encode_samples` / `collect_representations` / `print_output_summary` |

`analyze_representation` 靠**就地修改**四个容器收集结果：`records`、`summaries`、`similarity_matrices`（dict）、`evaluation_rows`。加新统计量时在这四个里加，不要改成返回值。

## 输出布局

一切都在 `./outputs/`（**没有** `ROOT_DIR` / `DETAIL_DIR`，不要往项目根目录写文件）：

```
./outputs/
├── {representation}_all_layers_{pooling}.png      # 大图：全层总览
├── {representation}_similarity_curves_{pooling}.png
├── overview_{metric}.png                          # 4 个指标的总对比曲线
├── human_match_evaluation.csv / best_matching_layers.csv
├── all_pairwise_similarities.csv / target_relationships.csv
├── human_target_matrix.csv / samples.json
└── {representation}_{pooling}/                    # 小图：逐层
    └── layer_{NN}_cosine.csv / _heatmap.png / _target_comparison.png
```

`representation` ∈ {`update`, `residual`}，`pooling` ∈ `POOLING_METHODS`。
子目录名 = `{representation}_{pooling}`，由 `config.layer_output_dir()` 创建；目录内文件名不再带前缀。

## 约定与坑

- 改 `TARGET_MATCH` 后会被 `validate_target` 校验：5×5、对称、对角全 3、非对角只能 1/2/3；行列顺序必须与 `LABELS`（A–E）一致。
- `TARGET_PAIRS` / `PAIR_INDICES` 在 `evaluation.py` 里由 `TARGET_MATCH` 派生（上三角同一顺序取对），两者顺序必须保持一致。
- CSV 一律 `encoding="utf-8-sig"`（Excel 友好），json 用 `utf-8`。
- `plot_target_comparison` 的**最后一个参数是 `output_path`**，路径由调用方给；不要再让它自己拼目录。
- 要加新的 representation（如 attention-only），在 `main()` 的 `representations` 列表里加一项即可；新 pooling 加进 `POOLING_METHODS` 并在 `analyze_representation` 的 if/elif 里分支。
- `residual` 含 embedding 层，所以比 `update` 多一层（30 vs 29）。
- 每份 `analyze_representation` 结果都是几十 MB × 层数级的 CPU 张量，别把多个 representation 同时留在内存里。
