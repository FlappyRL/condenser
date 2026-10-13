# Condenser

把 B 站用户画像文本用 **Qwen3-Embedding-0.6B** 编码成向量,降维后画成 3D 点云,
用来观察同一视频(`BVID`)下的用户是不是真的聚在一起。

```
data/raw.txt  ──解析──>  17 条文本  ──嵌入──>  (17, 1024) 向量
                                                      │
                                                   PCA 降维
                                                      ▼
                                         按 BVID 着色的 3D 点云
```

## 快速开始

```bash
conda activate transformer
cd condenser

python main.py                 # 全流程(首次会下 ~1.2GB 模型)
python main.py --viz-only      # 复用已存的向量重画,秒出,不加载模型
```

产出在 `outputs/`:

| 文件 | 内容 |
|---|---|
| `embeddings.npz` | 向量、PCA 坐标、各主成分方差占比、BVID、原文 |
| `pca3d.html` | **可自由旋转的交互点云**(plotly) |
| `pca3d_coords.csv` | 坐标表,供精确查数或无障碍替代 |

## 项目结构

按职责分文件,每个模块只干一件事:

| 文件 | 职责 | 依赖 |
|---|---|---|
| `config.py` | 路径、模型 ID、超参数 | 纯标准库 |
| `model_loader.py` | **模型加载 helper**,设备/精度自动决策 | torch, transformers |
| `main.py` | **主文件**:`embed_texts()` 核心 + `main()` 编排 | config, model_loader |
| `dataset.py` | `raw.txt` → `Record` 列表(BVID 分组) | 纯标准库 |
| `projection.py` | PCA(numpy SVD 手写) | numpy |
| `visualization.py` | 3D 点云,按 BVID 着色 | numpy + plotly/matplotlib |

## 移植这个算法

核心是 `main.embed_texts()`:**一堆文本进,`(n, dim)` 的归一化向量出**。

```python
from main import embed_texts

vectors = embed_texts(["任意文本", "另一条"])   # -> (2, 1024) float32,已 L2 归一化
```

归一化后向量点积即余弦相似度,可直接喂给聚类或检索。

**只需复制 3 个文件**:`config.py` + `model_loader.py` + `main.py`。
其余模块(dataset / projection / visualization)与此无关 —— 它们只在 `main()`
内部被导入,所以:

```python
import main          # 0.1 秒,不加载 torch,不加载任何画图/解析代码
main.embed_texts(...)  # 这时才按需拉起 torch 和模型
```

也可以绕开配置直接塞自己的 HuggingFace 对象:

```python
embed_texts(texts, model=my_model, tokenizer=my_tokenizer, device="cuda")
```

## 命令行参数

```
--data PATH          原始数据,默认 data/raw.txt
--out PATH           npz 存放路径,默认 outputs/embeddings.npz
--viz-only           不加载模型,复用 npz 直接出图
--backend {auto,plotly,matplotlib}
--device / --dtype   默认自动探测(xpu > cuda > cpu)
--highlight BVID     只高亮这一组,其余压暗
--label BVID=名称    给图例换可读名字,可重复
--light              浅色主题(默认深色)
--show               额外弹出窗口 / 浏览器预览
```

## 数据格式

`data/raw.txt` 用方括号包住的 BVID 当分组标题,组标题下每个非空行是一条样本:

```
[BV1Y8HS6HE94]
该用户是一个关注经济、科技与社会议题的理性思考型用户……
该用户兴趣广泛,偏理性务实……
[BV15gpo6GEd5]
……
```

同一 BVID 下的行来自同一个视频的评论区,在图上就是同色同形状的一组。

> `raw.txt` 带 UTF-8 BOM,代码用 `utf-8-sig` 读取,否则第一个 BVID 会带上不可见字符。

## 关于配色 —— 一个需要说明的限制

用了经过校验的 8 槽定性调色板,**按固定槽位顺序分配,绝不循环复用色相**。

但必须如实说明:散点类图形要求「任意两色都能分辨」,**8 种颜色做不到**。
实测最差一对(红↔橙)在正常色觉下 ΔE 仅 **7.1**,远低于 15 的可分辨下限;
红绿色盲下最差一对 ΔE 只有 **3.2**。也就是说,**光靠颜色区分不了 8 个组**。

所以额外做了三件事:

1. **形状作为第二编码通道** —— 每个 BVID 一个专属 marker(圆/方/菱/十字/叉/空心圆/空心方/空心菱)。
   形状比颜色可靠得多,读图请优先看形状。
2. **图例可点选隔离** —— 点掉其他组,只看一组。
3. **`--highlight BVxxx`** —— 压暗其余组。

另外 `pca3d_coords.csv` 是配套的表格视图,色觉障碍或需要精确数值时用它。

## 环境

`conda env transformer` — torch 2.14.1+**xpu**(Intel Arc 140T 走 XPU 加速)、
transformers 5.18、numpy、matplotlib、plotly。

⚠️ transformers v5 起 `torch_dtype` 已废弃,代码统一用 `dtype`。

### 首次下载模型会卡住?禁用 Xet

`huggingface_hub` 1.x 默认走 Xet 传输协议,在国内网络下**会在几十 MB 处卡死不动**
(实测停在 64MB 再也不涨)。关掉它走普通 HTTP 即可,速度正常:

```bash
HF_HUB_DISABLE_XET=1 python main.py
```

PowerShell 里是:

```powershell
$env:HF_HUB_DISABLE_XET=1; python main.py
```

只想先手动把模型拉下来,也可以单独执行一次:

```bash
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"
```

PCA 是手写的 numpy SVD,**刻意不依赖 scikit-learn** —— 17×1024 这种量级 SVD 就是精确解,
少一个依赖对移植是净收益。想换 t-SNE / UMAP 只需替换 `projection.py` 里那一个函数。

## 已知取舍

- `pca3d.html` 内联了 plotly.js(~4.8MB),换来**单文件、离线可用、发给别人不用装环境**。
  嫌大可以在 `visualization.py` 里把 `include_plotlyjs=True` 改成 `"cdn"`。
- 跨兴趣用户(既米哈游又 Minecraft)目前不做特殊处理,它们落在哪儿就是哪儿 ——
  这正是点云图要展示的东西。
- `data/README.md` 里有 BVID → 内容(经济学科普 / 娱乐明星 / 三角洲…)的对照表,
  想要中文图例可以用 `--label` 传进去。
