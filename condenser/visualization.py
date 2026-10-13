"""3D 点云可视化:颜色 + 形状共同标识 BVID 分组。

配色说明(这点很重要,不是随手挑的)
------------------------------------
用了一组经过校验的 8 槽定性调色板,并**按固定槽位顺序分配、绝不循环复用**。

但需要如实说明一个限制:散点类图形要按「任意两色都要能分开」的标准校验,
8 种颜色是**过不了**的 —— 实测最差一对(红↔橙)在正常色觉下 ΔE 仅 7.1,
远低于 15 的可分辨下限;红绿色盲下最差一对 ΔE 只有 3.2。
结论就是:**光靠颜色区分不了 8 个组**。

所以这里加了第二编码通道 —— **每个 BVID 一个专属 marker 形状**,
再配合可点击的图例(单组隔离)和 hover 显示原文。
形状是比颜色更可靠的身份通道,颜色只是辅助。

依赖
----
* plotly  —— 可选,产出可自由旋转的自包含 HTML(需要 `pip install plotly`)
* matplotlib —— 必装,产出静态 PNG,也能弹交互窗口
两个都没有时 plot_backend="auto" 会自动回退,并明确打印实际用了哪个。
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 调色板:8 个固定槽位。顺序即 CVD 安全性机制的一部分,不要重排、不要循环。
# --------------------------------------------------------------------------

SERIES_COLORS_LIGHT = [
    "#2a78d6",  # 1 蓝
    "#eb6834",  # 2 橙
    "#1baf7a",  # 3 青
    "#eda100",  # 4 黄
    "#e87ba4",  # 5 品红
    "#008300",  # 6 绿
    "#6250d6",  # 7 紫
    "#e34948",  # 8 红
]
SERIES_COLORS_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
# 超出 8 个分组时的兜底色。绝不新造色相,一律并入中性灰。
OTHER_COLOR_LIGHT = "#898781"
OTHER_COLOR_DARK = "#898781"

# 第二编码通道:形状。plotly 的 scatter3d **只认这 8 个 symbol**,
# 正好一组一个,和 matplotlib 的 marker 一一对应。
PLOTLY_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "circle-open",
    "square-open",
    "diamond-open",
]
# 对应关系:3 个实心形状、2 个线形、3 个同款空心(空心由 Style.open_marker 标记)
MPL_MARKERS = ["o", "s", "D", "+", "x", "o", "s", "D"]
# 线形 marker 没有"面",matplotlib 会忽略传给它们的 edgecolors
MPL_LINE_MARKERS = {"+", "x", "|", "_", "1", "2", "3", "4"}

# 被 --highlight 压暗的点:0.1 太狠,浅色底上基本看不见;0.22 还能看出轮廓。
DIM_OPACITY = 0.22

# 界面配色
CHROME = {
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
}


# --------------------------------------------------------------------------
# 分组 -> (颜色, 形状) 的映射
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """一个分组的视觉身份。

    形状是主通道,颜色是辅助 —— 8 色在散点里分辨不可靠,形状可靠。
    open_marker 让 matplotlib 把后三个槽位画成空心(和 plotly 的
    circle-open / square-open / diamond-open 对应)。
    """

    color: str
    plotly_symbol: str
    mpl_marker: str
    open_marker: bool = False


def assign_styles(bvids: Sequence[str], dark: bool = True) -> dict[str, Style]:
    """给每个 BVID 定一份稳定的视觉身份。

    按 BVID **首次出现顺序**分配槽位(不是字母序),这样同一份数据反复跑,
    颜色和形状都不会变 —— 换个筛选条件重画时,活下来的组也不会被重新上色。

    超过 8 个分组时,多出来的组一律用中性灰 + 循环复用形状(形状本身还能区分),
    并给出警告 —— 不新造色相。
    """
    palette = SERIES_COLORS_DARK if dark else SERIES_COLORS_LIGHT
    other_color = OTHER_COLOR_DARK if dark else OTHER_COLOR_LIGHT

    ordered = list(dict.fromkeys(bvids))  # 去重且保序
    if len(ordered) > len(palette):
        logger.warning(
            "有 %d 个分组,超过调色板的 %d 个槽位;第 %d 组起改用中性灰"
            "(不循环复用色相)。建议改用分组隔离或分面查看。",
            len(ordered), len(palette), len(palette) + 1,
        )

    styles: dict[str, Style] = {}
    for slot, bvid in enumerate(ordered):
        idx = slot % len(PLOTLY_SYMBOLS)
        styles[bvid] = Style(
            color=palette[slot] if slot < len(palette) else other_color,
            plotly_symbol=PLOTLY_SYMBOLS[idx],
            mpl_marker=MPL_MARKERS[idx],
            open_marker=slot % len(PLOTLY_SYMBOLS) >= 5,
        )
    return styles


def legend_labels(
    bvids: Sequence[str], labels: Mapping[str, str] | None = None
) -> dict[str, str]:
    """图例文字。默认就是 BVID 原文,也可以用 labels 换成可读名。"""
    ordered = list(dict.fromkeys(bvids))
    counts = {b: sum(1 for x in bvids if x == b) for b in ordered}
    out = {}
    for b in ordered:
        name = labels.get(b, b) if labels else b
        out[b] = f"{name} ({counts[b]})"
    return out


# --------------------------------------------------------------------------
# 重绘辅助
# --------------------------------------------------------------------------


def _wrap(text: str, width: int = 30) -> str:
    """把长中文按宽度折行,给 hover 用。textwrap 对中文够用。"""
    return "<br>".join(textwrap.wrap(text, width=width)) or ""


def _axis_title(name: str, ratio: Sequence[float] | None, i: int) -> str:
    """轴标题带上主成分解释方差占比 —— 少了它,读者会误判点云的形状。"""
    if ratio is not None and i < len(ratio):
        return f"{name} ({ratio[i] * 100:.1f}%)"
    return name


def resolve_backend(backend: str = "auto") -> str:
    """把 "auto" 解析成实际可用的后端名。

    公开出来是给调用方用的:输出文件后缀必须按**实际**后端决定,
    否则会出现「以为在写 .html、其实 matplotlib 往 .html 名字里存了张 PNG」。
    """
    if backend != "auto":
        return backend
    try:
        import plotly  # noqa: F401

        return "plotly"
    except ImportError:
        pass
    try:
        import matplotlib  # noqa: F401

        return "matplotlib"
    except ImportError as exc:  # pragma: no cover - 两个都没有才会走到
        raise RuntimeError(
            "plotly 和 matplotlib 都不可用,装任意一个即可:"
            "pip install plotly  或  pip install matplotlib"
        ) from exc


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def plot_pointcloud_3d(
    coords: np.ndarray,
    bvids: Sequence[str],
    texts: Sequence[str] | None = None,
    *,
    backend: str = "auto",
    out_path: str | Path | None = None,
    title: str = "嵌入向量 PCA 3D 点云",
    labels: Mapping[str, str] | None = None,
    dark: bool = True,
    point_size: float = 5.0,
    point_opacity: float = 0.85,
    highlight: str | None = None,
    explained: Sequence[float] | None = None,
    aspect: str = "data",
    show: bool = False,
) -> Path | None:
    """把 (n, 3) 的坐标画成按 BVID 着色的 3D 点云。

    参数
    ----
    coords : (n, 3) 的 PCA 坐标。
    bvids : 长度 n 的分组标签,决定颜色和形状。
    texts : 长度 n 的原文,hover 时显示。
    backend : "auto" | "plotly" | "matplotlib"。
    out_path : 输出路径;plotly 出 .html,matplotlib 出 .png。
        matplotlib 且 out_path=None 时只弹窗不落盘。
    highlight : 只高亮这个 BVID,其余压暗 —— 因为 8 色难以互相分辨,
        这是实际读图时的主要手段。
    explained : 各主成分解释方差占比,写进轴标题。
    aspect : "data"(按真实比例,诚实)或 "cube"(各轴等长,结构更明显但会失真)。

    返回实际写出的文件路径;只在内存里画时返回 None。
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"coords 需要 (n, 3),收到 shape={coords.shape}")
    if len(bvids) != coords.shape[0]:
        raise ValueError(f"bvids 长度 {len(bvids)} 与点数 {coords.shape[0]} 不一致")
    if texts is not None and len(texts) != coords.shape[0]:
        raise ValueError(f"texts 长度 {len(texts)} 与点数 {coords.shape[0]} 不一致")

    if highlight is not None and highlight not in set(bvids):
        raise ValueError(f"highlight={highlight!r} 不在数据的分组里")

    resolved = resolve_backend(backend)
    logger.info("可视化后端:%s", resolved)

    styles = assign_styles(bvids, dark=dark)
    names = legend_labels(bvids, labels)
    theme = CHROME["dark" if dark else "light"]
    axis_titles = [_axis_title(f"PC{i + 1}", explained, i) for i in range(3)]

    if resolved == "plotly":
        return _plot_plotly(
            coords, bvids, texts, styles, names, theme, axis_titles,
            out_path=out_path, title=title, point_size=point_size,
            point_opacity=point_opacity, highlight=highlight,
            aspect=aspect, show=show,
        )
    if resolved == "matplotlib":
        return _plot_matplotlib(
            coords, bvids, texts, styles, names, theme, axis_titles,
            out_path=out_path, title=title, point_size=point_size,
            point_opacity=point_opacity, highlight=highlight,
            aspect=aspect, show=show,
        )
    raise ValueError(f"未知的后端 {backend!r},可选:auto / plotly / matplotlib")


# --------------------------------------------------------------------------
# 后端实现:plotly(交互 HTML)
# --------------------------------------------------------------------------


def _plot_plotly(
    coords, bvids, texts, styles, names, theme, axis_titles, *,
    out_path, title, point_size, point_opacity, highlight, aspect, show,
) -> Path | None:
    import plotly.graph_objects as go

    fig = go.Figure()
    bvids = list(bvids)
    ordered = list(styles.keys())

    for bvid in ordered:
        mask = np.array([b == bvid for b in bvids])
        if not mask.any():
            continue
        style = styles[bvid]

        hover = None
        if texts is not None:
            hover = [
                f"<b>{bvid}</b><br>{_wrap(t)}"
                for t, keep in zip(texts, mask)
                if keep
            ]

        dimmed = highlight is not None and bvid != highlight
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=names[bvid],
                legendgroup=bvid,
                hovertext=hover,
                hovertemplate=(
                    "%{hovertext}<extra></extra>" if hover
                    else f"<b>{bvid}</b><extra></extra>"
                ),
                marker=dict(
                    size=point_size * (1.2 if dimmed else 2.4),
                    color=style.color,
                    symbol=style.plotly_symbol,
                    opacity=DIM_OPACITY if dimmed else point_opacity,
                    line=dict(width=0.5, color=theme["surface"]),
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["ink"], size=16)),
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(color=theme["ink_secondary"], size=12),
        legend=dict(
            title=dict(text="BVID 分组", font=dict(color=theme["ink_secondary"])),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=theme["grid"],
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            bgcolor=theme["surface"],
            aspectmode=aspect,
            xaxis=_plotly_axis(axis_titles[0], theme),
            yaxis=_plotly_axis(axis_titles[1], theme),
            zaxis=_plotly_axis(axis_titles[2], theme),
        ),
    )

    if out_path is None:
        if show:
            fig.show()
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs=True:把 plotly.js 内联进去,产出单文件、离线可用,
    # 发给别人不用装环境也不用联网。
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    logger.info("已写出 %s", out_path)
    if show:
        fig.show()
    return out_path


# matplotlib 自带的 DejaVu Sans 没有中文字形,标题和图例会渲染成方块。
# 按优先级找一个装了的中文字体(Windows 上一般命中 Microsoft YaHei)。
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "WenQuanYi Zen Hei",
    "Microsoft JhengHei",
    "Arial Unicode MS",
]


def _configure_cjk_font() -> str | None:
    """把 matplotlib 的默认字体换成能画中文的。找不到就返回 None 并给出警告。

    plotly 那边不需要这个 —— HTML 用的是浏览器字体,中文天然正常。
    """
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            # DejaVu Sans 留在后面兜底:中文字体缺的符号(如某些数学记号)由它补
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            # 中文字体的负号字形常有问题,用 ASCII 连字符代替
            rcParams["axes.unicode_minus"] = False
            logger.debug("matplotlib 中文字体:%s", name)
            return name

    logger.warning(
        "没找到可用的中文字体,matplotlib 图上的中文会显示成方块。"
        "装一个即可,例如 Noto Sans CJK SC / SimHei。"
    )
    return None


def _plotly_axis(title: str, theme: dict) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=theme["ink_secondary"], size=12)),
        backgroundcolor=theme["surface"],
        showbackground=False,
        gridcolor=theme["grid"],
        zeroline=False,
        showline=False,
        tickfont=dict(color=theme["muted"], size=10),
    )


# --------------------------------------------------------------------------
# 后端实现:matplotlib(静态 PNG / 本地窗口)
# --------------------------------------------------------------------------


def _plot_matplotlib(
    coords, bvids, texts, styles, names, theme, axis_titles, *,
    out_path, title, point_size, point_opacity, highlight, aspect, show,
) -> Path | None:
    import matplotlib

    if out_path is not None and not show:
        matplotlib.use("Agg")  # 只落盘就不开窗口
    import matplotlib.pyplot as plt

    _configure_cjk_font()

    fig = plt.figure(figsize=(10, 9), facecolor=theme["surface"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(theme["surface"])
    # 3D 轴默认只占画布中间一小块,四周全是空白。手动撑满,点才不至于挤成一小撮。
    # 顶部留 8% 给标题,左右留一点点给轴标签。
    ax.set_position([0.02, 0.0, 0.96, 0.92])

    bvids = list(bvids)
    for bvid in styles:
        mask = np.array([b == bvid for b in bvids])
        if not mask.any():
            continue
        style = styles[bvid]
        dimmed = highlight is not None and bvid != highlight

        # 三类 marker 的上色方式不同,不能一套参数打天下:
        #   空心   -> 面透明,颜色只走描边
        #   线形   -> 没有"面",颜色只能走 facecolor,传 edgecolors 会被忽略
        #   实心   -> 分组色填充 + 一圈背景色描边,让重叠的点仍分得开
        common = dict(
            s=point_size * 18 * (0.45 if dimmed else 1.0),
            marker=style.mpl_marker,
            alpha=DIM_OPACITY if dimmed else point_opacity,
            label=names[bvid],
            depthshade=False,
        )
        if style.open_marker:
            style_kwargs = dict(c="none", edgecolors=style.color, linewidths=1.2)
        elif style.mpl_marker in MPL_LINE_MARKERS:
            style_kwargs = dict(c=style.color, linewidths=1.2)
        else:
            style_kwargs = dict(
                c=style.color, edgecolors=theme["surface"], linewidths=0.8
            )

        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            **common, **style_kwargs,
        )

    ax.set_xlabel(axis_titles[0], color=theme["ink_secondary"], fontsize=10)
    ax.set_ylabel(axis_titles[1], color=theme["ink_secondary"], fontsize=10)
    ax.set_zlabel(axis_titles[2], color=theme["ink_secondary"], fontsize=10)
    ax.set_title(title, color=theme["ink"], fontsize=14, pad=18)

    ax.tick_params(colors=theme["muted"], labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(theme["surface"])
        axis.pane.set_edgecolor(theme["grid"])
        axis.line.set_color(theme["axis"])
        axis._axinfo["grid"].update(color=theme["grid"], linewidth=0.6)

    if aspect == "cube":
        ax.set_box_aspect((1, 1, 1))
    else:
        # 按各轴真实跨度设比例,不放大 PC3
        ranges = np.ptp(coords, axis=0)
        ranges[ranges == 0] = 1.0
        ax.set_box_aspect(ranges)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=9,
        labelcolor=theme["ink_secondary"],
    )
    legend.get_frame().set_facecolor(theme["surface"])
    legend.get_frame().set_edgecolor(theme["grid"])

    # 这里既不能用 tight_layout 也不能用 subplots_adjust:两者都会重新计算
    # 子图位置,把上面 set_position 的布局覆盖掉(实测会把标题直接顶出画布)。

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, facecolor=theme["surface"])
        logger.info("已写出 %s", out_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# --------------------------------------------------------------------------
# 表格视图 —— 图的一个无障碍替代品,也方便直接查数值
# --------------------------------------------------------------------------


def write_coords_table(
    path: str | Path,
    coords: np.ndarray,
    bvids: Sequence[str],
    texts: Sequence[str] | None = None,
) -> Path:
    """把坐标和标签导成 CSV。

    图表只能靠颜色和形状传达信息,对色觉障碍或需要精确数值的场景不够用;
    这张表就是配套的替代路径。
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=float)

    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        header = ["index", "bvid", "pc1", "pc2", "pc3"]
        if texts is not None:
            header.append("text")
        writer.writerow(header)
        for i in range(coords.shape[0]):
            row = [i, bvids[i], *(f"{v:.6f}" for v in coords[i, :3])]
            if texts is not None:
                row.append(texts[i])
            writer.writerow(row)

    logger.info("已写出 %s", path)
    return path
