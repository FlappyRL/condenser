"""把 raw.txt 解析成带 BVID 标签的样本列表。

raw.txt 的格式很简单:方括号包起来的 BVID 当分组标题,组标题下的每个非空行
是一条样本人设描述::

    [BV1Y8HS6HE94]
    该用户是一个关注经济、科技与社会议题的理性思考型用户……
    该用户兴趣广泛,偏理性务实……
    [BV15gpo6GEd5]
    ……

同一 BVID 下的行来自同一个视频的评论区,所以在可视化里它们是同一组、同色。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 组标题:整行就是 [BVID]。用 fullmatch,免得把正文里的中括号误判成标题。
_HEADER_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class Record:
    """一条样本。

    index 是全局序号(0 起),用来把嵌入向量、坐标、原文对齐;
    保留 line_no 是为了报错时能直接定位回 raw.txt 的哪一行。
    """

    bvid: str
    text: str
    index: int
    line_no: int


def load_records(path: str | Path) -> list[Record]:
    """读 raw.txt,返回按出现顺序排列的 Record 列表。

    注意用 utf-8-sig 打开:raw.txt 开头有 BOM,按普通 utf-8 读的话
    第一个 BVID 会变成 '\\ufeffBV1Y8HS6HE94',和一个正常 BVID 对不上。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件:{path}")

    records: list[Record] = []
    current_bvid: str | None = None

    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            header = _HEADER_RE.fullmatch(line)
            if header:
                current_bvid = header.group(1).strip()
                continue

            if current_bvid is None:
                raise ValueError(
                    f"{path}:{line_no} 出现了分组标题之前的内容,"
                    f"文件应以 [BVID] 开头。该行:{line[:40]!r}"
                )

            records.append(
                Record(
                    bvid=current_bvid,
                    text=line,
                    index=len(records),
                    line_no=line_no,
                )
            )

    if not records:
        raise ValueError(f"{path} 里没解析出任何样本")

    logger.info("从 %s 解析出 %d 条样本,分属 %d 个 BVID",
                path.name, len(records), len(group_by_bvid(records)))
    return records


def group_by_bvid(records: list[Record]) -> dict[str, list[int]]:
    """{BVID: [全局 index, ...]},按 BVID 首次出现的顺序排列。

    用 dict(3.7+ 保序)+ 首次出现顺序,而不是 sorted(),是为了让
    调色板分配稳定:同一个文件跑多少次,第一个 BVID 永远是 1 号色。
    """
    groups: dict[str, list[int]] = {}
    for record in records:
        groups.setdefault(record.bvid, []).append(record.index)
    return groups


def texts_of(records: list[Record]) -> list[str]:
    """按 index 顺序取出纯文本,直接喂给 embed_texts。"""
    return [r.text for r in sorted(records, key=lambda r: r.index)]


def bvids_of(records: list[Record]) -> list[str]:
    """按 index 顺序取出 BVID,直接喂给可视化着色。"""
    return [r.bvid for r in sorted(records, key=lambda r: r.index)]
