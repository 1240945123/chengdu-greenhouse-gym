# -*- coding: utf-8 -*-
"""把章节 Markdown 导出为符合川农格式的 .docx。

流程：pandoc（套样式模板）→ 后处理（三线表 / 图表题样式 / 表格文字）
用法：
    python 导出学位论文docx.py 第2章-玻璃温室物理数字孪生的构建与可信性诊断.md
    python 导出学位论文docx.py --all          # 导出全部章节
    python 导出学位论文docx.py --all --no-toc # 不生成目录
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

PAPER = Path(__file__).resolve().parents[1]
REF = PAPER / "学位论文-样式模板.docx"
PANDOC = Path("C:/Users/12409/AppData/Local/Microsoft/WinGet/Packages/"
              "JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe/pandoc-3.9/pandoc.exe")

# 图表题识别（中/英），文本形式如：
#   **图 2-1 冠层过热诊断**          或  Figure 2-1 Canopy overheating diagnosis
RE_CAP_CN = re.compile(r"^\s*\*{0,2}\s*(图|表)\s*(\d+)\s*[-.]\s*(\d+)\s*(\S.*?)\s*\*{0,2}\s*$")
RE_CAP_EN = re.compile(r"^\s*\*{0,2}\s*(Figure|Table|Fig\.?)\s*(\d+)\s*[-.]\s*(\d+)\s*(\S.*?)\s*\*{0,2}\s*$",
                       re.IGNORECASE)
# 章节标题：第 2 章 / 第二章
RE_CHAP = re.compile(r"^\s*(?:第\s*(?:\d+|[一二三四五六七八九十]+)\s*章)")
# 表注/数据来源的显式前缀（推荐写法：`注：…` / `数据来源：…`）
RE_NOTE_PREFIX = re.compile(r"^\s*(注\s*[:：]|数据来源\s*[:：]|来源\s*[:：]|表中|表中数据)")


def _set_borders(el, kind: str, sz: int | None, color: str = "000000") -> None:
    """在给定元素下写一条 `w:{kind}` 边框。sz=0/None 表示 none。"""
    tag = "w:" + kind
    for old in el.findall(qn(tag)):
        el.remove(old)
    b = OxmlElement(tag)
    if sz:
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), str(sz))
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), color)
    else:
        b.set(qn("w:val"), "none")
        b.set(qn("w:sz"), "0")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "auto")
    el.append(b)


def to_three_line_table(table) -> None:
    """把表格改为**三线表**：顶线粗、表头下线细、底线粗，其余边框全去。"""
    tbl_pr = table._tbl.tblPr
    for old in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    _set_borders(borders, "top", 12)          # 1.5 pt
    _set_borders(borders, "bottom", 12)       # 1.5 pt
    _set_borders(borders, "left", None)
    _set_borders(borders, "right", None)
    _set_borders(borders, "insideH", None)
    _set_borders(borders, "insideV", None)
    tbl_pr.append(borders)

    # 表头行底部细线（0.75 pt）
    if len(table.rows) >= 1:
        for cell in table.rows[0].cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            for old in tc_pr.findall(qn("w:tcBorders")):
                tc_pr.remove(old)
            cb = OxmlElement("w:tcBorders")
            _set_borders(cb, "bottom", 6)     # 0.75 pt
            tc_pr.append(cb)


def _para_text(p_el) -> str:
    return "".join(t.text or "" for t in p_el.iter(qn("w:t")))


def _has_image(p_el) -> bool:
    return p_el.find(".//" + qn("w:drawing")) is not None


def _set_para_text(p_el, text: str) -> None:
    """清空段落的 runs，写入单 run，并继承原首个带 rPr 的 run 的字体属性。"""
    runs = p_el.findall(qn("w:r"))
    rpr = None
    for r in runs:
        rp = r.find(qn("w:rPr"))
        if rp is not None:
            rpr = deepcopy(rp)
            break
    for r in list(runs):
        p_el.remove(r)
    new_r = OxmlElement("w:r")
    if rpr is not None:
        new_r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    new_r.append(t)
    p_el.append(new_r)


def style_captions(doc) -> tuple[int, int, list[str]]:
    """题注处理，返回 (题注数, 表注数, 待人工确认清单)。

    判定规则（按正文元素顺序，而非"开头是不是'表/图'字"—— 因为**正文叙述句也常以
    "图 2-1 给出了…"开头**，实测会误判，故一律以**相邻性**为准）：
      · 段落的下一个元素是表格        → **表题**（Table Caption）
      · 段落的上一段含图片            → **图题**（Image Caption）
      · 段落的上一元素是表格          → **表注/数据来源**（Table Note）
      · 其余（正文引用图的句子等）    → **保持原样**
    中英文对照按学校要求**中文在上、英文在下**，故拆成两个段落。
    """
    body = doc.element.body
    kids = [c for c in body.iterchildren() if c.tag in (qn("w:p"), qn("w:tbl"))]
    n_cap = n_note = 0
    review: list[str] = []

    for i, el in enumerate(kids):
        if el.tag != qn("w:p"):
            continue
        txt = _para_text(el).strip()
        if not txt:
            continue
        if not (RE_CAP_CN.match(txt) or RE_CAP_EN.match(txt)):
            continue

        nxt = kids[i + 1] if i + 1 < len(kids) else None
        prv = kids[i - 1] if i > 0 else None
        is_table_cap = nxt is not None and nxt.tag == qn("w:tbl")
        is_fig_cap = (not is_table_cap) and \
                     prv is not None and prv.tag == qn("w:p") and _has_image(prv)

        if not (is_table_cap or is_fig_cap):
            # 紧邻表格之后、且**显式带注记前缀或足够短** → 表注；否则视为正文引用句
            after_tbl = prv is not None and prv.tag == qn("w:tbl")
            looks_note = bool(RE_NOTE_PREFIX.match(txt)) or \
                         (after_tbl and len(txt) <= 40 and "。" not in txt)
            if after_tbl and looks_note:
                Paragraph(el, doc).style = doc.styles["Table Note"]
                n_note += 1
            elif after_tbl:
                review.append(txt[:70])
            continue

        style_name = "Table Caption" if is_table_cap else "Image Caption"
        # 拆分中英文（英文起点 = "Table/Figure X-Y"）
        m_split = re.search(r"\b(Table|Figure|Fig\.?)\s*\d+\s*[-.]\s*\d+", txt)
        cn_part, en_part = (txt, "") if not m_split or m_split.start() == 0 else \
                           (txt[:m_split.start()].strip(), txt[m_split.start():].strip())

        _set_para_text(el, cn_part)
        p_cn = Paragraph(el, doc)
        p_cn.style = doc.styles[style_name]
        p_cn.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        n_cap += 1

        if en_part:
            new_el = deepcopy(el)
            _set_para_text(new_el, en_part)
            el.addnext(new_el)
            p_en = Paragraph(new_el, doc)
            p_en.style = doc.styles[style_name]
            p_en.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            n_cap += 1

    return n_cap, n_note, review


def style_tables(doc) -> tuple[int, int]:
    """三线表化 + 单元格文字套 Table Text 样式。"""
    n_tbl = n_cell = 0
    for table in doc.tables:
        # 跳过封面/题名页等官方模板自带表（无表格标题的宽表不动）
        to_three_line_table(table)
        n_tbl += 1
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if p.text.strip():
                        p.style = doc.styles["Table Text"]
                        n_cell += 1
    return n_tbl, n_cell


def export(md: Path, toc: bool) -> Path | None:
    out = md.with_suffix(".docx")
    if not PANDOC.exists():
        print(f"[ERROR] 找不到 pandoc：{PANDOC}", file=sys.stderr)
        return None
    if not REF.exists():
        print(f"[ERROR] 找不到样式模板：{REF}（先跑 生成样式模板.py）", file=sys.stderr)
        return None

    tmp = Path(tempfile.mkdtemp()) / "raw.docx"
    cmd = [str(PANDOC), "-f", "gfm", "-t", "docx",
           f"--reference-doc={REF}", "--resource-path", str(md.parent),
           "-o", str(tmp)]
    if toc:
        cmd += ["--toc", "--toc-depth=3"]
    cmd.append(str(md))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"[pandoc 失败] {md.name}\n{r.stderr[:800]}", file=sys.stderr)
        return None

    doc = Document(str(tmp))
    nc, nn, review = style_captions(doc)
    nt, ncell = style_tables(doc)
    doc.save(str(out))
    shutil.rmtree(tmp.parent, ignore_errors=True)
    print(f"  ✅ {out.name}  题注 {nc} 处（中英分离）| 表注 {nn} 处 | 三线表 {nt} 张（{ncell} 单元格）")
    if review:
        print(f"  ℹ️ {len(review)} 处紧跟表格、但按正文处理（若确为表注，改写成 `注：…` 开头即可）：")
        for t in review:
            print(f"       · {t}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="要导出的 md 文件（可多个）")
    ap.add_argument("--all", action="store_true", help="导出论文目录下全部章节 md")
    ap.add_argument("--no-toc", action="store_true", help="不生成目录")
    a = ap.parse_args()

    targets: list[Path] = []
    if a.all:
        targets = sorted(p for p in PAPER.glob("*.md")
                         if re.match(r"^第\d+章-", p.name) or p.name.startswith("00-前置"))
    targets += [PAPER / f for f in a.files if not Path(f).is_absolute()]

    if not targets:
        print(__doc__)
        return 0

    print(f"套用样式模板：{REF.name}")
    ok = 0
    for md in targets:
        if not md.exists():
            print(f"  ⚠️ 跳过（不存在）：{md.name}")
            continue
        if export(md, toc=not a.no_toc):
            ok += 1
    print(f"\n完成 {ok}/{len(targets)} 个文件")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
