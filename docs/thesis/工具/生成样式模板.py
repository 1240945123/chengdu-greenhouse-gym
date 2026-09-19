# -*- coding: utf-8 -*-
"""生成 Pandoc 可用的学位论文样式模板 `学位论文-样式模板.docx`。

为什么需要它：
    学校官方模板（附录 D）的正文段落几乎全部使用 `Normal` 样式 + **直接格式**
    （只有 toc 1/toc 2 是命名样式），而 Pandoc 的 `--reference-doc` 靠**样式名映射**
    继承格式 —— 直接用会失效，导致 5 章写完还要逐段手工套字体字号。

做法：
    以官方 `4 专业型硕士学位论文-终稿格式.docx` 为底（保留页面设置/页眉页脚/表格），
    把规范**固化成命名样式**，之后 Pandoc 导出即自动正确。

规范来源（《四川农业大学研究生学位论文写作指南》2.3.5 / 2.3.10 等）：
    正文      小四(12pt) 宋体 / Times New Roman，两端对齐，首行缩进 2 字符，1.5 倍行距，段前后 0
    一级标题  三号(16pt) 黑体 / Arial，居中，单倍行距，段前 0、段后 18 磅，序号用中文汉字
    二级标题  小三(15pt) 黑体 / Arial，居左，固定 20 磅，段前 12、段后 6
    三级标题  四号(14pt) 黑体 / Arial，居左，固定 20 磅，段前 12、段后 6
    图题/表题 五号(10.5pt) 加粗 宋体 / Times New Roman，居中，无缩进，段前后 0，多倍行距 1.25
    表注      五号 宋体 / Times New Roman，两端对齐，首行缩进 2 字符，段前后 0，多倍行距 1.25
    参考文献  五号 宋体 / Times New Roman，固定 20 磅，段前后 0，悬挂缩进
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Pt

PAPER = Path(__file__).resolve().parents[1]
TPL_DIR = PAPER / "论文格式/附录D 论文格式模板"
SRC = TPL_DIR / "4 专业型硕士学位论文-终稿格式.docx"
DST = PAPER / "学位论文-样式模板.docx"

SONG, HEI, MONO = "宋体", "黑体", "Consolas"
TNR, ARIAL = "Times New Roman", "Arial"

# 字号（中文号 → 磅）
SZ = {"小二": 18, "三号": 16, "小三": 15, "四号": 14, "小四": 12, "五号": 10.5, "小五": 9}

# 样式名 → (中文体, 西文体, 磅, 加粗, 对齐, 行距, 段前, 段后, 首行缩进字符, 悬挂磅)
STYLES: dict[str, dict] = {
    # ---- 正文 ----
    "Body Text": dict(cn=SONG, en=TNR, sz=SZ["小四"], bold=False,
                      align="justify", line=("multiple", 1.5), before=0, after=0, indent_chars=2),
    "First Paragraph": dict(cn=SONG, en=TNR, sz=SZ["小四"], bold=False,
                            align="justify", line=("multiple", 1.5), before=0, after=0, indent_chars=2),
    "Compact": dict(cn=SONG, en=TNR, sz=SZ["小四"], bold=False,
                    align="justify", line=("multiple", 1.5), before=0, after=0, indent_chars=2),
    "Body Text Indent": dict(cn=SONG, en=TNR, sz=SZ["小四"], bold=False,
                             align="justify", line=("multiple", 1.5), before=0, after=0, indent_chars=2),
    # ---- 标题 ----
    "Heading 1": dict(cn=HEI, en=ARIAL, sz=SZ["三号"], bold=False,
                      align="center", line=("single", 1.0), before=0, after=18),
    "Heading 2": dict(cn=HEI, en=ARIAL, sz=SZ["小三"], bold=False,
                      align="left", line=("exact", 20), before=12, after=6),
    "Heading 3": dict(cn=HEI, en=ARIAL, sz=SZ["四号"], bold=False,
                      align="left", line=("exact", 20), before=12, after=6),
    "Heading 4": dict(cn=HEI, en=ARIAL, sz=SZ["小四"], bold=True,
                      align="left", line=("exact", 20), before=12, after=6),
    # ---- 图表题（五号加粗，居中，多倍行距 1.25）----
    "Image Caption": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=True,
                          align="center", line=("multiple", 1.25), before=0, after=0),
    "Table Caption": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=True,
                          align="center", line=("multiple", 1.25), before=0, after=0),
    "Caption": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=True,
                    align="center", line=("multiple", 1.25), before=0, after=0),
    # ---- 表注 / 数据来源 ----
    "Table Note": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=False,
                       align="justify", line=("multiple", 1.25), before=0, after=0, indent_chars=2),
    # ---- 表格内文字（指南未单列，取五号宋体居中：宽表最稳）----
    "Table Text": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=False,
                       align="center", line=("single", 1.0), before=0, after=0),
    # ---- 参考文献（五号，固定 20 磅，悬挂缩进）----
    "Bibliography": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=False,
                         align="justify", line=("exact", 20), before=0, after=0,
                         hanging=Pt(21)),          # ≈ 3 字符 + 序号
    # ---- 其他 ----
    "Block Text": dict(cn=SONG, en=TNR, sz=SZ["五号"], bold=False,
                       align="justify", line=("multiple", 1.25), before=0, after=0, indent_chars=2),
    "Source Code": dict(cn=MONO, en=MONO, sz=SZ["五号"], bold=False,
                        align="left", line=("single", 1.0), before=0, after=0),
    "Abstract": dict(cn=SONG, en=TNR, sz=SZ["小四"], bold=False,
                     align="justify", line=("multiple", 1.5), before=0, after=0, indent_chars=2),
    "Title": dict(cn=HEI, en=ARIAL, sz=SZ["三号"], bold=False,
                  align="center", line=("single", 1.0), before=0, after=18),
}

LINE_RULE = {"single": WD_LINE_SPACING.SINGLE,
             "exact": WD_LINE_SPACING.EXACTLY,
             "multiple": WD_LINE_SPACING.MULTIPLE}
ALIGN = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
         "right": WD_ALIGN_PARAGRAPH.RIGHT, "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}


def set_fonts(style, cn: str, en: str) -> None:
    """同时设置 ascii / hAnsi / eastAsia 字体（python-docx 不直接支持中文字体）。"""
    rpr = style.element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    rf.set(qn("w:ascii"), en)
    rf.set(qn("w:hAnsi"), en)
    rf.set(qn("w:eastAsia"), cn)
    rf.set(qn("w:cs"), en)
    # 关闭"自动调整西文与中文间距"干扰时保持默认；此处显式声明语言
    style.font.name = en


def set_hanging(style, pt: Pt) -> None:
    ppr = style.element.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = ppr.makeelement(qn("w:ind"), {})
        ppr.append(ind)
    ind.set(qn("w:left"), str(int(pt.pt * 20)))
    ind.set(qn("w:hanging"), str(int(pt.pt * 20)))


def set_first_line_chars(style, chars: int) -> None:
    """按"字符"设置首行缩进（Word 的 firstLineChars，比 Pt 更符合中文排版）。"""
    ppr = style.element.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = ppr.makeelement(qn("w:ind"), {})
        ppr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars * 100))
    ind.set(qn("w:firstLine"), "0")
    ind.attrib.pop(qn("w:firstLine"))


def build() -> int:
    if not SRC.exists():
        print(f"[ERROR] 找不到官方模板：{SRC}", file=sys.stderr)
        return 1

    shutil.copy2(SRC, DST)
    doc = Document(str(DST))
    existing = {s.name for s in doc.styles}

    created, updated = [], []
    for name, cfg in STYLES.items():
        if name in existing:
            st = doc.styles[name]
            updated.append(name)
        else:
            st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            st.base_style = doc.styles["Normal"]
            created.append(name)

        set_fonts(st, cfg["cn"], cfg["en"])
        st.font.size = Pt(cfg["sz"])
        st.font.bold = cfg["bold"]

        pf = st.paragraph_format
        pf.alignment = ALIGN[cfg["align"]]
        rule, val = cfg["line"]
        pf.line_spacing_rule = LINE_RULE[rule]
        pf.line_spacing = val if rule != "exact" else Pt(val)
        pf.space_before = Pt(cfg["before"])
        pf.space_after = Pt(cfg["after"])
        pf.first_line_indent = None
        pf.left_indent = None
        if cfg.get("indent_chars"):
            set_first_line_chars(st, cfg["indent_chars"])
        if cfg.get("hanging"):
            set_hanging(st, cfg["hanging"])

    doc.save(str(DST))
    print(f"✅ 已生成 {DST.name}")
    print(f"   新建 {len(created)} 个样式：{', '.join(created)}")
    if updated:
        print(f"   更新 {len(updated)} 个：{', '.join(updated)}")
    print("\n   页面设置（沿用官方模板）：")
    for i, s in enumerate(doc.sections[:3]):
        print(f"     [{i}] {s.page_width.cm:.1f}×{s.page_height.cm:.1f} cm | "
              f"边距 上{s.top_margin.cm:.2f}/下{s.bottom_margin.cm:.2f}/"
              f"左{s.left_margin.cm:.2f}/右{s.right_margin.cm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
