"""基于川农官方模板生成《学位论文-骨架.docx》：填入本论文信息 + 符号说明表；并另出《正文骨架（仅标题）.docx》。"""
import copy
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

TITLE_ZH = "面向成都玻璃温室的数字孪生建模与强化学习控制基准"
TITLE_EN = ("Digital Twin Modeling and Reinforcement Learning Control Benchmark "
            "for a Glass Greenhouse in Chengdu")

INFO_ZH = {
    "姓名": "谢添臣",
    "学号": "【待填】",
    "指导教师": "【待填】  教授",
    "合作导师": "【待填】  高级工程师",
    "专业学位类别": "电子信息（0854）",
    "领域名称": "计算机技术（085404）",
    "研究方向": "农业人工智能与智能控制",
    "培养单位名称": "四川农业大学【学院全称待核对】",
}
INFO_EN = {
    "name": "Xie Tianchen",
    "student": "【TBD】",
    "supervisor": "【TBD】",
    "co-supervisor": "【TBD】",
    "degree": "Electronic Information (0854)",
    "major": "Computer Technology (085404)",
    "research": "Agricultural Artificial Intelligence and Intelligent Control",
    "institute": "Sichuan Agricultural University",
}

SYMBOLS = [
    ("一、温室热环境与水汽", "", ""),
    ("T_air", "室内空气温度", "°C"),
    ("T_can", "作物冠层温度", "°C"),
    ("T_out", "室外空气温度", "°C"),
    ("T_cov", "覆盖层温度", "°C"),
    ("T_scr", "保温幕温度", "°C"),
    ("RH_air", "室内空气相对湿度", "%"),
    ("VP_air", "室内空气水汽压", "Pa"),
    ("VP_can", "冠层水汽压", "Pa"),
    ("二、能量与物质通量", "", ""),
    ("Q_sun", "太阳辐射得热", "W·m⁻²"),
    ("Q_lamp", "补光得热", "W·m⁻²"),
    ("Q_heat", "加热管道供热", "W·m⁻²"),
    ("Q_vent", "通风显热交换", "W·m⁻²"),
    ("Q_pad", "湿帘风机冷却量", "W·m⁻²"),
    ("Q_tr", "作物蒸腾潜热", "W·m⁻²"),
    ("Q_cov", "覆盖层长波辐射换热", "W·m⁻²"),
    ("U_cov", "覆盖层传热系数", "W·m⁻²·K⁻¹"),
    ("三、作物生长与光合", "", ""),
    ("LAI", "叶面积指数", "m²·m⁻²"),
    ("PAR", "光合有效辐射", "μmol·m⁻²·s⁻¹"),
    ("C_leaf", "叶片碳量", "mg·m⁻²"),
    ("C_stem", "茎碳量", "mg·m⁻²"),
    ("C_fruit", "果实碳量", "mg·m⁻²"),
    ("C_buf", "缓冲池碳量", "mg·m⁻²"),
    ("DVS", "发育阶段（0=定植，1=成熟）", "—"),
    ("c_air", "空气 CO₂ 浓度", "μmol·mol⁻¹"),
    ("四、控制与评价", "", ""),
    ("u_vent", "天窗开度档位", "—"),
    ("u_pad", "湿帘风机档位", "—"),
    ("u_scr", "外遮阳网档位", "—"),
    ("u_th_scr", "顶部保温幕档位", "—"),
    ("a_t", "t 时刻的动作向量", "—"),
    ("s_t", "t 时刻的状态向量", "—"),
    ("r_t", "t 时刻的即时奖励", "—"),
    ("γ", "折扣因子", "—"),
    ("π_θ", "参数为 θ 的策略", "—"),
    ("V(s) / Q(s,a)", "状态价值函数 / 动作价值函数", "—"),
    ("RMSE", "均方根误差", "°C"),
]


def set_cell(cell, text):
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)


def fill_table_by_label(table, mapping, keys):
    """table 为 2 列；按第 0 列标签匹配 keys 后写入 mapping。"""
    filled = 0
    for row in table.rows:
        label = row.cells[0].text.strip()
        low = label.lower()
        for k in keys:
            if k in label or k.lower() in low:
                set_cell(row.cells[1], mapping[k])
                filled += 1
                break
    return filled


def main():
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    target = out_dir / "学位论文-骨架.docx"
    shutil.copyfile(src, target)

    d = Document(str(target))
    t = d.tables
    print(f"表格数 {len(t)}")

    for i in (0, 4):
        if len(t) > i and len(t[i].rows) and len(t[i].columns):
            set_cell(t[i].rows[0].cells[0], f"论文题目：{TITLE_ZH}" if i == 0 else TITLE_EN)

    zh = fill_table_by_label(t[1], INFO_ZH, list(INFO_ZH))
    print("中文信息表填入", zh, "项")

    en_keys = list(INFO_EN)
    en_label_map = {}
    row_labels = [r.cells[0].text.strip() for r in t[5].rows]
    order = ["name", "student", "supervisor", "co-supervisor", "degree", "major",
             "research", "institute"]
    if len(row_labels) == len(order):
        en_label_map = dict(zip(order, row_labels))
    filled_en = 0
    for i, row in enumerate(t[5].rows):
        if i < len(order):
            set_cell(row.cells[1], INFO_EN[order[i]])
            filled_en += 1
    print("英文信息表填入", filled_en, "项")

    sym = t[6]
    tr_list = sym._tbl.tr_lst
    last_tr = tr_list[-1]
    while len(sym.rows) < len(SYMBOLS) + 1:
        new_tr = copy.deepcopy(last_tr)
        last_tr.addnext(new_tr)
        last_tr = new_tr
    for i, (s, mean, unit) in enumerate(SYMBOLS, start=1):
        cells = sym.rows[i].cells
        set_cell(cells[0], s)
        set_cell(cells[1], mean)
        set_cell(cells[2], unit)
    print(f"符号说明表：{len(sym.rows) - 1} 行数据（含 4 组分组标题）")

    for p in d.paragraphs:
        txt = p.text.strip()
        if txt.startswith("文本文本文本"):
            if p.runs:
                p.runs[0].text = "【摘要待写：500–800 字，四段式（目的/方法/结果/结论），不得含图表。详见 00-前置部分.md §2】"
                for r in p.runs[1:]:
                    r.text = ""
        elif txt.startswith("关键词：关键词1"):
            if p.runs:
                p.runs[0].text = "关键词：玻璃温室；数字孪生；模型可信性诊断；强化学习控制；控制基准"
                for r in p.runs[1:]:
                    r.text = ""
        elif txt.startswith("Text text text"):
            if p.runs:
                p.runs[0].text = ("[Abstract to be written: 500-800 words, strictly mirroring the "
                                  "Chinese abstract. See 00-前置部分.md §3.]")
                for r in p.runs[1:]:
                    r.text = ""
        elif txt.startswith("Keywords: keyword1"):
            if p.runs:
                p.runs[0].text = ("Keywords: glass greenhouse; digital twin; model credibility "
                                  "diagnosis; reinforcement learning control; control benchmark")
                for r in p.runs[1:]:
                    r.text = ""
        elif txt.startswith("。") and len(txt) < 5:
            if p.runs:
                p.runs[0].text = "【符号说明正文：见下方三线表】"

    d.save(str(target))
    print("已生成", target)


if __name__ == "__main__":
    main()
