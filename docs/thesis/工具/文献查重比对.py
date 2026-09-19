"""把 文献补充/ 的 25 份与 综述文献打包/ 的 152 份做合并查重。
匹配优先级：① DOI 精确匹配；② 题名归一化前缀匹配；③ 无匹配 = 新增。
"""
import re
from pathlib import Path

from pypdf import PdfReader

LIB = Path("E:/school/final paper/论文/综述文献/综述文献打包")
ADD = Path("E:/school/final paper/论文/文献补充")

# N编号 -> (DOI 或等价标识, 年份, 题名)
NEW = {
    "N29": ("10.3390/agriculture15202135", 2025, "The Intelligentization Process of Agricultural Greenhouse: A Review of Control Strategies and Modeling Techniques"),
    "N30": ("10.1371/journal.pone.0344946", 2026, "Enhancing autonomous agriculture control systems in greenhouses for sustainable resource usage using deep learning techniques"),
    "N31": ("arXiv:2506.13278", 2025, "RL-Guided MPC for Autonomous Greenhouse Control"),
    "N32": ("10.1016/j.atech.2025.101327", 2025, "Adaptive robust greenhouse climate control: Combining deep reinforcement learning and economic optimization"),
    "N33": ("10.1016/j.atech.2024.100512", 2024, "Current applications and potential future directions of reinforcement learning-based Digital Twins in agriculture"),
    "N34": ("arXiv:2510.23882", 2025, "Hybrid Modeling, Sim-to-Real Reinforcement Learning, and Large Language Model Driven Control for Digital Twins"),
    "N35": ("hal-05401361", 2025, "Reinforcement Learning-Enhanced MPC for Sustainable Greenhouse Climate Management"),
    "N36": ("10.1016/j.atech.2026.102009", 2026, "Advancing Digital Twin development for Controlled-Environment Agriculture: Comparative evaluation of two greenhouse energy models"),
    "N37": ("10.1016/j.buildenv.2025.114036", 2026, "A grey-box temperature model for digital twins of building-integrated rooftop greenhouses"),
    "N38": ("10.1007/s11831-020-09473-7", 2021, "Unified Framework and Survey for Model Verification, Validation and Uncertainty Quantification"),
    "N41": ("10.1016/j.apenergy.2014.10.061", 2015, "Experimental performance of evaporative cooling pad systems in greenhouses in humid subtropical climates"),
    "N42": ("10.11975/j.issn.1002-6819.202508079", 2026, "装备化正压通风降温系统在连栋温室的应用"),
    "N43": ("10.1016/j.compag.2025.111025", 2025, "Data-driven localization of the TOMGRO model: Cultivar-specific parameter optimization for Shanghai greenhouse tomato production"),
    "N44": ("10.1016/j.compag.2024.109324", 2024, "Uncertainty in greenhouse tomato growth models"),
    "N45": ("10.1016/j.biosystemseng.2026.104439", 2026, "A validated climate-crop model for the prediction of yield of dwarf tomatoes in Controlled Environment Agriculture"),
    "N46": ("10.1016/j.agsy.2022.103388", 2022, "Process-based greenhouse climate models: genealogy, current status, and future directions"),
    "N47": ("10.1016/j.compag.2023.108578", 2024, "Chance-constrained stochastic MPC of greenhouse production systems with parametric uncertainty"),
    "N48": ("10.1016/j.apenergy.2023.121190", 2023, "Data-driven robust model predictive control for greenhouse temperature control and energy utilisation assessment"),
    "N49": ("10.1016/j.compag.2018.02.001", 2018, "Adaptive two time-scale receding horizon optimal control for greenhouse lettuce cultivation"),
    "N50": ("arXiv:2108.13264", 2021, "Deep Reinforcement Learning at the Edge of the Statistical Precipice"),
    "N51": ("PMLR-v119", 2020, "Evaluating the Performance of Reinforcement Learning Algorithms"),
    "N52": ("10.3390/agriculture13010141", 2023, "Energy-Saving Control Algorithm of Venlo Greenhouse Skylight and Wet Curtain Fan Based on Reinforcement Learning with Soft Action Mask"),
    "N53": ("scnync-2021", 2021, "春茬大棚番茄如何控温"),
    "N54": ("scnync-2018", 2018, "设施番茄越夏栽培田间管理技术"),
    "N55": ("cnki-2026-115", 2025, "AI温室环境控制技术的应用及发展趋势"),
}


def doi_of(pdf: Path, pages: int = 2) -> str:
    try:
        r = PdfReader(str(pdf))
        txt = "".join((r.pages[i].extract_text() or "") for i in range(min(pages, len(r.pages))))
    except Exception:
        return ""
    txt = txt.replace("\n", " ")
    m = re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", txt)
    if not m:
        return ""
    # 取出现频次最高者
    from collections import Counter
    d = Counter(x.rstrip(".,;)") for x in m).most_common(1)[0][0]
    return d.lower()


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return s


print("=== 1) 扫描底座 152 篇的 DOI ===")
lib_dois = {}
lib_files = sorted([p for p in LIB.iterdir() if p.suffix.lower() == ".pdf"])
for i, p in enumerate(lib_files, 1):
    d = doi_of(p)
    if d:
        lib_dois.setdefault(d, p.name)
print(f"底座 {len(lib_files)} 篇，成功提取 DOI 的 {len(lib_dois)} 篇")

print()
print("=== 2) 逐条比对新增 25 条 ===")
dup, new = [], []
for n, (doi, year, title) in NEW.items():
    hit = None
    reason = ""
    if doi.lower() in lib_dois:
        hit = lib_dois[doi.lower()]
        reason = "DOI 精确匹配"
    else:
        # 题名前 40 字符的归一化前缀在底座文件名中找
        key = norm(title)[:40]
        for f in lib_files:
            if key and key[:32] in norm(f.name):
                hit = f.name
                reason = "题名近似匹配"
                break
    if hit:
        dup.append((n, reason, hit))
        print(f"⚠️  {n} 重复 | {reason}\n    底座: {hit}")
    else:
        new.append((n, year, title))
        print(f"✅ {n} 新增")

print()
print("=== 3) 汇总 ===")
print(f"重复 {len(dup)} 条 / 新增 {len(new)} 条")
print()
print("重复清单:")
for n, r, h in dup:
    print(f"  {n}  [{r}]  {h}")
print()
print("应新增清单:")
for n, y, t in new:
    print(f"  {n}  ({y})  {t[:80]}")
