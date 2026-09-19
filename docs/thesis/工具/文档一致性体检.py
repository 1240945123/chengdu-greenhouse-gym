"""论文文档一致性体检：N编号引用、图引用、字数达标、跨文档冲突。"""
import re
from pathlib import Path

P = Path("E:/school/final paper/论文")
DOCS = {
    "框架": P / "大论文整体框架.md",
    "格式": P / "川农格式硬约束速查.md",
    "架构": P / "文献架构与核心清单.md",
    "检索": P / "文献补充检索_20260915.md",
    "清单": P / "文献补充_获取清单.md",
    "索引": P / "文献库" / "00-索引.md",
    "前置": P / "00-前置部分.md",
}
CH = {f"第{i}章": list(P.glob(f"第{i}章-*.md")) for i in range(1, 7)}

print("=" * 72)
print("① 各文档中出现的 N 编号范围")
for name, p in DOCS.items():
    if not p.exists():
        print(f"  {name}: ❌ 文件缺失 {p.name}")
        continue
    t = p.read_text(encoding="utf-8")
    ns = sorted({int(x) for x in re.findall(r"\bN(\d{1,2})\b", t)})
    print(f"  {name:<4} N编号 {len(ns):>3} 个  范围 {ns[0] if ns else '-'}–{ns[-1] if ns else '-'}")

print()
print("=" * 72)
print("② 图引用 vs 实际文件")
figs = {p.name for p in (P / "figures").glob("*.png")}
for ch, files in CH.items():
    if not files:
        print(f"  {ch}: 无文件")
        continue
    t = files[0].read_text(encoding="utf-8")
    refs = re.findall(r"\(figures/([^)]+)\)", t)
    miss = [r for r in refs if r not in figs]
    print(f"  {ch}: 引用 {len(refs)} 张 → 缺失 {miss if miss else '无 ✅'}")

print()
print("=" * 72)
print("③ 各章字数 vs 目标")
TARGET = {1: 5000, 2: 9500, 3: 5500, 4: 10500, 5: 6500, 6: 3000}
for i in range(1, 7):
    f = CH[f"第{i}章"]
    if not f:
        continue
    t = f[0].read_text(encoding="utf-8")
    zh = len(re.findall(r"[\u4e00-\u9fff]", t))
    est = int(zh * 1.2)
    ok = "✅" if est >= TARGET[i] * 0.95 else "⚠️"
    print(f"  {f[0].name[:24]:<26} 中文{zh:>6}  估{est:>6} / 目标{TARGET[i]:>6}  {ok}  ({est/TARGET[i]*100:.0f}%)")

print()
print("=" * 72)
print("④ 跨文档关键词冲突（同一事实的不同表述）")
checks = [
    ("引用配额", [r"100[–-]110", r"110[–-]125", r"110[–-]120"]),
    ("增量条数", [r"55 条", r"27 条", r"21 条", r"28 条"]),
    ("物理上限措辞", [r"无制冷物理上限", r"既定配置", r"非技术极限"]),
    ("日间RMSE", [r"4\.4\s*°C", r"日间 RMSE"]),
    ("单seed声明", [r"单一随机种子", r"单 seed", r"seed\s*=\s*0"]),
]
allt = {n: (p.read_text(encoding="utf-8") if p.exists() else "") for n, p in DOCS.items()}
for label, pats in checks:
    hits = {}
    for pat in pats:
        who = [n for n, t in allt.items() if re.search(pat, t)]
        if who:
            hits[pat] = who
    print(f"  {label}:")
    for pat, who in hits.items():
        print(f"     /{pat}/ → {', '.join(who)}")

print()
print("=" * 72)
print("⑤ 章节骨架 vs 已写正文（判断进度）")
for i in range(1, 7):
    f = CH[f"第{i}章"]
    if not f:
        continue
    t = f[0].read_text(encoding="utf-8")
    n_notes = len(re.findall(r"【写作要点】|【待写】|【待绘】|【待制】", t))
    n_body = len(re.findall(r"^[^#\n>|].{40,}$", t, re.M))
    print(f"  {f[0].name[:22]:<24} 占位标记 {n_notes:>3}  疑似正文段 {n_body:>4}")
