# -*- coding: utf-8 -*-
"""references.bib 维护工具：① 注入/刷新 nnum 字段  ② 与底座 152 篇查重。

背景（2026-09-19 体检报告 P1-3 / P1-4）：
    - P1-3：bib 原无 N 编号 ↔ key 映射，定稿时"替换为正式顺序号"只能手工比对
    - P1-4：N29–N55 有 6/27 与综述底座重合，N1–N28 未查重

用法：
    python 维护参考文献bib.py              # 注入 nnum + 查重
    python 维护参考文献bib.py --check      # 只做 BibTeX 结构校验
    python 维护参考文献bib.py --refresh    # 清掉旧 nnum 后重新注入
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
BIB = PAPER / "references.bib"
BASE_DIR = PAPER / "综述文献" / "综述文献打包"
CACHE = Path("E:/tmp/base152_index.json")

# N 编号 ↔ bib key（按 bib 条目顺序人工核对，含 N21×2、N22×4 多条目）
NMAP = {
    "sutton2018rl": 1, "mnih2015dqn": 2, "mnih2016a3c": 3,
    "schulman2015trpo": 4, "schulman2016gae": 5, "fujimoto2018td3": 6,
    "kuznetsov2020tqc": 7, "bhatt2024crossq": 8, "huang2020masking": 9,
    "ross2011dagger": 10, "raffin2021sb3": 11, "silver2018residual": 12,
    "johannink2019residual": 13, "sb3contrib": 14, "kirk2023generalisation": 15,
    "oberkampf2010vv": 16, "grieves2017dt": 17, "kritzinger2018dt": 18,
    "tao2019five": 19,
    "stanghellini1987": 20,
    "rafiq2019thermal": 21, "thermalscreen_uvalue2022": 21,
    "kotilainen2018light": 22, "nijskens1985radiation": 22,
    "ahemd2016shading": 22, "tanny2013screens": 22,
    "ashrae2023ch25": 23,
    "vanthoor2011thesis": 24, "vanhenten1994thesis": 25, "vanstraten2010": 26,
    "heuvelink1996thesis": 27, "dekoning1994": 28,
    "li2025intelligentization": 29, "hindi2026autonomous": 30, "msaad2025rlguided": 31,
    "mansour2025adaptiverobust": 32, "goldenits2024rldt": 33, "rasheed2025hybrid": 34,
    "pourghavam2025rlmpc": 35, "costantino2026dtreadiness": 36,
    "lopezcarreno2026greybox": 37, "riedmaier2021vv": 38,
    "oberkampf2025vvuq": 39, "oberkampf2003predictive": 40,
    "xu2015evaporative": 41, "sun2026positivepressure": 42, "sun2025tomgro": 43,
    "deoliveira2024uncertainty": 44, "righini2026dwarftomato": 45,
    "katzin2022genealogy": 46, "svensen2024chance": 47, "mahmood2023ddrmpc": 48,
    "xu2018twotimescale": 49, "agarwal2021statistical": 50,
    "jordan2020evaluating": 51, "chen2023softmask": 52,
    "scnync2021tomato": 53, "scnync2018summer": 54, "liu2025aigreenhouse": 55,
}

# 已知与底座重复的条目（2026-09-19 查重结论，正文照引、不重复下载）
KNOWN_DUP = {29, 31, 32, 47, 48, 52}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s.lower())


def entries_of(txt: str) -> list[dict]:
    out = []
    for b in re.split(r"(?=^@)", txt, flags=re.M):
        m = re.match(r"^@\w+\{([^,]+),", b)
        if not m:
            continue

        def fld(name, blk=b):
            mm = re.search(rf"^\s*{name}\s*=\s*\{{(.*?)\}}\s*,?\s*$", blk, re.M | re.S)
            return " ".join(mm.group(1).split()) if mm else ""

        nn = fld("nnum")
        out.append({"key": m.group(1),
                    "n": int(nn[1:]) if re.fullmatch(r"N\d+", nn) else None,
                    "doi": fld("doi").lower(), "title": fld("title")})
    return out


def inject(refresh: bool = False) -> tuple[int, int]:
    txt = BIB.read_text(encoding="utf-8")
    if refresh:
        txt = re.sub(r"\n\s*nnum\s*=\s*\{[^}]*\},?", "", txt)
    blocks = re.split(r"(?=^@)", txt, flags=re.M)
    out, inj, unk = [], 0, 0
    for b in blocks:
        if not b.startswith("@"):
            out.append(b)
            continue
        m = re.match(r"^(@\w+\{[^,]+,)", b)
        if not m:
            out.append(b)
            continue
        if re.search(r"^\s*nnum\s*=", b, re.M):
            out.append(b)
            continue
        key = re.match(r"^@\w+\{([^,]+),", b).group(1)
        n = NMAP.get(key)
        if n is None:
            unk += 1
            out.append(b)
            continue
        # ⚠️ 必须插在 key 行之后：插在末尾字段后易漏逗号（BibTeX 语法错误）
        out.append(f"{m.group(1)}\n  nnum = {{N{n}}},{b[m.end():]}")
        inj += 1
    BIB.write_text("".join(out), encoding="utf-8")
    return inj, unk


def validate(txt: str) -> list[str]:
    bad = []
    for b in re.split(r"(?=^@)", txt, flags=re.M):
        if not b.startswith("@"):
            continue
        m = re.match(r"^@\w+\{([^,]+),", b)
        if not m:
            bad.append(f"条目头无法解析: {b[:40]!r}")
            continue
        body = b[m.end():].rstrip()
        body = body[:body.rfind("}")] if "}" in body else body
        lines = [ln.rstrip() for ln in body.split("\n")
                 if ln.strip() and not ln.strip().startswith("%")]
        for i, ln in enumerate(lines):
            if re.match(r"^\s*[a-zA-Z]+\s*=", ln) and i < len(lines) - 1 and not ln.endswith(","):
                bad.append(f"{m.group(1)}: 字段缺逗号 → {ln[:60]!r}")
    return bad


def base_index() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    from pypdf import PdfReader
    from collections import Counter
    idx = {}
    files = sorted(p for p in BASE_DIR.iterdir() if p.suffix.lower() == ".pdf")
    print(f"扫描底座 {len(files)} 篇 PDF（首次，约 2-3 分钟）…", flush=True)
    for i, p in enumerate(files, 1):
        try:
            r = PdfReader(str(p))
            t = "".join((r.pages[j].extract_text() or "")
                        for j in range(min(2, len(r.pages)))).replace("\n", " ")
        except Exception:
            t = ""
        ds = re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", t)
        idx[p.name] = {"doi": Counter(x.rstrip(".,;)").lower() for x in ds).most_common(1)[0][0]
                       if ds else ""}
        if i % 25 == 0:
            print(f"  … {i}/{len(files)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    return idx


def dedup() -> int:
    idx = base_index()
    names = list(idx)
    dois = {v["doi"]: k for k, v in idx.items() if v["doi"]}
    ents = sorted([e for e in entries_of(BIB.read_text(encoding="utf-8"))],
                  key=lambda e: e["n"] or 999)

    dup = {}
    for e in ents:
        if e["doi"] and e["doi"] in dois:
            dup[e["n"]] = ("DOI 精确匹配", dois[e["doi"]])
            continue
        t = norm(e["title"])
        if len(t) < 30:          # ⚠️ 短题名（教科书/专著）易误报，直接跳过题名匹配
            continue
        best, score = "", 0.0
        for nm in names:
            nt = norm(nm)
            if len(nt) < 30:
                continue
            cover = len(t) / len(nt)
            if t in nt and cover >= 0.60:                    # 双向包含需覆盖 ≥60%
                s = 1.0
            elif nt in t and (1 / cover) >= 0.60:
                s = 1.0
            else:
                s = SequenceMatcher(None, t[:60], nt[:60]).ratio() if cover >= 0.60 else 0.0
            if s > score:
                best, score = nm, s
        if score >= 0.88:
            dup[e["n"]] = (f"题名匹配 {score:.2f}", best)

    print(f"\n=== 查重结果 ===\n  与底座重复 {len(dup)} 条：")
    for n in sorted(k for k in dup if k):
        r, h = dup[n]
        print(f"    N{n:<3} [{r}]  → 底座 {h[:66]}")
    print(f"\n  ✅ 真新增 {55 - len(dup)} 条（N1–N55 编号制）")
    extra = sorted(set(dup) - KNOWN_DUP)
    if extra:
        print(f"  ⚠️ 新增重复项（超出已知 {sorted(KNOWN_DUP)}）：{extra}")
    else:
        print(f"  ✓ 与已知重复清单一致 {sorted(KNOWN_DUP)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验 bib 结构")
    ap.add_argument("--refresh", action="store_true", help="清掉旧 nnum 后重注入")
    a = ap.parse_args()

    if not BIB.exists():
        print(f"[ERROR] 找不到 {BIB}", file=sys.stderr)
        return 1

    if a.check:
        bad = validate(BIB.read_text(encoding="utf-8"))
        print("✅ BibTeX 结构校验通过" if not bad else "\n".join("⚠️ " + p for p in bad))
        return 0 if not bad else 2

    inj, unk = inject(refresh=a.refresh)
    print(f"=== ① nnum 注入 ===\n  注入 {inj} 条 | 未映射 {unk} 条")
    bad = validate(BIB.read_text(encoding="utf-8"))
    print(f"\n=== ② 结构校验 ===\n  {'✅ 通过' if not bad else chr(10).join('  ⚠️ ' + p for p in bad)}")
    return dedup()


if __name__ == "__main__":
    raise SystemExit(main())
