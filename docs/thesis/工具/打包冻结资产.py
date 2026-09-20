# -*- coding: utf-8 -*-
"""打包「不可再生的冻结资产」，供异地备份（移动硬盘 / 云盘）。

分层依据（2026-09-19 决策 D2）：
    results/ 里 7.5 GB 是中间产物（可用 git 里的脚本重跑）→ **不备份**
    但模型权重重跑会因随机性**无法精确复现**（第 4 章"8/8 精确复现"的前提）→ **必须冻**

用法：
    python 打包冻结资产.py                 # 打模型冻结集
    python 打包冻结资产.py --with-pdf      # 附带打包 173 篇文献 PDF（约 940 MB，较慢）
    python 打包冻结资产.py --list          # 只列清单不打包
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path("E:/school/final paper")
REPO = ROOT / "chengdu-greenhouse-gym"
RL = REPO / "results/chengdu_agri_greenhouse_001/real_greenhouse/rl"
V5 = REPO / "results/chengdu_agri_greenhouse_001/controller_benchmark/v5_full_training/training"
PAPER = ROOT / "论文"
BACKUP = ROOT / "_backup"
STAMP = date.today().strftime("%Y%m%d")

# 第 4 章统一矩阵（9 执行器 · 102 天）—— 15 个学习型策略
MATRIX = [
    ("ppo_matrix600k", RL / "algorithms_fullseason/ppo/model.zip", "同协议基准（矩阵内 PPO）"),
    ("a2c", RL / "algorithms_fullseason/a2c/model.zip", "退化（1 种动作）"),
    ("dqn", RL / "algorithms_fullseason/dqn/model.zip", "退化"),
    ("sac", RL / "algorithms_fullseason/sac/model.zip", "退化"),
    ("ddpg", RL / "algorithms_fullseason/ddpg/model.zip", "退化（与 TD3 逐位相同）"),
    ("td3", RL / "algorithms_fullseason/td3/model.zip", "退化（与 DDPG 逐位相同）"),
    ("trpo", RL / "algorithms_fullseason/trpo/model.zip", "RL 侧舒适率上界 56.0%"),
    ("recurrent_ppo", RL / "algorithms_fullseason/recurrent_ppo/model.zip", ""),
    ("tqc", RL / "tqc/model.zip", "修好 SAC 安全短板"),
    ("crossq", RL / "crossq/model.zip", "600k 版（450k 版果实 8.59 为全表最高）"),
    ("masked_ppo", RL / "masked_ppo/model.zip", "动作掩码 1536→384"),
    ("multiagent_cooling", RL / "multiagent/cooling/model.zip", "多智能体 IPPO · 降温子智能体"),
    ("multiagent_insulation", RL / "multiagent/insulation/model.zip", "多智能体 IPPO · 保温子智能体"),
    ("multiagent_pad", RL / "multiagent/pad/model.zip", "多智能体 IPPO · 湿帘子智能体"),
    ("dagger_v1_policy", RL / "distill_mpc/policy.pt", "DAgger 蒸馏 MPC（PyTorch）"),
    ("dagger_v1_norm", RL / "distill_mpc/norm.json", "DAgger 归一化参数"),
    ("ppo_v6_obs8", RL / "ppo_final_v6/model.zip", "观测增强对照：obs8"),
    ("ppo_v9_obs16", RL / "ppo_final_v9/model.zip", "观测增强对照：obs16 + 未来 4h 天气"),
]

# V5 线（2 执行器 · 6 季 benchmark）—— 已冻结的两个基线
V5_FROZEN = [
    ("v5_ppo_seed0_ck307200", V5 / "ppo/seed_0/checkpoint_00307200", "V5 线 PPO 冻结基线"),
    ("v5_sac_seed0_ck102400", V5 / "sac/seed_0/checkpoint_00102400", "V5 线 SAC 冻结基线"),
]


def sha256(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def collect() -> list[dict]:
    items = []
    for label, src, note in MATRIX + V5_FROZEN:
        if not src.exists():
            print(f"  ⚠️ 缺失：{label}  ← {src}")
            continue
        items.append({"label": label, "src": src, "note": note,
                      "bytes": size_of(src), "sha256": sha256(src) if src.is_file() else ""})
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-pdf", action="store_true", help="附带打包文献 PDF")
    ap.add_argument("--list", action="store_true", help="只列清单")
    a = ap.parse_args()

    print("=== ① 冻结资产清单 ===")
    items = collect()
    total = sum(i["bytes"] for i in items)
    for i in items:
        print(f"  {i['label']:26} {i['bytes'] / 1024:>9.0f} KB  {i['note']}")
    print(f"  {'合计':26} {total / 1024 / 1024:>9.1f} MB  共 {len(items)} 项")
    if a.list:
        return 0

    # ② 复制到 _backup/（保持可读结构）
    dest_dir = BACKUP / f"模型冻结集_{STAMP}"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i in items:
        d = dest_dir / i["label"]
        shutil.copytree(i["src"], d) if i["src"].is_dir() else shutil.copy2(i["src"], d)
    print(f"\n=== ② 已复制到 {dest_dir} ===")

    # ③ 生成 manifest
    manifest = {"generated": date.today().isoformat(), "purpose":
                "第 4 章统一矩阵（9 执行器/102 天）+ V5 线（2 执行器/6 季）的冻结模型",
                "why": "模型权重重跑因随机性无法精确复现；results/ 其余 7.5 GB 中间产物可重跑，不备份",
                "items": [{**{k: i[k] for k in ("label", "note", "bytes", "sha256")},
                           "source": str(i["src"]).replace(str(REPO), "<repo>")} for i in items]}
    (dest_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# 模型冻结集清单（{STAMP}）", "",
          f"> 用途：{manifest['purpose']}",
          f"> 理由：{manifest['why']}", "",
          f"共 **{len(items)}** 项，合计 **{total / 1024 / 1024:.1f} MB**", "",
          "| # | 名称 | 体积 | SHA-256（前 16） | 说明 |", "|---|---|---:|---|---|"]
    for n, i in enumerate(items, 1):
        md.append(f"| {n} | `{i['label']}` | {i['bytes'] / 1024:.0f} KB | "
                  f"`{i['sha256'][:16] or '—'}` | {i['note']} |")
    md += ["", "## 恢复方式", "",
           "```bash",
           "# 还原到仓库（路径按 manifest.json 的 source 字段）",
           f"cp -r 模型冻结集_{STAMP}/* <repo>/results/chengdu_agri_greenhouse_001/real_greenhouse/rl/",
           "```"]
    (dest_dir / "清单.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ④ 打 zip
    zpath = BACKUP / f"模型冻结集_{STAMP}.zip"
    if zpath.exists():
        zpath.unlink()
    print(f"=== ③ 压缩 → {zpath.name} ===")
    files = [(f, f.relative_to(BACKUP)) for f in dest_dir.rglob("*") if f.is_file()]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f, rel in files:
            z.write(f, rel)
    print(f"  {len(files)} 个文件 → {zpath.stat().st_size / 1024 / 1024:.1f} MB")

    # ⑤ 可选：文献 PDF
    if a.with_pdf:
        lib = PAPER / "文献库"
        pdfs = sorted(p for p in lib.glob("*.pdf"))
        # ⚠️ 篇数动态计算，不硬编码——曾写死"173篇"，文献增至 175 篇后即失效
        pdf_zip = BACKUP / f"文献库_{len(pdfs)}篇_{STAMP}.zip"
        print(f"\n=== ④ 打包文献库（{len(pdfs)} 篇）→ {pdf_zip.name} ===")
        with zipfile.ZipFile(pdf_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for i, p in enumerate(pdfs, 1):
                z.write(p, f"文献库/{p.name}")
                if i % 40 == 0:
                    print(f"  … {i}/{len(pdfs)}", flush=True)
            for extra in (lib / "00-索引.md",):
                if extra.exists():
                    z.write(extra, f"文献库/{extra.name}")
        print(f"  → {pdf_zip.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n✅ 完成。备份目录：{BACKUP}")
    print("   请把 *.zip 拷到移动硬盘 / 云盘（本地副本仅作中转）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
