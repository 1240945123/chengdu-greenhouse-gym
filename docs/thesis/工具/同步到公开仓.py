# -*- coding: utf-8 -*-
"""把学位论文的**元数据层**同步到公开代码仓 `chengdu-greenhouse-gym/docs/thesis/`。

设计依据（2026-09-19 决策 D2「分层」）：
    公开仓只放「研究设计 / 格式规范 / 文献体系 / 工具脚本」，
    **不放论文正文文字**——避免未答辩稿提前进入查重比对库。

用法：
    python 同步到公开仓.py            # 同步
    python 同步到公开仓.py --dry-run  # 只打印将要同步的内容
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 论文/工具/同步到公开仓.py → 论文/ = parents[1]，final paper/ = parents[2]
PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
DEST = ROOT / "chengdu-greenhouse-gym" / "docs" / "thesis"

# 源（相对 PAPER）→ 目标（相对 DEST）
FILES = {
    "大论文整体框架.md": "论文整体框架.md",
    "川农格式硬约束速查.md": "川农格式硬约束速查.md",
    "文献架构与核心清单.md": "文献架构与核心清单.md",
    "文献库/00-索引.md": "文献库索引_173篇.md",
}

# 目录整体同步（源目录 → 目标目录）
DIRS = {
    "工具": "工具",
}

# 明确排除（正文与含原创判断 / 内部批判的内部文档）
NEVER = [
    # 正文（答辩前不公开）
    "第1章-", "第2章-", "第3章-", "第4章-", "第5章-", "第6章-",
    "00-前置部分",
    # 内部批判与自评文档（含未公开的实验缺陷分析，不宜提前公开）
    "项目整体体检报告", "实验设计缺陷分析",
    # 研究判断类文档
    "文献补充检索", "文献补充_获取清单",
    "references.bib",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写入")
    args = ap.parse_args()

    if not PAPER.exists():
        print(f"[ERROR] 找不到论文目录：{PAPER}", file=sys.stderr)
        return 1
    if not (ROOT / "chengdu-greenhouse-gym" / ".git").exists():
        print(f"[ERROR] 找不到代码仓：{ROOT / 'chengdu-greenhouse-gym'}", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    plan: list[tuple[Path, Path]] = []

    for src_rel, dst_rel in FILES.items():
        src = PAPER / src_rel
        if not src.exists():
            print(f"  [跳过·缺失] {src_rel}")
            continue
        plan.append((src, DEST / dst_rel))

    for src_rel, dst_rel in DIRS.items():
        src_dir = PAPER / src_rel
        if not src_dir.exists():
            print(f"  [跳过·缺失] {src_rel}/")
            continue
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and f.suffix in {".py", ".md"}:
                plan.append((f, DEST / dst_rel / f.name))

    # 安全闸：任何命中 NEVER 的路径都不允许同步
    blocked = [str(dst.relative_to(DEST)) for _, dst in plan
               if any(k in dst.name for k in NEVER)]
    if blocked:
        print(f"[ABORT] 以下文件命中排除名单，拒绝同步：{blocked}", file=sys.stderr)
        return 2

    print(f"目标：{DEST}")
    print(f"将同步 {len(plan)} 个文件：")
    for src, dst in plan:
        rel = dst.relative_to(DEST)
        try:
            size = f"{src.stat().st_size / 1024:.1f} KB"
        except OSError:
            size = "?"
        print(f"  {str(rel):44} ← {src.name}  ({size})")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print()
    print("（README.md 为手写，不在同步范围，需手动维护）" if not args.dry_run
          else "（dry-run，未写入任何文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
