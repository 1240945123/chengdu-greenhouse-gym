"""构建统一文献库：HTML→PDF，统一命名为 [序号] 年份 题名.pdf，与底座合并，重复项归档。"""
import os
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("E:/school/final paper/论文")
LIB_SRC = PAPER / "综述文献" / "综述文献打包"
ADD = PAPER / "文献补充"
LIB = PAPER / "文献库"
DUP_DIR = ADD / "重复项-已在底座"

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

# N编号 -> (年份, 题名, 源文件名)
NEW19 = {
    "N30": (2026, "Enhancing autonomous agriculture control systems in greenhouses using deep learning techniques", "N30_Hindi2026_RL温室控制_深度学习_PLOSONE.pdf"),
    "N33": (2024, "Current applications and potential future directions of reinforcement learning-based Digital Twins in agriculture", "N33  1-s2.0-S2772375524001175-main.pdf"),
    "N34": (2025, "Hybrid Modeling, Sim-to-Real Reinforcement Learning, and LLM Driven Control for Digital Twins", "N34_Rasheed2025_混合建模Sim2Real_LLM_数字孪生_arXiv.pdf"),
    "N35": (2025, "Reinforcement Learning-Enhanced MPC for Sustainable Greenhouse Climate Management", "N35 Reinforcement Learning-Enhanced MPC for Sustainable Greenhouse Climate Management.pdf"),
    "N36": (2026, "Advancing Digital Twin Development for Controlled-Environment Agriculture - comparative evaluation of two greenhouse energy models", "N36 1-s2.0-S2772375526002327-main.pdf"),
    "N37": (2026, "A grey-box temperature model for digital twins of building-integrated rooftop greenhouses", "N37 1-s2.0-S0360132325015021-main.pdf"),
    "N38": (2021, "Unified Framework and Survey for Model Verification, Validation and Uncertainty Quantification", "N38_Riedmaier2021_模型VV_UQ统一框架.pdf"),
    "N41": (2015, "Experimental performance of evaporative cooling pad systems in greenhouses in humid subtropical climates", "N41 1-s2.0-S0306261914011118-main.pdf"),
    "N42": (2026, "装备化正压通风降温系统在连栋温室的应用", "N42_孙维拓2026_装备化正压通风降温系统_农业工程学报"),
    "N43": (2025, "Data-driven localization of the TOMGRO model - cultivar-specific parameter optimization for Shanghai greenhouse tomato production", "N43 1-s2.0-S0168169925011317-main.pdf"),
    "N44": (2024, "Uncertainty in greenhouse tomato growth models", "N44 1-s2.0-S0168169924007154-main.pdf"),
    "N45": (2026, "A validated climate-crop model for the prediction of yield of dwarf tomatoes in Controlled Environment Agriculture", "N45 1-s2.0-S1537511026000577-main.pdf"),
    "N46": (2022, "Process-based greenhouse climate models - genealogy, current status, and future directions", "N46 1-s2.0-S0308521X22000245-main.pdf"),
    "N49": (2018, "Adaptive two time-scale receding horizon optimal control for greenhouse lettuce cultivation", "N49 1-s2.0-S0168169917310347-main.pdf"),
    "N50": (2021, "Deep Reinforcement Learning at the Edge of the Statistical Precipice", "N50_Agarwal2021_统计悬崖上的深度RL_NeurIPS.pdf"),
    "N51": (2020, "Evaluating the Performance of Reinforcement Learning Algorithms", "N51_Jordan2020_RL算法性能评估_ICML.pdf"),
    "N53": (2021, "春茬大棚番茄如何控温（四川省农业农村厅）", "N53_四川省农业农村厅_春茬大棚番茄如何控温.html"),
    "N54": (2018, "设施番茄越夏栽培田间管理技术（四川省农业农村厅）", "N54_四川省农业农村厅_设施番茄越夏栽培田间管理技术.html"),
    "N55": (2025, "AI温室环境控制技术的应用及发展趋势", "N55 AI温室环境控制技术的应用及发展趋势_刘伟华.pdf"),
}

DUP6 = {
    "N29": ("[005] 2025 The Intelligentization Process of Agricultural Greenhouse - A Review of Control Strategies and Model....pdf", "N29_Li2025_温室控制与建模综述_Agriculture.pdf"),
    "N31": ("[052] 2025 RL-Guided MPC for Autonomous Greenhouse Control.pdf", "N31_Msaad2025_RL-Guided_MPC_arXiv.pdf"),
    "N32": ("[060] 2025 Adaptive robust greenhouse climate control - Combining deep reinforcement learning and economic opti....pdf", "N32adaptive robust.pdf"),
    "N47": ("[046] 2024 Chance-constrained stochastic MPC of greenhouse production systems with parametric uncertainty.pdf", "N47 1-s2.0-S0168169923009663-main.pdf"),
    "N48": ("[043] 2023 Data-driven robust model predictive control for greenhouse temperature control and energy utilisatio....pdf", "N48 1-s2.0-S0306261923005548-main.pdf"),
    "N52": ("[054] 2023 Energy-Saving Control Algorithm of Venlo Greenhouse Skylight and Wet Curtain Fan Based on Reinforcem....pdf", "N52_Chen2023_软动作掩码_维洛温室天窗湿帘_Agriculture.pdf"),
}


def safe(title: str, limit: int = 95) -> str:
    t = title.replace(":", " -").replace("?", "")
    t = re.sub(r'[<>"/\\|*]', " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        t = t[:limit].rstrip() + "...."
    return t


def html2pdf(src: Path, dst: Path) -> bool:
    if not EDGE.exists():
        return False
    cmd = [str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={dst}", src.resolve().as_uri()]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return False
    return dst.exists() and dst.stat().st_size > 10000


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def main():
    LIB.mkdir(exist_ok=True)
    DUP_DIR.mkdir(exist_ok=True)

    print("=== 1) HTML → PDF ===")
    for n, (year, title, fname) in NEW19.items():
        if fname.lower().endswith(".html"):
            src = ADD / fname
            pdf = ADD / (src.stem + ".pdf")
            if pdf.exists():
                print(f"  {n} 已有 PDF，跳过")
                continue
            ok = html2pdf(src, pdf)
            print(f"  {n} {'✅ 转换成功' if ok else '❌ 转换失败（保留 HTML）'}  {pdf.name}")
            if ok:
                NEW19[n] = (year, title, pdf.name)

    print()
    print("=== 2) 底座 152 篇 → 文献库（硬链接，不占额外空间）===")
    base_files = sorted([p for p in LIB_SRC.iterdir() if p.suffix.lower() == ".pdf"])
    ok = 0
    for p in base_files:
        link_or_copy(p, LIB / p.name)
        ok += 1
    print(f"  已链入 {ok} 篇")

    print()
    print("=== 3) 新增 19 篇 → 文献库（统一命名 [153]-[171]）===")
    idx = 153
    manifest = []
    for n, (year, title, fname) in NEW19.items():
        src = ADD / fname
        if not src.exists():
            print(f"  ❌ {n} 源文件缺失: {fname}")
            continue
        if src.suffix.lower() == ".html":
            newname = f"[{idx}] {year} {safe(title)}.html"
        else:
            newname = f"[{idx}] {year} {safe(title)}.pdf"
        kind = link_or_copy(src, LIB / newname)
        manifest.append((idx, n, year, title, newname))
        print(f"  [{idx}] ← {n}  ({kind})")
        idx += 1

    print()
    print("=== 4) 重复项 6 篇 → 归档 ===")
    for n, (libname, myfile) in DUP6.items():
        src = ADD / myfile
        if src.exists():
            target = DUP_DIR / f"{n}（=底座{libname[:5]}）{src.name}"
            if target.exists():
                target.unlink()
            shutil.move(str(src), str(target))
            print(f"  {n} → 重复项/  （底座已有：{libname[:48]}...）")
        else:
            print(f"  {n} 源文件未找到（可能已处理）")

    print()
    total = len(list(LIB.glob("*.pdf"))) + len(list(LIB.glob("*.html")))
    print(f"=== 完成：文献库共 {total} 篇 ===")
    (PAPER / "_lib_manifest.txt").write_text(
        "\n".join(f"{i}\t{n}\t{y}\t{t}" for i, n, y, t, _ in manifest),
        encoding="utf-8")


if __name__ == "__main__":
    main()
