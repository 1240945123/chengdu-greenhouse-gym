"""P0 出图：把四条 P0 的结果画成四张图（读 p0_bundle.json + horizon_frontier.json）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "p0_bundle"


def smooth(x, w=15):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def steps_to_frac(ts, r, frac=0.8):
    if len(r) < 10:
        return None, None
    sm = smooth(r)
    k = max(3, len(sm) // 20)
    r0, r1 = sm[:k].mean(), sm[-k:].mean()
    thr = r0 + frac * (r1 - r0)
    idx = int(np.argmax(sm >= thr)) if (sm >= thr).any() else len(sm) - 1
    return ts[idx], r1


def bar_labels(ax, bars, vals, fmt="{:.1f}"):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)


def main() -> None:
    b = json.loads((OUT / "p0_bundle.json").read_text(encoding="utf-8"))
    R = b["results"]
    C = b.get("curves", {})

    # ------------------------------------------------ 图1：P0-1 动作掩码
    a_lab, b_lab = "PPO v6（无掩码）", "MaskablePPO（掩码384）"
    if a_lab in R and b_lab in R:
        A, B = R[a_lab], R[b_lab]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axes[0]
        for lab, col in [(a_lab, "#2c6fbb"), (b_lab, "#e67e22")]:
            c = C.get(lab, {})
            if c.get("timestep"):
                ts = np.array(c["timestep"]) / 1000
                r = np.array(c["ep_rew_mean"])
                ax.plot(ts, r, color=col, alpha=0.25, lw=1)
                ax.plot(ts, smooth(r), color=col, lw=2.2, label=lab)
        ax.set_xlabel("训练步数 (×1000)")
        ax.set_ylabel("平均 episode reward")
        ax.set_title("① 训练收敛对比（掩码 vs 无掩码）")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1]
        keys = [("comfort_pct", "舒适率 %"), ("fruit_kg", "果实 kg/m²"),
                ("max_temp", "最高温 °C")]
        x = np.arange(len(keys))
        va = [A[k] for k, _ in keys]
        vb = [B[k] for k, _ in keys]
        bar_labels(ax, ax.bar(x - 0.2, va, 0.38, color="#2c6fbb", label=a_lab), va)
        bar_labels(ax, ax.bar(x + 0.2, vb, 0.38, color="#e67e22", label=b_lab), vb)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in keys], fontsize=9)
        ax.set_title("② 最终指标对比（102 天）")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[2]
        waste_a = A.get("action_usage_pct", {}).get("补光", 0) + A.get("action_usage_pct", {}).get("CO2", 0)
        waste_b = B.get("action_usage_pct", {}).get("补光", 0) + B.get("action_usage_pct", {}).get("CO2", 0)
        sb = [(a_lab, steps_to_frac(C.get(a_lab, {}).get("timestep", []),
                                    C.get(a_lab, {}).get("ep_rew_mean", []))[0]),
              (b_lab, steps_to_frac(C.get(b_lab, {}).get("timestep", []),
                                    C.get(b_lab, {}).get("ep_rew_mean", []))[0])]
        sb = [(lab, v / 1000) for lab, v in sb if v]
        if sb:
            bars = ax.bar([l for l, _ in sb], [v for _, v in sb],
                          color=["#2c6fbb", "#e67e22"][:len(sb)])
            bar_labels(ax, bars, [v for _, v in sb], "{:.0f}k")
            ax.set_ylabel("达最终 reward 80% 所需步数 (×1000)")
        ax.set_title(f"③ 样本效率 ｜ 无效动作(补光+CO₂)使用率\n"
                     f"无掩码 {waste_a:.0f}% → 掩码 {waste_b:.0f}%")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("P0-1 动作掩码：屏蔽无效执行器通道（1536 → 384 组合）", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / "p0_masked_ppo.png", dpi=130)
        plt.close(fig)
        print("已生成 p0_masked_ppo.png")

    # ------------------------------------------------ 图2：P0-2 off-policy 家族
    fam = [("SAC v3", "#95a5a6"), ("TQC", "#c0392b"), ("CrossQ", "#27ae60")]
    have = [(l, c) for l, c in fam if l in R]
    if have:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axes[0]
        for lab, col in have:
            c = C.get(lab, {})
            if c.get("timestep"):
                ts = np.array(c["timestep"]) / 1000
                r = np.array(c["ep_rew_mean"])
                ax.plot(ts, r, color=col, alpha=0.25, lw=1)
                ax.plot(ts, smooth(r), color=col, lw=2.2, label=lab)
        ax.set_xlabel("训练步数 (×1000)")
        ax.set_ylabel("平均 episode reward")
        ax.set_title("① off-policy 训练收敛")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1]
        keys = [("comfort_pct", "舒适率 %"), ("fruit_kg", "果实 kg/m²"),
                ("max_temp", "最高温 °C")]
        x = np.arange(len(keys))
        w = 0.8 / len(have)
        for i, (lab, col) in enumerate(have):
            vals = [R[lab][k] for k, _ in keys]
            bars = ax.bar(x + (i - (len(have) - 1) / 2) * w, vals, w, color=col, label=lab)
            bar_labels(ax, bars, vals)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in keys], fontsize=9)
        ax.set_title("② 最终指标对比（102 天）")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[2]
        labs, vals = [], []
        for lab, _ in have:
            s, _f = steps_to_frac(C.get(lab, {}).get("timestep", []),
                                  C.get(lab, {}).get("ep_rew_mean", []))
            if s:
                labs.append(lab)
                vals.append(s / 1000)
        if labs:
            bars = ax.bar(labs, vals, color=[c for (l, c) in have if l in labs])
            bar_labels(ax, bars, vals, "{:.0f}k")
        ax.set_ylabel("达最终 reward 80% 所需步数 (×1000)")
        ax.set_title("③ 样本效率")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("P0-2 off-policy 家族补强：TQC / CrossQ vs SAC（同协议）", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / "p0_offpolicy.png", dpi=130)
        plt.close(fig)
        print("已生成 p0_offpolicy.png")

    # ------------------------------------------------ 图3：P0-3 DAgger 蒸馏
    dm = b.get("distill_meta", {})
    if dm.get("final_per_dim_acc"):
        names = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]
        acc = dm["final_per_dim_acc"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        ax = axes[0]
        bars = ax.bar(names, [a * 100 for a in acc], color="#8e44ad")
        bar_labels(ax, bars, [a * 100 for a in acc], "{:.0f}%")
        ax.axhline(100, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("逐维动作一致率 %")
        ax.set_title(f"① 学生 vs MPC 逐维一致率（DAgger，均值 {np.mean(acc)*100:.1f}%）")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1]
        pts = [(l, R[l]) for l in ("MPC（原实现）", "DAgger蒸馏", "PPO v6（无掩码）",
                                   "MaskablePPO（掩码384）") if l in R]
        for lab, r in pts:
            if "decision_ms_mean" not in r or "comfort_pct" not in r:
                continue
            ax.scatter(r["decision_ms_mean"], r["comfort_pct"], s=90, zorder=3)
            ax.annotate(f"{lab}\n{r['decision_ms_mean']:.2f} ms / {r['comfort_pct']:.1f}%",
                        (r["decision_ms_mean"], r["comfort_pct"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("决策延迟 ms（对数轴）")
        ax.set_ylabel("舒适率 %")
        ax.set_title("② 性能-延迟权衡（蒸馏把 MPC 搬进 1 ms 级）")
        ax.grid(alpha=0.3)
        fig.suptitle("P0-3 DAgger 蒸馏 MPC", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / "p0_dagger.png", dpi=130)
        plt.close(fig)
        print("已生成 p0_dagger.png")

    # ------------------------------------------------ 图4：P0-4 前瞻前沿
    hf = b.get("horizon_frontier", {})
    H = hf.get("horizons", {})
    if H:
        hs = sorted(int(k) for k in H)
        comf = [H[str(h)]["comfort_pct"] for h in hs]
        lat = [H[str(h)]["decision_ms_mean"] for h in hs]
        fruit = [H[str(h)]["fruit_kg"] for h in hs]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        ax = axes[0]
        ax.plot(hs, comf, "o-", color="#2c6fbb", lw=2, label="等算力（N≈192/H）")
        # 等候选数对照（N 固定，算力随 H 增长）——若存在则叠加
        fx = RL / "horizon_frontier_fixedN" / "horizon_frontier.json"
        if fx.exists():
            fd = json.loads(fx.read_text(encoding="utf-8")).get("horizons", {})
            fh = sorted(int(k) for k in fd)
            if fh:
                ax.plot(fh, [fd[str(h)]["comfort_pct"] for h in fh], "D--",
                        color="#8e44ad", lw=2, label="等候选数（N=192 固定）")
        ax.set_xlabel("前瞻步数 H")
        ax.set_ylabel("舒适率 %", color="#2c6fbb")
        ax.set_xticks(hs)
        for h, v in zip(hs, comf):
            ax.annotate(f"{v:.1f}", (h, v), textcoords="offset points", xytext=(0, 7),
                        fontsize=9, color="#2c6fbb")
        ax2 = ax.twinx()
        ax2.plot(hs, lat, "s--", color="#c0392b", lw=2, label="决策延迟 ms")
        ref = hf.get("glass_mpc_reference")
        if ref:
            ax.scatter([1], [ref["comfort_pct"]], marker="*", s=200, color="#f39c12",
                       zorder=5, label="原 GlassMPC（自定义评分）")
        ax2.set_ylabel("决策延迟 ms", color="#c0392b")
        ax.set_title("① H → 性能 → 延迟（等算力：N×H 恒定）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax2.legend(fontsize=8, loc="upper left")

        ax = axes[1]
        ax.plot(hs, fruit, "o-", color="#27ae60", lw=2)
        for h, v in zip(hs, fruit):
            ax.annotate(f"{v:.2f}", (h, v), textcoords="offset points", xytext=(0, 7), fontsize=9)
        ax.set_xticks(hs)
        ax.set_xlabel("前瞻步数 H")
        ax.set_ylabel("果实 kg/m²")
        ax.set_title("② H → 果实产量")
        ax.grid(alpha=0.3)
        fig.suptitle("P0-4 前瞻步数前沿：H 步搜索 vs 原一步枚举", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / "p0_frontier.png", dpi=130)
        plt.close(fig)
        print("已生成 p0_frontier.png")


if __name__ == "__main__":
    main()
