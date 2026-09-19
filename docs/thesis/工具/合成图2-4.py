"""把 figures_ablation 的两张图合成为论文图 2-4（四联/双联）。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
SRC = RL / "figures_ablation"
OUT_PAPER = Path("E:/school/final paper/论文/figures")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

imgs = [mpimg.imread(SRC / "param_sensitivity.png"),
        mpimg.imread(SRC / "action_ablation.png")]
titles = ["(a) 关键参数敏感性 / Parameter sensitivity",
          "(b) 动作消融 / Action ablation"]

fig, axes = plt.subplots(2, 1, figsize=(9, 10))
for ax, im, t in zip(axes, imgs, titles):
    ax.imshow(im)
    ax.set_title(t, fontsize=12, fontweight="bold")
    ax.axis("off")
fig.tight_layout()

OUT_PAPER.mkdir(parents=True, exist_ok=True)
target = OUT_PAPER / "fig2_4_sensitivity_ablation.png"
fig.savefig(target, dpi=150, bbox_inches="tight")
plt.close(fig)

also = RL / "figures_chapter2"
also.mkdir(parents=True, exist_ok=True)
fig2 = None
import shutil
shutil.copyfile(target, also / target.name)
print("已生成:", target)
print("尺寸:", mpimg.imread(target).shape)
