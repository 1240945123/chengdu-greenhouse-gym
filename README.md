# glassgym

**GreenLight-Gym2 research fork — 面向成都郫都玻璃温室（GH63）的强化学习控制研究代码库。**

*A research codebase forked from [GreenLight-Gym2](https://github.com/BartvLaatum/GreenLight-Gym2): physics-based modeling, crop-yield modeling and RL control for a real glass greenhouse in Chengdu (Pidu), China.*

> 中文主线说明见下文 [研究内容](#研究内容-research-lines)；交接细节见 `docs/notes/03-传承手册_玻璃温室RL.md`。

## Summary

本项目是 [GreenLight-Gym2](https://github.com/BartvLaatum/GreenLight-Gym2)（AgriControl 2025 会议论文《GreenLight-Gym: Reinforcement learning benchmark environment for control of greenhouse production systems》的实验代码）的深度定制 fork。底层作物-气候模型为瓦格宁根大学验证的 **GreenLight**（CasADi 实现），本项目在其上叠加了成都本地化研究管线。

**主线（当前核心）**：针对郫都 GH63 **真实玻璃温室**（192 m²、9 个离散档位执行器）训练「综合最优」强化学习控制器，并与真实人工操作对比。

**最新结论（2026-09-13 口径：修复后模型 + 郫都作物标定 + 完整生长季 102 天 / 4-1 定植）**：

| 策略 | 舒适率 % | 果实 kg/m² |
|---|---:|---:|
| 一步前瞻模型搜索（H=1）| **67.9** | 7.15 |
| MPC（原实现，H=1 枚举）| 67.4 | 7.12 |
| **TRPO**（同协议 60 万步）| **56.0** | 7.50 |
| **PPO（obs 含未来 4h 天气）** | 54.8 | 7.71 |
| 规则控制 | 48.8 | 6.94 |
| **CrossQ** | 47.4 | **8.59** |
| 多智能体 IPPO | 47.0 | 6.51 |
| MaskablePPO（掩码 1536→384）| 44.9 | 8.37 |
| SAC | 35.9 | 6.58 |
| 人工（真实操作日志重放）| 33.7 | 3.06 |

> 完整 **21 策略**统一矩阵（含 a2c/dqn/ddpg/td3/recurrent_ppo/DAgger/TQC 等）与四联图见
> `docs/notes/17-全算法统一矩阵_21策略_20260913.md`；P0 结论见 `docs/notes/18`。

结论应当分三层读：① **模型级**——修复 4 处物理/作物缺陷（冠层过热、外遮阳透光、`solar_gain_scale` 误标定、保温幕缺长波反射）后，作物标定误差 40%→8.2%，人工对照组产量从失真的 0.12 恢复到 3.06 kg/m²；② **控制级**——**模型预测类（MPC / 前瞻搜索）性能最高但单次决策慢两个数量级**（~80 ms vs RL ~1 ms）；③ **RL 级**——RL 的价值是**以约 1/80 的推理成本逼近模型预测控制**，且其瓶颈不在算法复杂度，而在**观测信息与目标设定**（详见 `docs/notes/13`、`15`、`16`、`17`、`18`）。

⚠️ 早期文档 `clean_benchmark_result.md`（2026-09-09）的数字（舒适率 48% vs 17%、reward 6.5×）**早于上述 4 处模型修复与郫都作物标定，已被本表取代**，仅作溯源保留。

## 研究内容 Research Lines

| 线 | 内容 | 关键结果 |
|---|---|---|
| ① 气候建模 | 机理（ChengduPhysics）+ 灰箱残差混合模型，防泄漏验证 | V5 模型 24h 温度 MAE 1.2°C / 72h 1.39°C |
| ② 作物/产量 | Vanthoor 作物模型 + 物候验证 + 碳同化 + 源域迁移 | De Koning 穗率 WMAPE 19.9%；郫都迁移产量 5.8 kg/m² |
| ③ 控制器基准 | Baseline/PID/MPC + PPO/SAC 冻结协议对比 | six-season 与 v5_full_training 基准 |
| ④ 玻璃温室 RL ★ | GlassGreenhouse 模型 + 9 离散执行器环境 + PPO | RL 超人工（见上）；含 CO2 单位 bug 修复 |

## Repository Structure

```
├── glassgym/          # 核心包：环境、模型（ChengduPhysics 系列 / GlassGreenhouse / GreenLight）、组件
├── experiments/       # 实验脚本：controllers/（含 glass_rl 主线 + archive 归档）、crop/、predictors/、reports/、weather/
├── processing/        # 数据管线（chengdu_*：传感器/控制/收获/轨迹构建）
├── common/            # 共享评估（指标/物候/收获评估）
├── configs/           # 环境/数据集/作物/基准配置
├── data/              # 数据（gitignored，见 data/README.md 获取方式）
├── results/           # 实验产物（gitignored，见 results/README.md 索引）
├── docs/              # design/（设计审计文档）experiments/（结论）notes/（论文与交接笔记）
└── tests/             # 测试套件（~100 个文件）
```

## Installation

```bash
# Python >= 3.11；本项目已在 .venv（Windows）验证
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[train,data]"
```

## 快速复现（玻璃温室 RL 主线）

```bash
cd <repo> && PY=.venv/Scripts/python.exe

# 0) 环境冒烟
$PY -c "import sys; sys.path.insert(0,'.'); from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv; print(GlassGreenhouseEnv(episode_days=1, start_day_index=40, crop_start='early_fruiting', disable_supplements=True).action_space)"

# 1) 干净重训（60 万步 ≈ 30 分钟）
$PY experiments/controllers/glass_rl/train_glass_clean.py --timesteps 600000

# 2) 干净评估（人工 vs RL）
$PY experiments/controllers/glass_rl/full_season_benchmark_v3.py --model results/chengdu_agri_greenhouse_001/real_greenhouse/rl/glass_ppo_clean/model.zip

# 3) 碳平衡通量诊断
$PY experiments/controllers/glass_rl/carbon_diag.py
```

**重要口径**（详见交接手册 §1/§4）：① 训练/评估必须 `disable_supplements=True`（禁补光/CO2，否则 RL 会 exploit 灌产量）；② 作物起点用 `early_fruiting`（成熟株起点会亏碳）；③ 人工对比用精确档位数据 `aligned_actuator_states.csv`；④ 栽培窗口 04-01~07-12（清棚后为空棚）。

## Citation

```bibtex
@inproceedings{vanlaatum2025greenlightgym,
  title     = {GreenLight-Gym: Reinforcement learning benchmark environment for control of greenhouse production systems},
  author    = {van Laatum, Bart and others},
  booktitle = {AgriControl 2025},
  year      = {2025},
  doi       = {10.1016/j.ifacol.2025.11.827}
}
```

## License

AGPL-3.0-or-later（继承自上游），见 `LICENSE`。
