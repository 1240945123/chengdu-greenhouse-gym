# experiments/ —— 实验流程、文件用途与依赖导航

> 生成：2026-09-19 ｜ 表格数据来自依赖图谱（模块 import + 文档交叉引用自动扫描）
> 配套：`docs/实验流程与文件组织.md`（结构调整说明）、`docs/README.md`（文档索引）

---

## 1. 五条实验线（本目录的整体划分）

| 线 | 目录 | 论文用途 | 状态 |
|---|---|---|---|
| **① 玻璃温室 RL 线** | `controllers/glass_rl/` | ⭐ **论文主实验**（第 2/4/5 章） | **在用** |
| ② V5 控制器基准线 | `controllers/`（`v5_*`、`benchmark_*`） | 跨 3 年 6 季 × 多种子基准；论文**未使用**，但为补充实验 A2/A3 的基础设施 | 可用（待接入） |
| ③ 作物与收获标定线 | `crop/` | 第 3 章作物模型标定与产量先验 | 在用 |
| ④ 物理标定线 | `reports/` | 第 2 章物理模型识别/验证 | 在用 |
| ⑤ 预测器线（PINN/Transformer） | `predictors/` | 未进入论文 | **已停用**（保留溯源） |

## 2. 论文主线的完整管线（`controllers/glass_rl/`）

```
环境与控制器（第 ① 组，库）
        │
        ├─► 训练 ──► results/chengdu_agri_greenhouse_001/real_greenhouse/rl/<算法>/model.zip
        │            （第 ②③ 组；旧世代见第 ⑦ 组）
        │
        ├─► 评估 ──► algorithm_matrix.json / p0_bundle.json / latency_idle.json
        │            （第 ④ 组；诊断见第 ⑤ 组）
        │
        └─► 绘图 ──► algorithm_matrix.png / figures_chapter2/ / p0_bundle/
                     （第 ⑥ 组；旧图为第 ⑧ 组）
```

**三步复现（论文第 4 章全部数字）**：

```bash
# ① 训练（可选，权重已冻结在 _backup/模型冻结集_20260919.zip）
python -m experiments.controllers.glass_rl.train_all_algorithms_fullseason --algorithms ppo --timesteps 600000
# ② 评估（21 策略同协议矩阵）
python -m experiments.controllers.glass_rl.eval_algorithm_matrix
# ③ 出图
python -m experiments.controllers.glass_rl.plot_algorithm_matrix
```

> **延迟必须空闲测**：`measure_latency.py`（并发时绝对值会膨胀 1.4–3.4 倍）。
> **协议保真性自检**：`eval_algorithm_matrix.py` 会对历史冻结值做 8/8 复现校验。

---

## 3. `controllers/glass_rl/` 逐文件说明

### ① 环境与控制器（库，不单独运行）

被全流程依赖的底层模块。**改动会连带影响数十个脚本**，动前必跑全量测试。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `glass_env.py` | 17.8 KB | 35 | 10 | ⭐核心库（35 脚本依赖）；文档 10 处引用 |
| `env_variants.py` | 7.6 KB | 13 | 1 | ⭐核心库（13 脚本依赖）；文档 1 处引用 |
| `multiagent_env.py` | 4.8 KB | 4 | 1 | 库（4 依赖）；文档 1 处引用 |
| `residual_env.py` | 2.8 KB | 2 | 2 | 文档 2 处引用 |
| `classical_controllers.py` | 8.3 KB | 12 | 3 | ⭐核心库（12 脚本依赖）；文档 3 处引用 |
| `plotting.py` | 12.3 KB | 1 | 2 | 文档 2 处引用 |

### ② 训练 · 第 4 章统一矩阵

21 策略矩阵的训练入口（同协议 60 万步）。产物 → `rl/algorithms_fullseason/`。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `train_all_algorithms_fullseason.py` | 7.3 KB | 0 | 3 | 文档 3 处引用 |
| `train_offpolicy_extra.py` | 7.2 KB | 1 | 1 | 文档 1 处引用 |
| `resume_offpolicy.py` | 3.7 KB | 0 | 3 | 文档 3 处引用 |

### ③ 训练 · 第 5 章提升路线

四条提升路线的训练入口。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `train_masked_ppo.py` | 6.0 KB | 0 | 1 | 文档 1 处引用 |
| `train_multiagent.py` | 3.9 KB | 0 | 1 | 文档 1 处引用 |
| `train_ppo_final.py` | 9.1 KB | 4 | 1 | 库（4 依赖）；文档 1 处引用 |

### ④ 评估 · 第 4 章主表与 P0

论文正文直接引用的评估入口。**优先级最高的入口脚本**。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `eval_algorithm_matrix.py` | 12.0 KB | 1 | 4 | 文档 4 处引用 |
| `measure_latency.py` | 3.6 KB | 0 | 3 | 文档 3 处引用 |
| `eval_p0_bundle.py` | 10.7 KB | 0 | 2 | 文档 2 处引用 |
| `eval_horizon_frontier.py` | 6.5 KB | 0 | 1 | 文档 1 处引用 |
| `horizon_search.py` | 8.2 KB | 2 | 2 | 文档 2 处引用 |
| `full_season_benchmark_v3.py` | 6.0 KB | 9 | 3 | 库（9 依赖）；文档 3 处引用 |

### ⑤ 课题专项诊断

物理/作物可信性诊断与策略蒸馏。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `distill_mpc.py` | 13.1 KB | 3 | 2 | 库（3 依赖）；文档 2 处引用 |
| `distill_mpc_v2.py` | 9.9 KB | 0 | 1 | 文档 1 处引用 |
| `full_season_growth_diag.py` | 5.3 KB | 0 | 3 | 文档 3 处引用 |
| `ablation_glass.py` | 5.2 KB | 0 | 2 | 文档 2 处引用 |
| `carbon_diag.py` | 4.2 KB | 0 | 2 | 文档 2 处引用 |
| `benchmark_all.py` | 9.2 KB | 2 | 0 | 入口/一次性 |
| `benchmark_all_fullseason.py` | 8.3 KB | 1 | 1 | 文档 1 处引用 |

### ⑥ 绘图 · 论文在用

第四章/第二章/P0 三套图的可复现脚本。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `plot_algorithm_matrix.py` | 6.8 KB | 0 | 4 | 文档 4 处引用 |
| `plot_chapter2_physics.py` | 10.6 KB | 0 | 2 | 文档 2 处引用 |
| `plot_p0.py` | 13.3 KB | 0 | 0 | 入口/一次性 |

### ⑦ 训练与评估 · 旧世代（PPO/SAC 优化，论文仅作溯源）

产出 `rl/algorithms/`（8 策略早期表）与 PPO v6/v7/v8、SAC v3/v4 世代；对应 `docs/notes/10/11/12`。保留用于结论溯源，**新实验不要再用**。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `train_all_algorithms.py` | 5.6 KB | 0 | 0 | 入口/一次性 |
| `train_ppo_optimized.py` | 5.9 KB | 1 | 4 | 文档 4 处引用 |
| `train_sac_final.py` | 6.0 KB | 0 | 4 | 文档 4 处引用 |
| `train_glass_clean.py` | 2.2 KB | 0 | 2 | 文档 2 处引用 |
| `continue_train_ppo.py` | 1.8 KB | 0 | 1 | 文档 1 处引用 |
| `train_residual_ppo.py` | 4.1 KB | 0 | 1 | 文档 1 处引用 |
| `eval_systematic.py` | 10.2 KB | 0 | 3 | 文档 3 处引用 |
| `eval_ppo_final.py` | 4.7 KB | 0 | 2 | 文档 2 处引用 |
| `eval_final_comparison.py` | 4.9 KB | 0 | 2 | 文档 2 处引用 |
| `eval_four_way.py` | 6.2 KB | 0 | 2 | 文档 2 处引用 |
| `eval_mpc_pid.py` | 3.5 KB | 0 | 1 | 文档 1 处引用 |
| `summarize_results.py` | 2.8 KB | 0 | 1 | 文档 1 处引用 |

### ⑧ 绘图 · 旧世代

早期图表脚本，对应 `docs/notes/10-绘图工作流`。

| 文件 | 体积 | 被 import | 文档引用 | 角色 |
|---|---:|---:|---:|---|
| `plot_ppo_final.py` | 5.2 KB | 0 | 1 | 文档 1 处引用 |
| `plot_ppo_variants.py` | 3.5 KB | 0 | 0 | 入口/一次性 |
| `plot_systematic.py` | 5.4 KB | 0 | 1 | 文档 1 处引用 |
| `plot_full_season.py` | 7.3 KB | 0 | 1 | 文档 1 处引用 |
| `plot_60day_all.py` | 7.4 KB | 0 | 1 | 文档 1 处引用 |
| `plot_ab_comparison.py` | 3.0 KB | 0 | 1 | 文档 1 处引用 |
| `plot_four_way_full.py` | 13.1 KB | 0 | 2 | 文档 2 处引用 |
| `plot_optimization_and_core.py` | 9.5 KB | 0 | 1 | 文档 1 处引用 |
| `plot_env_trajectory.py` | 6.1 KB | 0 | 1 | 文档 1 处引用 |
| `plot_training_curve.py` | 3.5 KB | 0 | 1 | 文档 1 处引用 |

---

## 4. 其他实验线（简表）

| 目录 | 代表文件 | 用途 |
|---|---|---|
| `controllers/`（V5/基准线） | `benchmark_protocol.py`、`benchmark_metrics.py`、`evaluate_six_season_controllers.py`、`run_chengdu_benchmark.py` | **跨 3 年 6 季 × 多种子**控制器基准；协议抽象完备。⚠️ 其舒适带定义与 glass_rl 线**不一致**（温湿分离 + 安全界 50–85），见 `论文/实验设计缺陷分析与补充实验清单_20260919.md` |
| `crop/` | `calibrate_harvest_model.py`、`run_chengdu_harvest_pipeline.py` | 第 3 章：作物参数标定、GH2024 实测收获先验 |
| `reports/` | `evaluate_chengdu_physics.py`、`chengdu_residual_correction.py` | 第 2 章：物理模型识别、残差校正、数据质量审计 |
| `predictors/` | `train_pinn.py`、`train_transformer.py` | ⚠️ **已停用**：代理模型探索，未进入论文 |
| `weather/` | `fit_forecast_emulator.py` | 天气预测模拟器 |
| `archive/` | `archive/_retired_20260919/README.md` | 归档（**移动非删除**），含归档理由 |

## 5. 目录约定

| 约定 | 说明 |
|---|---|
| 脚本命名 | `train_*` 训练入口 · `eval_*`/`evaluate_*` 评估入口 · `plot_*` 绘图 · 其余为库或诊断 |
| 结果落盘 | 一律写 `results/chengdu_agri_greenhouse_001/...`，不进 git（见 `.gitignore`） |
| 冻结权重 | `model.zip` 与关键 `checkpoint_*.zip` 已打包为 `_backup/模型冻结集_20260919.zip` |
| 测试 | `tests/` 下**全部**以 `test_` 开头（2026-09-19 修正 15 个漏收集文件） |
| 归档 | 退役脚本移入 `archive/`，**不删除**；`git log` 仍可追溯 |

## 6. 三条使用纪律（踩过的坑）

1. **`nohup cmd &` 会被挂起**（日志只写一行就冻结）→ 用工具自带的后台机制。
2. **off-policy 续训必须预热回放池**（`resume_offpolicy.py`），否则空回放池冷启动直接 NaN 发散。
3. **训练活跃度不能看 `training_curve.csv` 的 mtime**（仅段结束写盘）→ 看 `tb_log/*/*.tfevents.*`。
