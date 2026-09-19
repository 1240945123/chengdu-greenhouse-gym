# 归档清单（2026-09-19）

> 归档 = **移动**，未删除；如需恢复，按下列路径反向移动即可。
> 判定依据：无人 import + 文档零引用 + 产物被取代或引用失效文件。

| 原路径 | 归档后 | 理由 |
|---|---|---|
| `experiments/controllers/glass_rl/full_season_benchmark_full.py` | `experiments/archive/_retired_20260919/full_season_benchmark_full.py` | 引用了不存在的 results/.../aligned_actuator_states.csv；零 import、零文档引用 → 已失效 |
| `experiments/controllers/glass_rl/train_glass_fullseason.py` | `experiments/archive/_retired_20260919/train_glass_fullseason.py` | 产物目录 glass_ppo_fullseason/ 已被 train_all_algorithms_fullseason.py 取代；零文档引用 |
| `experiments/controllers/stochastic_rl.py` | `experiments/archive/_retired_20260919/stochastic_rl.py` | 零 import、零文档引用、无产物 → 未接入任何流程的探针 |
| `experiments/controllers/evaluate_rl.py` | `experiments/archive/_retired_20260919/evaluate_rl.py` | 零 import、零文档引用、无产物 → 未接入任何流程的探针 |
