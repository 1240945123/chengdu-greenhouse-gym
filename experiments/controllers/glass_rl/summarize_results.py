"""全算法对比结果汇总：读 benchmark_all JSON，生成 markdown 对比表。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
JSON = OUT / "benchmark_all_d30_60.json"
MD = OUT / "algorithm_comparison.md"

# 展示顺序与分类
ORDER = ["human", "baseline", "rule", "pid", "mpc",
         "ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo",
         "multiagent"]
CLASSICAL = {"human", "baseline", "rule", "pid", "mpc"}
RL = {"ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo", "multiagent"}


def fmt_row(name: str, s: dict) -> str:
    tag = "RL" if name in RL else "经典"
    return (f"| {name:14s} | {tag} | {s['mean_temp']:>5.1f} | {s['max_temp']:>5.1f} "
            f"| {s['comfort']:>5.1f} | {s['total_reward']:>8.1f} | {s['mean_effort']:>6.4f} "
            f"| {s['fruit_growth_kg_m2']:>+6.3f} | {s['integration_failures']:>4d} |")


def main() -> None:
    if not JSON.exists():
        print(f"未找到 {JSON}，先运行 benchmark_all")
        sys.exit(1)
    data = json.loads(JSON.read_text(encoding="utf-8"))

    lines = []
    lines.append("# 玻璃温室全算法控制对比（72 天生长季，day30 起）")
    lines.append("")
    lines.append("口径：CO2 修复后环境 · early_fruiting 坐果初株 · 禁补光/CO2 · yield_weight=1.0")
    lines.append("")
    lines.append("| 策略 | 类型 | 均温°C | 最高°C | 舒适率% | 总reward | 能耗 | 果实kg/m² | 积分失败 |")
    lines.append("|:---|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ORDER:
        if name in data:
            lines.append(fmt_row(name, data[name]))
    lines.append("")

    # 分月表（舒适率）
    lines.append("## 分月舒适率（%）")
    lines.append("")
    lines.append("| 策略 | 5月 | 6月 | 7月 |")
    lines.append("|:---|---:|---:|---:|")
    for name in ORDER:
        if name in data and data[name].get("monthly"):
            m = {x["month"]: x["comfort"] for x in data[name]["monthly"]}
            lines.append(f"| {name:14s} | {m.get('5月','—'):>5} | {m.get('6月','—'):>5} | {m.get('7月','—'):>5} |")
    lines.append("")

    # 分月最高温
    lines.append("## 分月最高温（°C）")
    lines.append("")
    lines.append("| 策略 | 5月 | 6月 | 7月 |")
    lines.append("|:---|---:|---:|---:|")
    for name in ORDER:
        if name in data and data[name].get("monthly"):
            m = {x["month"]: x["max_temp"] for x in data[name]["monthly"]}
            lines.append(f"| {name:14s} | {m.get('5月','—'):>5} | {m.get('6月','—'):>5} | {m.get('7月','—'):>5} |")
    lines.append("")

    MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {MD}")


if __name__ == "__main__":
    main()
