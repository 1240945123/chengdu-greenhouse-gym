from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.controllers.benchmark_metrics import (
    aggregate_algorithms,
    benchmark_completeness,
)
from experiments.controllers.benchmark_protocol import BenchmarkConfig, load_benchmark_config


ALGORITHM_ORDER = ("baseline", "pid", "mpc", "ppo", "sac")
DISPLAY_NAMES = {
    "baseline": "Rule-based",
    "pid": "PID",
    "mpc": "MPC",
    "ppo": "PPO",
    "sac": "SAC",
}
COLORS = ["#4477AA", "#228833", "#CCBB44", "#EE6677", "#AA3377"]

PAPER_METRICS = (
    ("reward", "reward", "Cumulative reward"),
    ("temperature_mae", "temperature_mae", "Temperature MAE (deg C)"),
    ("temperature_rmse", "temperature_rmse", "Temperature RMSE (deg C)"),
    ("humidity_mae", "humidity_mae", "RH MAE (%RH)"),
    ("humidity_rmse", "humidity_rmse", "RH RMSE (%RH)"),
    ("temperature_comfort_fraction", "temperature_comfort_fraction", "Temperature comfort (%)"),
    ("humidity_comfort_fraction", "humidity_comfort_fraction", "RH comfort (%)"),
    ("joint_comfort_fraction", "joint_comfort_fraction", "Joint comfort (%)"),
    ("temperature_violation_hours", "temperature_violation_hours", "Temperature violation (h)"),
    ("humidity_violation_hours", "humidity_violation_hours", "RH violation (h)"),
    ("mean_actuator_effort", "mean_actuator_effort", "Mean actuator effort"),
    ("total_action_variation", "total_action_variation", "Total action variation"),
    ("switching_count", "switching_count", "Switching count"),
    ("lamp_use_hours", "lamp_use_hours", "Lamp use (h)"),
    ("thermal_screen_command", "uThScr_mean", "Thermal screen mean"),
    ("vent_command", "uVent_mean", "Vent mean"),
    ("lamp_command", "uLamp_mean", "Lamp mean"),
    ("blackout_screen_command", "uBlScr_mean", "Blackout screen mean"),
    ("mean_wall_time_ms", "mean_wall_time_ms", "Compute time (ms/step)"),
    ("harvested_dry_matter_kg_m2", "harvested_dry_matter_kg_m2", "Harvested dry matter (kg/m2)"),
    ("fresh_yield_kg_m2", "fresh_yield_kg_m2", "Fresh yield (kg/m2)"),
    ("simulated_fresh_fruit_production_kg_m2", "simulated_fresh_fruit_production_kg_m2", "Simulated fresh fruit production (kg/m2)"),
    ("final_unharvested_fresh_fruit_kg_m2", "final_unharvested_fresh_fruit_kg_m2", "Final unharvested fruit (kg/m2)"),
    ("harvest_onset_day", "harvest_onset_day", "Harvest onset (day)"),
    ("peak_daily_fresh_yield_kg_m2", "peak_daily_fresh_yield_kg_m2", "Peak daily fresh yield (kg/m2)"),
    ("final_leaf_dry_matter_kg_m2", "final_leaf_dry_matter_kg_m2", "Final leaf dry matter (kg/m2)"),
    ("final_stem_dry_matter_kg_m2", "final_stem_dry_matter_kg_m2", "Final stem dry matter (kg/m2)"),
    ("final_fruit_dry_matter_kg_m2", "final_fruit_dry_matter_kg_m2", "Final fruit dry matter (kg/m2)"),
)

PERCENT_METRICS = {
    "temperature_comfort_fraction",
    "humidity_comfort_fraction",
    "joint_comfort_fraction",
}


def validate_result_artifacts(root: Path, config: BenchmarkConfig) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(f"manifest.json: {exc}")
    else:
        missing.append("manifest.json")

    for algorithm in ALGORITHM_ORDER:
        seeds = (0,) if algorithm in {"baseline", "pid", "mpc"} else config.evaluation_seeds
        for seed in seeds:
            status = manifest.get("runs", {}).get(algorithm, {}).get(str(seed), {}).get("status")
            if status != "complete":
                invalid.append(f"manifest:{algorithm}/seed_{seed}: status={status}")
            trajectory_path = root / "trajectories" / f"{algorithm}_seed_{seed}.csv"
            if not trajectory_path.exists():
                missing.append(str(trajectory_path.relative_to(root)))
                continue
            try:
                trajectory = pd.read_csv(trajectory_path)
                required = {
                    "uBoil", "uCO2", "reward", "terminated", "truncated",
                    "start_day", "growth_year", "location",
                }
                if not required.issubset(trajectory.columns):
                    invalid.append(f"{trajectory_path.name}: missing required columns")
                    continue
                if len(trajectory) != config.episode_days * 24 * 4:
                    invalid.append(f"{trajectory_path.name}: unexpected step count {len(trajectory)}")
                if set(trajectory["start_day"].astype(int)) != {config.test_start_day}:
                    invalid.append(f"{trajectory_path.name}: wrong test start day")
                if set(trajectory["growth_year"].astype(int)) != {config.growth_year}:
                    invalid.append(f"{trajectory_path.name}: wrong growth year")
                if set(trajectory["location"].astype(str)) != {config.location}:
                    invalid.append(f"{trajectory_path.name}: wrong location")
                numeric = trajectory.select_dtypes(include=[np.number])
                if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                    invalid.append(f"{trajectory_path.name}: non-finite values")
                if not np.allclose(trajectory[["uBoil", "uCO2"]], 0.0):
                    invalid.append(f"{trajectory_path.name}: disabled controls are nonzero")
                if not bool(trajectory["terminated"].iloc[-1]) or bool(trajectory["truncated"].any()):
                    invalid.append(f"{trajectory_path.name}: incomplete episode")
            except Exception as exc:
                invalid.append(f"{trajectory_path.name}: {type(exc).__name__}: {exc}")

            if algorithm in {"ppo", "sac"}:
                training_dir = root / "training" / algorithm / f"seed_{seed}"
                for name in (
                    "best_model.zip",
                    "best_vecnormalize.pkl",
                    "final_model.zip",
                    "vecnormalize.pkl",
                    "training_metadata.json",
                ):
                    artifact = training_dir / name
                    if not artifact.exists() or artifact.stat().st_size == 0:
                        missing.append(str(artifact.relative_to(root)))
                metadata_path = training_dir / "training_metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata.get("status") != "complete":
                            invalid.append(f"{algorithm}/seed_{seed}: training is not complete")
                    except (OSError, json.JSONDecodeError) as exc:
                        invalid.append(f"{algorithm}/seed_{seed}: invalid metadata: {exc}")

    for algorithm in ("pid", "mpc"):
        tuning_path = root / "tuning" / algorithm / "selected_params.json"
        if not tuning_path.exists() or tuning_path.stat().st_size == 0:
            missing.append(str(tuning_path.relative_to(root)))

    return {
        "complete": not missing and not invalid,
        "missing": sorted(set(missing)),
        "invalid": sorted(set(invalid)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )


def _ordered(summary: pd.DataFrame) -> pd.DataFrame:
    available = [name for name in ALGORITHM_ORDER if name in summary.index]
    return summary.loc[available]


def _reward_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    means = data["reward_mean"].to_numpy(dtype=float)
    low = means - data["reward_ci95_low"].to_numpy(dtype=float)
    high = data["reward_ci95_high"].to_numpy(dtype=float) - means
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        range(len(data)),
        means,
        yerr=np.vstack([low, high]),
        capsize=4,
        color=COLORS[: len(data)],
    )
    ax.set_xticks(range(len(data)), [DISPLAY_NAMES[name] for name in data.index])
    ax.set_ylabel("Cumulative reward (higher is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _comfort_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    x = np.arange(len(data))
    width = 0.36
    temp = data.get("temperature_mae_mean", pd.Series(0.0, index=data.index))
    humidity = data.get("humidity_mae_mean", pd.Series(0.0, index=data.index))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, temp, width, label="Temperature MAE (deg C)", color="#4477AA")
    ax.bar(x + width / 2, humidity, width, label="RH MAE (%RH)", color="#EE6677")
    ax.set_xticks(x, [DISPLAY_NAMES[name] for name in data.index])
    ax.set_ylabel("Mean absolute error")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _tracking_error_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    x = np.arange(len(data))
    labels = [DISPLAY_NAMES[name] for name in data.index]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, prefix, ylabel in (
        (axes[0], "temperature", "Temperature error (deg C)"),
        (axes[1], "humidity", "Relative-humidity error (%RH)"),
    ):
        ax.bar(x - 0.18, data[f"{prefix}_mae_mean"], 0.36, label="MAE", color="#4477AA")
        ax.bar(x + 0.18, data[f"{prefix}_rmse_mean"], 0.36, label="RMSE", color="#EE6677")
        ax.set_xticks(x, labels, rotation=20)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _comfort_rates_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    x = np.arange(len(data))
    labels = [DISPLAY_NAMES[name] for name in data.index]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    width = 0.25
    for offset, metric, label, color in (
        (-width, "temperature_comfort_fraction_mean", "Temperature", "#4477AA"),
        (0.0, "humidity_comfort_fraction_mean", "RH", "#228833"),
        (width, "joint_comfort_fraction_mean", "Joint", "#EE6677"),
    ):
        axes[0].bar(x + offset, data[metric] * 100.0, width, label=label, color=color)
    axes[0].set_ylabel("Time within comfort bounds (%)")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, data["temperature_violation_hours_mean"], 0.36,
                label="Temperature", color="#4477AA")
    axes[1].bar(x + 0.18, data["humidity_violation_hours_mean"], 0.36,
                label="RH", color="#EE6677")
    axes[1].set_ylabel("Violation duration (h / 96 h)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _runtime_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    runtime = data["mean_wall_time_ms_mean"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(len(data)), runtime, color=COLORS[: len(data)])
    ax.set_xticks(range(len(data)), [DISPLAY_NAMES[name] for name in data.index])
    ax.set_ylabel("Controller computation time (ms/step)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_paper_tables(summary: pd.DataFrame, root: Path) -> None:
    records: list[dict[str, float | int | str]] = []
    for algorithm, row in _ordered(summary).iterrows():
        record: dict[str, float | int | str] = {
            "algorithm": DISPLAY_NAMES[algorithm],
            "runs": int(row["episodes"]),
        }
        for output_name, source_name, _ in PAPER_METRICS:
            mean_column = f"{source_name}_mean"
            std_column = f"{source_name}_std"
            if mean_column not in row:
                continue
            scale = 100.0 if source_name in PERCENT_METRICS else 1.0
            record[f"{output_name}_mean"] = float(row[mean_column]) * scale
            record[f"{output_name}_std"] = float(row.get(std_column, 0.0)) * scale
        records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(root / "paper_metrics.csv", index=False)

    lines = [
        "# Chengdu Controller Benchmark Metrics",
        "",
        "Values are mean +/- sample standard deviation across runs. Comfort rates use the dynamic reward bounds.",
    ]
    groups = (
        ("Tracking and comfort", PAPER_METRICS[:10]),
        ("Control usage and computation", PAPER_METRICS[10:19]),
        ("Crop and yield", PAPER_METRICS[19:]),
    )
    for title, metrics in groups:
        available = [(key, label) for key, _, label in metrics if f"{key}_mean" in table.columns]
        lines.extend(["", f"## {title}", ""])
        headers = ["Algorithm", "Runs", *[label for _, label in available]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in table.iterrows():
            values = [str(row["algorithm"]), str(int(row["runs"]))]
            for key, _ in available:
                values.append(f"{row[f'{key}_mean']:.3f} +/- {row[f'{key}_std']:.3f}")
            lines.append("| " + " | ".join(values) + " |")
    (root / "paper_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _effort_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    effort = data.get("mean_actuator_effort_mean", pd.Series(0.0, index=data.index))
    variation = data.get("total_action_variation_mean", pd.Series(0.0, index=data.index))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].bar(range(len(data)), effort, color=COLORS[: len(data)])
    axes[0].set_ylabel("Mean actuator effort (0-1)")
    axes[1].bar(range(len(data)), variation, color=COLORS[: len(data)])
    axes[1].set_ylabel("Total action variation")
    for ax in axes:
        ax.set_xticks(range(len(data)), [DISPLAY_NAMES[name] for name in data.index], rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _yield_plot(summary: pd.DataFrame, path: Path) -> None:
    data = _ordered(summary)
    fresh_production = data["simulated_fresh_fruit_production_kg_m2_mean"]
    config_path = Path(__file__).resolve().parents[2] / "configs" / "crops" / "vanthoor_tomato.yml"
    crop_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference_low, reference_high = crop_config["external_validation"]["fresh_yield_kg_m2"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(data)), fresh_production, color=COLORS[: len(data)])
    ax.axhspan(
        reference_low,
        reference_high,
        color="#999999",
        alpha=0.25,
        label="Pengzhou external reference",
    )
    ax.set_xticks(range(len(data)), [DISPLAY_NAMES[name] for name in data.index])
    ax.set_ylabel("Simulated fresh tomato yield (kg/m2)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _trajectory_plot(root: Path, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    plotted = 0
    for algorithm, color in zip(ALGORITHM_ORDER, COLORS):
        trajectory_path = root / "trajectories" / f"{algorithm}_seed_0.csv"
        if not trajectory_path.exists():
            continue
        trajectory = pd.read_csv(trajectory_path)
        x = trajectory["timestep"].to_numpy(dtype=float) * 0.25
        axes[0].plot(x, trajectory["air_temperature"], label=DISPLAY_NAMES[algorithm], color=color)
        axes[1].plot(x, trajectory["relative_humidity"], label=DISPLAY_NAMES[algorithm], color=color)
        plotted += 1
    axes[0].set_ylabel("Air temperature (deg C)")
    axes[1].set_ylabel("Relative humidity (%RH)")
    axes[1].set_xlabel("Test time (hours)")
    for ax in axes:
        ax.grid(alpha=0.25)
    if plotted:
        axes[0].legend(ncol=min(plotted, 5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_report(
    result_dir: str | Path,
    config: BenchmarkConfig,
    *,
    make_plots: bool = True,
) -> dict[str, Any]:
    root = Path(result_dir)
    episodes_path = root / "episode_metrics.csv"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episode metrics: {episodes_path}")
    episodes = pd.read_csv(episodes_path)
    summary = aggregate_algorithms(episodes)
    summary = _ordered(summary)
    summary.to_csv(root / "comparison.csv")
    _write_paper_tables(summary, root)
    completeness = benchmark_completeness(
        episodes,
        required_algorithms=ALGORITHM_ORDER,
        required_seeds=config.evaluation_seeds,
        deterministic_algorithms=("baseline", "pid", "mpc"),
    )
    artifact_completeness = validate_result_artifacts(root, config)
    payload = {
        "profile": config.profile,
        "paper_ready": bool(
            config.profile == "paper"
            and completeness["paper_ready"]
            and artifact_completeness["complete"]
        ),
        "completeness": completeness,
        "artifact_completeness": artifact_completeness,
        "test_weather": {
            "location": config.location,
            "growth_year": config.growth_year,
            "start_day": config.test_start_day,
            "episode_days": config.episode_days,
        },
        "limitation": (
            "Seed repetitions quantify controller or model-initialization variability "
            "on one fixed test-weather window; they do not establish weather generalization."
        ),
        "algorithms": summary.reset_index().to_dict(orient="records"),
    }
    _write_json(root / "comparison.json", payload)
    if make_plots:
        _reward_plot(summary, root / "reward_comparison.png")
        _comfort_plot(summary, root / "comfort_comparison.png")
        if {"temperature_rmse_mean", "humidity_rmse_mean"}.issubset(summary.columns):
            _tracking_error_plot(summary, root / "tracking_errors.png")
        if {"temperature_comfort_fraction_mean", "humidity_comfort_fraction_mean"}.issubset(summary.columns):
            _comfort_rates_plot(summary, root / "comfort_rates.png")
        _effort_plot(summary, root / "control_effort.png")
        if "mean_wall_time_ms_mean" in summary.columns:
            _runtime_plot(summary, root / "runtime_comparison.png")
        if "simulated_fresh_fruit_production_kg_m2_mean" in summary.columns:
            _yield_plot(summary, root / "yield_comparison.png")
        _trajectory_plot(root, root / "test_trajectories.png")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chengdu controller benchmark report")
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--result_dir", default=None)
    args = parser.parse_args()
    config = load_benchmark_config(args.profile)
    result_dir = Path(args.result_dir) if args.result_dir else config.output_dir
    generate_report(result_dir, config)
    print(f"Report generated in {result_dir}")


if __name__ == "__main__":
    main()
