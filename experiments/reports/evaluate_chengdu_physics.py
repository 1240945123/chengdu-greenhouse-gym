from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd

from glassgym.configs.default_params import init_default_params
from glassgym.environments.utils import rh2vaporDens, satVp, vaporDens2pres, vaporPres2rh


TARGETS = {
    "air_temperature": ("x_air_temperature", "next_x_air_temperature"),
    "relative_humidity": ("x_relative_humidity", "next_x_relative_humidity"),
    "co2_concentration": ("x_co2_concentration", "next_x_co2_concentration"),
}
DEFAULT_TARGETS = ["air_temperature", "relative_humidity"]
CONTROL_COLUMNS = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]
CONTROL_SCHEMAS = {
    "ChengduPhysics": CONTROL_COLUMNS,
    "ChengduPhysicsLegacy": CONTROL_COLUMNS,
    "ChengduPhysicsV2": CONTROL_COLUMNS,
    "ChengduPhysicsV3": ["uBoil", "uCO2", "uThScr", "uRoofVent", "uLamp", "uBlScr", "uFan", "uPad"],
    "ChengduPhysicsV4": ["uBoil", "uCO2", "uThScr", "uRoofVent", "uLamp", "uBlScr", "uFan", "uPad"],
    # Real 192 m2 Pidu glass greenhouse: 10 controls (no boiler; uBoil kept as 0)
    "GlassGreenhouse": [
        "uBoil", "uCO2", "uThScr", "uRoofVent", "uLamp", "uBlScr",
        "uFan", "wet_pad_pump", "side_thermal_screen", "wet_pad_roll_film",
    ],
}


def default_parameter_vector(n_params: int = 216) -> np.ndarray:
    params = np.asarray(init_default_params(n_params), dtype=np.float64)
    params[208:216] = 1.0
    return params


def _row_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row[column] if column in row and pd.notna(row[column]) else default
    return float(value)


def estimate_clear_sky_temperature(air_temperature_c: float, relative_humidity: float) -> float:
    """Estimate effective clear-sky temperature with Brutsaert emissivity."""
    air_kelvin = float(np.clip(air_temperature_c + 273.15, 180.0, 330.0))
    vapor_pressure_hpa = (
        float(np.clip(relative_humidity, 0.0, 100.0)) / 100.0 * float(satVp(air_temperature_c)) / 100.0
    )
    emissivity = float(np.clip(1.24 * (max(vapor_pressure_hpa, 1e-6) / air_kelvin) ** (1.0 / 7.0), 0.2, 1.0))
    sky_temperature = air_kelvin * emissivity ** 0.25 - 273.15
    return float(np.clip(sky_temperature, -80.0, air_temperature_c))


def row_to_model_inputs(
    row: pd.Series,
    model_backend: str = "ChengduPhysics",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    air_temperature = _row_value(row, "x_air_temperature", 20.0)
    relative_humidity = _row_value(row, "x_relative_humidity", 70.0)
    co2_concentration = _row_value(row, "x_co2_concentration", 650.0)

    x = np.zeros(28, dtype=np.float64)
    x[0] = co2_concentration
    x[1] = co2_concentration
    x[2] = air_temperature
    x[3] = air_temperature
    x[4] = air_temperature
    x[5] = air_temperature
    x[6] = _row_value(row, "d_air_temperature", air_temperature)
    x[7] = air_temperature
    x[8] = air_temperature
    x[9] = air_temperature
    x[10:15] = _row_value(row, "x_soil_temperature", air_temperature)
    x[15] = vaporDens2pres(air_temperature, rh2vaporDens(air_temperature, relative_humidity))
    x[16] = x[15]
    x[17:21] = air_temperature
    x[21] = air_temperature
    x[23] = 9.5283e4
    x[24] = 2.5107e5
    x[25] = 5.5338e4

    try:
        control_columns = CONTROL_SCHEMAS[model_backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported Chengdu evaluation backend: {model_backend}") from exc
    u = np.array([_row_value(row, column, 0.0) for column in control_columns], dtype=np.float64)

    outdoor_temperature = _row_value(row, "d_air_temperature", air_temperature)
    outdoor_rh = _row_value(row, "d_relative_humidity", relative_humidity)
    d = np.zeros(10, dtype=np.float64)
    d[0] = _row_value(row, "d_global_radiation", 0.0)
    d[1] = outdoor_temperature
    d[2] = vaporDens2pres(outdoor_temperature, rh2vaporDens(outdoor_temperature, outdoor_rh))
    d[3] = _row_value(row, "d_co2_concentration", 720.0)
    d[4] = _row_value(row, "d_wind_speed", 0.0)
    if "d_sky_temperature" in row and pd.notna(row["d_sky_temperature"]):
        d[5] = float(row["d_sky_temperature"])
    elif model_backend in ("ChengduPhysicsV4", "GlassGreenhouse"):
        d[5] = estimate_clear_sky_temperature(outdoor_temperature, outdoor_rh)
    else:
        d[5] = outdoor_temperature - 6.0
    d[6] = _row_value(row, "x_soil_temperature", outdoor_temperature)
    d[8] = 1.0 if d[0] > 0.0 else 0.0
    d[9] = d[8]
    return x, u, d


def predict_one_step(
    row: pd.Series,
    integrator,
    params: np.ndarray,
    model_backend: str = "ChengduPhysics",
) -> dict[str, float]:
    x, u, d = row_to_model_inputs(row, model_backend=model_backend)
    p_dyn = ca.vertcat(ca.DM(d), ca.DM(params))
    result = integrator(x0=ca.DM(x), u=ca.DM(u), p=p_dyn)
    next_x = result["xf"].full().flatten()
    return {
        "pred_air_temperature": float(next_x[2]),
        "pred_relative_humidity": float(vaporPres2rh(next_x[2], next_x[15])),
        "pred_co2_concentration": float(next_x[0]),
    }


def evaluate_one_step_predictions(
    trajectories: pd.DataFrame,
    parameter_vector: np.ndarray | None = None,
    sample_limit: int | None = None,
    dt: float = 3600.0,
    target_names: list[str] | None = None,
    model_backend: str = "ChengduPhysics",
) -> tuple[dict[str, float], pd.DataFrame]:
    data = trajectories.head(sample_limit).copy() if sample_limit is not None else trajectories.copy()
    params = np.asarray(parameter_vector, dtype=np.float64) if parameter_vector is not None else default_parameter_vector()
    define_backend_model = import_module(f"glassgym.models.{model_backend}.utils").define_model
    integrator = define_backend_model(
        nx=28,
        nu=len(CONTROL_SCHEMAS[model_backend]),
        nd=10,
        n_params=len(params),
        dt=dt,
    )

    rows = []
    for _, row in data.iterrows():
        prediction = predict_one_step(
            row,
            integrator=integrator,
            params=params,
            model_backend=model_backend,
        )
        rows.append({**row.to_dict(), **prediction})
    predictions = pd.DataFrame(rows)

    selected_targets = target_names or DEFAULT_TARGETS
    unknown_targets = sorted(set(selected_targets).difference(TARGETS))
    if unknown_targets:
        raise ValueError(f"Unknown target names: {unknown_targets}")

    metrics: dict[str, float] = {"num_samples": float(len(predictions))}
    maes = []
    mses = []
    for target in selected_targets:
        _, next_column = TARGETS[target]
        pred_column = f"pred_{target}"
        error = predictions[pred_column].astype(float) - predictions[next_column].astype(float)
        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(np.square(error)))
        metrics[f"{target}_mae"] = mae
        metrics[f"{target}_mse"] = mse
        maes.append(mae)
        mses.append(mse)
    metrics["mean_mae"] = float(np.mean(maes))
    metrics["mean_mse"] = float(np.mean(mses))
    return metrics, predictions


def load_parameter_vector(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("multipliers", payload)
    declared_size = int(payload.get("num_params", 216)) if isinstance(payload, dict) else 216
    max_index = max((int(key) for key in values), default=215)
    params = default_parameter_vector(max(declared_size, max_index + 1, 216))
    for key, value in values.items():
        params[int(key)] = float(value)
    return params


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChengduPhysics one-step error on real greenhouse trajectories.")
    parser.add_argument("--trajectory_csv", default="data/processed/chengdu_agri/greenhouse_001/trajectories/test.csv")
    parser.add_argument("--params_json", default=None)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--targets", default="air_temperature,relative_humidity")
    parser.add_argument("--model_backend", default="ChengduPhysics")
    parser.add_argument("--save_dir", default="results/chengdu_agri_greenhouse_001/physics_calibration/")
    args = parser.parse_args()

    trajectories = pd.read_csv(args.trajectory_csv)
    params = load_parameter_vector(args.params_json)
    target_names = [target.strip() for target in args.targets.split(",") if target.strip()]
    metrics, predictions = evaluate_one_step_predictions(
        trajectories,
        parameter_vector=params,
        sample_limit=args.sample_limit,
        target_names=target_names,
        model_backend=args.model_backend,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "chengdu_physics_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(save_dir / "chengdu_physics_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
