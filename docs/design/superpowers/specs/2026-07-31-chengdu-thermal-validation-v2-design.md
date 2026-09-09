# Chengdu Thermal Validation V2 Design

## Objective

Restore a scientifically defensible Chengdu greenhouse thermal-model workflow before controller ranking. The workflow must use measured outdoor disturbances, preserve chronological train/validation/test separation, and reject models that look acceptable for one step but become physically implausible in continuous simulation.

## Evidence And Root Cause

The aligned database contains 2,650 hourly rows from 2026-04-01 through 2026-07-20, including distinct indoor and outdoor temperature and relative humidity. The currently published trajectory splits contain only 743 transitions from 2026-04-05 through 2026-05-05. In those splits, `d_air_temperature == x_air_temperature` and `d_relative_humidity == x_relative_humidity` for every row because the trajectory builder silently substituted indoor measurements when outdoor columns were absent.

The original calibration then optimized one-step mean MAE on only the first 240 training rows. This makes envelope and ventilation heat exchange weakly identifiable and does not constrain accumulated solar energy. A 24-hour controller smoke test consequently reached about 95 C.

## Chosen Approach

Use a staged, evidence-first repair:

1. Make measured outdoor temperature and humidity mandatory for physics-calibration trajectories. Missing disturbances must fail loudly rather than copy indoor state.
2. Rebuild chronological trajectories from `aligned/greenhouse_1h.csv`, retaining only adjacent one-hour transitions with complete required measurements and control-quality metadata.
3. Audit train, validation, and test spans, missingness, indoor/outdoor equality, control coverage, and source hashes.
4. Evaluate the existing model on one-step and free-running 6/12/24/72-hour horizons. Calibrate on train, select solely on validation, and open test only once.
5. Require physical gates for finite states, plausible extrema, and continuous-rollout error. Do not use state clipping as evidence that a model is valid.
6. Introduce a versioned multi-node `ChengduPhysicsV2` only if the corrected-data evaluation shows the existing model cannot meet the gates. Keep the legacy backend available for reproducibility.

This is preferred over immediate coefficient retuning because it fixes the invalid experimental input first. It is preferred over a purely learned model because controller comparisons require stable extrapolation and interpretable energy flows beyond the short measured period.

## Data Contract

Required measured columns are timestamp, indoor temperature/RH, outdoor temperature/RH, global radiation, wind speed, and controls. Indoor substitution is prohibited. Consecutive rows must be exactly one hour apart. Rows with missing required values or unknown critical controls are excluded and counted in a manifest.

Splits are chronological 70/15/15 percent after valid-transition construction. There is no shuffling. The manifest records source SHA-256, row counts, date ranges, exclusion counts, column mappings, and split boundaries.

## Validation Gates

- Data integrity: outdoor fields exist, are non-null, and are not identically equal to indoor fields.
- Numerical stability: all predicted temperature and humidity values are finite.
- Physical envelope: continuous simulations must remain within -10 to 60 C and 0 to 100 percent RH without post-step clipping.
- Multistep reporting: temperature and RH MAE/RMSE/bias at 1, 6, 12, 24, and 72 hours, plus predicted/observed minima and maxima.
- Selection discipline: training data fits parameters; validation data chooses model/parameters; test data is report-only.
- Controller gate: PID/MPC/RL ranking remains blocked until the selected plant model passes the 24-hour and 72-hour physical gates on corrected disturbances.

The numerical accuracy thresholds will be reported against persistence and outdoor-following baselines. A model is not called accurate merely because it stays inside the physical envelope.

## Model V2 Boundary

If needed, `ChengduPhysicsV2` will use explicit air, canopy, floor/soil, and cover thermal capacities, with solar energy partitioned among nodes. Roof-window ventilation remains rain-sensitive, while infiltration and any documented side/fan exchange remain independent. Parameters are constrained to positive physically interpretable ranges. The environment selects the backend by configuration; legacy code and artifacts are not overwritten.

## Outputs

The stage produces versioned trajectory splits and manifest, one-step and multistep metrics, rollout CSV files, selected parameter metadata, plots, and an audit decision of GO or NO-GO. Every report includes data and model fingerprints.

