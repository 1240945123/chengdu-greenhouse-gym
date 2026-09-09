# Chengdu SQL Control and Sensor Alignment Design

## Objective

Build a reproducible, read-only pipeline that extracts greenhouse 63 control commands from the Chengdu MySQL backup, reconstructs device-level command states, aligns them with minute sensor observations, and emits both one-minute and one-hour model datasets.

The pipeline must preserve real device semantics while also producing the normalized GreenLight actuator columns used by the existing PID, MPC, PPO, SAC, PINN, Transformer, and physics-model workflows.

## Scope

### Included

- Parse only controller definitions and operation logs needed for controller IDs 43 through 51.
- Reconstruct command states for shading, thermal screens, roof window, fan stages, lamp, wet-pad pump, and wet-pad roll film.
- Ignore heating and force `uBoil` to zero because the greenhouse has no heating system.
- Ignore CO2 control for experiments and force `uCO2` to zero, while retaining raw CO2-generator events only in the extracted event audit.
- Stream daily sensor CSV files for `greenhouseId=63` and align them by Asia/Shanghai local time.
- Produce device-level columns, GreenLight-compatible aggregate columns, quality flags, and a JSON quality report.
- Produce one-minute canonical data and one-hour derived data.

### Excluded

- Importing the SQL backup into MySQL.
- Copying the complete SQL backup into the repository.
- Exporting account, phone, password, SMS, or other personal data.
- Inventing actuator feedback, irrigation events, crop measurements, or physical parameters that were not observed.
- Claiming that reconstructed command states are measured actuator positions.

## Architecture

### `processing/chengdu_sql_controls.py`

- Read the SQL backup as UTF-8 without modifying it.
- Parse `equipment_controller_dev` and `equipment_controller_operator_log` INSERT statements.
- Reject malformed rows with an error that identifies the table and line number.
- Filter controller IDs 43 through 51 before constructing output records.
- Return normalized event columns such as log ID, timestamp, controller ID, device name, action, operator ID, and source content.
- Never parse or export unrelated sensitive tables.

### `processing/chengdu_control_states.py`

- Sort events by timestamp and log ID.
- Remove exact duplicate records while retaining repeated commands that have different log IDs or timestamps.
- Reconstruct one-minute device command states and uncertainty flags.
- Derive one-hour time-weighted command levels and known-state coverage.
- Generate GreenLight-compatible actuator proxies without discarding the device-level columns.

### `processing/build_chengdu_dataset.py`

- Stream or chunk daily sensor CSV files so the approximately 12 million raw rows are not loaded at once.
- Filter `greenhouseId=63` and identify channels by stable equipment and sensor IDs.
- Apply configured physical range checks and convert invalid values to missing values while counting them.
- Aggregate observations to one minute, derive robust indoor summaries, and interpolate only eligible short gaps.
- Join sensor and command tables on the minute timestamp.
- Derive the one-hour table from the aligned one-minute table.
- Write datasets and the quality report to the configured processed-data directory.

## Output Layout

```text
data/processed/chengdu_agri/greenhouse_001/
|-- controls/
|   |-- control_events.csv
|   |-- controls_1min.csv
|   `-- controls_1h.csv
|-- aligned/
|   |-- greenhouse_1min.csv
|   `-- greenhouse_1h.csv
`-- reports/
    `-- alignment_quality.json
```

The complete SQL backup remains outside the repository and is supplied with a command-line argument.

## Device State Model

| Controller | Device | Normalized command state |
|---|---|---|
| 43 | External shade | retracted `0`, deployed `1` |
| 44 | Roof thermal screen | retracted `0`, deployed `1` |
| 45 | Side thermal screen | retracted `0`, deployed `1` |
| 46 | Roof window | closed `0`, half-open `0.5`, fully open `1` |
| 47 | Wet-pad fan | off `0`, stages 1/2/3 as `1/3`, `2/3`, `1` |
| 48 | Supplemental lamp | off `0`, on `1` |
| 49 | CO2 generator | retained in event audit, excluded from model controls |
| 50 | Wet-pad pump | off `0`, on `1` |
| 51 | Wet-pad roll film | closed `0`, open `1` |

Before the first command, a numeric default of zero is emitted for compatibility, but the corresponding known-state flag is false. The value must not be described as an observed off state.

For motorized windows, screens, and roll film, `STOP` retains the previous commanded target and sets the device uncertainty flag. The flag remains set until a later explicit open, close, or supported partial-position command establishes a new target. This avoids inventing a physical position when motor travel time and position feedback are unavailable.

## Aggregate Controls

The output retains all real device columns and adds compatibility proxies:

- `uBoil = 0`.
- `uCO2 = 0` for the current experimental scope.
- `uVent = max(roof_window_level, fan_level_normalized)`.
- `uPad = min(fan_level_normalized, wet_pad_pump, wet_pad_roll_film)`.
- `uThScr` is the area-weighted average of the roof and side thermal-screen command states, using documented areas of 192 and 196 square metres.
- `uLamp = supplemental_lamp`.
- `uBlScr = external_shade`.

`uPad` is a Chengdu-specific feature and is added to the optional control schema. These proxies are model inputs, not measured actuator outputs.

## Sensor Processing

- Use timestamps in `Asia/Shanghai` and store output timestamps in an unambiguous ISO-like format.
- Filter records by `greenhouseId=63`.
- Preserve stable equipment and sensor IDs in the channel mapping instead of relying only on mutable display names.
- Keep individual indoor sensor columns where available.
- Generate robust indoor air-temperature and relative-humidity summaries using the median of valid indoor channels.
- Preserve outdoor temperature, humidity, radiation, wind speed, illumination, rainfall, soil temperature, soil moisture, soil conductivity, canopy infrared temperatures, and other available channels as separate features.
- Apply variable-specific plausible-range checks. Invalid samples become missing and remain counted in the quality report.
- Interpolate only internal gaps of at most five consecutive minutes. Do not extrapolate across dataset boundaries or fill longer outages.
- Emit observed/interpolated/missing flags so training code can filter or weight samples.

## Hourly Derivation

- Continuous sensor variables use the mean of valid minute values, with valid-minute coverage recorded.
- Robust indoor summary variables are recomputed consistently from valid minute data.
- Device command levels use a time-weighted mean, equivalent to hourly duty or average stage.
- Unknown-state and uncertainty coverage are reported for every device.
- An hourly row is retained even when incomplete; downstream trajectory generation decides whether to drop it based on explicit completeness thresholds.

## Quality Report

`alignment_quality.json` records:

- Input paths, file counts, and processing timestamps.
- SQL event count, event count by device and action, duplicate count, repeated-command count, and conflicting commands in the same minute.
- First and last command time per device.
- Unknown-state and uncertain-state coverage per device.
- Sensor row counts, valid rates, invalid-range counts, first and last timestamps, and longest missing gaps.
- Control and sensor overlap interval.
- One-minute and one-hour output row counts.
- Complete-row rates and interpolation rates.
- Chronological train, validation, and test boundaries when trajectory splits are generated.

## Error Handling

- Missing SQL tables or required columns cause a clear failure.
- Unsupported action types cause a clear failure unless explicitly configured as audit-only.
- Missing sensor identity mappings are reported rather than silently merged by display name.
- Duplicate timestamps use deterministic ordering based on timestamp and log ID.
- Output directories are created only under the configured processed-data root.
- Raw source files are never modified.

## Testing Strategy

Implementation follows test-driven development. Tests are written and observed failing before production code is added.

Unit tests cover:

1. SQL values containing `NULL`, escaped JSON, commas, and Chinese text.
2. Filtering that excludes account and other sensitive tables.
3. Roof-window half-open, fan stages, and `STOP` uncertainty behavior.
4. Unknown state before the first command.
5. Deterministic handling of repeated and same-minute commands.
6. Time-weighted controls spanning hour boundaries.
7. Sensor range filtering and interpolation limited to five minutes.
8. Robust aggregation across multiple indoor sensors.

An integration test uses a small synthetic SQL fixture and sensor fixture to verify event extraction, state reconstruction, minute alignment, hourly derivation, and quality-report generation end to end.

## Acceptance Criteria

- The parser extracts the expected 974 target-controller events from the supplied 2026-07-24 SQL backup before scope filtering.
- CO2 and heating model-control outputs are zero while CO2 events remain auditable.
- Roof-window half-open and fan stages survive in device-level and normalized outputs.
- No personal account, phone, password, or SMS data appears in generated files.
- One-minute and one-hour datasets cover the sensor/control overlap and contain explicit known, uncertain, observed, and interpolated flags.
- The quality report reconciles all input, rejected, and output records.
- Existing tests and all new pipeline tests pass.

## Known Limitations

- Commands prove that an instruction was issued, not that the actuator reached the requested state.
- `STOP` positions remain uncertain without travel-time calibration or position feedback.
- The aggregate GreenLight controls are documented proxies and do not replace device-specific features.
- Irrigation and crop-transpiration records remain unavailable.
- Greenhouse construction type and crop-density metadata conflicts must be resolved separately before publication.
