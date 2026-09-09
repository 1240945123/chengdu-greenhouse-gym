# Chengdu SQL Control and Sensor Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, read-only pipeline that extracts greenhouse 63 control commands from the SQL backup and aligns device-level and GreenLight-compatible controls with minute sensor observations, producing one-minute and one-hour datasets plus a quality report.

**Architecture:** A focused SQL event parser feeds a device-state reconstruction module. A separate chunked sensor processor creates minute observations, and an orchestration module joins both streams, derives hourly data, and writes auditable outputs without copying sensitive source data into the repository.

**Tech Stack:** Python 3.12, standard library, pandas, NumPy, PyYAML, unittest.

---

## File Map

- Create `processing/chengdu_sql_controls.py`: safe SQL INSERT parsing and target controller-event extraction.
- Create `processing/chengdu_control_states.py`: device state reconstruction, uncertainty flags, aggregate controls, and hourly control summaries.
- Create `processing/chengdu_sensor_stream.py`: chunked long-format sensor reading, filtering, minute pivoting, range checks, interpolation, and sensor quality metrics.
- Create `processing/build_chengdu_dataset.py`: command-line orchestration, minute alignment, hourly derivation, output writing, and report assembly.
- Create `tests/test_chengdu_sql_controls.py`: SQL parser tests.
- Create `tests/test_chengdu_control_states.py`: action semantics and hourly weighting tests.
- Create `tests/test_chengdu_sensor_stream.py`: sensor filtering, pivoting, interpolation, and aggregation tests.
- Create `tests/test_chengdu_dataset_integration.py`: end-to-end fixture test and sensitive-data exclusion test.
- Modify `processing/chengdu_controls.py`: consume the new canonical device states while retaining the existing public functions.
- Modify `configs/datasets/chengdu_agri_greenhouse_001.yml`: source identifiers, sensor ranges, interpolation limit, and new processed paths.
- Modify `data/schemas/control_schema.yml`: device-level controls, `uPad`, and quality flags.
- Modify `data/schemas/trajectory_schema.yml`: optional Chengdu device-state and quality columns.
- Modify `data/raw/chengdu_agri/greenhouse_001/metadata.yml`: replace template status with audited source coverage and known limitations.
- Modify `data/README.md`: exact pipeline command and output description.

The current folder has no `.git` directory. Each task therefore ends with a verification checkpoint rather than a commit. When Git metadata is restored, commit each completed task separately using the message shown at its checkpoint.

### Task 1: Safe SQL Controller Event Parser

**Files:**
- Create: `processing/chengdu_sql_controls.py`
- Create: `tests/test_chengdu_sql_controls.py`

- [ ] **Step 1: Write failing parser tests**

Create a temporary SQL fixture with controller definitions, escaped JSON, `NULL`, a comma inside a quoted string, and an account row. Assert that only IDs 43 through 51 are returned and that source content is decoded without exposing account data.

```python
class ChengduSqlControlsTest(unittest.TestCase):
    def test_extract_target_control_events_handles_null_and_escaped_json(self):
        sql = """
INSERT INTO `account` VALUES (1, NULL, 'private');
INSERT INTO `equipment_controller_dev` VALUES (43, NULL, NULL, 0, 0, 0, 3037, 'dev', 'code', '外遮阳', '遮阳网', NULL, NULL, NULL, NULL, NULL, 0, 0, NULL, NULL, NULL, 'pidu1', 'open', 'close', 'stop', NULL, NULL, '10');
INSERT INTO `equipment_controller_operator_log` VALUES (100, '2026-04-02 15:12:05', NULL, 0, 0, 0, 43, 'TURN_ON', NULL, '{\"id\":43,\"note\":\"a,b\"}', 2);
"""
        path = self.write_fixture(sql)
        events = extract_control_events(path)
        self.assertEqual(events.loc[0, "controller_id"], 43)
        self.assertEqual(events.loc[0, "action"], "TURN_ON")
        self.assertIn('"note":"a,b"', events.loc[0, "source_content"])
        self.assertNotIn("private", events.to_csv(index=False))

    def test_extract_target_control_events_rejects_unsupported_action(self):
        path = self.write_fixture(self.log_row(controller_id=46, action="TURN_ON_9"))
        with self.assertRaisesRegex(ValueError, "TURN_ON_9"):
            extract_control_events(path)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chengdu_sql_controls -v
```

Expected: import failure for missing `processing.chengdu_sql_controls`.

- [ ] **Step 3: Implement the minimal parser**

Implement these public interfaces exactly:

- `TARGET_CONTROLLER_IDS = frozenset(range(43, 52))`
- `SUPPORTED_ACTIONS = {"TURN_ON", "TURN_OFF", "STOP", "TURN_ON_2", "TURN_ON_4"}`
- `split_mysql_values(values_text: str) -> list[str | None]`
- `extract_control_events(sql_path: str | Path) -> pd.DataFrame`

Read line by line using UTF-8 with BOM support. Match only `equipment_controller_dev` and `equipment_controller_operator_log`; do not build a general database dump in memory. Return columns:

```python
[
    "log_id", "timestamp", "controller_id", "device",
    "action", "params", "source_content", "operator_id"
]
```

Sort by `timestamp`, then `log_id`. Raise a line-numbered `ValueError` for malformed target rows and unsupported target actions.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 command. Expected: all parser tests pass.

- [ ] **Step 5: Verify against the real backup**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from processing.chengdu_sql_controls import extract_control_events; x=extract_control_events(r'E:\school\NKY\玻璃温室数据库信息\tomato_full_backup_20260724.sql'); print(len(x)); print(x.groupby(['controller_id','action']).size())"
```

Expected: `974` total target events; output includes `TURN_ON_2` for controller 46 and `TURN_ON_2`/`TURN_ON_4` for controller 47.

- [ ] **Step 6: Checkpoint**

Record test output. Future commit message: `feat: parse Chengdu controller events safely`.

### Task 2: Device State Reconstruction

**Files:**
- Create: `processing/chengdu_control_states.py`
- Create: `tests/test_chengdu_control_states.py`

- [ ] **Step 1: Write failing state tests**

Test command mappings, unknown prehistory, partial settings, and `STOP` uncertainty.

```python
def test_roof_window_half_open_and_stop_marks_uncertain():
    events = event_frame([
        (1, "2026-04-01 00:01:00", 46, "TURN_ON_2"),
        (2, "2026-04-01 00:03:00", 46, "STOP"),
        (3, "2026-04-01 00:05:00", 46, "TURN_OFF"),
    ])
    index = pd.date_range("2026-04-01 00:00", periods=7, freq="min")
    states = reconstruct_minute_controls(events, index)
    self.assertEqual(states["roof_window"].tolist(), [0, .5, .5, .5, .5, 0, 0])
    self.assertEqual(states["roof_window_known"].tolist(), [False, True, True, True, True, True, True])
    self.assertEqual(states["roof_window_uncertain"].tolist(), [False, False, False, True, True, False, False])

def test_fan_stages_are_preserved():
    events = event_frame([
        (1, "2026-04-01 00:00:00", 47, "TURN_ON"),
        (2, "2026-04-01 00:01:00", 47, "TURN_ON_2"),
        (3, "2026-04-01 00:02:00", 47, "TURN_ON_4"),
        (4, "2026-04-01 00:03:00", 47, "TURN_OFF"),
    ])
    states = reconstruct_minute_controls(events, pd.date_range("2026-04-01", periods=4, freq="min"))
    np.testing.assert_allclose(states["fan"], [1/3, 2/3, 1, 0])
```

- [ ] **Step 2: Run state tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chengdu_control_states -v
```

Expected: import failure for missing state module.

- [ ] **Step 3: Implement mappings and minute reconstruction**

Define immutable device specifications for IDs 43 through 51 and implement `reconstruct_minute_controls(events: pd.DataFrame, minute_index: pd.DatetimeIndex) -> pd.DataFrame` and `add_aggregate_controls(states: pd.DataFrame) -> pd.DataFrame`.

Use deterministic `(timestamp, log_id)` ordering. Exact duplicate rows may be removed, but repeated commands with unique log IDs remain auditable. Numeric state before the first command is zero with `<device>_known=False`. `STOP` changes uncertainty, not the last target level. Controllers 48 and 49 accept their recorded `TURN_OFF` events even though device definitions have no explicit off command string.

Add aggregate columns with:

```python
uBoil = 0.0
uCO2 = 0.0
uVent = maximum(roof_window, fan)
uPad = minimum(minimum(fan, wet_pad_pump), wet_pad_roll_film)
uThScr = (192.0 * roof_thermal_screen + 196.0 * side_thermal_screen) / 388.0
uLamp = supplemental_lamp
uBlScr = external_shade
```

- [ ] **Step 4: Run state tests and verify GREEN**

Expected: all state and aggregate mapping tests pass.

- [ ] **Step 5: Checkpoint**

Future commit message: `feat: reconstruct Chengdu device command states`.

### Task 3: Hourly Time-Weighted Controls

**Files:**
- Modify: `processing/chengdu_control_states.py`
- Modify: `tests/test_chengdu_control_states.py`

- [ ] **Step 1: Write failing hourly tests**

```python
def test_hourly_controls_use_minute_weighted_mean_and_quality_coverage():
    minute = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01 00:00", periods=60, freq="min"),
            "fan": [0.0] * 30 + [1.0] * 30,
            "fan_known": [False] * 10 + [True] * 50,
            "fan_uncertain": [False] * 40 + [True] * 20,
        }
    )
    hourly = derive_hourly_controls(minute)
    self.assertAlmostEqual(hourly.loc[0, "fan"], 0.5)
    self.assertAlmostEqual(hourly.loc[0, "fan_known_fraction"], 50 / 60)
    self.assertAlmostEqual(hourly.loc[0, "fan_uncertain_fraction"], 20 / 60)
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: missing `derive_hourly_controls`.

- [ ] **Step 3: Implement hourly derivation**

```python
def derive_hourly_controls(minute_controls: pd.DataFrame) -> pd.DataFrame:
    """Return hourly mean levels and known/uncertain minute fractions."""
```

Require a unique minute timestamp. Aggregate levels by arithmetic mean because each row represents one equal-duration minute. Aggregate boolean quality columns as fractions. Recompute aggregate GreenLight controls after hourly device aggregation rather than averaging a precomputed maximum.

- [ ] **Step 4: Run all control-state tests and verify GREEN**

Run Task 2's full test command. Expected: all pass.

- [ ] **Step 5: Checkpoint**

Future commit message: `feat: derive hourly Chengdu control levels`.

### Task 4: Chunked Sensor Stream Processing

**Files:**
- Create: `processing/chengdu_sensor_stream.py`
- Create: `tests/test_chengdu_sensor_stream.py`
- Modify: `configs/datasets/chengdu_agri_greenhouse_001.yml`

- [ ] **Step 1: Write failing sensor tests**

Create fixture rows using the real raw columns:

```python
RAW_COLUMNS = [
    "_id", "devCode", "paramId", "timeString", "paramName", "paramType",
    "code", "paramCode", "paramNo", "paramVal", "booleanIndoor", "paramUnit",
    "ctime", "ip", "topic", "plantingAreaId", "equipmentSensorId",
    "equipmentId", "greenhouseId", "extraFieldsJson",
]

def test_clean_sensor_chunk_filters_greenhouse_and_ranges():
    raw = sensor_rows([
        (63, 3036, 1500, "AIR_TEMPERATURE", "2026-04-01 00:00:00:000", 18.0, False),
        (63, 3044, 1558, "AIR_HUMIDITY", "2026-04-01 00:00:00:000", 125.0, True),
        (1, 3013, 1322, "AIR_TEMPERATURE", "2026-04-01 00:00:00:000", 20.0, False),
    ])
    clean, quality = clean_sensor_chunk(raw, greenhouse_id=63, ranges=SENSOR_RANGES)
    self.assertEqual(len(clean), 2)
    self.assertTrue(np.isnan(clean.loc[clean.paramType.eq("AIR_HUMIDITY"), "value"]).all())
    self.assertEqual(quality["invalid_range"], 1)
```

Also test two indoor temperature sensors produce individual columns and a median, and that a five-minute internal gap is interpolated while a six-minute gap remains missing.

- [ ] **Step 2: Run sensor tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chengdu_sensor_stream -v
```

Expected: import failure for missing sensor module.

- [ ] **Step 3: Add explicit dataset configuration**

Extend the Chengdu dataset YAML with:

```yaml
source:
  greenhouse_id: 63
  timezone: Asia/Shanghai
  sql_path_env: CHENGDU_SQL_BACKUP
  sensor_csv_glob: "**/*.csv"
processing:
  chunksize: 250000
  interpolation_limit_minutes: 5
  indoor_equipment_ids: [3038, 3044]
  outdoor_equipment_ids: [3036]
sensor_ranges:
  AIR_TEMPERATURE: [-30.0, 60.0]
  AIR_HUMIDITY: [0.0, 100.0]
  CO2: [250.0, 5000.0]
  WIND_SPEED: [0.0, 50.0]
  PHOTO_ELECTRIC_RADIATION: [0.0, 2500.0]
  ILLUMINATION: [0.0, 200000.0]
  SOIL_TEMPERATURE: [-10.0, 60.0]
  SOIL_HUMIDITY: [0.0, 100.0]
```

Keep ranges configurable and report them in output metadata.

- [ ] **Step 4: Implement chunk cleaning and minute aggregation**

Implement these exact interfaces:

- `clean_sensor_chunk(raw: pd.DataFrame, greenhouse_id: int, ranges: Mapping[str, tuple[float, float]]) -> tuple[pd.DataFrame, dict[str, int]]`
- `read_sensor_files(sensor_root: str | Path, greenhouse_id: int, ranges: Mapping[str, tuple[float, float]], chunksize: int = 250_000) -> tuple[pd.DataFrame, dict[str, object]]`
- `build_minute_sensor_table(clean_long: pd.DataFrame, interpolation_limit: int = 5) -> tuple[pd.DataFrame, dict[str, object]]`

Call `pd.read_csv` with the explicit 20-column raw schema, the configured chunk size, and only the columns needed for filtering and aggregation. Parse `timeString` with the millisecond suffix normalized. Build stable column names from lower-case `paramType`, `equipmentId`, and `equipmentSensorId`. Add robust indoor medians named `air_temperature`, `relative_humidity`, and, where available, `co2_concentration`. Add `<variable>_observed` and `<variable>_interpolated` flags. Interpolate internal gaps only with `limit=5`, `limit_area="inside"`, and prevent a six-minute run from being partially filled by identifying full missing runs before interpolation.

- [ ] **Step 5: Run sensor tests and verify GREEN**

Run Task 4's test command. Expected: all pass.

- [ ] **Step 6: Checkpoint**

Future commit message: `feat: stream and clean Chengdu sensor data`.

### Task 5: End-to-End Alignment and Quality Report

**Files:**
- Create: `processing/build_chengdu_dataset.py`
- Create: `tests/test_chengdu_dataset_integration.py`

- [ ] **Step 1: Write a failing integration test**

Use temporary SQL and daily sensor CSV fixtures. Assert exact output paths, minute and hourly row counts, nonzero roof-window and fan controls, explicit flags, and absence of a private marker placed in an account row.

```python
def test_build_dataset_writes_aligned_outputs_and_report(self):
    outputs = build_dataset(
        sql_path=self.sql_path,
        sensor_root=self.sensor_root,
        output_root=self.output_root,
        config=self.config,
    )
    minute = pd.read_csv(outputs["aligned_1min"])
    report = json.loads(Path(outputs["quality_report"]).read_text(encoding="utf-8"))
    self.assertIn("roof_window", minute.columns)
    self.assertIn("uVent", minute.columns)
    self.assertIn("air_temperature_observed", minute.columns)
    self.assertEqual(report["controls"]["target_events"], 4)
    for output in outputs.values():
        self.assertNotIn("private-marker", Path(output).read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run integration test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chengdu_dataset_integration -v
```

Expected: import failure for missing build module.

- [ ] **Step 3: Implement alignment and reporting**

Implement `build_dataset(sql_path: str | Path, sensor_root: str | Path, output_root: str | Path, config: Mapping[str, object]) -> dict[str, Path]`, `derive_hourly_dataset(minute: pd.DataFrame) -> pd.DataFrame`, and `main() -> None`.

Build the canonical minute index from the sensor coverage. Left-join reconstructed controls to sensor minutes. Write:

```text
controls/control_events.csv
controls/controls_1min.csv
controls/controls_1h.csv
aligned/greenhouse_1min.csv
aligned/greenhouse_1h.csv
reports/alignment_quality.json
```

The report must reconcile raw event count, target event count, action counts, exact duplicates, repeated commands, same-minute conflicts, device first/last times, unknown and uncertain fractions, sensor valid/invalid/interpolated counts, longest gaps, overlap interval, and output row counts.

Add CLI arguments:

```text
--sql
--sensor_root
--dataset_config
--output_root
```

- [ ] **Step 4: Run integration test and verify GREEN**

Expected: test passes and all generated fixture files exclude the private marker.

- [ ] **Step 5: Checkpoint**

Future commit message: `feat: align Chengdu controls and sensors`.

### Task 6: Integrate Existing Control and Trajectory APIs

**Files:**
- Modify: `processing/chengdu_controls.py`
- Modify: `tests/data_pipeline.py`
- Modify: `data/schemas/control_schema.yml`
- Modify: `data/schemas/trajectory_schema.yml`

- [ ] **Step 1: Write failing compatibility tests**

Add a test that passes canonical device-state columns to a new `build_hourly_controls_from_states(states: pd.DataFrame) -> pd.DataFrame` helper and verifies that `uVent`, `uPad`, `uThScr`, `uLamp`, and `uBlScr` survive. Add a trajectory test confirming optional device columns do not alter required existing trajectory columns.

- [ ] **Step 2: Run existing data-pipeline tests and verify RED for the new behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.data_pipeline -v
```

Expected: new compatibility assertions fail because `uPad` and device-level inputs are not supported.

- [ ] **Step 3: Update compatibility code and schemas**

Repair the mojibake device labels in `processing/chengdu_controls.py` and its tests. Preserve `normalize_device_events`, `build_hourly_controls`, and `merge_controls_with_hourly_wide` for older Excel/CSV workflows. Implement `build_hourly_controls_from_states` by validating the timestamp and device-state columns, calling `derive_hourly_controls`, and then calling `add_aggregate_controls`. Include `uPad` as an optional Chengdu control rather than adding it to the six mandatory GreenLight controls.

Extend `control_schema.yml` with device-level columns and known/uncertain flags. Extend `trajectory_schema.yml` only with optional Chengdu fields so Amsterdam experiments remain compatible.

- [ ] **Step 4: Run pipeline and new module tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.data_pipeline tests.test_chengdu_sql_controls tests.test_chengdu_control_states tests.test_chengdu_sensor_stream tests.test_chengdu_dataset_integration -v
```

Expected: all pass.

- [ ] **Step 5: Checkpoint**

Future commit message: `refactor: integrate canonical Chengdu controls`.

### Task 7: Documentation and Audited Metadata

**Files:**
- Modify: `data/raw/chengdu_agri/greenhouse_001/metadata.yml`
- Modify: `data/README.md`

- [ ] **Step 1: Add a failing documentation assertion**

Add to the integration test:

```python
def test_data_readme_documents_external_sql_and_sensor_inputs():
    text = Path("data/README.md").read_text(encoding="utf-8")
    self.assertIn("build_chengdu_dataset", text)
    self.assertIn("tomato_full_backup_20260724.sql", text)
    self.assertIn("command states are not actuator feedback", text)
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: documentation strings are missing.

- [ ] **Step 3: Document exact execution and limitations**

Add the production command:

```powershell
.\.venv\Scripts\python.exe -m processing.build_chengdu_dataset `
  --sql "E:\school\NKY\玻璃温室数据库信息\tomato_full_backup_20260724.sql" `
  --sensor_root "E:\school\NKY\玻璃温室数据库信息\tomato_2026-04-01_to_2026-07-20_csv" `
  --dataset_config configs/datasets/chengdu_agri_greenhouse_001.yml `
  --output_root data/processed/chengdu_agri/greenhouse_001
```

Update metadata to record greenhouse ID 63, sensor coverage 2026-04-01 through 2026-07-20, SQL dump date 2026-07-24, command-not-feedback limitation, no heating/CO2 experimental scope, unresolved glass-versus-plastic metadata conflict, and no actual irrigation/crop-response records.

- [ ] **Step 4: Run documentation and integration tests**

Expected: pass.

- [ ] **Step 5: Checkpoint**

Future commit message: `docs: document Chengdu aligned dataset pipeline`.

### Task 8: Real Data Build and Full Verification

**Files:**
- Generate: `data/processed/chengdu_agri/greenhouse_001/controls/*.csv`
- Generate: `data/processed/chengdu_agri/greenhouse_001/aligned/*.csv`
- Generate: `data/processed/chengdu_agri/greenhouse_001/reports/alignment_quality.json`

- [ ] **Step 1: Run the complete focused test suite**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chengdu_sql_controls tests.test_chengdu_control_states tests.test_chengdu_sensor_stream tests.test_chengdu_dataset_integration tests.data_pipeline -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run the pipeline on the real external sources**

Use the documented Task 7 command. Expected outputs are created only under `data/processed/chengdu_agri/greenhouse_001/`.

- [ ] **Step 3: Verify acceptance counts and privacy**

Run a read-only verification script that asserts:

```python
assert report["controls"]["target_events"] == 974
assert report["controls"]["by_device"]["46"]["TURN_ON_2"] == 80
assert report["controls"]["by_device"]["47"]["TURN_ON_4"] == 99
assert report["outputs"]["aligned_1min_rows"] > 100_000
assert report["outputs"]["aligned_1h_rows"] > 2_000
```

Search generated text files for account table names, known phone-field labels, `password`, and `sms_history`; expected result is no matches.

- [ ] **Step 4: Run existing repository tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "*.py" -v
```

If discovery does not load legacy non-`test_` modules, also run the repository's previously used explicit test commands. Report any pre-existing failure separately from new pipeline failures.

- [ ] **Step 5: Inspect the quality report**

Confirm sensor/control overlap, known-state coverage, uncertain-state coverage, invalid-range counts, long gaps, interpolation counts, and final row counts. Do not describe the build as suitable for training until these values are reviewed.

- [ ] **Step 6: Final checkpoint**

Future commit message: `data: build aligned Chengdu greenhouse dataset`.

## Execution Notes

- Do not install dependencies or create caches on the C: drive. Use the existing repository `.venv`, `.cache`, `.pip-cache`, and `.tmp` locations on E:.
- Do not copy the full SQL backup into `data/raw` because it contains personal and authentication-related records.
- Do not modify the 111 daily source CSV files or the SQL backup.
- Generated datasets may be large; inspect disk space before the real build and write incrementally where practical.
- Every production function is introduced only after its corresponding test has been observed failing for the intended reason.
