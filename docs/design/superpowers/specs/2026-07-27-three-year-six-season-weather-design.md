# Three-Year Six-Season Chengdu Weather Design

## Goal

Build an auditable outdoor-weather dataset for six 120-day greenhouse tomato
seasons across the three complete calendar years 2023 through 2025. Use the
dataset for controller training, validation, and held-out testing without
representing reanalysis weather as greenhouse observations.

## Source

Use the Open-Meteo Historical Weather API as the reproducible delivery layer
and explicitly request the ERA5 reanalysis model. Retrieve hourly:

- 2 m air temperature
- 2 m relative humidity
- 10 m wind speed
- shortwave global horizontal radiation
- precipitation
- shallow and deep soil temperature when available

Requests use the greenhouse-base coordinates and `Asia/Shanghai`. Preserve the
request URL, retrieval time, API/model name, coordinates, units, raw response
hash, and processed output hashes. If exact base coordinates cannot be
recovered, use an explicitly documented Pidu reference coordinate.

## Seasons

For each of 2023, 2024, and 2025, build:

- spring: March 1 through June 28, 120 days
- autumn: August 15 through December 12, 120 days

Each season contains 2,880 continuous hourly rows. The complete set contains
17,280 season rows. Full calendar-year source files are retained separately.

## GreenLight Conversion

Map ERA5 fields to the existing GreenLight weather schema. Keep timestamps as
seconds since January 1 so soil-temperature forcing remains calendar aligned.
Derive sky temperature with the repository's documented approximation, retain
constant outdoor CO2 at 400 ppm, and preserve the source day-of-year.

## Local Validation And Bias Correction

Compare the ERA5 series with the audited April-July 2026 outdoor observations
at matching hours. Produce raw validation metrics and a correction artifact.
Use conservative variable-specific corrections fitted only to the overlap:

- temperature: month-hour additive correction
- relative humidity: bounded month-hour additive correction
- radiation: daylight-only multiplicative correction with night fixed at zero
- wind: non-negative quantile or robust scale correction

Do not extrapolate unsupported month-specific corrections into autumn. Use a
global robust correction for months absent from the local overlap and label it
as lower confidence. Store both uncorrected and corrected statistics.

## Experiment Split

- training: spring and autumn 2023-2024, four seasons
- validation: spring 2025
- test: autumn 2025
- external local check: the observed 2026 greenhouse weather interval

Synthetic augmentation may draw only from the four training seasons. Validation,
test, and 2026 local-check weather are never synthetic-source data.

## Layout

```text
data/external/weather/era5/chengdu_greenhouse_001/
  raw/
  manifest.json
data/processed/chengdu_agri/greenhouse_001/weather_era5/
  full_years/
  seasons/
  quality_report.json
  bias_correction.json
```

## Failure Handling

Use request timeouts and bounded retries. Download to a temporary file on the
workspace drive, validate JSON and row counts, then atomically publish outputs.
Never silently interpolate missing hours. Abort with a precise list of missing
or duplicated timestamps.

## Verification

Tests cover API parsing, units, leap-year row counts, calendar-aligned time,
season boundaries, 2,880-row season lengths, physical bounds, deterministic
correction, provenance hashes, and strict split isolation. A benchmark smoke
run must load all configured years and evaluate the held-out autumn 2025 season.

## Scientific Scope

ERA5 is reanalysis outdoor forcing, not site observations. The six seasons
increase weather diversity but do not create measured actuator behavior,
greenhouse states, irrigation, transpiration, crop biomass, or harvested yield.
