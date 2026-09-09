# Chengdu Quality-Aware Climate Data Design

## Objective

Improve the Chengdu greenhouse climate dataset without deleting physically plausible extreme heat. Preserve canonical temperature and RH values, add channel-aware confidence metadata, create transition-level training weights, and report model performance by sensor quality and heat regime.

## Rejected Alternatives

- Hard clipping or deleting values above 40 C is rejected because the July record is physically coherent with high radiation, no shade, no fan and no wet pad.
- Physics-model replacement of measured values is rejected because it would contaminate the observations used to validate that same model.
- Requiring two sensors for every training row is rejected because 1,383 of 1,787 train rows have only one effective temperature sensor.

## Hourly Quality Features

For indoor air temperature and relative humidity, use channel columns belonging to configured indoor equipment IDs. Channel coverage comes from the existing hourly `*_observed_fraction` columns. A channel is effective when at least 80% of the hour is directly observed. Interpolated values do not count as direct coverage.

For each variable record:

- available and effective channel counts;
- maximum effective-channel coverage;
- effective-channel disagreement range;
- quality tier and numeric quality weight.

Temperature tiers use disagreement limits 1 C for high and 3 C for medium. RH uses 6 and 10 percentage points. High requires at least two effective agreeing channels. Medium accepts one well-covered channel or multiple channels within the medium limit. Low covers insufficient coverage, missing channels or excessive disagreement. Weights are 1.0, 0.6 and 0.0.

## Transition Quality

Create a V4 trajectory bundle without overwriting V3. Every transition carries current and target temperature/RH quality metadata. Its training weight is the minimum of the four current/target variable weights. Identification eligibility additionally requires known roof/fan states under the existing control-quality policy.

No row is removed solely because it is single-sensor or extreme. Existing finite-measurement and one-hour continuity requirements remain unchanged.

## Heat Regimes

Label target temperature as:

- `normal`: below 35 C;
- `high`: 35 C to below 40 C;
- `extreme`: 40 C or above.

These labels are evaluation strata, not outlier filters. Extreme rows also record whether outdoor temperature is at least 30 C and radiation at least 500 W/m2 as physical-support context, without declaring the measurement true or false.

## Outputs And Audit

Write chronological train/validation/test CSVs, a manifest with hashes and quality counts, and a quality audit summarizing sensor support, extreme events and usable training weight. Generate stratified metrics from frozen validation/test rollouts where available. The primary aggregate test remains unchanged.

## Success Criteria

- Original canonical temperature/RH values and row counts are unchanged.
- High-temperature rows are retained.
- Single-sensor hours are distinguishable from verified multi-sensor hours.
- Target quality is included so a transition cannot be considered high quality only from its origin.
- Every threshold and source hash is recorded.
- Existing V3 artifacts and controller configuration remain untouched.

