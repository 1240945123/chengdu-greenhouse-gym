# Three-Month Weather And Training Augmentation Design

## Goal

Rebuild the Chengdu GreenLight weather input from the complete audited April 1 through July 20 hourly table and add reproducible synthetic training weather without treating generated samples as observations.

## Real Weather

The source of truth is `aligned/greenhouse_1h.csv`, not the older 744-hour wide table. Outdoor radiation, wind speed, air temperature, and relative humidity map explicitly to GreenLight weather columns. Sky temperature is outdoor temperature minus 6 deg C and outdoor CO2 is fixed at 400 ppm because the controller benchmark disables CO2 control and the loader already assumes 400 ppm.

The real output retains all 2,650 hourly rows. Benchmark metadata uses 110 complete days. Four-day training episodes start on days 0 through 65, validation starts on day 74, and the held-out real test starts on day 105. These intervals do not overlap.

## Synthetic Training Weather

Synthetic weather uses multivariate moving-block bootstrap at whole-day boundaries:

- Convert only the real training interval, days 0 through 69, into 24-hour daily blocks.
- Sample consecutive 3-day blocks with replacement until 110 synthetic days are assembled.
- Blend each sampled-block join over six hours to avoid artificial temperature and humidity jumps.
- Keep timestamps aligned to the real April calendar position used by the soil-temperature model.
- Apply bounded, smoothly varying daily perturbations to temperature, RH, radiation, and wind while preserving all variables within each sampled hour as a joint vector.
- Keep radiation and wind nonnegative, RH in 0-100%, and outdoor temperatures within configured physical limits.
- Generate deterministic files from explicit seeds and write a provenance manifest containing source hash, method, parameters, and output hashes.

Synthetic years are training-only and may not sample any source block from the
real validation or test intervals. Validation and test always use real 2026
weather. The implementation initially creates ten synthetic weather years and
exposes them to PPO/SAC through the existing random weather sampler. Classical
controllers remain tuned on the real validation interval.

## Why This Method

Moving-block bootstrap preserves short-range temporal dependence and multivariate relationships better than independent row shuffling. A learned GAN or diffusion generator is not selected because 110 days are too few for reliable tail-distribution learning and validation. Simulator parameter randomization remains complementary and can be added after the weather augmentation experiment is measured.

## Outputs

- Real weather: `data/processed/chengdu_agri/greenhouse_001/weather/Chengdu/2026.csv`
- Synthetic weather: numeric year files under the same location directory.
- Provenance: `data/processed/chengdu_agri/greenhouse_001/weather/synthetic_weather_manifest.json`
- Configuration: the benchmark records real evaluation year and training weather years separately.

## Validation

Tests cover explicit outdoor-column mapping, hourly continuity, physical bounds, deterministic seeds, different-seed diversity, preservation of day blocks, provenance, non-overlapping benchmark windows, and training-only use of synthetic years. A smoke PPO/SAC run verifies environment loading and model training before any paper-scale retraining.
