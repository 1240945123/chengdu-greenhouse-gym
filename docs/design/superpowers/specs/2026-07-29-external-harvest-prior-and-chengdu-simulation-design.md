# External Harvest Prior and Chengdu Simulation Design

## Goal

Provide enough traceable tomato harvest data to run complete controller and
harvest-model experiments while the Pidu greenhouse has no recorded harvest
events. Public observations and simulations must never be presented as target
greenhouse validation evidence.

## Evidence Classes

Every row and artifact has exactly one evidence class:

- `target_observed`: measurements from greenhouse 63. These alone may satisfy
  target calibration and validation gates.
- `external_observed`: public measurements from another site, cultivar, or
  greenhouse. These may estimate priors and test ingestion code only.
- `simulated_prior`: model-generated Chengdu scenarios. These may train
  controllers, exercise the pipeline, and support sensitivity analysis only.

The production readiness assessor continues to require qualifying
`target_observed` harvest events from complete target seasons. Neither of the
other classes can set `target_validated=true`.

## Sources

### WUR Autonomous Greenhouse Challenge, Second Edition

- Publisher: 4TU.ResearchData / Wageningen University & Research.
- DOI: `10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7`.
- Article ID: `12764777`, version 2, CC0.
- Scope: six months of cherry tomato production in six Dutch high-tech
  glasshouse compartments, including climate, controls, irrigation, harvest,
  crop observations, quality, and resource consumption.
- Use: observed harvest cadence, yield-shape priors, dry-matter and management
  plausibility checks, and parser validation.
- Limitation: different country, greenhouse construction, cultivar, climate,
  and management; never copied into the Pidu identity or target split.

The downloader records the API metadata, URL, retrieval time, size, MD5, and
SHA-256. Raw files are immutable inputs under `data/external/crops/wur_agc2/`.

### Chengdu Outdoor Weather

The existing hourly ERA5-derived 2023-2025 data supply six Chengdu spring and
autumn weather seasons. Their existing local overlap correction and provenance
audits remain authoritative for simulation inputs. NASA POWER may be used as
an independent weather sensitivity source, not as indoor observations.

## Data Flow

1. Download and checksum the WUR archive through the public 4TU API.
2. Inspect the archive without assuming workbook names or columns.
3. Normalize public harvest records into an external schema retaining team,
   compartment, date, batch/cumulative mass semantics, area basis, source file,
   and evidence class.
4. Estimate source priors only when units and semantics are explicit. Store the
   estimate, interval, sample count, and source fields in a JSON prior artifact.
5. Run GreenLight with each of the six existing Chengdu weather seasons and a
   declared management protocol.
6. Generate a scenario ensemble over accepted uncertainty dimensions:
   weather season, greenhouse climate parameters, initial thermal sum,
   maturity thermal time, fruit dry-matter fraction, and picking schedule.
7. Write synthetic harvest events to a separate `simulated_prior` directory
   with synthetic greenhouse/planting identifiers and `target_eligible=false`.
8. Compare simulated distributions with WUR external observations and current
   Pidu standing-crop observations. Report plausibility diagnostics, not target
   harvest accuracy.

## Required Outputs

- Immutable WUR archive and source manifest.
- Archive inventory and normalization audit.
- Canonical external harvest observations and source-prior JSON.
- At least six Chengdu weather-season scenario groups with multiple parameter
  draws per season.
- Synthetic driver, harvest-event, season-summary, and uncertainty artifacts.
- A coverage matrix listing every missing target field and whether it is
  observed, externally informed, simulated, or still unavailable.

## Gates

- Checksums must match repository metadata before extraction.
- Unknown WUR columns, units, cumulative-versus-batch semantics, or area bases
  fail normalization instead of being guessed.
- Synthetic identifiers must not match target greenhouse 63 or its code.
- All synthetic harvest rows have `target_eligible=false` and
  `evidence_class=simulated_prior`.
- Calibration and readiness loaders reject external or synthetic rows as target
  evidence even if another field is edited incorrectly.
- No simulation output is written to the target raw harvest-event directory.
- Scenario generation uses fixed random seeds and records all sampled values.

## Validation

Tests cover metadata parsing, checksum failure, archive inventory, WUR harvest
normalization, evidence-class rejection, deterministic scenario generation,
mass balance, six-season coverage, and readiness remaining false. Acceptance
for this work means the experimental pipeline runs with plausible synthetic
distributions and complete provenance. It does not mean the Pidu harvest model
has been calibrated or validated.

## Deferred Target Evidence

Accurate Pidu harvest claims still require dated picking records with fresh
mass and area, confirmed management protocols, end-of-season evidence, and at
least two complete target seasons with a chronological held-out season. Public
or simulated data can reduce prior uncertainty but cannot replace these rows.
