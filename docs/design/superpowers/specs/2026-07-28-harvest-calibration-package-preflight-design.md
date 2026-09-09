# Harvest Calibration Package Preflight Design

## Purpose

Add a fast, standalone check for the complete measured-harvest calibration
package before the expensive parameter and bootstrap runs. The check combines
the canonical Chengdu harvest workbook with a GreenLight harvest-driver CSV and
uses the same target identity and minimum-coverage rules as production.

## Interface

The command is:

```powershell
python -m experiments.crop.validate_chengdu_harvest_package `
  --harvest-workbook <harvest.xlsx> `
  --drivers <drivers.csv> `
  --output-root <directory>
```

The Python API is `run_package_preflight(...) -> dict`. It loads repository
dataset and harvest configuration by default and accepts explicit paths for
tests and reproducible experiments.

## Validation Model

Canonical harvest loaders and season binding remain authoritative for measured
data. A shared, non-fitting production validator checks driver and event
identity, strict booleans, source declarations, fixed intervals, temporal
coverage, initial inventory, planting identity, complete seasons, minimum
90-day driver span, minimum eight harvest events, minimum 30-day harvest span,
and driver coverage of every harvest event.

Structural contract errors raise `ValueError`. Valid files that are not yet
eligible produce a report with season-level blocking reasons. Ongoing seasons
remain visible but cannot enter calibration. Only seasons present and
qualifying in both sources enter the chronological split.

## Outputs And Semantics

The command writes `harvest_calibration_package_preflight.json`. The report
contains:

- target greenhouse identity and fixed normalization area;
- event, manifest-season, and driver-season counts;
- a season readiness table with measured and driver coverage;
- qualifying complete seasons and chronological train/validation/test split;
- `calibration_package_ready` for at least one qualifying season;
- `independent_validation_package_ready` for at least two qualifying seasons;
- global blocking reasons and artifact paths.

Both readiness flags describe input-package coverage only. The report always
sets `harvest_parameters_fitted=false` and `target_harvest_validated=false`.
Actual model acceptance still requires untouched held-out metrics and the
configured uncertainty gates.

## Testing

Tests build real XLSX and CSV inputs. A three-season valid package must return
the production chronological split. A package with an ongoing season and a
diagnostic driver model must retain both blockers and must not claim readiness.
Production calibration tests continue to prove that accepted packages fit and
score only held-out seasons.
