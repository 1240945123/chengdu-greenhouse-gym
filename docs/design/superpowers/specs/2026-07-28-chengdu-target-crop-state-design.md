# Chengdu Target Crop-State Validation Design

## Purpose

Initialize the GreenLight crop states from the dated Pidu plant samples and
evaluate simulated standing fruit before using the trajectory as a harvest
driver. This is a crop-development validation gate, not a substitute for true
picking records.

## Data Semantics

- Use the three Pidu samples dated 2026-03-30 as the baseline population.
- Aggregate replicate plants by date and report mean, sample standard
  deviation, standard error, and a small-sample 95% confidence interval.
- Estimate leaf and stem dry-matter fractions only from positive paired Xindu
  fresh/dry observations. Mark these conversions as source-to-target transfer.
- Set target fruit dry matter to zero when all baseline fresh-fruit
  measurements are observed zero. Do not infer fruit inventory from undated
  truss-yield cells.
- Do not map target root mass into GreenLight because the implemented crop
  model has no root-biomass state.

## Components

1. `processing/chengdu_crop_initialization.py` builds a validated crop-state
   initialization record in GreenLight units (`mg m-2`) and an audit report.
2. `GreenLightEnv.reset` accepts an optional `initial_crop_state_mg_m2`
   mapping for `cBuf`, `cLeaf`, `cStem`, `cFruit`, and `tCanSum`. The override
   is local to one reset and values must be finite and non-negative.
3. A standing-crop evaluator aligns model predictions to observation dates,
   reports MAE, RMSE, bias, WMAPE, R2 where defined, and per-date residuals.
4. The Chengdu target replay uses measured indoor climate, real 2026 outdoor
   weather, and recorded controls from 2026-04-01 onward. Results identify
   measured, transferred, simulated, and unavailable uncertainty sources
   separately. Coupled climate trajectories are not used for crop scoring
   when they violate physical temperature bounds or fail integration.

## Acceptance

- A reset with no override remains bit-for-bit compatible with current model
  initialization.
- The target baseline produces zero initial fruit and substantially smaller
  leaf/stem states than the mature GreenLight defaults.
- Evaluation never calls standing fruit a harvest event or marks target
  harvest validation complete.
- Tests cover invalid state values, replicate aggregation, time alignment,
  zero-observation WMAPE behavior, and report semantics.
