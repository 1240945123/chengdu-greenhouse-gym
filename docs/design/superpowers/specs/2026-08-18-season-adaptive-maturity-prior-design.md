# Season-Adaptive Maturity Prior Design

## Objective

Improve the existing 120-day Chengdu tomato yield projection without extending
the experiment horizon. Correct the cross-season maturity bias while preserving
the existing controller trajectories, external-evidence labels, and cohort dry
matter balance.

## Problem

The v1 regional prior pools degree-day thresholds from all six seasons into one
642-1122 degree-day interval. A sampled autumn threshold can therefore be used
for a spring episode and vice versa. This makes spring harvest systematically
late and autumn harvest early even though the external timing target is the same
65-75 crop-day interval.

## Design

The calibration artifact will retain the global interval as a compatibility
fallback and add one ordered low, central, and high maturity interval per season.
Each interval is derived from that season's complete baseline temperature
trajectory at crop days 65, 70, and 75, with a 10 C base and 30 C upper cutoff.

The scenario sampler will use the matching season interval when present. Older
prior payloads without season intervals will continue to use the global bounds.
No yield multiplier, controller trajectory modification, heating, or CO2 control
is introduced.

V2 projections will use a distinct `regional_v2_` artifact prefix so the v1
projection remains available for direct sensitivity analysis. The audit will
report first-harvest MAE, bias, and the fraction of seasons whose projected first
harvest falls in the externally supported 65-75 day window, alongside the
existing spring yield-envelope coverage and mass-balance checks.

## Data Flow

1. Read the six complete deterministic baseline controller trajectories.
2. Derive season-specific degree-day bounds at crop days 65, 70, and 75.
3. Store both season-specific bounds and the legacy global fallback.
4. Sample each episode only from its matching season maturity interval.
5. Re-run the cohort projection over the unchanged 78 controller episodes.
6. Audit v2 outputs and compare them with preserved v1 outputs.

## Acceptance Criteria

- Every canonical season has a positive ordered maturity interval.
- Projection loading rejects missing or unknown season mappings when a seasonal
  prior is declared.
- Legacy global-only prior payloads remain supported.
- V2 produces 1,248 scenarios for 78 episodes and closes dry matter balance below
  `1e-9 kg/m2`.
- First-harvest timing metrics include bias, MAE, and 65-75 day coverage.
- V1 artifacts are not overwritten.
- Evidence remains `external_transfer_prior`, `target_eligible=false`, and the
  result remains regional plausibility rather than target-site validation.

