# Chengdu Regional Yield Transfer Design

## Objective

Add a traceable regional evidence layer for the Chengdu tomato harvest model. The
layer may constrain external-transfer priors and test plausibility, but it must
never be treated as measured harvest from target greenhouse 63.

## Evidence hierarchy

1. A 2024 peer-reviewed Sichuan Academy of Agricultural Sciences study conducted
   in a multi-span plastic greenhouse in Pengzhou, Chengdu is the primary regional
   yield source. Its three treatments produced 10.1625-10.3140 kg/m2 during a
   January-July 2022 crop at about 2.943 plants/m2.
2. A 2023 Chinese Academy of Agricultural Sciences institute report from an AI
   greenhouse in Chongzhou, Chengdu is the primary regional timing source. It
   reports about 10 weeks from transplanting to first harvest and 18-20 kg/m2
   annual cherry-tomato yield. The yield is retained as secondary context because
   cultivar and production system differ from the target.
3. The existing WUR Autonomous Greenhouse Challenge data remain the source of
   observed batch-harvest shape, inter-pick interval, and uncertainty. They are
   not target-domain observations.
4. The 2025 peer-reviewed CSGtom validation paper supplies external error
   benchmarks only. It does not calibrate this greenhouse.

Every record carries source URL, publication type, location, facility, cultivar,
measurement semantics, transfer role, limitations, and `target_eligible=false`.

## Calibration

The regional timing source is transferred through thermal time rather than by
hard-coding a harvest day. For each deterministic baseline trajectory, accumulated
degree-days above 10 C are measured at 65, 70, and 75 crop days. Temperature is
capped at 30 C, following the published commercial-tomato GDD convention, so
extreme heat does not accelerate phenology without limit. The observed minimum
and maximum form the conservative maturity-threshold sampling interval, and 70
days is the central timing target. This preserves controller and season effects
on harvest timing while correcting the unrealistically early autumn harvests
produced by the WUR-only threshold.

The Pengzhou total yield is not fitted directly to a 120-day model. The six WUR
compartment curves are normalized by total harvest and time-warped from their
180-day crop to the 212-day January-July regional crop. Their cumulative fraction
at day 120 is multiplied by each of the three Pengzhou treatment totals. Quantiles
of these 18 combinations form a spring-only 120-day plausibility envelope. This
envelope is an external transfer assumption, not a synthetic observation and not
an autumn target.

No post-hoc multiplicative yield correction is permitted because it would break
the cohort dry-matter balance. Yield differences continue to arise from the
GreenLight fruit driver, maturity, picking, and fresh dry-matter conversion.

## Outputs

- `data/external/crops/chengdu_regional_tomato/source_registry.csv`
- `configs/crops/chengdu_regional_yield_transfer.yml`
- `data/processed/external/chengdu_regional_tomato/regional_transfer_priors.json`
- `regional_cohort_yield_scenarios.csv` and aggregate CSVs beside the controller
  benchmark
- `regional_yield_reasonableness.json` and a concise Markdown report

## Acceptance

- Source records remain externally scoped and target-ineligible.
- Maturity samples are positive, deterministic for a fixed seed, and derived only
  from complete baseline trajectories.
- The regional 120-day envelope is positive, ordered, and spring-only.
- Every projected scenario closes dry-matter mass balance below 1e-9 kg/m2.
- The audit reports onset error, spring envelope coverage, season and algorithm
  summaries, and explicit blockers to target validation.
