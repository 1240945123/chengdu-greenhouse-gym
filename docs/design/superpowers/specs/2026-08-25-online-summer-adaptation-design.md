# Online Summer Adaptation Design

## Objective

Test whether chronological online adaptation can reduce extreme-heat error after the first measured summer heat event, without using future rows in model fitting.

## Protocol

Use the already fixed `thermal_dynamics`, Ridge alpha 0.1, residual gain 1.0, and heat weights `normal=1`, `high=5`, `extreme=5`. Hyperparameters are not re-selected.

Temperature feedback gain remains fixed at 1.0. Because the first replay shows a target trade-off, the calibration day may choose the humidity feedback gain from the predeclared set `[0, 0.25, 0.5, 0.75, 1]` using one-step humidity MAE. This changes no fitted coefficient and does not inspect evaluation rows.

- Historical fit: V4 train and validation plus test transitions whose target time is before 2026-07-13 00:00.
- Online calibration: transitions from 2026-07-13 00:00 through 2026-07-13 23:59. Calibration updates uncertainty intervals only, not model coefficients.
- Forward evaluation: transitions from 2026-07-14 00:00 onward.

The split uses target timestamps and must enforce strict chronological non-overlap. The generated artifact records row counts, ranges, hashes, and data roles.

## Evaluation

Run identical 1, 6, 24, and 72-hour rollouts for the frozen full-range reference, the fixed thermal candidate, and the online-adapted model on the forward evaluation block. Report aggregate and normal/high/extreme regime metrics.

## Acceptance

The adapted model is useful only if extreme temperature and RH MAE both improve against the frozen reference, normal-regime normalized MAE does not degrade by more than 2%, and all predictions remain finite and physical.

## Evidence scope

This is a retrospective prequential replay because the July block was inspected earlier. It can validate the adaptation mechanism but cannot establish independent production certification. A later unopened heat season is still required.
