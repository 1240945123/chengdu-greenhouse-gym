# Chengdu Cultivation Window Design

## Objective

Exclude all transitions whose target timestamp is on or after 2026-07-12 00:00 from crop-environment model training, validation, testing, and reported accuracy because tomato cultivation had ended.

## Data policy

The aligned source and V4 quality-aware trajectory remain immutable for provenance. A new V5 cultivation-window trajectory is derived from V4 and contains only rows with `next_timestamp < 2026-07-12 00:00`.

Filtering uses the target timestamp. A transition beginning before the cutoff but ending at or after the cutoff is excluded. The V5 manifest records the source hashes, cutoff, retained and excluded rows, chronological split ranges, split hashes, and heat-regime counts.

## Split policy

Chronologically re-split retained cultivation rows using 70% train, 15% validation, and 15% test. No random shuffle is allowed.

## Experiment policy

Create new cultivation-window model configurations and result directories. Do not overwrite prior artifacts. Any analysis using July 12 or later is marked invalid for the tomato cultivation environment objective, including the summer online adaptation replay.

## Acceptance

All V5 rows must precede the cutoff; retained observation, action, and quality values must exactly match V4; row accounting must balance; focused and full test suites must pass. Model metrics are recomputed only on V5 splits.
