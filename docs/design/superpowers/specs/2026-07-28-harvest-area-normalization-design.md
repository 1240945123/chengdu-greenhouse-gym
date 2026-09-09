# Harvest Area Normalization Design

## Purpose

Ensure every measured batch and cumulative yield uses one fixed modeled-area
basis instead of summing incompatible local harvested-area ratios.

## Semantics

`normalization_area_m2` is the fixed production area configured for each
greenhouse. `fresh_kg_m2` and `harvested_fresh_kg_m2` equal total batch fresh
mass divided by that fixed area and are the only values used for model fitting
and evaluation. `local_harvested_fresh_kg_m2` equals mass divided by the area
actually picked during the event and is audit-only. `harvest_coverage_fraction`
equals actual picked area divided by fixed production area.

## Validation

The loader requires a positive finite normalization-area mapping for every
greenhouse code in the event file. Actual harvested area must be positive and
cannot exceed normalization area. The mapping comes from controlled greenhouse
configuration, not event rows. Identity and duplicate checks remain unchanged.

## Pipeline

The Chengdu pipeline supplies the target site's configured 192 m2 greenhouse
area to the loader. Processed events retain both area bases and coverage. The
event template continues to request actual harvested area; users do not enter
the normalization denominator.

## Evidence Boundary

This corrects unit consistency but does not turn subarea sample weights into
whole-greenhouse harvest totals. `harvested_fresh_kg` must still be the total
fruit physically removed in that event, not a sample extrapolation.
