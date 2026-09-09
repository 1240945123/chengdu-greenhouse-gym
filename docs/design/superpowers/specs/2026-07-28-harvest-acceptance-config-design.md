# Harvest Acceptance Configuration Design

## Goal

Make the Chengdu harvest YAML acceptance section authoritative for both the
held-out experiment result and the final readiness report.

## Contract

The YAML retains directional names such as `_max`, `_abs_max`, and `_min`.
A strict adapter maps all 16 configured keys to the canonical metric names used
by `evaluate_target_acceptance`. Missing keys, unknown keys, non-finite values,
negative error limits, invalid coverage values, or out-of-range R2 thresholds
fail before calibration.

The seasonal calibration function accepts canonical thresholds and returns them
inside its structured result. The readiness report recomputes the gates from
held-out metrics using those exact returned thresholds. It does not silently
substitute process defaults. Direct unit callers may omit thresholds and retain
the documented defaults.

## Evidence

`calibration_and_validation.json` records the canonical threshold map and each
gate result. Tests prove that changing a YAML-derived timing or yield threshold
changes both experiment acceptance and final readiness, while malformed
configuration is rejected.
