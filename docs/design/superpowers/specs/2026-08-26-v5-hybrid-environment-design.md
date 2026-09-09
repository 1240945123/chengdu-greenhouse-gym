# V5 Hybrid Greenhouse Environment Design

## Goal

Create a preflight control environment that retains the GreenLight crop model and
mass-balance states while using ChengduPhysicsV4 and the frozen V5 Ridge residual
model for indoor temperature and relative humidity endpoint correction.

## Alternatives

1. A standalone two-state learned climate Gym environment is simple but cannot
   evaluate crop growth or yield.
2. Replacing the GreenLight climate ODE entirely risks breaking crop-state coupling
   and mass conservation.
3. Selected: integrate ChengduPhysicsV4 normally, then apply a bounded endpoint
   correction before crop endpoint fluxes, observations, reward, and metrics are
   calculated. This preserves the crop subsystem and keeps the correction optional.

## Environment

The preflight environment uses 28 states, 8 controls, 11 disturbances, 225
parameters, and a 900-second step. The only optimized controls are roof ventilation
(index 3) and fan (index 6). Heating, CO2, lamp, and wet-pad pump remain zero;
screens remain fixed in this first preflight. The common eight-control safety layer
continues to enforce equipment bounds, slew rates, rain/wind closure, emergency
ventilation, and fan/pump interlock.

## Climate correction

An optional state postprocessor receives the previous state, raw integrated state,
executed controls, current disturbance, timestep duration, and simulation clock.
The V5 implementation constructs the exact residual feature sources used during
selection. It applies the frozen model with gain scaled by `dt / 3600`, capped at
one, updates air temperature and vapor pressure consistently, and reports raw and
corrected climate plus fallback status.

The postprocessor must reject wrong state/control dimensions, non-finite values,
and unsupported residual schemas. Corrected temperature remains within -10 to 60 C
and RH within 0 to 100%. With no postprocessor, GreenLightEnv behavior is unchanged.

## Provenance

The builder loads the frozen 225-parameter artifact and V5 residual model, records
their SHA-256 hashes, and refuses artifacts with incompatible dimensions or schema.
The environment is a control preflight model, not new independent validation data.

## Preflight experiment

Run deterministic low, mid, and high ventilation/fan policies for one simulated
day on the same Chengdu weather block. Report completion, finite-state status,
reward, temperature/RH MAE and comfort, safety interventions, control effort,
crop-state minima, and final fruit-state change. Promotion requires all scenarios
to complete without numerical failure, all states/rewards to remain finite, and
crop carbon states to remain nonnegative.

This phase validates environment mechanics only. Algorithm ranking and yield claims
remain prohibited until PID/MPC/PPO/SAC are trained and evaluated under a frozen
long-horizon protocol.
