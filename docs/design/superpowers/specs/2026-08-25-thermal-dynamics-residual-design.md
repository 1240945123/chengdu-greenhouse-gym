# Thermal-Dynamics Residual Model Design

## Objective

Reduce Chengdu greenhouse high-temperature prediction error without target leakage, synthetic truth generation, or degradation of normal operating conditions.

## Context

The V4 chronological splits contain 25 train, 6 validation, and 106 test transitions at or above 35 C. The existing residual model is accurate in normal conditions but under-predicts extreme summer heat. The already-opened test split is diagnostic only and cannot be used for model selection.

## Model inputs

Add a `thermal_dynamics` Ridge feature set on top of the current periodic interaction set. Derived features use only information available at prediction time:

- `solar_trapping`: radiation retained when roof ventilation, fan, and wet pad are inactive.
- `ventilation_heat_exchange`: roof opening, wind speed, and predicted indoor-outdoor temperature difference.
- `fan_cooling_demand`: fan action multiplied by positive predicted indoor-outdoor temperature difference.
- `pad_evaporative_potential`: wet-pad action, positive temperature difference, and outdoor humidity deficit.
- `hot_solar_load`: radiation multiplied by positive outdoor temperature above 25 C.

All action values are clipped to [0, 1] when deriving interactions. The canonical observations and physical model remain unchanged.

## Selection protocol

Fit Ridge candidates on the chronological train prefix and calibrate on the train suffix. Select hyperparameters using validation only. The existing frozen unweighted model is the reference.

A thermal candidate may replace the reference only when:

1. all predictions are finite and inside the physical envelope;
2. the aggregate normalized 24/72-hour validation score improves;
3. validation high-temperature temperature MAE and RH MAE both improve;
4. validation normal-regime normalized MAE does not degrade by more than 2%.

The test split is evaluated once after the decision and is reported as a non-independent diagnostic because it was opened in an earlier project stage.

If the unweighted thermal candidate improves aggregate and normal conditions but fails one high-temperature target, run one predeclared capped heat-weight experiment with weights `normal=1`, `high=5`, and `extreme=5`. This uses training labels only, keeps all rows, and avoids inverse-frequency weights that would amplify two extreme samples by several dozen times.

## Outputs

Store the candidate model, validation and diagnostic test rollouts, quality/heat-stratified metrics, and an explicit promotion decision in a new result directory. Never overwrite the current primary model.

## Limitations

Six high-temperature validation transitions cannot establish production readiness. Even if the candidate passes, final industry acceptance requires additional measured summer data and a new unopened seasonal holdout.
