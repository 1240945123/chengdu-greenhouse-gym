from experiments.controllers.run_v5_full_pipeline import pipeline_stage_names


def test_full_pipeline_orders_training_selection_extension_and_evaluation():
    assert pipeline_stage_names() == (
        "minimum_training",
        "minimum_validation",
        "extended_training",
        "final_validation",
        "temporal_holdout_evaluation",
        "six_season_evaluation",
    )
