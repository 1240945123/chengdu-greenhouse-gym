from __future__ import annotations

from experiments.controllers.evaluation_failures import append_environment_failure_row


def test_environment_failure_row_preserves_last_state_and_records_cause():
    rows = [{
        "timestep": 11,
        "reward": -1.0,
        "air_temperature": 24.0,
        "terminated": False,
        "truncated": False,
        "environment_failure": False,
    }]
    info = {
        "reward": -999.0,
        "failure": True,
        "failure_kind": "numerical_integration",
        "failure_phase": "integration",
        "failure_exception": "RuntimeError",
        "failure_message": "mxstep",
        "failure_timestep": 12,
    }

    append_environment_failure_row(rows, info, timestep=12)

    assert len(rows) == 2
    failure = rows[-1]
    assert failure["air_temperature"] == 24.0
    assert failure["reward"] == -999.0
    assert failure["environment_failure"] is True
    assert failure["environment_failure_kind"] == "numerical_integration"
    assert failure["environment_failure_message"] == "mxstep"
    assert failure["truncated"] is True
    assert failure["terminated"] is False
