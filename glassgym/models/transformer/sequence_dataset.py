from __future__ import annotations

import numpy as np

from glassgym.models.pinn.dataset import generate_mixed_transitions


def build_sequence_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    previous_states: np.ndarray,
    seq_len: int,
) -> dict[str, np.ndarray]:
    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    previous_states = np.asarray(previous_states, dtype=np.float32)
    seq_len = int(seq_len)

    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    if len(inputs) < seq_len:
        raise ValueError("not enough transitions to build one sequence")

    sequence_inputs = []
    sequence_targets = []
    sequence_previous_states = []
    for end in range(seq_len - 1, len(inputs)):
        start = end - seq_len + 1
        sequence_inputs.append(inputs[start:end + 1])
        sequence_targets.append(targets[end])
        sequence_previous_states.append(previous_states[end])

    return {
        "inputs": np.asarray(sequence_inputs, dtype=np.float32),
        "targets": np.asarray(sequence_targets, dtype=np.float32),
        "previous_states": np.asarray(sequence_previous_states, dtype=np.float32),
    }


def generate_mixed_sequences(
    env_kwargs: dict,
    source_counts: dict[str, int],
    max_steps_per_episode: int,
    seed: int,
    seq_len: int,
) -> dict[str, np.ndarray]:
    transitions = generate_mixed_transitions(
        env_kwargs=env_kwargs,
        source_counts=source_counts,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )
    return build_sequence_dataset(
        inputs=transitions["inputs"],
        targets=transitions["targets"],
        previous_states=transitions["previous_states"],
        seq_len=seq_len,
    )
