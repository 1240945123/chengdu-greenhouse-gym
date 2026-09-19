import unittest

import numpy as np
import torch

from glassgym.models.pinn.dataset import generate_mixed_transitions
from glassgym.models.pinn.surrogate import (
    DynamicsSurrogate,
    Normalizer,
    physics_loss,
    train_one_epoch,
)
from RL.utils import build_env_kwargs, load_env_params


class TestPINNSurrogate(unittest.TestCase):
    def test_model_maps_input_to_next_state(self):
        model = DynamicsSurrogate(input_dim=44, output_dim=28, hidden_layers=[16, 16])
        x = torch.zeros((3, 44), dtype=torch.float32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (3, 28))

    def test_normalizer_round_trip(self):
        values = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
        normalizer = Normalizer.fit(values)
        reconstructed = normalizer.inverse_transform(normalizer.transform(values))
        np.testing.assert_allclose(reconstructed, values, atol=1e-5)

    def test_physics_loss_is_finite_and_nonnegative(self):
        pred = torch.zeros((4, 28), dtype=torch.float32)
        prev = torch.zeros((4, 28), dtype=torch.float32)
        loss = physics_loss(
            pred,
            prev,
            nonnegative_indices=[22, 23, 24, 25],
            temperature_indices=[2, 3, 4, 5, 6, 7, 8, 9],
            lambda_nonnegative=1.0,
            lambda_temperature=1.0,
            lambda_delta=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_generate_mixed_transitions_small(self):
        env_kwargs = load_env_params("GreenLightEnv", "configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        transitions = generate_mixed_transitions(
            env_kwargs=env_kwargs,
            source_counts={"random": 1, "baseline": 1, "pid": 1},
            max_steps_per_episode=2,
            seed=123,
        )
        self.assertEqual(transitions["inputs"].shape[1], 44)
        self.assertEqual(transitions["targets"].shape[1], 28)
        self.assertEqual(len(transitions["sources"]), 6)

    def test_tiny_training_loop_updates_without_crashing(self):
        model = DynamicsSurrogate(input_dim=44, output_dim=28, hidden_layers=[16])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.randn((8, 44), dtype=torch.float32)
        targets = torch.randn((8, 28), dtype=torch.float32)
        previous_states = torch.randn((8, 28), dtype=torch.float32)
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            previous_states=previous_states,
            batch_size=4,
            loss_weights={
                "lambda_nonnegative": 0.1,
                "lambda_temperature": 0.1,
                "lambda_delta": 0.01,
            },
        )
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()


