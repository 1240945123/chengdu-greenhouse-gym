import os
import tempfile
import unittest

import numpy as np
import torch

from glassgym.models.transformer.sequence_dataset import build_sequence_dataset
from glassgym.models.transformer.surrogate import (
    TransformerDynamicsSurrogate,
    train_one_epoch,
)
from glassgym.models.pinn.surrogate import Normalizer, load_normalizers, save_normalizers


class TransformerSurrogateTest(unittest.TestCase):
    def test_forward_returns_next_state_shape(self):
        model = TransformerDynamicsSurrogate(
            input_dim=44,
            output_dim=28,
            seq_len=4,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )

        y = model(torch.zeros(3, 4, 44))

        self.assertEqual(tuple(y.shape), (3, 28))

    def test_build_sequence_dataset_uses_sliding_windows(self):
        inputs = np.arange(6 * 44, dtype=np.float32).reshape(6, 44)
        targets = np.arange(6 * 28, dtype=np.float32).reshape(6, 28)
        previous_states = inputs[:, :28].copy()

        sequences = build_sequence_dataset(inputs, targets, previous_states, seq_len=3)

        self.assertEqual(tuple(sequences["inputs"].shape), (4, 3, 44))
        self.assertEqual(tuple(sequences["targets"].shape), (4, 28))
        np.testing.assert_array_equal(sequences["inputs"][0], inputs[:3])
        np.testing.assert_array_equal(sequences["targets"][0], targets[2])
        np.testing.assert_array_equal(sequences["previous_states"][0], previous_states[2])

    def test_build_sequence_dataset_rejects_short_inputs(self):
        inputs = np.zeros((2, 44), dtype=np.float32)
        targets = np.zeros((2, 28), dtype=np.float32)
        previous_states = np.zeros((2, 28), dtype=np.float32)

        with self.assertRaises(ValueError):
            build_sequence_dataset(inputs, targets, previous_states, seq_len=3)

    def test_normalizers_round_trip_for_sequence_inputs(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(5, 4, 44)).astype(np.float32)
        targets = rng.normal(size=(5, 28)).astype(np.float32)
        input_norm = Normalizer.fit(values.reshape(-1, values.shape[-1]))
        target_norm = Normalizer.fit(targets)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "normalizer.npz")
            save_normalizers(path, input_norm, target_norm)
            loaded_input, loaded_target = load_normalizers(path)

        np.testing.assert_allclose(loaded_input.transform(values), input_norm.transform(values))
        np.testing.assert_allclose(loaded_target.transform(targets), target_norm.transform(targets))

    def test_train_one_epoch_updates_through_valid_loss_path(self):
        torch.manual_seed(0)
        model = TransformerDynamicsSurrogate(
            input_dim=44,
            output_dim=28,
            seq_len=3,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        inputs = torch.randn(8, 3, 44)
        targets = torch.randn(8, 28)

        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            batch_size=4,
        )

        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()


