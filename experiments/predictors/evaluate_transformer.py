import argparse
import json
import os

import numpy as np
import torch

from glassgym.models.pinn.surrogate import load_normalizers
from glassgym.models.transformer.sequence_dataset import generate_mixed_sequences
from glassgym.models.transformer.surrogate import TransformerDynamicsSurrogate
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


def _predict(model, input_norm, target_norm, sequence: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred_norm = model(torch.tensor(input_norm.transform(sequence[None, :, :]), dtype=torch.float32)).numpy()
    return target_norm.inverse_transform(pred_norm)[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Transformer dynamics surrogate.")
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="configs/envs/")
    parser.add_argument("--model_path", default="train_data/amsterdam_reference/transformer/transformer_model.pt")
    parser.add_argument("--normalizer_path", default="train_data/amsterdam_reference/transformer/normalizer.npz")
    parser.add_argument("--save_dir", default="results/amsterdam_reference/transformer/")
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    params = load_model_hyperparams("transformer", args.env_id)
    eval_params = params["evaluation"]

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = TransformerDynamicsSurrogate(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_norm, target_norm = load_normalizers(args.normalizer_path)

    sequences = generate_mixed_sequences(
        env_kwargs=env_kwargs,
        source_counts=eval_params["source_counts"],
        max_steps_per_episode=int(eval_params["max_steps_per_episode"]),
        seed=int(params["seed"]) + 999,
        seq_len=int(model_config["seq_len"]),
    )
    inputs = sequences["inputs"]
    targets = sequences["targets"]

    with torch.no_grad():
        pred_norm = model(torch.tensor(input_norm.transform(inputs), dtype=torch.float32)).numpy()
    preds = target_norm.inverse_transform(pred_norm)
    errors = preds - targets

    rollout_steps = min(int(eval_params["multi_step_rollout_steps"]), len(inputs))
    current_sequence = inputs[0].copy()
    multi_errors = []
    for i in range(rollout_steps):
        pred_state = _predict(model, input_norm, target_norm, current_sequence)
        multi_errors.append(pred_state - targets[i])
        if i + 1 < len(inputs):
            next_token = inputs[i + 1, -1].copy()
            next_token[:28] = pred_state.astype(np.float32)
            current_sequence = np.vstack([current_sequence[1:], next_token]).astype(np.float32)

    multi_errors = np.asarray(multi_errors, dtype=np.float32)
    metrics = {
        "one_step_mse": float(np.mean(errors ** 2)),
        "one_step_mae": float(np.mean(np.abs(errors))),
        "multi_step_mse": float(np.mean(multi_errors ** 2)),
        "multi_step_mae": float(np.mean(np.abs(multi_errors))),
        "num_eval_sequences": int(len(inputs)),
        "multi_step_rollout_steps": int(rollout_steps),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "transformer_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()


