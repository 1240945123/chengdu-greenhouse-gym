import argparse
import json
import os

import numpy as np
import torch

from glassgym.models.pinn.dataset import generate_mixed_transitions
from glassgym.models.pinn.surrogate import DynamicsSurrogate, load_normalizers
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PINN dynamics surrogate.")
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="configs/envs/")
    parser.add_argument("--pinn_config", default="configs/agents/")
    parser.add_argument("--model_path", default="train_data/amsterdam_reference/pinn/pinn_model.pt")
    parser.add_argument("--normalizer_path", default="train_data/amsterdam_reference/pinn/normalizer.npz")
    parser.add_argument("--save_dir", default="results/amsterdam_reference/pinn/")
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    params = load_model_hyperparams("pinn", args.env_id)
    eval_params = params["evaluation"]

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = DynamicsSurrogate(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_norm, target_norm = load_normalizers(args.normalizer_path)

    transitions = generate_mixed_transitions(
        env_kwargs=env_kwargs,
        source_counts=eval_params["source_counts"],
        max_steps_per_episode=int(eval_params["max_steps_per_episode"]),
        seed=int(params["seed"]) + 999,
    )
    inputs = transitions["inputs"]
    targets = transitions["targets"]

    with torch.no_grad():
        pred_norm = model(torch.tensor(input_norm.transform(inputs), dtype=torch.float32)).numpy()
    preds = target_norm.inverse_transform(pred_norm)
    errors = preds - targets

    one_step_mse = float(np.mean(errors ** 2))
    one_step_mae = float(np.mean(np.abs(errors)))

    rollout_steps = min(int(eval_params["multi_step_rollout_steps"]), len(inputs))
    current_state = inputs[0, :28].copy()
    multi_errors = []
    for i in range(rollout_steps):
        model_input = inputs[i].copy()
        model_input[:28] = current_state
        with torch.no_grad():
            pred_norm_i = model(torch.tensor(input_norm.transform(model_input[None, :]), dtype=torch.float32)).numpy()
        pred_state = target_norm.inverse_transform(pred_norm_i)[0]
        target_state = targets[i]
        multi_errors.append(pred_state - target_state)
        current_state = pred_state.astype(np.float32)

    multi_errors = np.asarray(multi_errors, dtype=np.float32)
    metrics = {
        "one_step_mse": one_step_mse,
        "one_step_mae": one_step_mae,
        "multi_step_mse": float(np.mean(multi_errors ** 2)),
        "multi_step_mae": float(np.mean(np.abs(multi_errors))),
        "num_eval_transitions": int(len(inputs)),
        "multi_step_rollout_steps": int(rollout_steps),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "pinn_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()


