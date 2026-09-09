import argparse
import json
import os

import numpy as np
import torch

from glassgym.models.pinn.surrogate import Normalizer, save_normalizers
from glassgym.models.transformer.sequence_dataset import generate_mixed_sequences
from glassgym.models.transformer.surrogate import TransformerDynamicsSurrogate, train_one_epoch
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer dynamics surrogate.")
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="configs/envs/")
    parser.add_argument("--save_dir", default="train_data/amsterdam_reference/transformer/")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    params = load_model_hyperparams("transformer", args.env_id)
    seed = int(params["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_config = params["model"]
    sequences = generate_mixed_sequences(
        env_kwargs=env_kwargs,
        source_counts=params["source_counts"],
        max_steps_per_episode=int(params["max_steps_per_episode"]),
        seed=seed,
        seq_len=int(model_config["seq_len"]),
    )

    inputs = sequences["inputs"]
    targets = sequences["targets"]

    n = len(inputs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    test_size = max(1, int(n * float(params["test_fraction"])))
    test_idx = order[:test_size]
    train_idx = order[test_size:]

    input_norm = Normalizer.fit(inputs[train_idx].reshape(-1, inputs.shape[-1]))
    target_norm = Normalizer.fit(targets[train_idx])

    device = torch.device(params["training"].get("device", "cpu"))
    train_inputs = torch.tensor(input_norm.transform(inputs[train_idx]), dtype=torch.float32, device=device)
    train_targets = torch.tensor(target_norm.transform(targets[train_idx]), dtype=torch.float32, device=device)
    test_inputs = torch.tensor(input_norm.transform(inputs[test_idx]), dtype=torch.float32, device=device)
    test_targets = torch.tensor(target_norm.transform(targets[test_idx]), dtype=torch.float32, device=device)

    model = TransformerDynamicsSurrogate(**model_config).to(device)
    training = params["training"]
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            inputs=train_inputs,
            targets=train_targets,
            batch_size=int(training["batch_size"]),
        )
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = torch.mean((model(test_inputs) - test_targets) ** 2).item()
            print(f"epoch={epoch} train_loss={train_loss:.6f} test_mse_norm={test_loss:.6f}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "transformer_model.pt")
    normalizer_path = os.path.join(args.save_dir, "normalizer.npz")
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_config": model_config,
        },
        model_path,
    )
    save_normalizers(normalizer_path, input_norm, target_norm)
    print(json.dumps({"model_path": model_path, "normalizer_path": normalizer_path, "sequences": int(n)}))


if __name__ == "__main__":
    main()


