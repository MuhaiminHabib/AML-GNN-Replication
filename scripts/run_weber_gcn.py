from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.elliptic import (
    add_weber_2019_masks,
    build_elliptic_pyg_data,
    describe_data,
)
from src.evaluation.metrics import compute_binary_metrics, format_metrics
from src.models.gcn import GCN
from src.utils.reproducibility import set_seed


def main():
    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Elliptic dataset...")
    data = build_elliptic_pyg_data(
        data_dir="data/raw/elliptic",
        make_undirected=False,
        include_time_as_feature=False,
    )
    data = add_weber_2019_masks(data)

    print("\nDataset summary:")
    for key, value in describe_data(data).items():
        print(f"{key}: {value}")

    data = data.to(device)

    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=100,
        out_channels=2,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    epochs = 1000

    print("Using weighted cross entropy with class weights: licit=0.3, illicit=0.7")
    print("\nTraining Weber 2019-style GCN replication...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        class_weights = torch.tensor([0.3, 0.7], dtype=torch.float, device=device)

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                train_metrics = compute_binary_metrics(
                    logits, data.y, data.train_mask
                )
                test_metrics = compute_binary_metrics(
                    logits, data.y, data.test_mask
                )


            print(
                f"Epoch {epoch:04d} | "
                f"Loss: {loss.item():.5f} | "
                f"Train illicit_f1: {train_metrics['illicit_f1']:.4f} | "
                f"Test illicit_f1: {test_metrics['illicit_f1']:.4f}"

            )

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        train_metrics = compute_binary_metrics(logits, data.y, data.train_mask)
        test_metrics = compute_binary_metrics(logits, data.y, data.test_mask)

    print("\nFinal train metrics:")
    print(format_metrics(train_metrics))

    print("\nFinal test metrics:")
    print(format_metrics(test_metrics))

    output_dir = PROJECT_ROOT / "outputs" / "results" / "weber_2019_gcn"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "weber_2019_gcn_replication",
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "model": "GCN",
        "split": "Weber 2019 temporal split: train 1-34, test 35-49",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    output_path = output_dir / "seed_42_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()