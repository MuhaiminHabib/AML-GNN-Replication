import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from models.gcn import GCN
from utils.elliptic_loader import load_elliptic, print_elliptic_summary
from utils.metrics import evaluate


DATA_DIR = PROJECT_ROOT / "data" / "elliptic"

SEED = 42
EPOCHS = 100
LR = 0.005
WEIGHT_DECAY = 5e-4
HIDDEN_CHANNELS = 128
DROPOUT = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_elliptic(DATA_DIR)
    print_elliptic_summary(data)
    data = data.to(device)

    train_labels = data.y[data.train_mask]
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = class_weights.to(device)

    print("Class weights:", class_weights.detach().cpu().numpy())

    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            val_metrics = evaluate(model, data, data.val_mask, device)

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Val illicit F1: {val_metrics['illicit_f1']:.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
            )

            if val_metrics["illicit_f1"] > best_val_f1:
                best_val_f1 = val_metrics["illicit_f1"]
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    test_metrics = evaluate(model, data, data.test_mask, device)

    print("\nFinal Test Metrics")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()