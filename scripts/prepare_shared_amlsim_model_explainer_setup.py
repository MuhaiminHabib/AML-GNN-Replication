from pathlib import Path
import sys
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import build_ibm_amlsim_graph


# =============================================================================
# Global settings
# =============================================================================

SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

EPOCHS = 1000
PATIENCE = 100

FRAUD_LABEL = 1
NUM_EXPLANATION_NODES = 10

DEFAULT_LR = 0.005
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_DROPOUT = 0.5

MODEL_CONFIGS = {
    "gcn": {
        "hidden_dim": 64,
        "dropout": 0.5,
        "lr": 0.005,
        "weight_decay": 5e-4,
    },
    "graphsage": {
        "hidden_dim": 64,
        "dropout": 0.5,
        "lr": 0.005,
        "weight_decay": 5e-4,
    },
    "gatv2": {
        "hidden_dim": 32,
        "heads": 8,
        "dropout": 0.5,
        "lr": 0.005,
        "weight_decay": 5e-4,
    },
}


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Models
# =============================================================================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, dropout=0.5):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x


class GATv2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=2,
        heads=8,
        dropout=0.5,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x


def build_model(model_name: str, in_channels: int, config: dict):
    model_name = model_name.lower()

    if model_name == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=config["hidden_dim"],
            out_channels=2,
            dropout=config["dropout"],
        )

    if model_name == "graphsage":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=config["hidden_dim"],
            out_channels=2,
            dropout=config["dropout"],
        )

    if model_name == "gatv2":
        return GATv2(
            in_channels=in_channels,
            hidden_channels=config["hidden_dim"],
            out_channels=2,
            heads=config["heads"],
            dropout=config["dropout"],
        )

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def evaluate_split(model, data, mask, split_name):
    model.eval()

    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, FRAUD_LABEL]
    preds = logits.argmax(dim=1)

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = preds[mask].detach().cpu().numpy()
    y_prob = probs[mask].detach().cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[FRAUD_LABEL],
        average=None,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = np.nan

    return {
        f"{split_name}_accuracy": float(accuracy),
        f"{split_name}_fraud_precision": float(precision[0]),
        f"{split_name}_fraud_recall": float(recall[0]),
        f"{split_name}_fraud_f1": float(f1[0]),
        f"{split_name}_roc_auc": float(roc_auc),
        f"{split_name}_pr_auc": float(pr_auc),
    }


@torch.no_grad()
def select_explanation_nodes(model, data, num_nodes: int):
    """
    Select correctly predicted fraud nodes from the test set.

    We choose the highest-confidence fraud predictions, same style as the
    GraphSAGE-only AMLSim setup.
    """

    model.eval()

    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

    rows = []

    for node_idx in test_idx.detach().cpu().tolist():
        true_label = int(data.y[node_idx].detach().cpu().item())
        pred_label = int(preds[node_idx].detach().cpu().item())
        fraud_prob = float(probs[node_idx, FRAUD_LABEL].detach().cpu().item())

        if true_label == FRAUD_LABEL and pred_label == FRAUD_LABEL:
            rows.append(
                {
                    "node_idx": int(node_idx),
                    "split": "test",
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "fraud_probability": float(fraud_prob),
                }
            )

    nodes_df = pd.DataFrame(rows)

    if nodes_df.empty:
        raise RuntimeError(
            "No correctly predicted fraud test nodes found. "
            "Cannot select explanation nodes."
        )

    nodes_df = (
        nodes_df
        .sort_values("fraud_probability", ascending=False)
        .head(num_nodes)
        .reset_index(drop=True)
    )

    nodes_df.insert(0, "rank", np.arange(1, len(nodes_df) + 1))

    return nodes_df


# =============================================================================
# Training
# =============================================================================

def train_shared_model(model_name: str, data, device, config: dict):
    model = build_model(
        model_name=model_name,
        in_channels=data.num_features,
        config=config,
    ).to(device)

    y_train = data.y[data.train_mask]
    class_counts = torch.bincount(y_train, minlength=2).float()

    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = class_weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    history = []

    print("\nTraining settings:")
    print(f"Model:        {model_name}")
    print(f"Hidden dim:   {config['hidden_dim']}")
    print(f"Dropout:      {config['dropout']}")
    print(f"LR:           {config['lr']}")
    print(f"Weight decay: {config['weight_decay']}")

    if model_name == "gatv2":
        print(f"Heads:        {config['heads']}")

    print(f"Class counts:  {class_counts.detach().cpu().tolist()}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        train_metrics = evaluate_split(model, data, data.train_mask, "train")
        val_metrics = evaluate_split(model, data, data.val_mask, "val")

        val_f1 = val_metrics["val_fraud_f1"]

        row = {
            "epoch": int(epoch),
            "loss": float(loss.detach().cpu().item()),
            **train_metrics,
            **val_metrics,
        }

        history.append(row)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"loss={loss.detach().cpu().item():.6f} | "
                f"train_f1={train_metrics['train_fraud_f1']:.4f} | "
                f"val_f1={val_metrics['val_fraud_f1']:.4f} | "
                f"best_val_f1={best_val_f1:.4f}"
            )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed. No best model state was stored.")

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    return model, best_state, best_epoch, pd.DataFrame(history)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["gcn", "graphsage", "gatv2"],
        help="AMLSim model to train and prepare for explainers.",
    )
    parser.add_argument(
        "--num-explanation-nodes",
        type=int,
        default=NUM_EXPLANATION_NODES,
        help="Number of correctly predicted fraud test nodes to explain.",
    )

    args = parser.parse_args()

    model_name = args.model.lower()
    config = MODEL_CONFIGS[model_name]

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = PROJECT_ROOT / "outputs" / "explainers" / f"amlsim_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"shared_amlsim_{model_name}_model.pt"
    metrics_path = output_dir / f"shared_amlsim_{model_name}_model_metrics.csv"
    history_path = output_dir / f"shared_amlsim_{model_name}_training_history.csv"
    nodes_path = output_dir / f"shared_amlsim_{model_name}_explanation_nodes.csv"
    graph_summary_path = output_dir / f"shared_amlsim_{model_name}_graph_summary.csv"

    print("=" * 100)
    print(f"Preparing shared AMLSim {model_name.upper()} explainer setup")
    print("=" * 100)
    print(f"Using device: {device}")

    print("\nLoading AMLSim graph...")
    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    data = data.to(device)

    graph_summary = {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "model": model_name,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.num_features),
        "num_classes": int(data.y.max().detach().cpu().item() + 1),
        "train_nodes": int(data.train_mask.sum().detach().cpu().item()),
        "val_nodes": int(data.val_mask.sum().detach().cpu().item()),
        "test_nodes": int(data.test_mask.sum().detach().cpu().item()),
        "train_fraud": int((data.y[data.train_mask] == FRAUD_LABEL).sum().detach().cpu().item()),
        "val_fraud": int((data.y[data.val_mask] == FRAUD_LABEL).sum().detach().cpu().item()),
        "test_fraud": int((data.y[data.test_mask] == FRAUD_LABEL).sum().detach().cpu().item()),
        "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
        "seed": SEED,
    }

    pd.DataFrame([graph_summary]).to_csv(graph_summary_path, index=False)

    print("\nGraph summary:")
    for key, value in graph_summary.items():
        print(f"{key}: {value}")

    model, best_state, best_epoch, history_df = train_shared_model(
        model_name=model_name,
        data=data,
        device=device,
        config=config,
    )

    print("\nEvaluating best checkpoint...")
    train_metrics = evaluate_split(model, data, data.train_mask, "train")
    val_metrics = evaluate_split(model, data, data.val_mask, "val")
    test_metrics = evaluate_split(model, data, data.test_mask, "test")

    metrics = {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "model": model_name,
        "seed": SEED,
        "best_epoch": int(best_epoch),
        **config,
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }

    metrics_df = pd.DataFrame([metrics])

    print("\nFinal shared checkpoint metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSelecting explanation nodes...")
    nodes_df = select_explanation_nodes(
        model=model,
        data=data,
        num_nodes=args.num_explanation_nodes,
    )

    print(nodes_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    checkpoint = {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "model": model_name,
        "state_dict": best_state,
        "seed": SEED,
        "best_epoch": int(best_epoch),
        "config": config,
        "hidden_dim": config["hidden_dim"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
        "fraud_label": FRAUD_LABEL,
    }

    if model_name == "gatv2":
        checkpoint["heads"] = config["heads"]

    torch.save(checkpoint, checkpoint_path)

    metrics_df.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)

    print("\nSaved files:")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Metrics:          {metrics_path}")
    print(f"Training history: {history_path}")
    print(f"Explanation nodes:{nodes_path}")
    print(f"Graph summary:    {graph_summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()