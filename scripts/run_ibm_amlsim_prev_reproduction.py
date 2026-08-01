from pathlib import Path
import sys
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

from src.data.ibm_amlsim_prev_reproduction import (
    build_ibm_amlsim_graph,
    describe_ibm_amlsim_data,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ibm_amlsim_prev_reproduction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEEDS = [42, 43, 44, 45, 46]
MODELS = ["GCN", "GraphSAGE", "GATv2"]

EPOCHS = 1000
PATIENCE = 100
LR = 0.005
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
DROPOUT = 0.5

VAL_SIZE = 0.15
TEST_SIZE = 0.20

# Try False first because this matches the recovered previous-code leakage-control note.
# If it does not match your old table, change this to True and rerun.
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

FRAUD_LABEL = 1


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    def __init__(self, in_channels, hidden_channels, out_channels=2, dropout=0.5):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=1,
            dropout=dropout,
        )

        self.conv2 = GATv2Conv(
            hidden_channels,
            out_channels,
            heads=1,
            dropout=dropout,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def build_model(model_name: str, in_channels: int):
    if model_name == "GCN":
        return GCN(
            in_channels=in_channels,
            hidden_channels=HIDDEN_DIM,
            dropout=DROPOUT,
        )

    if model_name == "GraphSAGE":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=HIDDEN_DIM,
            dropout=DROPOUT,
        )

    if model_name == "GATv2":
        return GATv2(
            in_channels=in_channels,
            hidden_channels=HIDDEN_DIM,
            dropout=DROPOUT,
        )

    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()

    logits = model(data.x, data.edge_index)

    probs = torch.softmax(logits[mask], dim=1)[:, FRAUD_LABEL]
    preds = logits[mask].argmax(dim=1)
    labels = data.y[mask]

    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    accuracy = accuracy_score(labels_np, preds_np)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np,
        preds_np,
        pos_label=FRAUD_LABEL,
        average="binary",
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(labels_np, probs_np)
    except ValueError:
        pr_auc = np.nan

    return {
        "accuracy": accuracy,
        "fraud_precision": precision,
        "fraud_recall": recall,
        "fraud_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def get_class_weights(data, device):
    y_train = data.y[data.train_mask]
    counts = torch.bincount(y_train, minlength=2).float()

    weights = counts.sum() / (len(counts) * counts.clamp(min=1))

    return weights.to(device)


def train_one_model(seed: int, model_name: str, device: torch.device):
    print("\n" + "=" * 100)
    print(f"Dataset: IBM AMLSim previous reproduction | Model: {model_name} | Seed: {seed}")
    print("=" * 100)

    seed_everything(seed)

    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=seed,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    split_summary = describe_ibm_amlsim_data(data)

    print("\nGraph summary:")
    for key, value in split_summary.items():
        print(f"{key}: {value}")

    data = data.to(device)

    model = build_model(
        model_name=model_name,
        in_channels=data.num_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    class_weights = get_class_weights(data, device)

    print("\nClass weights:")
    print(class_weights.detach().cpu().numpy())

    best_val_f1 = -1.0
    best_state = None
    bad_epochs = 0
    best_epoch = 0

    history_rows = []

    for epoch in range(EPOCHS):
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

        val_metrics = evaluate(model, data, data.val_mask)
        val_f1 = val_metrics["fraud_f1"]

        history_rows.append(
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction",
                "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "loss": float(loss.item()),
                "val_fraud_precision": val_metrics["fraud_precision"],
                "val_fraud_recall": val_metrics["fraud_recall"],
                "val_fraud_f1": val_metrics["fraud_f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 0 or epoch % 50 == 0:
            train_metrics = evaluate(model, data, data.train_mask)

            print(
                f"Epoch {epoch:04d} | "
                f"Loss {loss.item():.4f} | "
                f"Train F1 {train_metrics['fraud_f1']:.4f} | "
                f"Val F1 {val_f1:.4f} | "
                f"Val PR-AUC {val_metrics['pr_auc']:.4f}"
            )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No best state found for model={model_name}, seed={seed}")

    model.load_state_dict(best_state)
    model = model.to(device)

    train_metrics = evaluate(model, data, data.train_mask)
    val_metrics = evaluate(model, data, data.val_mask)
    test_metrics = evaluate(model, data, data.test_mask)

    result = {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "task": "account_node_classification",
        "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
        "model": model_name,
        "seed": seed,
        "num_nodes": split_summary["num_nodes"],
        "num_edges": split_summary["num_edges"],
        "num_features": split_summary["num_features"],
        "num_transactions": split_summary["num_transactions"],
        "num_fraud_transactions": split_summary["num_fraud_transactions"],
        "num_fraud_nodes": split_summary["num_fraud_nodes"],
        "train_samples": split_summary["train_samples"],
        "val_samples": split_summary["val_samples"],
        "test_samples": split_summary["test_samples"],
        "train_fraud": split_summary["train_fraud"],
        "val_fraud": split_summary["val_fraud"],
        "test_fraud": split_summary["test_fraud"],
        "best_epoch": best_epoch,
        "train_accuracy": train_metrics["accuracy"],
        "train_fraud_precision": train_metrics["fraud_precision"],
        "train_fraud_recall": train_metrics["fraud_recall"],
        "train_fraud_f1": train_metrics["fraud_f1"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_pr_auc": train_metrics["pr_auc"],
        "val_accuracy": val_metrics["accuracy"],
        "val_fraud_precision": val_metrics["fraud_precision"],
        "val_fraud_recall": val_metrics["fraud_recall"],
        "val_fraud_f1": val_metrics["fraud_f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_fraud_precision": test_metrics["fraud_precision"],
        "test_fraud_recall": test_metrics["fraud_recall"],
        "test_fraud_f1": test_metrics["fraud_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
    }

    model_name_safe = model_name.lower().replace(" ", "_")

    history_path = (
        OUTPUT_DIR
        / f"ibm_amlsim_prev_reproduction_{model_name_safe}_seed{seed}_history.csv"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / f"ibm_amlsim_prev_reproduction_{model_name_safe}_seed{seed}_model.pt"
    )

    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    checkpoint = {
        "dataset": "IBM_AMLSim",
        "setting": "previous_reproduction",
        "task": "account_node_classification",
        "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
        "model_name": model_name,
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "in_channels": data.num_features,
        "out_channels": 2,
        "state_dict": model.state_dict(),
        "metrics": result,
    }

    torch.save(checkpoint, checkpoint_path)

    print("\nFinal test result:")
    print(
        pd.DataFrame([result]).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nSaved history to: {history_path}")
    print(f"Saved checkpoint to: {checkpoint_path}")

    del model
    del data

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Setting: previous_reproduction")
    print("Split: stratified 65/15/20")
    print(f"Include fraud tx count features: {INCLUDE_FRAUD_TX_COUNT_FEATURES}")

    all_results = []
    partial_path = OUTPUT_DIR / "ibm_amlsim_prev_reproduction_partial_results.csv"

    for seed in SEEDS:
        for model_name in MODELS:
            result = train_one_model(
                seed=seed,
                model_name=model_name,
                device=device,
            )

            all_results.append(result)

            pd.DataFrame(all_results).to_csv(partial_path, index=False)

            print(f"\nPartial results saved to: {partial_path}")

    results_df = pd.DataFrame(all_results)

    summary_df = (
        results_df
        .groupby(
            [
                "dataset",
                "setting",
                "task",
                "include_fraud_tx_count_features",
                "model",
            ],
            as_index=False,
        )
        .agg(
            runs=("seed", "nunique"),
            train_samples=("train_samples", "first"),
            val_samples=("val_samples", "first"),
            test_samples=("test_samples", "first"),
            train_fraud=("train_fraud", "first"),
            val_fraud=("val_fraud", "first"),
            test_fraud=("test_fraud", "first"),
            test_fraud_f1_mean=("test_fraud_f1", "mean"),
            test_fraud_f1_std=("test_fraud_f1", "std"),
            test_fraud_precision_mean=("test_fraud_precision", "mean"),
            test_fraud_precision_std=("test_fraud_precision", "std"),
            test_fraud_recall_mean=("test_fraud_recall", "mean"),
            test_fraud_recall_std=("test_fraud_recall", "std"),
            test_pr_auc_mean=("test_pr_auc", "mean"),
            test_pr_auc_std=("test_pr_auc", "std"),
            test_roc_auc_mean=("test_roc_auc", "mean"),
            test_roc_auc_std=("test_roc_auc", "std"),
        )
    )

    results_path = OUTPUT_DIR / "ibm_amlsim_prev_reproduction_results.csv"
    summary_path = OUTPUT_DIR / "ibm_amlsim_prev_reproduction_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("IBM AMLSim previous reproduction multi-seed summary")
    print("=" * 100)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nSaved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()