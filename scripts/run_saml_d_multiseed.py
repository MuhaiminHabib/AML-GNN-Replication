from pathlib import Path
import sys
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

from src.data.saml_d import build_saml_d_graph, describe_saml_d_data


OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

EPOCHS = 500
PATIENCE = 50
LR = 0.005
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
DROPOUT = 0.5

USE_COALESCED_EDGES = True


class GCN(nn.Module):
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


class GraphSAGE(nn.Module):
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


class GATv2(nn.Module):
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
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(model_name: str, in_channels: int):
    if model_name == "GCN":
        return GCN(in_channels, HIDDEN_DIM, dropout=DROPOUT)
    if model_name == "GraphSAGE":
        return GraphSAGE(in_channels, HIDDEN_DIM, dropout=DROPOUT)
    if model_name == "GATv2":
        return GATv2(in_channels, HIDDEN_DIM, dropout=DROPOUT)

    raise ValueError(f"Unknown model: {model_name}")


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    best_idx = int(np.nanargmax(f1_scores))

    if best_idx >= len(thresholds):
        return 0.5

    return float(thresholds[best_idx])


@torch.no_grad()
def get_scores(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)

    y_true = data.y[mask].detach().cpu().numpy()
    y_score = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()

    return y_true, y_score


def compute_metrics_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    return {
        "accuracy": acc,
        "laundering_precision": precision,
        "laundering_recall": recall,
        "laundering_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
    }


@torch.no_grad()
def evaluate(model, data, mask, threshold=0.5):
    y_true, y_score = get_scores(model, data, mask)
    return compute_metrics_from_scores(y_true, y_score, threshold)


def train_one_model(model_name, seed, data, device):
    set_seed(seed)

    model = make_model(model_name, data.num_features).to(device)

    train_y = data.y[data.train_mask]
    class_counts = torch.bincount(train_y, minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = class_weights.to(device)

    print(f"Class counts: {class_counts.detach().cpu().tolist()}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_score = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

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

        # Early stopping is based on validation PR-AUC,
        # not fixed-threshold F1.
        val_metrics_default = evaluate(model, data, data.val_mask, threshold=0.5)
        val_score = val_metrics_default["pr_auc"]

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{model_name} | seed={seed} | epoch={epoch:03d} | "
                f"loss={loss.item():.5f} | "
                f"val_f1@0.5={val_metrics_default['laundering_f1']:.4f} | "
                f"val_pr_auc={val_metrics_default['pr_auc']:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------
    # Tune threshold on validation set.
    # Then apply the same threshold to train/val/test.
    # ------------------------------------------------------------
    val_y_true, val_y_score = get_scores(model, data, data.val_mask)
    best_threshold = find_best_threshold(val_y_true, val_y_score)

    print(f"Best validation threshold: {best_threshold:.6f}")

    train_metrics = evaluate(model, data, data.train_mask, threshold=best_threshold)
    val_metrics = evaluate(model, data, data.val_mask, threshold=best_threshold)
    test_metrics = evaluate(model, data, data.test_mask, threshold=best_threshold)

    # Also keep default 0.5 test F1 for comparison.
    test_metrics_default = evaluate(model, data, data.test_mask, threshold=0.5)

    return {
        "dataset": "SAML-D",
        "task": "account_node_classification",
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "early_stop_metric": "val_pr_auc",
        "best_val_pr_auc": best_val_score,
        "best_threshold": best_threshold,
        "coalesced_edges": USE_COALESCED_EDGES,
        "train_samples": int(data.train_mask.sum()),
        "val_samples": int(data.val_mask.sum()),
        "test_samples": int(data.test_mask.sum()),
        "train_f1": train_metrics["laundering_f1"],
        "val_f1": val_metrics["laundering_f1"],
        "test_f1": test_metrics["laundering_f1"],
        "test_f1_at_0_5": test_metrics_default["laundering_f1"],
        "test_precision": test_metrics["laundering_precision"],
        "test_recall": test_metrics["laundering_recall"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_saml_d_graph(coalesce_edges=USE_COALESCED_EDGES)
    summary = describe_saml_d_data(data)

    print("\nSAML-D summary")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")

    data = data.to(device)

    results = []

    models = ["GCN", "GraphSAGE", "GATv2"]

    for model_name in models:
        for seed in SEEDS:
            print("\n" + "=" * 80)
            print(f"Running {model_name} | seed={seed}")
            print("=" * 80)

            result = train_one_model(model_name, seed, data, device)
            results.append(result)

            partial_df = pd.DataFrame(results)
            partial_df.to_csv(
                OUTPUT_DIR / "saml_d_threshold_multiseed_partial_results.csv",
                index=False,
            )

            print("\nTest result:")
            for k, v in result.items():
                print(f"{k}: {v}")

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "saml_d_threshold_multiseed_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["dataset", "task", "model"])
        .agg(
            runs=("seed", "count"),
            train_samples=("train_samples", "first"),
            val_samples=("val_samples", "first"),
            test_samples=("test_samples", "first"),
            best_threshold_mean=("best_threshold", "mean"),
            best_threshold_std=("best_threshold", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            test_f1_at_0_5_mean=("test_f1_at_0_5", "mean"),
            test_f1_at_0_5_std=("test_f1_at_0_5", "std"),
            precision_mean=("test_precision", "mean"),
            precision_std=("test_precision", "std"),
            recall_mean=("test_recall", "mean"),
            recall_std=("test_recall", "std"),
            pr_auc_mean=("test_pr_auc", "mean"),
            pr_auc_std=("test_pr_auc", "std"),
            roc_auc_mean=("test_roc_auc", "mean"),
            roc_auc_std=("test_roc_auc", "std"),
        )
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "saml_d_threshold_multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSAML-D threshold-tuned multi-seed summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print(f"\nSaved detailed results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()