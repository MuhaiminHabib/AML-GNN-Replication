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

from torch_geometric.nn import SAGEConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import (
    build_ibm_amlsim_graph,
    describe_ibm_amlsim_data,
)


# =============================================================================
# Fixed shared AMLSim explainer setup
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

MODEL_NAME = "GraphSAGE"
DATASET_NAME = "IBM AMLSim"
SETTING_NAME = "previous_reproduction"

EPOCHS = 1000
PATIENCE = 100
LR = 0.005
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
DROPOUT = 0.5

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

FRAUD_LABEL = 1
TOP_N_EXPLANATION_NODES = 10


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def select_explanation_nodes(model, data, top_n: int):
    """
    Select correctly predicted fraud nodes from the test set.

    These nodes will be shared by:
        - GNNExplainer
        - PGExplainer
        - SubgraphX

    This gives a fixed node list for fair explainer comparison.
    """

    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, FRAUD_LABEL]
        preds = logits.argmax(dim=1)

    test_mask = data.test_mask
    fraud_mask = data.y == FRAUD_LABEL
    correct_mask = preds == data.y

    candidate_mask = test_mask & fraud_mask & correct_mask
    candidate_nodes = torch.where(candidate_mask)[0]

    if candidate_nodes.numel() == 0:
        raise RuntimeError("No correctly predicted fraud test nodes found.")

    candidate_probs = probs[candidate_nodes]

    sorted_order = torch.argsort(candidate_probs, descending=True)
    selected_nodes = candidate_nodes[sorted_order][:top_n]

    rows = []

    for rank, node_idx in enumerate(selected_nodes.detach().cpu().tolist(), start=1):
        rows.append(
            {
                "rank": rank,
                "node_idx": int(node_idx),
                "true_label": int(data.y[node_idx].detach().cpu().item()),
                "pred_label": int(preds[node_idx].detach().cpu().item()),
                "fraud_probability": float(probs[node_idx].detach().cpu().item()),
                "split": "test",
                "selection_rule": "correctly_predicted_fraud_test_nodes_sorted_by_probability",
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("Preparing shared AMLSim GraphSAGE explainer setup")
    print("=" * 100)

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Setting: {SETTING_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {SEED}")

    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    summary = describe_ibm_amlsim_data(data)

    print("\nGraph summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    data = data.to(device)

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=2,
        dropout=DROPOUT,
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
    best_epoch = 0
    bad_epochs = 0

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
                "dataset": DATASET_NAME,
                "setting": SETTING_NAME,
                "model": MODEL_NAME,
                "seed": SEED,
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
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

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
        raise RuntimeError("No best model checkpoint was found.")

    model.load_state_dict(best_state)
    model = model.to(device)

    train_metrics = evaluate(model, data, data.train_mask)
    val_metrics = evaluate(model, data, data.val_mask)
    test_metrics = evaluate(model, data, data.test_mask)

    metrics = {
        "dataset": DATASET_NAME,
        "setting": SETTING_NAME,
        "task": "account_node_classification",
        "model": MODEL_NAME,
        "seed": SEED,
        "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
        "num_nodes": summary["num_nodes"],
        "num_edges": summary["num_edges"],
        "num_features": summary["num_features"],
        "train_samples": summary["train_samples"],
        "val_samples": summary["val_samples"],
        "test_samples": summary["test_samples"],
        "train_fraud": summary["train_fraud"],
        "val_fraud": summary["val_fraud"],
        "test_fraud": summary["test_fraud"],
        "best_epoch": best_epoch,
        "train_fraud_f1": train_metrics["fraud_f1"],
        "val_fraud_f1": val_metrics["fraud_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_fraud_precision": test_metrics["fraud_precision"],
        "test_fraud_recall": test_metrics["fraud_recall"],
        "test_fraud_f1": test_metrics["fraud_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
    }

    node_df = select_explanation_nodes(
        model=model,
        data=data,
        top_n=TOP_N_EXPLANATION_NODES,
    )

    checkpoint_path = OUTPUT_DIR / "shared_amlsim_graphsage_model.pt"
    metrics_path = OUTPUT_DIR / "shared_amlsim_graphsage_model_metrics.csv"
    history_path = OUTPUT_DIR / "shared_amlsim_graphsage_training_history.csv"
    node_path = OUTPUT_DIR / "shared_amlsim_graphsage_explanation_nodes.csv"
    graph_summary_path = OUTPUT_DIR / "shared_amlsim_graphsage_graph_summary.csv"

    checkpoint = {
        "dataset": DATASET_NAME,
        "setting": SETTING_NAME,
        "task": "account_node_classification",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "in_channels": data.num_features,
        "out_channels": 2,
        "include_fraud_tx_count_features": INCLUDE_FRAUD_TX_COUNT_FEATURES,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "best_epoch": best_epoch,
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    node_df.to_csv(node_path, index=False)
    pd.DataFrame([summary]).to_csv(graph_summary_path, index=False)

    print("\nFinal shared checkpoint metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSelected shared explanation nodes:")
    print(node_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSaved files:")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics:    {metrics_path}")
    print(f"History:    {history_path}")
    print(f"Nodes:      {node_path}")
    print(f"Summary:    {graph_summary_path}")


if __name__ == "__main__":
    main()
