from pathlib import Path
import sys
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch_geometric.utils import k_hop_subgraph

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "shared_graphsage_model.pt"
NODE_LIST_PATH = OUTPUT_DIR / "shared_graphsage_explanation_nodes.csv"
METRICS_PATH = OUTPUT_DIR / "shared_graphsage_model_metrics.csv"

SEED = 42
MODEL_NAME = "graphsage"

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

ILLICIT_LABEL = 0
LICIT_LABEL = 1

NUM_SHARED_NODES = 10
EGO_HOPS = 2
MIN_EGO_NODES = 4


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    return build_marasi_elliptic_data(
        feature_setting="tx+agg",
        seed=SEED,
    )


@torch.no_grad()
def evaluate_model(model, data, mask):
    model.eval()

    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = pred[mask].detach().cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "illicit_precision": precision_score(
            y_true,
            y_pred,
            pos_label=ILLICIT_LABEL,
            zero_division=0,
        ),
        "illicit_recall": recall_score(
            y_true,
            y_pred,
            pos_label=ILLICIT_LABEL,
            zero_division=0,
        ),
        "illicit_f1": f1_score(
            y_true,
            y_pred,
            pos_label=ILLICIT_LABEL,
            zero_division=0,
        ),
    }


def train_model(data, device):
    print("\n" + "=" * 80)
    print("Training shared GraphSAGE model")
    print("=" * 80)

    model = build_marasi_model(
        model_name=MODEL_NAME,
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        val_metrics = evaluate_model(model, data, data.val_mask)
        val_f1 = val_metrics["illicit_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"epoch={epoch:04d} | "
                f"loss={loss.item():.5f} | "
                f"val_illicit_f1={val_f1:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_model(model, data, data.train_mask)
    val_metrics = evaluate_model(model, data, data.val_mask)
    test_metrics = evaluate_model(model, data, data.test_mask)

    print("\nShared GraphSAGE final metrics")
    print("-" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Train illicit F1: {train_metrics['illicit_f1']:.4f}")
    print(f"Val illicit F1:   {val_metrics['illicit_f1']:.4f}")
    print(f"Test illicit F1:  {test_metrics['illicit_f1']:.4f}")

    metrics = {
        "model": MODEL_NAME,
        "best_epoch": best_epoch,
        "train_accuracy": train_metrics["accuracy"],
        "train_illicit_precision": train_metrics["illicit_precision"],
        "train_illicit_recall": train_metrics["illicit_recall"],
        "train_illicit_f1": train_metrics["illicit_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_illicit_precision": val_metrics["illicit_precision"],
        "val_illicit_recall": val_metrics["illicit_recall"],
        "val_illicit_f1": val_metrics["illicit_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_illicit_precision": test_metrics["illicit_precision"],
        "test_illicit_recall": test_metrics["illicit_recall"],
        "test_illicit_f1": test_metrics["illicit_f1"],
    }

    return model, metrics


@torch.no_grad()
def get_predictions(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    return logits, probs, pred


def get_ego_node_count(data, node_id: int):
    subset, _, _, _ = k_hop_subgraph(
        node_idx=int(node_id),
        num_hops=EGO_HOPS,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        flow="source_to_target",
    )
    return int(subset.numel())


@torch.no_grad()
def select_shared_nodes(model, data, num_nodes: int):
    print("\n" + "=" * 80)
    print("Selecting shared explanation nodes")
    print("=" * 80)

    _, probs, pred = get_predictions(model, data)

    candidate_mask = (
        data.test_mask
        & (data.y == ILLICIT_LABEL)
        & (pred == ILLICIT_LABEL)
    )

    candidate_nodes = candidate_mask.nonzero(as_tuple=False).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError("No correctly predicted illicit test nodes found.")

    illicit_probs = probs[candidate_nodes, ILLICIT_LABEL]
    sorted_idx = torch.argsort(illicit_probs, descending=True)

    rows = []
    skipped_tiny = 0

    for idx in sorted_idx:
        node_id = int(candidate_nodes[idx])
        prob_illicit = float(probs[node_id, ILLICIT_LABEL])
        ego_nodes = get_ego_node_count(data, node_id)

        if ego_nodes < MIN_EGO_NODES:
            skipped_tiny += 1
            continue

        rows.append(
            {
                "dataset": "Elliptic",
                "feature_setting": "tx+agg",
                "model": MODEL_NAME,
                "node_id": node_id,
                "true_label": int(data.y[node_id]),
                "pred_label": int(pred[node_id]),
                "pred_prob_illicit": prob_illicit,
                "ego_hops": EGO_HOPS,
                "num_ego_nodes": ego_nodes,
            }
        )

        print(
            f"Selected node={node_id} | "
            f"prob_illicit={prob_illicit:.6f} | "
            f"ego_nodes={ego_nodes}"
        )

        if len(rows) >= num_nodes:
            break

    if len(rows) == 0:
        raise RuntimeError("No suitable shared explanation nodes found.")

    print("\nShared node selection summary")
    print("-" * 80)
    print(f"Selected nodes: {len(rows)}")
    print(f"Skipped tiny ego graphs: {skipped_tiny}")

    return pd.DataFrame(rows)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Preparing shared GraphSAGE checkpoint and shared explanation nodes.")

    data = load_data().to(device)

    print("\nElliptic Marasi-style graph")
    print("=" * 80)
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.edge_index.size(1)}")
    print(f"num_features: {data.num_features}")
    print(f"train_nodes: {int(data.train_mask.sum())}")
    print(f"val_nodes: {int(data.val_mask.sum())}")
    print(f"test_nodes: {int(data.test_mask.sum())}")
    print(f"illicit nodes label 0: {int((data.y == ILLICIT_LABEL).sum())}")
    print(f"licit nodes label 1: {int((data.y == LICIT_LABEL).sum())}")

    model, metrics = train_model(data, device)

    checkpoint = {
        "model_name": MODEL_NAME,
        "feature_setting": "tx+agg",
        "seed": SEED,
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": 2,
        "in_channels": data.num_features,
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"\nSaved shared model checkpoint to: {MODEL_PATH}")

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    print(f"Saved shared model metrics to: {METRICS_PATH}")

    shared_nodes_df = select_shared_nodes(
        model=model,
        data=data,
        num_nodes=NUM_SHARED_NODES,
    )

    shared_nodes_df.to_csv(NODE_LIST_PATH, index=False)
    print(f"\nSaved shared explanation nodes to: {NODE_LIST_PATH}")

    print("\nShared explanation nodes")
    print("=" * 80)
    print(shared_nodes_df.to_string(index=False))


if __name__ == "__main__":
    main()
