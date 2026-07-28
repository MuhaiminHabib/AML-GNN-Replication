from pathlib import Path
import sys
import copy
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.utils import subgraph

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


EXPLAINER_RESULTS_PATH = Path("outputs/explainers/gnnexplainer_elliptic_results.csv")
OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

ILLICIT_LABEL = 0
LICIT_LABEL = 1


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
def evaluate_illicit_f1(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = pred[mask].detach().cpu().numpy()

    return f1_score(
        y_true,
        y_pred,
        pos_label=ILLICIT_LABEL,
        zero_division=0,
    )


def train_model(model_name: str, data, device):
    print("\n" + "=" * 80)
    print(f"Training model for faithfulness evaluation: {model_name}")
    print("=" * 80)

    model = build_marasi_model(
        model_name=model_name,
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
        )

        loss.backward()
        optimizer.step()

        val_f1 = evaluate_illicit_f1(model, data, data.val_mask)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"{model_name} | epoch={epoch:04d} | "
                f"loss={loss.item():.5f} | val_illicit_f1={val_f1:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_f1 = evaluate_illicit_f1(model, data, data.train_mask)
    val_f1 = evaluate_illicit_f1(model, data, data.val_mask)
    test_f1 = evaluate_illicit_f1(model, data, data.test_mask)

    print("\nFinal model metrics")
    print("-" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Train illicit F1: {train_f1:.4f}")
    print(f"Val illicit F1:   {val_f1:.4f}")
    print(f"Test illicit F1:  {test_f1:.4f}")

    return model, {
        "best_epoch": best_epoch,
        "train_illicit_f1": train_f1,
        "val_illicit_f1": val_f1,
        "test_illicit_f1": test_f1,
    }


@torch.no_grad()
def predict_for_node(model, x, edge_index, node_id: int):
    model.eval()

    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[node_id].argmax().detach().cpu())
    illicit_prob = float(probs[node_id, ILLICIT_LABEL].detach().cpu())
    licit_prob = float(probs[node_id, LICIT_LABEL].detach().cpu())

    return pred_label, illicit_prob, licit_prob


def remove_edges(edge_index: torch.Tensor, edge_ids: list[int]):
    num_edges = edge_index.size(1)

    keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)

    if edge_ids:
        edge_ids_tensor = torch.tensor(edge_ids, dtype=torch.long, device=edge_index.device)
        edge_ids_tensor = edge_ids_tensor[
            (edge_ids_tensor >= 0) & (edge_ids_tensor < num_edges)
        ]
        keep_mask[edge_ids_tensor] = False

    return edge_index[:, keep_mask]


def keep_only_edges(edge_index: torch.Tensor, edge_ids: list[int]):
    num_edges = edge_index.size(1)

    if not edge_ids:
        return edge_index[:, :0]

    edge_ids_tensor = torch.tensor(edge_ids, dtype=torch.long, device=edge_index.device)
    edge_ids_tensor = edge_ids_tensor[
        (edge_ids_tensor >= 0) & (edge_ids_tensor < num_edges)
    ]

    if edge_ids_tensor.numel() == 0:
        return edge_index[:, :0]

    return edge_index[:, edge_ids_tensor]


def evaluate_one_explanation(model, data, row):
    node_id = int(row["node_id"])
    model_name = row["model"]

    top_edge_indices = json.loads(row["top_edge_indices"])

    original_pred, original_illicit_prob, original_licit_prob = predict_for_node(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        node_id=node_id,
    )

    deletion_edge_index = remove_edges(
        edge_index=data.edge_index,
        edge_ids=top_edge_indices,
    )

    deletion_pred, deletion_illicit_prob, deletion_licit_prob = predict_for_node(
        model=model,
        x=data.x,
        edge_index=deletion_edge_index,
        node_id=node_id,
    )

    insertion_edge_index = keep_only_edges(
        edge_index=data.edge_index,
        edge_ids=top_edge_indices,
    )

    insertion_pred, insertion_illicit_prob, insertion_licit_prob = predict_for_node(
        model=model,
        x=data.x,
        edge_index=insertion_edge_index,
        node_id=node_id,
    )

    deletion_drop = original_illicit_prob - deletion_illicit_prob
    insertion_gap = original_illicit_prob - insertion_illicit_prob

    deletion_label_flip = int(deletion_pred != original_pred)
    insertion_preservation = int(insertion_pred == original_pred)

    return {
        "dataset": row["dataset"],
        "model": model_name,
        "explainer": row["explainer"],
        "node_id": node_id,
        "true_label": int(row["true_label"]),
        "original_pred_label": original_pred,
        "original_prob_illicit": original_illicit_prob,
        "deletion_pred_label": deletion_pred,
        "deletion_prob_illicit": deletion_illicit_prob,
        "deletion_drop": deletion_drop,
        "deletion_label_flip": deletion_label_flip,
        "insertion_pred_label": insertion_pred,
        "insertion_prob_illicit": insertion_illicit_prob,
        "insertion_gap": insertion_gap,
        "insertion_preservation": insertion_preservation,
        "num_explanation_edges": int(row["num_explanation_edges"]),
        "num_explanation_nodes": int(row["num_explanation_nodes"]),
        "sparsity_top_k": float(row["sparsity_top_k"]),
        "edge_mask_mean": float(row["edge_mask_mean"]),
        "edge_mask_max": float(row["edge_mask_max"]),
    }


def main():
    set_seed(SEED)

    if not EXPLAINER_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {EXPLAINER_RESULTS_PATH}. "
            "Run scripts/run_gnnexplainer.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_data()

    print("\nElliptic Marasi-style graph")
    print("=" * 80)
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.edge_index.size(1)}")
    print(f"num_features: {data.num_features}")
    print(f"train_nodes: {int(data.train_mask.sum())}")
    print(f"val_nodes: {int(data.val_mask.sum())}")
    print(f"test_nodes: {int(data.test_mask.sum())}")

    data = data.to(device)

    explanation_df = pd.read_csv(EXPLAINER_RESULTS_PATH)

    all_rows = []

    for model_name in sorted(explanation_df["model"].unique()):
        model_df = explanation_df[explanation_df["model"] == model_name].copy()

        model, model_metrics = train_model(model_name, data, device)
        model.eval()

        for _, row in model_df.iterrows():
            print(
                f"Evaluating faithfulness | model={model_name} | "
                f"node_id={int(row['node_id'])}"
            )

            faithfulness_row = evaluate_one_explanation(
                model=model,
                data=data,
                row=row,
            )

            faithfulness_row["model_best_epoch"] = model_metrics["best_epoch"]
            faithfulness_row["model_test_illicit_f1"] = model_metrics["test_illicit_f1"]

            all_rows.append(faithfulness_row)

            partial_path = OUTPUT_DIR / "gnnexplainer_elliptic_faithfulness_partial.csv"
            pd.DataFrame(all_rows).to_csv(partial_path, index=False)

    results_df = pd.DataFrame(all_rows)

    results_path = OUTPUT_DIR / "gnnexplainer_elliptic_faithfulness.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(["dataset", "model", "explainer"])
        .agg(
            nodes_explained=("node_id", "count"),
            deletion_drop_mean=("deletion_drop", "mean"),
            deletion_drop_std=("deletion_drop", "std"),
            insertion_prob_mean=("insertion_prob_illicit", "mean"),
            insertion_prob_std=("insertion_prob_illicit", "std"),
            insertion_gap_mean=("insertion_gap", "mean"),
            deletion_label_flip_rate=("deletion_label_flip", "mean"),
            insertion_preservation_rate=("insertion_preservation", "mean"),
            sparsity_mean=("sparsity_top_k", "mean"),
            num_explanation_edges_mean=("num_explanation_edges", "mean"),
            num_explanation_nodes_mean=("num_explanation_nodes", "mean"),
            edge_mask_mean=("edge_mask_mean", "mean"),
            edge_mask_max_mean=("edge_mask_max", "mean"),
        )
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "gnnexplainer_elliptic_faithfulness_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nGNNExplainer faithfulness results")
    print("=" * 80)
    print(
        results_df[
            [
                "model",
                "node_id",
                "original_prob_illicit",
                "deletion_prob_illicit",
                "deletion_drop",
                "deletion_label_flip",
                "insertion_prob_illicit",
                "insertion_preservation",
                "sparsity_top_k",
            ]
        ].to_string(index=False)
    )

    print("\nGNNExplainer faithfulness summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print(f"\nSaved detailed faithfulness results to: {results_path}")
    print(f"Saved faithfulness summary to: {summary_path}")


if __name__ == "__main__":
    main()