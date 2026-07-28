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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MODELS = ["graphsage", "gatv2"]

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

EXPLAINER_EPOCHS = 100
NUM_NODES_TO_EXPLAIN = 10
TOP_K_EDGES = 20

# Marasi label convention:
# illicit = 0
# licit = 1
ILLICIT_LABEL = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    """
    Loads the labelled Elliptic transaction graph using the Marasi-style setup:
    - unknown nodes removed
    - tx+agg features
    - random train/val/test masks
    - label convention: illicit=0, licit=1
    """
    data = build_marasi_elliptic_data(
        feature_setting="tx+agg",
        seed=SEED,
    )

    return data


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


def train_model(model_name: str, data, device):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
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
    best_epoch = 0
    best_state = None
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
                f"{model_name} | epoch={epoch:04d} | "
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

    print("\nFinal metrics")
    print("-" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Train illicit F1: {train_metrics['illicit_f1']:.4f}")
    print(f"Val illicit F1:   {val_metrics['illicit_f1']:.4f}")
    print(f"Test illicit F1:  {test_metrics['illicit_f1']:.4f}")

    return model, {
        "best_epoch": best_epoch,
        "train_illicit_f1": train_metrics["illicit_f1"],
        "val_illicit_f1": val_metrics["illicit_f1"],
        "test_illicit_f1": test_metrics["illicit_f1"],
    }


@torch.no_grad()
def select_nodes_to_explain(model, data, num_nodes: int):
    """
    Select correctly predicted illicit test nodes.
    """
    model.eval()

    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)

    candidate_mask = (
        data.test_mask
        & (data.y == ILLICIT_LABEL)
        & (pred == ILLICIT_LABEL)
    )

    candidate_nodes = candidate_mask.nonzero(as_tuple=False).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError("No correctly predicted illicit test nodes found.")

    # Pick highest-confidence illicit predictions.
    illicit_probs = probs[candidate_nodes, ILLICIT_LABEL]
    sorted_idx = torch.argsort(illicit_probs, descending=True)
    selected = candidate_nodes[sorted_idx[:num_nodes]]

    return selected.detach().cpu().tolist(), probs.detach().cpu(), pred.detach().cpu()


def make_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=EXPLAINER_EPOCHS,
            lr=0.01,
        ),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def summarise_explanation(
    explanation,
    data,
    node_id: int,
    model_name: str,
    model_metrics: dict,
    probs,
    pred,
):
    edge_mask = explanation.edge_mask.detach().cpu()

    num_edges = int(edge_mask.numel())
    top_k = min(TOP_K_EDGES, num_edges)

    top_values, top_indices = torch.topk(edge_mask, k=top_k)

    top_edge_indices = top_indices.tolist()
    top_edge_scores = [float(v) for v in top_values.tolist()]

    explanation_edges = data.edge_index[:, top_indices.to(data.edge_index.device)]
    explanation_nodes = torch.unique(explanation_edges).detach().cpu().tolist()

    sparsity = 1.0 - (top_k / num_edges)

    return {
        "dataset": "Elliptic",
        "feature_setting": "tx+agg",
        "explainer": "GNNExplainer",
        "model": model_name,
        "node_id": int(node_id),
        "true_label": int(data.y[node_id].detach().cpu()),
        "pred_label": int(pred[node_id]),
        "pred_prob_illicit": float(probs[node_id, ILLICIT_LABEL]),
        "model_best_epoch": int(model_metrics["best_epoch"]),
        "model_test_illicit_f1": float(model_metrics["test_illicit_f1"]),
        "num_graph_nodes": int(data.num_nodes),
        "num_graph_edges": int(data.edge_index.size(1)),
        "num_explanation_edges": int(top_k),
        "num_explanation_nodes": int(len(explanation_nodes)),
        "sparsity_top_k": float(sparsity),
        "edge_mask_mean": float(edge_mask.mean()),
        "edge_mask_max": float(edge_mask.max()),
        "edge_mask_min": float(edge_mask.min()),
        "top_edge_indices": json.dumps(top_edge_indices),
        "top_edge_scores": json.dumps(top_edge_scores),
        "explanation_nodes": json.dumps([int(n) for n in explanation_nodes]),
    }


def run_gnnexplainer_for_model(model_name: str, data, device):
    model, model_metrics = train_model(model_name, data, device)

    model.eval()

    selected_nodes, probs, pred = select_nodes_to_explain(
        model=model,
        data=data,
        num_nodes=NUM_NODES_TO_EXPLAIN,
    )

    print("\nSelected illicit test nodes to explain:")
    print(selected_nodes)

    explainer = make_explainer(model)

    rows = []

    for i, node_id in enumerate(selected_nodes, start=1):
        print("\n" + "-" * 80)
        print(f"{model_name} | explaining node {i}/{len(selected_nodes)} | node_id={node_id}")
        print("-" * 80)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=int(node_id),
        )

        row = summarise_explanation(
            explanation=explanation,
            data=data,
            node_id=int(node_id),
            model_name=model_name,
            model_metrics=model_metrics,
            probs=probs,
            pred=pred,
        )

        rows.append(row)

        partial_df = pd.DataFrame(rows)
        partial_path = OUTPUT_DIR / f"gnnexplainer_elliptic_{model_name}_partial.csv"
        partial_df.to_csv(partial_path, index=False)

    return rows


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    data = load_data()

    print("\nElliptic Marasi-style graph")
    print("=" * 80)
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.edge_index.size(1)}")
    print(f"num_features: {data.num_features}")
    print(f"train_nodes: {int(data.train_mask.sum())}")
    print(f"val_nodes: {int(data.val_mask.sum())}")
    print(f"test_nodes: {int(data.test_mask.sum())}")
    print(f"illicit nodes label 0: {int((data.y == ILLICIT_LABEL).sum())}")
    print(f"licit nodes label 1: {int((data.y == 1).sum())}")

    data = data.to(device)

    all_rows = []

    for model_name in MODELS:
        rows = run_gnnexplainer_for_model(model_name, data, device)
        all_rows.extend(rows)

        model_path = OUTPUT_DIR / f"gnnexplainer_elliptic_{model_name}.csv"
        pd.DataFrame(rows).to_csv(model_path, index=False)
        print(f"\nSaved {model_name} explanations to: {model_path}")

    final_df = pd.DataFrame(all_rows)
    final_path = OUTPUT_DIR / "gnnexplainer_elliptic_results.csv"
    final_df.to_csv(final_path, index=False)

    print("\nGNNExplainer summary")
    print("=" * 80)
    print(
        final_df[
            [
                "model",
                "node_id",
                "true_label",
                "pred_label",
                "pred_prob_illicit",
                "num_explanation_edges",
                "num_explanation_nodes",
                "sparsity_top_k",
                "edge_mask_mean",
                "edge_mask_max",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved final results to: {final_path}")


if __name__ == "__main__":
    main()