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
from torch_geometric.explain.algorithm import PGExplainer

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MODELS = ["graphsage"]

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

# PGExplainer settings.
# Keep this conservative first to avoid NaN loss.
PG_EXPLAINER_EPOCHS = 20
PG_EXPLAINER_LR = 0.00001
NUM_EXPLAINER_TRAIN_NODES = 20
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
def get_predictions(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    return logits, probs, pred


@torch.no_grad()
def select_correct_illicit_nodes(model, data, mask, num_nodes: int):
    """
    Select high-confidence correctly predicted illicit nodes.
    """
    _, probs, pred = get_predictions(model, data)

    candidate_mask = (
        mask
        & (data.y == ILLICIT_LABEL)
        & (pred == ILLICIT_LABEL)
    )

    candidate_nodes = candidate_mask.nonzero(as_tuple=False).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError("No correctly predicted illicit nodes found.")

    illicit_probs = probs[candidate_nodes, ILLICIT_LABEL]
    sorted_idx = torch.argsort(illicit_probs, descending=True)
    selected = candidate_nodes[sorted_idx[:num_nodes]]

    return selected.detach().cpu().tolist(), probs.detach().cpu(), pred.detach().cpu()


def make_pgexplainer(model, device):
    """
    PyG 2.8.0 PGExplainer only supports explanation_type='phenomenon'.

    To explain model behaviour, we pass the model's predicted labels as
    the target during PGExplainer training and explanation.
    """
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(
            epochs=PG_EXPLAINER_EPOCHS,
            lr=PG_EXPLAINER_LR,
        ),
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    # PGExplainer has its own neural network.
    # Move it to the same device as model/data.
    explainer.algorithm.to(device)

    return explainer


def train_pgexplainer(explainer, model, data, train_node_ids, target):
    """
    Train PGExplainer using PyG's official training interface.

    Since PGExplainer only supports phenomenon explanations in PyG 2.8.0,
    we use the model predictions as the target. This makes the explainer
    approximate the model's own decision behaviour.
    """
    print("\n" + "=" * 80)
    print("Training PGExplainer")
    print("=" * 80)
    print(f"PGExplainer training nodes: {len(train_node_ids)}")
    print(f"PGExplainer epochs: {PG_EXPLAINER_EPOCHS}")
    print(f"PGExplainer learning rate: {PG_EXPLAINER_LR}")

    model.eval()

    for epoch in range(PG_EXPLAINER_EPOCHS):
        epoch_losses = []

        for node_id in train_node_ids:
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=data.x,
                edge_index=data.edge_index,
                target=target,
                index=int(node_id),
            )

            loss_value = float(loss)

            if np.isnan(loss_value) or np.isinf(loss_value):
                raise RuntimeError(
                    f"PGExplainer produced invalid loss: {loss_value}. "
                    f"epoch={epoch + 1}, node_id={node_id}. "
                    "The current PGExplainer run should not be used."
                )

            epoch_losses.append(loss_value)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        print(
            f"PGExplainer | epoch={epoch + 1:03d}/{PG_EXPLAINER_EPOCHS} | "
            f"loss={mean_loss:.5f}"
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
    if explanation.edge_mask is None:
        raise RuntimeError(
            f"PGExplainer returned no edge_mask for model={model_name}, node={node_id}"
        )

    edge_mask = explanation.edge_mask.detach().cpu()

    if torch.isnan(edge_mask).any() or torch.isinf(edge_mask).any():
        raise RuntimeError(
            f"PGExplainer produced invalid edge mask for model={model_name}, node={node_id}"
        )

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
        "explainer": "PGExplainer",
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


def run_pgexplainer_for_model(model_name: str, data, device):
    model, model_metrics = train_model(model_name, data, device)
    model.eval()

    _, probs, pred = get_predictions(model, data)

    # PGExplainer in PyG 2.8.0 only supports phenomenon explanations.
    # We use the model predictions as the target, so it explains model behaviour.
    target = pred.to(device)

    train_node_ids, _, _ = select_correct_illicit_nodes(
        model=model,
        data=data,
        mask=data.train_mask,
        num_nodes=NUM_EXPLAINER_TRAIN_NODES,
    )

    test_node_ids, probs_cpu, pred_cpu = select_correct_illicit_nodes(
        model=model,
        data=data,
        mask=data.test_mask,
        num_nodes=NUM_NODES_TO_EXPLAIN,
    )

    print("\nSelected PGExplainer training nodes:")
    print(train_node_ids)

    print("\nSelected illicit test nodes to explain:")
    print(test_node_ids)

    explainer = make_pgexplainer(model, device)

    train_pgexplainer(
        explainer=explainer,
        model=model,
        data=data,
        train_node_ids=train_node_ids,
        target=target,
    )

    rows = []

    for i, node_id in enumerate(test_node_ids, start=1):
        print("\n" + "-" * 80)
        print(
            f"{model_name} | PGExplainer explaining node "
            f"{i}/{len(test_node_ids)} | node_id={node_id}"
        )
        print("-" * 80)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=target,
            index=int(node_id),
        )

        row = summarise_explanation(
            explanation=explanation,
            data=data,
            node_id=int(node_id),
            model_name=model_name,
            model_metrics=model_metrics,
            probs=probs_cpu,
            pred=pred_cpu,
        )

        rows.append(row)

        partial_df = pd.DataFrame(rows)
        partial_path = OUTPUT_DIR / f"pgexplainer_elliptic_{model_name}_partial.csv"
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
        rows = run_pgexplainer_for_model(model_name, data, device)
        all_rows.extend(rows)

        model_path = OUTPUT_DIR / f"pgexplainer_elliptic_{model_name}.csv"
        pd.DataFrame(rows).to_csv(model_path, index=False)
        print(f"\nSaved {model_name} PGExplainer explanations to: {model_path}")

    final_df = pd.DataFrame(all_rows)
    final_path = OUTPUT_DIR / "pgexplainer_elliptic_results.csv"
    final_df.to_csv(final_path, index=False)

    print("\nPGExplainer summary")
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

    print(f"\nSaved final PGExplainer results to: {final_path}")


if __name__ == "__main__":
    main()