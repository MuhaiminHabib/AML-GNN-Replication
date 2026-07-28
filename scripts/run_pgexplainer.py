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

# Start with GraphSAGE only.
# Once this works, we can try adding "gatv2".
MODELS = ["graphsage"]

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

# PGExplainer settings.
# Very small LR because full-graph AML PGExplainer previously gave NaN.
PG_EXPLAINER_EPOCHS = 20
PG_EXPLAINER_LR = 0.000001

NUM_EXPLAINER_TRAIN_NODES = 30
NUM_NODES_TO_EXPLAIN = 10
TOP_K_EDGES = 20

# Avoid extremely saturated training predictions.
# Earlier PGExplainer crashed on very high-confidence nodes.
TRAIN_PROB_MIN = 0.55
TRAIN_PROB_MAX = 0.95

# Marasi label convention:
# illicit = 0
# licit = 1
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
def select_pgexplainer_training_nodes(model, data, num_nodes: int):
    """
    Select correctly predicted illicit training nodes for PGExplainer.

    This keeps the original full-graph PGExplainer setting, but avoids
    extremely saturated predictions because those caused NaN loss earlier.
    """
    model.eval()

    _, probs, pred = get_predictions(model, data)

    candidate_mask = (
        data.train_mask
        & (data.y == ILLICIT_LABEL)
        & (pred == ILLICIT_LABEL)
    )

    candidate_nodes = candidate_mask.nonzero(as_tuple=False).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError("No correctly predicted illicit training nodes found.")

    selected = []
    skipped_low = 0
    skipped_high = 0

    # Do not sort descending. We want stable medium-confidence examples.
    for node_id_tensor in candidate_nodes:
        node_id = int(node_id_tensor)
        prob_illicit = float(probs[node_id, ILLICIT_LABEL])

        if prob_illicit < TRAIN_PROB_MIN:
            skipped_low += 1
            continue

        if prob_illicit > TRAIN_PROB_MAX:
            skipped_high += 1
            continue

        selected.append(node_id)

        if len(selected) >= num_nodes:
            break

    if len(selected) == 0:
        raise RuntimeError(
            "No stable PGExplainer training nodes found. "
            "Try increasing TRAIN_PROB_MAX to 0.98."
        )

    print("\nSelected PGExplainer training nodes")
    print("-" * 80)

    for node_id in selected:
        print(
            f"node={node_id} | "
            f"prob_illicit={float(probs[node_id, ILLICIT_LABEL]):.6f}"
        )

    print("\nPGExplainer training-node selection summary")
    print("-" * 80)
    print(f"Selected: {len(selected)}")
    print(f"Skipped low confidence: {skipped_low}")
    print(f"Skipped high/saturated confidence: {skipped_high}")

    return selected


@torch.no_grad()
def select_correct_illicit_test_nodes(model, data, num_nodes: int):
    """
    Select correctly predicted illicit test nodes for explanation.

    For explanation targets, we can still use high-confidence illicit nodes,
    but we keep the same original full-graph setting.
    """
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
    selected = candidate_nodes[sorted_idx[:num_nodes]]

    print("\nSelected illicit test nodes to explain")
    print("-" * 80)

    for node_id_tensor in selected:
        node_id = int(node_id_tensor)
        print(
            f"node={node_id} | "
            f"prob_illicit={float(probs[node_id, ILLICIT_LABEL]):.6f}"
        )

    return selected.detach().cpu().tolist(), probs.detach().cpu(), pred.detach().cpu()


def make_pgexplainer(model, device):
    """
    PyG 2.8 PGExplainer supports phenomenon explanations.

    To explain model behaviour, we use the model's own predicted labels as
    the phenomenon target. This is the standard workaround for current PyG.
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

    explainer.algorithm.to(device)

    return explainer


def train_pgexplainer(explainer, model, data, train_node_ids, target):
    """
    Train official PyG PGExplainer on the full Elliptic graph.

    Difference from the earlier failing version:
    - smaller learning rate
    - medium-confidence training nodes
    - skip NaN/Inf nodes instead of crashing immediately
    """
    print("\n" + "=" * 80)
    print("Training PGExplainer on full graph")
    print("=" * 80)
    print(f"PGExplainer training nodes: {len(train_node_ids)}")
    print(f"PGExplainer epochs: {PG_EXPLAINER_EPOCHS}")
    print(f"PGExplainer learning rate: {PG_EXPLAINER_LR}")

    model.eval()

    for epoch in range(PG_EXPLAINER_EPOCHS):
        epoch_losses = []
        skipped_nan = 0

        for node_id in train_node_ids:
            try:
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
                    skipped_nan += 1
                    print(
                        f"WARNING: NaN/Inf PGExplainer loss skipped | "
                        f"epoch={epoch + 1}, node_id={node_id}"
                    )
                    continue

                epoch_losses.append(loss_value)

            except Exception as e:
                skipped_nan += 1
                print(
                    f"WARNING: PGExplainer skipped node {node_id} "
                    f"at epoch {epoch + 1} due to error: {repr(e)}"
                )
                continue

        if len(epoch_losses) == 0:
            raise RuntimeError(
                f"All PGExplainer training nodes failed at epoch {epoch + 1}. "
                "Try lowering PG_EXPLAINER_LR further or widening TRAIN_PROB_MAX."
            )

        mean_loss = float(np.mean(epoch_losses))

        print(
            f"PGExplainer | epoch={epoch + 1:03d}/{PG_EXPLAINER_EPOCHS} | "
            f"loss={mean_loss:.5f} | skipped={skipped_nan}"
        )


def summarise_explanation(
    explanation,
    data,
    node_id: int,
    model_name: str,
    model_metrics: dict,
    probs_cpu,
    pred_cpu,
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

    if top_k == 0:
        top_edge_indices = []
        top_edge_scores = []
        explanation_nodes = []
        explanation_edge_pairs = []
        sparsity = 1.0
    else:
        top_values, top_indices = torch.topk(edge_mask, k=top_k)

        top_edge_indices = top_indices.tolist()
        top_edge_scores = [float(v) for v in top_values.tolist()]

        explanation_edges = data.edge_index[:, top_indices.to(data.edge_index.device)]
        explanation_nodes_tensor = torch.unique(explanation_edges).detach().cpu()

        explanation_nodes = [int(n) for n in explanation_nodes_tensor.tolist()]

        explanation_edge_pairs = []

        explanation_edges_cpu = explanation_edges.detach().cpu()

        for i in range(explanation_edges_cpu.size(1)):
            src = int(explanation_edges_cpu[0, i])
            dst = int(explanation_edges_cpu[1, i])
            explanation_edge_pairs.append([src, dst])

        sparsity = 1.0 - (top_k / max(1, num_edges))

    return {
        "dataset": "Elliptic",
        "feature_setting": "tx+agg",
        "explainer": "PGExplainer",
        "model": model_name,
        "node_id": int(node_id),
        "true_label": int(data.y[node_id].detach().cpu()),
        "pred_label": int(pred_cpu[node_id]),
        "pred_prob_illicit": float(probs_cpu[node_id, ILLICIT_LABEL]),
        "model_best_epoch": int(model_metrics["best_epoch"]),
        "model_test_illicit_f1": float(model_metrics["test_illicit_f1"]),
        "num_graph_nodes": int(data.num_nodes),
        "num_graph_edges": int(data.edge_index.size(1)),
        "num_explanation_edges": int(top_k),
        "num_explanation_nodes": int(len(explanation_nodes)),
        "sparsity_top_k": float(sparsity),
        "edge_mask_mean": float(edge_mask.mean()) if edge_mask.numel() > 0 else 0.0,
        "edge_mask_max": float(edge_mask.max()) if edge_mask.numel() > 0 else 0.0,
        "edge_mask_min": float(edge_mask.min()) if edge_mask.numel() > 0 else 0.0,
        "top_edge_indices": json.dumps([int(i) for i in top_edge_indices]),
        "top_edge_scores": json.dumps(top_edge_scores),
        "explanation_nodes": json.dumps(explanation_nodes),
        "explanation_original_edge_pairs": json.dumps(explanation_edge_pairs),
    }


def run_pgexplainer_for_model(model_name: str, data, device):
    model, model_metrics = train_model(model_name, data, device)
    model.eval()

    _, probs, pred = get_predictions(model, data)

    # PyG PGExplainer requires phenomenon target.
    # We use the model's own predictions to explain model behaviour.
    target = pred.to(device)

    train_node_ids = select_pgexplainer_training_nodes(
        model=model,
        data=data,
        num_nodes=NUM_EXPLAINER_TRAIN_NODES,
    )

    test_node_ids, probs_cpu, pred_cpu = select_correct_illicit_test_nodes(
        model=model,
        data=data,
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

        try:
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
                probs_cpu=probs_cpu,
                pred_cpu=pred_cpu,
            )

            rows.append(row)

        except Exception as e:
            print(
                f"WARNING: failed to explain node {node_id} "
                f"for model {model_name}: {repr(e)}"
            )
            continue

        partial_df = pd.DataFrame(rows)
        partial_path = OUTPUT_DIR / f"pgexplainer_elliptic_{model_name}_partial.csv"
        partial_df.to_csv(partial_path, index=False)

    model_path = OUTPUT_DIR / f"pgexplainer_elliptic_{model_name}.csv"
    pd.DataFrame(rows).to_csv(model_path, index=False)

    print(f"\nSaved {model_name} PGExplainer explanations to: {model_path}")

    return rows


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print("Running official full-graph PyG PGExplainer.")

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

    all_rows = []

    for model_name in MODELS:
        rows = run_pgexplainer_for_model(model_name, data, device)
        all_rows.extend(rows)

    final_df = pd.DataFrame(all_rows)
    final_path = OUTPUT_DIR / "pgexplainer_elliptic_results.csv"
    final_df.to_csv(final_path, index=False)

    print("\nPGExplainer summary")
    print("=" * 80)

    if len(final_df) > 0:
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
    else:
        print("No PGExplainer explanations were produced.")

    print(f"\nSaved final PGExplainer results to: {final_path}")


if __name__ == "__main__":
    main()