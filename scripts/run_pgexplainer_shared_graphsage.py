from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "shared_graphsage_model.pt"
NODE_LIST_PATH = OUTPUT_DIR / "shared_graphsage_explanation_nodes.csv"

OUTPUT_FILE = OUTPUT_DIR / "pgexplainer_shared_graphsage_results.csv"
PARTIAL_FILE = OUTPUT_DIR / "pgexplainer_shared_graphsage_partial.csv"

SEED = 42
MODEL_NAME = "graphsage"

ILLICIT_LABEL = 0
LICIT_LABEL = 1

# Official full-graph PyG PGExplainer.
PG_EXPLAINER_EPOCHS = 20
PG_EXPLAINER_LR = 0.000001

# PGExplainer needs its own training nodes.
# These are NOT the explanation target nodes.
NUM_EXPLAINER_TRAIN_NODES = 30

# Avoid saturated training nodes because they caused NaN loss earlier.
TRAIN_PROB_MIN = 0.55
TRAIN_PROB_MAX = 0.95

TOP_K_EDGES = 20


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    return build_marasi_elliptic_data(
        feature_setting="tx+agg",
        seed=SEED,
    )


def load_shared_model(data, device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing shared model checkpoint: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_marasi_model(
        model_name=checkpoint["model_name"],
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


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


def make_pgexplainer(model, device):
    """
    PyG 2.8 PGExplainer supports phenomenon explanations.

    To explain model behaviour, we use the model's own predicted labels as
    the phenomenon target.
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

    This script keeps the original full-graph PGExplainer setting.
    To handle numerical instability, invalid training nodes are skipped
    instead of modifying the internal PyG PGExplainer loss.
    """
    print("\n" + "=" * 80)
    print("Training PGExplainer on full graph")
    print("=" * 80)
    print(f"PGExplainer training nodes: {len(train_node_ids)}")
    print(f"PGExplainer epochs: {PG_EXPLAINER_EPOCHS}")
    print(f"PGExplainer learning rate: {PG_EXPLAINER_LR}")

    model.eval()

    training_log_rows = []

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

        training_log_rows.append(
            {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "usable_nodes": len(epoch_losses),
                "skipped_nodes": skipped_nan,
                "total_training_nodes": len(train_node_ids),
            }
        )

        print(
            f"PGExplainer | epoch={epoch + 1:03d}/{PG_EXPLAINER_EPOCHS} | "
            f"loss={mean_loss:.5f} | "
            f"usable={len(epoch_losses)} | skipped={skipped_nan}"
        )

    training_log_df = pd.DataFrame(training_log_rows)
    training_log_path = OUTPUT_DIR / "pgexplainer_shared_graphsage_training_log.csv"
    training_log_df.to_csv(training_log_path, index=False)

    print(f"\nSaved PGExplainer training log to: {training_log_path}")


def summarise_explanation(
    explanation,
    data,
    node_id: int,
    probs_cpu,
    pred_cpu,
    checkpoint,
):
    if explanation.edge_mask is None:
        raise RuntimeError(f"PGExplainer returned no edge mask for node {node_id}")

    edge_mask = explanation.edge_mask.detach().cpu()

    if torch.isnan(edge_mask).any() or torch.isinf(edge_mask).any():
        raise RuntimeError(f"PGExplainer produced invalid edge mask for node {node_id}")

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

        top_edge_indices = [int(i) for i in top_indices.tolist()]
        top_edge_scores = [float(v) for v in top_values.tolist()]

        explanation_edges = data.edge_index[:, top_indices.to(data.edge_index.device)]
        explanation_nodes_tensor = torch.unique(explanation_edges).detach().cpu()

        explanation_nodes = [int(n) for n in explanation_nodes_tensor.tolist()]

        explanation_edges_cpu = explanation_edges.detach().cpu()
        explanation_edge_pairs = []

        for i in range(explanation_edges_cpu.size(1)):
            src = int(explanation_edges_cpu[0, i])
            dst = int(explanation_edges_cpu[1, i])
            explanation_edge_pairs.append([src, dst])

        sparsity = 1.0 - (top_k / max(1, num_edges))

    model_metrics = checkpoint.get("metrics", {})

    return {
        "dataset": "Elliptic",
        "feature_setting": "tx+agg",
        "explainer": "PGExplainer",
        "model": MODEL_NAME,
        "node_id": int(node_id),
        "true_label": int(data.y[node_id].detach().cpu()),
        "pred_label": int(pred_cpu[node_id]),
        "pred_prob_illicit": float(probs_cpu[node_id, ILLICIT_LABEL]),
        "model_best_epoch": int(model_metrics.get("best_epoch", -1)),
        "model_test_illicit_f1": float(model_metrics.get("test_illicit_f1", np.nan)),
        "num_graph_nodes": int(data.num_nodes),
        "num_graph_edges": int(data.edge_index.size(1)),
        "num_explanation_edges": int(top_k),
        "num_explanation_nodes": int(len(explanation_nodes)),
        "sparsity_top_k": float(sparsity),
        "edge_mask_mean": float(edge_mask.mean()) if edge_mask.numel() > 0 else 0.0,
        "edge_mask_max": float(edge_mask.max()) if edge_mask.numel() > 0 else 0.0,
        "edge_mask_min": float(edge_mask.min()) if edge_mask.numel() > 0 else 0.0,
        "top_edge_indices": json.dumps(top_edge_indices),
        "top_edge_scores": json.dumps(top_edge_scores),
        "explanation_nodes": json.dumps(explanation_nodes),
        "explanation_original_edge_pairs": json.dumps(explanation_edge_pairs),
    }


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print("Running shared-node full-graph PGExplainer on fixed GraphSAGE checkpoint.")

    if not NODE_LIST_PATH.exists():
        raise FileNotFoundError(f"Missing shared node list: {NODE_LIST_PATH}")

    shared_nodes_df = pd.read_csv(NODE_LIST_PATH)
    node_ids = [int(n) for n in shared_nodes_df["node_id"].tolist()]

    print("\nShared explanation target nodes:")
    print(node_ids)

    data = load_data().to(device)
    model, checkpoint = load_shared_model(data, device)
    model.eval()

    _, probs, pred = get_predictions(model, data)
    probs_cpu = probs.detach().cpu()
    pred_cpu = pred.detach().cpu()

    # PyG PGExplainer requires phenomenon target.
    # We use the model's own predictions to explain model behaviour.
    target = pred.to(device)

    train_node_ids = select_pgexplainer_training_nodes(
        model=model,
        data=data,
        num_nodes=NUM_EXPLAINER_TRAIN_NODES,
    )

    explainer = make_pgexplainer(model, device)

    train_pgexplainer(
        explainer=explainer,
        model=model,
        data=data,
        train_node_ids=train_node_ids,
        target=target,
    )

    rows = []

    for i, node_id in enumerate(node_ids, start=1):
        print("\n" + "-" * 80)
        print(f"PGExplainer node {i}/{len(node_ids)} | node_id={node_id}")
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
                node_id=node_id,
                probs_cpu=probs_cpu,
                pred_cpu=pred_cpu,
                checkpoint=checkpoint,
            )

            rows.append(row)

        except Exception as e:
            print(f"WARNING: failed to explain node {node_id}: {repr(e)}")
            continue

        pd.DataFrame(rows).to_csv(PARTIAL_FILE, index=False)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nPGExplainer shared summary")
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

    print(f"\nSaved shared PGExplainer results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()