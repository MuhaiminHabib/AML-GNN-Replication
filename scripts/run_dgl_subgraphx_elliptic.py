from pathlib import Path
import sys
import copy
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.explain import SubgraphX

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from torch_geometric.utils import k_hop_subgraph

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Start with GraphSAGE only.
# After this works, we can add "gatv2".
MODELS = ["graphsage"]

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

ILLICIT_LABEL = 0
LICIT_LABEL = 1

# Keep this small for the first successful test.
NUM_NODES_TO_EXPLAIN = 10

EGO_HOPS = 2

# DGL SubgraphX settings.
# These are intentionally small first because SubgraphX is expensive.
SUBGRAPHX_NUM_HOPS = 2
SUBGRAPHX_NUM_CHILD = 6
SUBGRAPHX_NUM_ROLLOUTS = 5
SUBGRAPHX_NODE_MIN = 3
SUBGRAPHX_SHAPLEY_STEPS = 5

TOP_K_EDGES = 20


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def get_ego_node_count(data, node_id: int, num_hops: int):
    subset, _, _, _ = k_hop_subgraph(
        node_idx=int(node_id),
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        flow="source_to_target",
    )
    return int(subset.numel())


@torch.no_grad()
def select_correct_illicit_nodes(model, data, num_nodes: int):
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

    selected = []
    skipped_tiny = 0

    for idx in sorted_idx:
        node_id = int(candidate_nodes[idx])

        ego_node_count = get_ego_node_count(
            data=data,
            node_id=node_id,
            num_hops=EGO_HOPS,
        )

        # DGL SubgraphX requires graph.num_nodes() > node_min.
        # Tiny ego graphs are also not useful for MCTS search.
        if ego_node_count > SUBGRAPHX_NODE_MIN:
            selected.append(node_id)
            print(
                f"Selected node {node_id} | "
                f"illicit_prob={float(probs[node_id, ILLICIT_LABEL]):.6f} | "
                f"ego_nodes={ego_node_count}"
            )
        else:
            skipped_tiny += 1

        if len(selected) >= num_nodes:
            break

    if len(selected) == 0:
        raise RuntimeError(
            "No correctly predicted illicit test nodes with large enough ego graphs found."
        )

    print(
        f"\nSelected {len(selected)} nodes with ego graph size > "
        f"{SUBGRAPHX_NODE_MIN}."
    )
    print(f"Skipped tiny ego graphs: {skipped_tiny}")

    return selected, probs.detach().cpu(), pred.detach().cpu()


class EgoGraphClassifier(nn.Module):
    """
    DGL SubgraphX expects graph classification:
        model(graph, feat) -> graph-level logits

    Our AML model is node classification:
        model(x, edge_index) -> node-level logits

    This wrapper returns the target node's logits as a graph-level output.
    """

    def __init__(self, pyg_model, target_local_idx):
        super().__init__()
        self.pyg_model = pyg_model
        self.target_local_idx = int(target_local_idx)

    def forward(self, graph, feat):
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0).long()

        logits = self.pyg_model(feat, edge_index)

        # Return shape [1, num_classes], like graph classification.
        return logits[self.target_local_idx].unsqueeze(0)


def extract_pyg_ego_to_dgl(data, node_id: int, num_hops: int):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=int(node_id),
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        flow="source_to_target",
    )

    target_local_idx = int(mapping.item())
    ego_x = data.x[subset].float()

    src = edge_index[0].long()
    dst = edge_index[1].long()

    dgl_graph = dgl.graph(
        (src, dst),
        num_nodes=ego_x.size(0),
    )

    dgl_graph = dgl.remove_self_loop(dgl_graph)
    dgl_graph = dgl.add_self_loop(dgl_graph)

    return dgl_graph, ego_x, subset, target_local_idx


def explanation_edges_from_nodes(dgl_graph, explanation_local_nodes):
    explanation_set = set(int(n) for n in explanation_local_nodes.detach().cpu().tolist())

    src, dst = dgl_graph.edges()
    src = src.detach().cpu()
    dst = dst.detach().cpu()

    selected_edges = []

    for i in range(src.numel()):
        s = int(src[i])
        d = int(dst[i])

        if s in explanation_set and d in explanation_set:
            selected_edges.append(i)

    return selected_edges


def run_subgraphx_for_node(model, data, node_id, model_name, model_metrics, probs_cpu, pred_cpu):
    print("\n" + "-" * 80)
    print(f"{model_name} | DGL SubgraphX explaining node_id={node_id}")
    print("-" * 80)

    true_label = int(data.y[node_id].detach().cpu())
    pred_label = int(pred_cpu[node_id])
    pred_prob_illicit = float(probs_cpu[node_id, ILLICIT_LABEL])

    dgl_graph, ego_x, original_node_ids, target_local_idx = extract_pyg_ego_to_dgl(
        data=data,
        node_id=int(node_id),
        num_hops=EGO_HOPS,
    )

    print(f"Ego nodes: {dgl_graph.num_nodes()}")
    print(f"Ego edges: {dgl_graph.num_edges()}")
    print(f"Target local index: {target_local_idx}")
    print(f"True label: {true_label}")
    print(f"Pred label: {pred_label}")
    print(f"Pred illicit prob: {pred_prob_illicit:.6f}")

    if dgl_graph.num_nodes() <= SUBGRAPHX_NODE_MIN:
        print(
            f"Skipping node {node_id}: ego graph has {dgl_graph.num_nodes()} nodes, "
            f"but DGL SubgraphX requires more than {SUBGRAPHX_NODE_MIN}."
        )
        return None

    wrapped_model = EgoGraphClassifier(
        pyg_model=model,
        target_local_idx=target_local_idx,
    )

    wrapped_model.eval()

    with torch.no_grad():
        wrapped_logits = wrapped_model(dgl_graph, ego_x)
        wrapped_probs = torch.softmax(wrapped_logits, dim=1)
        wrapped_pred = int(wrapped_logits.argmax(dim=1).item())
        wrapped_pred_prob_illicit = float(wrapped_probs[0, ILLICIT_LABEL])

    print(f"Wrapped prediction: {wrapped_pred}")
    print(f"Wrapped illicit prob: {wrapped_pred_prob_illicit:.6f}")

    if wrapped_pred != pred_label:
        print(
            "WARNING: Wrapped ego-graph prediction differs from original full-graph prediction. "
            "This may happen because the model is now evaluated only on the ego graph."
        )

    explainer = SubgraphX(
        wrapped_model,
        num_hops=SUBGRAPHX_NUM_HOPS,
        coef=10.0,
        high2low=True,
        num_child=SUBGRAPHX_NUM_CHILD,
        num_rollouts=SUBGRAPHX_NUM_ROLLOUTS,
        node_min=SUBGRAPHX_NODE_MIN,
        shapley_steps=SUBGRAPHX_SHAPLEY_STEPS,
        log=False,
    )

    explanation_local_nodes = explainer.explain_graph(
        dgl_graph,
        ego_x,
        target_class=int(pred_label),
    )

    explanation_local_nodes = explanation_local_nodes.detach().cpu().long()
    explanation_original_nodes = original_node_ids[explanation_local_nodes].detach().cpu().long()

    explanation_local_edges = explanation_edges_from_nodes(
        dgl_graph=dgl_graph,
        explanation_local_nodes=explanation_local_nodes,
    )

    src, dst = dgl_graph.edges()
    edge_pairs_original = []

    for edge_id in explanation_local_edges[:TOP_K_EDGES]:
        local_src = int(src[edge_id])
        local_dst = int(dst[edge_id])

        original_src = int(original_node_ids[local_src])
        original_dst = int(original_node_ids[local_dst])

        edge_pairs_original.append([original_src, original_dst])

    num_explanation_nodes = int(explanation_local_nodes.numel())
    num_explanation_edges = int(len(edge_pairs_original))

    sparsity_nodes = 1.0 - (num_explanation_nodes / max(1, int(dgl_graph.num_nodes())))
    sparsity_edges = 1.0 - (num_explanation_edges / max(1, int(dgl_graph.num_edges())))

    row = {
        "dataset": "Elliptic",
        "feature_setting": "tx+agg",
        "explainer": "DGL_SubgraphX",
        "model": model_name,
        "node_id": int(node_id),
        "true_label": true_label,
        "pred_label": pred_label,
        "pred_prob_illicit": pred_prob_illicit,
        "wrapped_pred_label": int(wrapped_pred),
        "wrapped_pred_prob_illicit": wrapped_pred_prob_illicit,
        "wrapped_matches_full_prediction": bool(wrapped_pred == pred_label),
        "model_best_epoch": int(model_metrics["best_epoch"]),
        "model_test_illicit_f1": float(model_metrics["test_illicit_f1"]),
        "ego_hops": int(EGO_HOPS),
        "num_ego_nodes": int(dgl_graph.num_nodes()),
        "num_ego_edges": int(dgl_graph.num_edges()),
        "subgraphx_num_hops": int(SUBGRAPHX_NUM_HOPS),
        "subgraphx_num_child": int(SUBGRAPHX_NUM_CHILD),
        "subgraphx_num_rollouts": int(SUBGRAPHX_NUM_ROLLOUTS),
        "subgraphx_node_min": int(SUBGRAPHX_NODE_MIN),
        "subgraphx_shapley_steps": int(SUBGRAPHX_SHAPLEY_STEPS),
        "num_explanation_nodes": num_explanation_nodes,
        "num_explanation_edges": num_explanation_edges,
        "sparsity_nodes": float(sparsity_nodes),
        "sparsity_edges": float(sparsity_edges),
        "explanation_local_nodes": json.dumps([int(n) for n in explanation_local_nodes.tolist()]),
        "explanation_original_nodes": json.dumps([int(n) for n in explanation_original_nodes.tolist()]),
        "explanation_original_edge_pairs": json.dumps(edge_pairs_original),
    }

    print("Explanation original nodes:")
    print(row["explanation_original_nodes"])

    print("Explanation original edge pairs:")
    print(row["explanation_original_edge_pairs"])

    return row


def run_for_model(model_name, data, device):
    model, model_metrics = train_model(model_name, data, device)
    model.eval()

    selected_nodes, probs_cpu, pred_cpu = select_correct_illicit_nodes(
        model=model,
        data=data,
        num_nodes=NUM_NODES_TO_EXPLAIN,
    )

    print("\nSelected illicit test nodes to explain:")
    print(selected_nodes)

    rows = []

    for i, node_id in enumerate(selected_nodes, start=1):
        print(f"\nNode {i}/{len(selected_nodes)}")

        row = run_subgraphx_for_node(
            model=model,
            data=data,
            node_id=node_id,
            model_name=model_name,
            model_metrics=model_metrics,
            probs_cpu=probs_cpu,
            pred_cpu=pred_cpu,
        )

        if row is not None:
            rows.append(row)

        partial_df = pd.DataFrame(rows)
        partial_path = OUTPUT_DIR / f"dgl_subgraphx_elliptic_{model_name}_partial.csv"
        partial_df.to_csv(partial_path, index=False)

    model_path = OUTPUT_DIR / f"dgl_subgraphx_elliptic_{model_name}.csv"
    pd.DataFrame(rows).to_csv(model_path, index=False)

    print(f"\nSaved {model_name} DGL SubgraphX explanations to: {model_path}")

    return rows


def main():
    set_seed(SEED)

    # Use CPU here because .venv-dgl is a CPU-compatible environment.
    device = torch.device("cpu")

    print(f"Using device: {device}")
    print("This script uses official DGL SubgraphX through a PyG-to-DGL ego-graph bridge.")

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
        rows = run_for_model(model_name, data, device)
        all_rows.extend(rows)

    final_df = pd.DataFrame(all_rows)

    final_path = OUTPUT_DIR / "dgl_subgraphx_elliptic_results.csv"
    final_df.to_csv(final_path, index=False)

    print("\nDGL SubgraphX summary")
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
                    "wrapped_pred_label",
                    "wrapped_pred_prob_illicit",
                    "wrapped_matches_full_prediction",
                    "num_ego_nodes",
                    "num_ego_edges",
                    "num_explanation_nodes",
                    "num_explanation_edges",
                    "sparsity_nodes",
                    "sparsity_edges",
                ]
            ].to_string(index=False)
        )
    else:
        print("No explanations were produced.")

    print(f"\nSaved final DGL SubgraphX results to: {final_path}")


if __name__ == "__main__":
    main()