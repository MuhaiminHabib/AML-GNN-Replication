from pathlib import Path
import sys
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch.explain import SubgraphX

from torch_geometric.utils import k_hop_subgraph

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SUPPORTED_MODELS = ["gcn", "graphsage", "gatv2"]

ILLICIT_LABEL = 0
LICIT_LABEL = 1

EGO_HOPS = 2

SUBGRAPHX_NUM_HOPS = 2
SUBGRAPHX_NUM_CHILD = 6
SUBGRAPHX_NUM_ROLLOUTS = 5
SUBGRAPHX_NODE_MIN = 3
SUBGRAPHX_SHAPLEY_STEPS = 5

TOP_K_EDGES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DGL SubgraphX on a shared model checkpoint and shared nodes."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model backbone: gcn, graphsage, or gatv2.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    return build_marasi_elliptic_data(
        feature_setting="tx+agg",
        seed=SEED,
    )


def get_paths(model_name: str):
    model_path = OUTPUT_DIR / f"shared_{model_name}_model.pt"
    node_list_path = OUTPUT_DIR / f"shared_{model_name}_explanation_nodes.csv"
    output_file = OUTPUT_DIR / f"dgl_subgraphx_shared_{model_name}_results.csv"
    partial_file = OUTPUT_DIR / f"dgl_subgraphx_shared_{model_name}_partial.csv"

    return model_path, node_list_path, output_file, partial_file


def load_shared_model(model_name: str, device):
    model_path, _, _, _ = get_paths(model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing shared model checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

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


class EgoGraphClassifier(nn.Module):
    """
    DGL SubgraphX expects graph classification:
        model(graph, feat) -> graph-level logits

    Your AML model is node classification:
        model(x, edge_index) -> node-level logits

    This wrapper returns the explained target node's logits as graph-level logits.
    """

    def __init__(self, pyg_model, target_local_idx):
        super().__init__()
        self.pyg_model = pyg_model
        self.target_local_idx = int(target_local_idx)

    def forward(self, graph, feat):
        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0).long()

        logits = self.pyg_model(feat, edge_index)

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
    explanation_set = set(
        int(n) for n in explanation_local_nodes.detach().cpu().tolist()
    )

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


def run_subgraphx_for_node(
    model,
    data,
    node_id,
    model_name,
    probs_cpu,
    pred_cpu,
    checkpoint,
):
    print("\n" + "-" * 80)
    print(f"DGL SubgraphX | model={model_name} | node_id={node_id}")
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

    wrapped_matches = bool(wrapped_pred == pred_label)

    print(f"Wrapped prediction: {wrapped_pred}")
    print(f"Wrapped illicit prob: {wrapped_pred_prob_illicit:.6f}")
    print(f"Wrapped matches full prediction: {wrapped_matches}")

    if not wrapped_matches:
        print(
            "WARNING: Wrapped ego-graph prediction differs from original full-graph "
            "prediction. This row will still be saved with "
            "wrapped_matches_full_prediction=False."
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
    explanation_original_nodes = (
        original_node_ids[explanation_local_nodes].detach().cpu().long()
    )

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

    sparsity_nodes = 1.0 - (
        num_explanation_nodes / max(1, int(dgl_graph.num_nodes()))
    )
    sparsity_edges = 1.0 - (
        num_explanation_edges / max(1, int(dgl_graph.num_edges()))
    )

    model_metrics = checkpoint.get("metrics", {})

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
        "wrapped_matches_full_prediction": wrapped_matches,
        "model_best_epoch": int(model_metrics.get("best_epoch", -1)),
        "model_test_illicit_f1": float(model_metrics.get("test_illicit_f1", np.nan)),
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
        "explanation_local_nodes": json.dumps(
            [int(n) for n in explanation_local_nodes.tolist()]
        ),
        "explanation_original_nodes": json.dumps(
            [int(n) for n in explanation_original_nodes.tolist()]
        ),
        "explanation_original_edge_pairs": json.dumps(edge_pairs_original),
    }

    print("Explanation original nodes:")
    print(row["explanation_original_nodes"])

    print("Explanation original edge pairs:")
    print(row["explanation_original_edge_pairs"])

    return row


def main():
    args = parse_args()
    model_name = args.model.lower()

    set_seed(SEED)

    device = torch.device("cpu")

    _, node_list_path, output_file, partial_file = get_paths(model_name)

    print(f"Using device: {device}")
    print(f"Running shared-node official DGL SubgraphX on fixed {model_name.upper()} checkpoint.")

    if not node_list_path.exists():
        raise FileNotFoundError(f"Missing shared node list: {node_list_path}")

    shared_nodes_df = pd.read_csv(node_list_path)
    node_ids = [int(n) for n in shared_nodes_df["node_id"].tolist()]

    print("\nShared explanation target nodes:")
    print(node_ids)

    data = load_data().to(device)
    model, checkpoint = load_shared_model(model_name, device)
    model.eval()

    _, probs, pred = get_predictions(model, data)
    probs_cpu = probs.detach().cpu()
    pred_cpu = pred.detach().cpu()

    rows = []

    for i, node_id in enumerate(node_ids, start=1):
        print("\n" + "=" * 80)
        print(f"SubgraphX {model_name} node {i}/{len(node_ids)}")
        print("=" * 80)

        row = run_subgraphx_for_node(
            model=model,
            data=data,
            node_id=node_id,
            model_name=model_name,
            probs_cpu=probs_cpu,
            pred_cpu=pred_cpu,
            checkpoint=checkpoint,
        )

        if row is not None:
            rows.append(row)

        pd.DataFrame(rows).to_csv(partial_file, index=False)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_file, index=False)

    print("\nDGL SubgraphX shared model summary")
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
        print("No DGL SubgraphX explanations were produced.")

    print(f"\nSaved shared DGL SubgraphX results to: {output_file}")


if __name__ == "__main__":
    main()