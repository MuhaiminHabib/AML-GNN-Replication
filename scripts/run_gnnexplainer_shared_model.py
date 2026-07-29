from pathlib import Path
import sys
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ILLICIT_LABEL = 0

SUPPORTED_MODELS = ["gcn", "graphsage", "gatv2"]

GNNEXPLAINER_EPOCHS = 200
GNNEXPLAINER_LR = 0.01
TOP_K_EDGES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GNNExplainer on a shared model checkpoint and shared nodes."
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
    torch.cuda.manual_seed_all(seed)


def load_data():
    return build_marasi_elliptic_data(
        feature_setting="tx+agg",
        seed=SEED,
    )


def get_paths(model_name: str):
    model_path = OUTPUT_DIR / f"shared_{model_name}_model.pt"
    node_list_path = OUTPUT_DIR / f"shared_{model_name}_explanation_nodes.csv"
    output_file = OUTPUT_DIR / f"gnnexplainer_shared_{model_name}_results.csv"
    partial_file = OUTPUT_DIR / f"gnnexplainer_shared_{model_name}_partial.csv"

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


def make_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(
            epochs=GNNEXPLAINER_EPOCHS,
            lr=GNNEXPLAINER_LR,
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
    node_id,
    model_name,
    probs_cpu,
    pred_cpu,
    checkpoint,
):
    if explanation.edge_mask is None:
        raise RuntimeError(f"GNNExplainer returned no edge mask for node {node_id}")

    edge_mask = explanation.edge_mask.detach().cpu()

    if torch.isnan(edge_mask).any() or torch.isinf(edge_mask).any():
        raise RuntimeError(f"GNNExplainer produced invalid edge mask for node {node_id}")

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
        "explainer": "GNNExplainer",
        "model": model_name,
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
    args = parse_args()
    model_name = args.model.lower()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, node_list_path, output_file, partial_file = get_paths(model_name)

    print(f"Using device: {device}")
    print(f"Running shared-node GNNExplainer on fixed {model_name.upper()} checkpoint.")

    if not node_list_path.exists():
        raise FileNotFoundError(f"Missing shared node list: {node_list_path}")

    shared_nodes_df = pd.read_csv(node_list_path)
    node_ids = [int(n) for n in shared_nodes_df["node_id"].tolist()]

    print("\nShared nodes:")
    print(node_ids)

    data = load_data().to(device)
    model, checkpoint = load_shared_model(model_name, device)

    _, probs, pred = get_predictions(model, data)
    probs_cpu = probs.detach().cpu()
    pred_cpu = pred.detach().cpu()

    explainer = make_explainer(model)

    rows = []

    for i, node_id in enumerate(node_ids, start=1):
        print("\n" + "-" * 80)
        print(f"GNNExplainer {model_name} node {i}/{len(node_ids)} | node_id={node_id}")
        print("-" * 80)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=int(node_id),
        )

        row = summarise_explanation(
            explanation=explanation,
            data=data,
            node_id=node_id,
            model_name=model_name,
            probs_cpu=probs_cpu,
            pred_cpu=pred_cpu,
            checkpoint=checkpoint,
        )

        rows.append(row)

        pd.DataFrame(rows).to_csv(partial_file, index=False)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_file, index=False)

    print("\nGNNExplainer shared model summary")
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

    print(f"\nSaved shared GNNExplainer results to: {output_file}")


if __name__ == "__main__":
    main()