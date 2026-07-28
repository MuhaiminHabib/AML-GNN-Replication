from pathlib import Path
import sys
import json
import ast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
MODEL_PATH = OUTPUT_DIR / "shared_graphsage_model.pt"

INPUT_FILES = [
    OUTPUT_DIR / "gnnexplainer_shared_graphsage_results.csv",
    OUTPUT_DIR / "pgexplainer_shared_graphsage_results.csv",
    OUTPUT_DIR / "dgl_subgraphx_shared_graphsage_results.csv",
    OUTPUT_DIR / "dgl_subgraphx_shared_graphsage_large_results.csv",
]

DETAIL_OUTPUT = OUTPUT_DIR / "shared_explainer_faithfulness.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "shared_explainer_faithfulness_summary.csv"

SEED = 42
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


def parse_json_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return []

        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return []

    return []


def build_edge_lookup(edge_index):
    edge_lookup = {}

    edge_index_cpu = edge_index.detach().cpu()

    for edge_id in range(edge_index_cpu.size(1)):
        src = int(edge_index_cpu[0, edge_id])
        dst = int(edge_index_cpu[1, edge_id])
        edge_lookup.setdefault((src, dst), []).append(edge_id)

    return edge_lookup


def get_explanation_edge_indices(row, data, edge_lookup):
    """
    Preferred input:
        explanation_original_edge_pairs

    Fallback:
        top_edge_indices
    """
    edge_pairs = parse_json_list(row.get("explanation_original_edge_pairs", "[]"))

    selected_edge_ids = []

    if len(edge_pairs) > 0:
        for pair in edge_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue

            src = int(pair[0])
            dst = int(pair[1])

            matching_ids = edge_lookup.get((src, dst), [])

            for edge_id in matching_ids:
                selected_edge_ids.append(edge_id)

    if len(selected_edge_ids) == 0:
        top_edge_indices = parse_json_list(row.get("top_edge_indices", "[]"))
        selected_edge_ids = [int(i) for i in top_edge_indices]

    selected_edge_ids = sorted(set(selected_edge_ids))

    selected_edge_ids = [
        edge_id for edge_id in selected_edge_ids
        if 0 <= edge_id < data.edge_index.size(1)
    ]

    return selected_edge_ids


@torch.no_grad()
def predict_node(model, x, edge_index, node_id):
    model.eval()
    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)

    return {
        "pred_label": int(pred[node_id].detach().cpu()),
        "prob_illicit": float(probs[node_id, ILLICIT_LABEL].detach().cpu()),
        "prob_licit": float(probs[node_id, LICIT_LABEL].detach().cpu()),
    }


def remove_edges(edge_index, selected_edge_ids):
    if len(selected_edge_ids) == 0:
        return edge_index

    device = edge_index.device
    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
    selected_edge_ids_tensor = torch.tensor(
        selected_edge_ids,
        dtype=torch.long,
        device=device,
    )
    mask[selected_edge_ids_tensor] = False

    return edge_index[:, mask]


def keep_only_edges(edge_index, selected_edge_ids):
    if len(selected_edge_ids) == 0:
        return edge_index[:, :0]

    device = edge_index.device
    selected_edge_ids_tensor = torch.tensor(
        selected_edge_ids,
        dtype=torch.long,
        device=device,
    )

    return edge_index[:, selected_edge_ids_tensor]


def evaluate_one_explanation(model, data, row, edge_lookup):
    node_id = int(row["node_id"])
    explainer = str(row["explainer"])
    model_name = str(row["model"])

    selected_edge_ids = get_explanation_edge_indices(
        row=row,
        data=data,
        edge_lookup=edge_lookup,
    )

    original = predict_node(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        node_id=node_id,
    )

    deletion_edge_index = remove_edges(
        edge_index=data.edge_index,
        selected_edge_ids=selected_edge_ids,
    )

    deletion = predict_node(
        model=model,
        x=data.x,
        edge_index=deletion_edge_index,
        node_id=node_id,
    )

    insertion_edge_index = keep_only_edges(
        edge_index=data.edge_index,
        selected_edge_ids=selected_edge_ids,
    )

    insertion = predict_node(
        model=model,
        x=data.x,
        edge_index=insertion_edge_index,
        node_id=node_id,
    )

    original_prob = original["prob_illicit"]
    deletion_prob = deletion["prob_illicit"]
    insertion_prob = insertion["prob_illicit"]

    deletion_drop = original_prob - deletion_prob
    insertion_gap = original_prob - insertion_prob

    deletion_label_flip = int(deletion["pred_label"] != original["pred_label"])
    insertion_preservation = int(insertion["pred_label"] == original["pred_label"])

    num_graph_edges = int(data.edge_index.size(1))
    num_explanation_edges = int(len(selected_edge_ids))

    sparsity_edges = 1.0 - (num_explanation_edges / max(1, num_graph_edges))

    explanation_nodes = set()

    if num_explanation_edges > 0:
        selected_edges = data.edge_index[:, selected_edge_ids].detach().cpu()

        for i in range(selected_edges.size(1)):
            explanation_nodes.add(int(selected_edges[0, i]))
            explanation_nodes.add(int(selected_edges[1, i]))

    return {
        "dataset": row.get("dataset", "Elliptic"),
        "feature_setting": row.get("feature_setting", "tx+agg"),
        "explainer": explainer,
        "model": model_name,
        "node_id": node_id,
        "true_label": int(row["true_label"]),
        "original_pred_label": int(original["pred_label"]),
        "original_prob_illicit": original_prob,
        "deletion_pred_label": int(deletion["pred_label"]),
        "deletion_prob_illicit": deletion_prob,
        "deletion_drop": deletion_drop,
        "deletion_label_flip": deletion_label_flip,
        "insertion_pred_label": int(insertion["pred_label"]),
        "insertion_prob_illicit": insertion_prob,
        "insertion_gap": insertion_gap,
        "insertion_preservation": insertion_preservation,
        "num_graph_edges": num_graph_edges,
        "num_explanation_edges": num_explanation_edges,
        "num_explanation_nodes": int(len(explanation_nodes)),
        "sparsity_edges": sparsity_edges,
    }


def load_all_explainer_outputs():
    frames = []

    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing explainer output file: {path}")

        df = pd.read_csv(path)
        frames.append(df)

        print(f"Loaded {len(df)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)

    return combined


def summarise_results(detail_df):
    summary = (
        detail_df
        .groupby(["explainer", "model"])
        .agg(
            nodes_evaluated=("node_id", "count"),
            deletion_drop_mean=("deletion_drop", "mean"),
            deletion_drop_std=("deletion_drop", "std"),
            deletion_label_flip_rate=("deletion_label_flip", "mean"),
            insertion_prob_mean=("insertion_prob_illicit", "mean"),
            insertion_prob_std=("insertion_prob_illicit", "std"),
            insertion_gap_mean=("insertion_gap", "mean"),
            insertion_gap_std=("insertion_gap", "std"),
            insertion_preservation_rate=("insertion_preservation", "mean"),
            sparsity_edges_mean=("sparsity_edges", "mean"),
            num_explanation_edges_mean=("num_explanation_edges", "mean"),
            num_explanation_nodes_mean=("num_explanation_nodes", "mean"),
        )
        .reset_index()
    )

    return summary


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Evaluating shared explainer faithfulness on fixed GraphSAGE checkpoint.")

    data = load_data().to(device)
    model, checkpoint = load_shared_model(data, device)

    all_outputs = load_all_explainer_outputs()

    edge_lookup = build_edge_lookup(data.edge_index)

    detail_rows = []

    for _, row in all_outputs.iterrows():
        result = evaluate_one_explanation(
            model=model,
            data=data,
            row=row,
            edge_lookup=edge_lookup,
        )

        detail_rows.append(result)

        print(
            f"{result['explainer']} | node={result['node_id']} | "
            f"orig={result['original_prob_illicit']:.4f} | "
            f"del={result['deletion_prob_illicit']:.4f} | "
            f"drop={result['deletion_drop']:.4f} | "
            f"ins={result['insertion_prob_illicit']:.4f} | "
            f"preserve={result['insertion_preservation']}"
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(DETAIL_OUTPUT, index=False)

    summary_df = summarise_results(detail_df)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("\nShared explainer faithfulness summary")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print(f"\nSaved detail results to: {DETAIL_OUTPUT}")
    print(f"Saved summary results to: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()