from pathlib import Path
import sys
import argparse
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import build_ibm_amlsim_graph


# =============================================================================
# Global settings
# =============================================================================

SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

FRAUD_LABEL = 1

DEFAULT_HOPS = 2
FALLBACK_HOPS = 1
MAX_SUBGRAPH_EDGES = 100000

GNNEXPLAINER_EPOCHS = 200
TOP_K_EDGES = 20


MODEL_CONFIGS = {
    "gcn": {
        "hidden_dim": 64,
        "dropout": 0.5,
    },
    "graphsage": {
        "hidden_dim": 64,
        "dropout": 0.5,
    },
    "gatv2": {
        "hidden_dim": 32,
        "heads": 8,
        "dropout": 0.5,
    },
}


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Models
# =============================================================================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, dropout=0.5):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x


class GATv2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=2,
        heads=8,
        dropout=0.5,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return x


def build_model(model_name: str, in_channels: int, checkpoint: dict):
    model_name = model_name.lower()

    hidden_dim = int(checkpoint.get("hidden_dim", MODEL_CONFIGS[model_name]["hidden_dim"]))
    dropout = float(checkpoint.get("dropout", MODEL_CONFIGS[model_name]["dropout"]))

    if model_name == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=2,
            dropout=dropout,
        )

    if model_name == "graphsage":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=2,
            dropout=dropout,
        )

    if model_name == "gatv2":
        heads = int(checkpoint.get("heads", MODEL_CONFIGS["gatv2"]["heads"]))

        return GATv2(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=2,
            heads=heads,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Helpers
# =============================================================================

def get_paths(model_name: str):
    output_dir = PROJECT_ROOT / "outputs" / "explainers" / f"amlsim_{model_name}"

    checkpoint_path = output_dir / f"shared_amlsim_{model_name}_model.pt"
    nodes_path = output_dir / f"shared_amlsim_{model_name}_explanation_nodes.csv"

    results_path = output_dir / f"gnnexplainer_shared_amlsim_{model_name}_results.csv"
    partial_path = output_dir / f"gnnexplainer_shared_amlsim_{model_name}_partial.csv"
    summary_path = output_dir / f"gnnexplainer_shared_amlsim_{model_name}_summary.csv"

    return {
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "nodes_path": nodes_path,
        "results_path": results_path,
        "partial_path": partial_path,
        "summary_path": summary_path,
    }


def load_data(device):
    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    data = data.to(device)

    return data


def load_shared_model(model_name: str, data, device, checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            f"Run this first:\n"
            f"python scripts\\prepare_shared_amlsim_model_explainer_setup.py --model {model_name}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        model_name=model_name,
        in_channels=data.num_features,
        checkpoint=checkpoint,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def get_prediction(model, x, edge_index, node_idx: int):
    model.eval()

    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[node_idx].argmax().detach().cpu().item())
    fraud_prob = float(probs[node_idx, FRAUD_LABEL].detach().cpu().item())

    return pred_label, fraud_prob


def make_local_subgraph(data, node_idx: int, requested_hops: int):
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=requested_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        flow="source_to_target",
    )

    return subset, sub_edge_index, mapping


def get_subgraph_with_fallback(data, node_idx: int):
    subset, sub_edge_index, mapping = make_local_subgraph(
        data=data,
        node_idx=node_idx,
        requested_hops=DEFAULT_HOPS,
    )

    used_hops = DEFAULT_HOPS

    if int(sub_edge_index.size(1)) > MAX_SUBGRAPH_EDGES:
        print(
            f"  2-hop subgraph too large "
            f"({int(sub_edge_index.size(1))} edges). "
            f"Falling back to {FALLBACK_HOPS}-hop."
        )

        subset, sub_edge_index, mapping = make_local_subgraph(
            data=data,
            node_idx=node_idx,
            requested_hops=FALLBACK_HOPS,
        )

        used_hops = FALLBACK_HOPS

    return subset, sub_edge_index, mapping, used_hops


def build_explainer(model):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=GNNEXPLAINER_EPOCHS),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    return explainer


def explain_one_node(
    model_name: str,
    model,
    explainer,
    data,
    node_idx: int,
    rank: int,
    true_label: int,
    original_pred_label: int,
    original_fraud_probability: float,
    device,
):
    subset, sub_edge_index, mapping, used_hops = get_subgraph_with_fallback(
        data=data,
        node_idx=node_idx,
    )

    subset = subset.to(device)
    sub_edge_index = sub_edge_index.to(device)
    mapping = mapping.to(device)

    x_sub = data.x[subset].to(device)
    local_node_idx = int(mapping.item())

    subgraph_pred_label, subgraph_fraud_probability = get_prediction(
        model=model,
        x=x_sub,
        edge_index=sub_edge_index,
        node_idx=local_node_idx,
    )

    print(
        f"  used_hops={used_hops} | "
        f"subgraph_nodes={int(subset.numel())} | "
        f"subgraph_edges={int(sub_edge_index.size(1))} | "
        f"full_prob={original_fraud_probability:.6f} | "
        f"sub_prob={subgraph_fraud_probability:.6f}"
    )

    explanation = explainer(
        x=x_sub,
        edge_index=sub_edge_index,
        index=local_node_idx,
    )

    edge_mask = explanation.edge_mask

    if edge_mask is None:
        raise RuntimeError("GNNExplainer returned no edge_mask.")

    edge_mask = edge_mask.detach().cpu()

    local_edges = sub_edge_index.detach().cpu()
    subset_cpu = subset.detach().cpu()

    rows = []

    if edge_mask.numel() == 0:
        rows.append(
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction",
                "model": model_name,
                "explainer": "GNNExplainer",
                "center_node_idx": int(node_idx),
                "rank": int(rank),
                "split": "test",
                "true_label": int(true_label),
                "original_pred_label": int(original_pred_label),
                "original_fraud_probability": float(original_fraud_probability),
                "subgraph_pred_label": int(subgraph_pred_label),
                "subgraph_fraud_probability": float(subgraph_fraud_probability),
                "used_hops": int(used_hops),
                "subgraph_num_nodes": int(subset.numel()),
                "subgraph_num_edges": int(sub_edge_index.size(1)),
                "local_center_node_idx": int(local_node_idx),
                "edge_rank": -1,
                "edge_pos": -1,
                "edge_mask": 0.0,
                "local_src": -1,
                "local_dst": -1,
                "src": -1,
                "dst": -1,
            }
        )

        return rows

    top_k = min(TOP_K_EDGES, int(edge_mask.numel()))

    top_values, top_positions = torch.topk(edge_mask, k=top_k)

    for edge_rank, (edge_value, edge_pos) in enumerate(
        zip(top_values.tolist(), top_positions.tolist()),
        start=1,
    ):
        local_src = int(local_edges[0, edge_pos].item())
        local_dst = int(local_edges[1, edge_pos].item())

        global_src = int(subset_cpu[local_src].item())
        global_dst = int(subset_cpu[local_dst].item())

        rows.append(
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction",
                "model": model_name,
                "explainer": "GNNExplainer",
                "center_node_idx": int(node_idx),
                "rank": int(rank),
                "split": "test",
                "true_label": int(true_label),
                "original_pred_label": int(original_pred_label),
                "original_fraud_probability": float(original_fraud_probability),
                "subgraph_pred_label": int(subgraph_pred_label),
                "subgraph_fraud_probability": float(subgraph_fraud_probability),
                "used_hops": int(used_hops),
                "subgraph_num_nodes": int(subset.numel()),
                "subgraph_num_edges": int(sub_edge_index.size(1)),
                "local_center_node_idx": int(local_node_idx),
                "edge_rank": int(edge_rank),
                "edge_pos": int(edge_pos),
                "edge_mask": float(edge_value),
                "local_src": int(local_src),
                "local_dst": int(local_dst),
                "src": int(global_src),
                "dst": int(global_dst),
            }
        )

    return rows


def summarise_results(results_df: pd.DataFrame):
    valid_df = results_df[results_df["edge_pos"].astype(int) >= 0].copy()

    if valid_df.empty:
        return pd.DataFrame(
            [
                {
                    "explainer": "GNNExplainer",
                    "explained_nodes": 0,
                    "total_rows": int(len(results_df)),
                    "mean_subgraph_nodes": np.nan,
                    "mean_subgraph_edges": np.nan,
                    "mean_edge_mask": np.nan,
                    "max_edge_mask": np.nan,
                }
            ]
        )

    summary = {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "model": valid_df["model"].iloc[0],
        "explainer": "GNNExplainer",
        "explained_nodes": int(valid_df["center_node_idx"].nunique()),
        "total_rows": int(len(valid_df)),
        "mean_used_hops": float(valid_df.groupby("center_node_idx")["used_hops"].first().mean()),
        "mean_subgraph_nodes": float(valid_df.groupby("center_node_idx")["subgraph_num_nodes"].first().mean()),
        "mean_subgraph_edges": float(valid_df.groupby("center_node_idx")["subgraph_num_edges"].first().mean()),
        "mean_edge_mask": float(valid_df["edge_mask"].mean()),
        "max_edge_mask": float(valid_df["edge_mask"].max()),
        "min_edge_mask": float(valid_df["edge_mask"].min()),
    }

    return pd.DataFrame([summary])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["gcn", "graphsage", "gatv2"],
        help="AMLSim model to explain.",
    )

    args = parser.parse_args()
    model_name = args.model.lower()

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = get_paths(model_name)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Running GNNExplainer on shared AMLSim {model_name.upper()} model")
    print("=" * 100)
    print(f"Using device: {device}")

    print("\nLoading AMLSim graph...")
    data = load_data(device=device)

    print("\nLoading shared checkpoint...")
    model, checkpoint = load_shared_model(
        model_name=model_name,
        data=data,
        device=device,
        checkpoint_path=paths["checkpoint_path"],
    )

    print(f"Checkpoint: {paths['checkpoint_path']}")

    if not paths["nodes_path"].exists():
        raise FileNotFoundError(
            f"Missing explanation nodes file: {paths['nodes_path']}\n"
            f"Run this first:\n"
            f"python scripts\\prepare_shared_amlsim_model_explainer_setup.py --model {model_name}"
        )

    nodes_df = pd.read_csv(paths["nodes_path"])

    required_cols = {
        "rank",
        "node_idx",
        "true_label",
        "pred_label",
        "fraud_probability",
    }

    missing_cols = required_cols - set(nodes_df.columns)

    if missing_cols:
        raise ValueError(
            f"Explanation nodes file is missing columns: {sorted(missing_cols)}"
        )

    print("\nExplanation nodes:")
    print(nodes_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBuilding GNNExplainer...")
    explainer = build_explainer(model)

    all_rows = []

    for i, row in nodes_df.iterrows():
        node_idx = int(row["node_idx"])
        rank = int(row["rank"])
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])
        fraud_probability = float(row["fraud_probability"])

        print("\n" + "-" * 100)
        print(
            f"[{i + 1}/{len(nodes_df)}] Explaining node {node_idx} "
            f"| rank={rank} | fraud_prob={fraud_probability:.6f}"
        )

        try:
            rows = explain_one_node(
                model_name=model_name,
                model=model,
                explainer=explainer,
                data=data,
                node_idx=node_idx,
                rank=rank,
                true_label=true_label,
                original_pred_label=pred_label,
                original_fraud_probability=fraud_probability,
                device=device,
            )

            all_rows.extend(rows)

            partial_df = pd.DataFrame(all_rows)
            partial_df.to_csv(paths["partial_path"], index=False)

            print(f"  Saved partial results: {paths['partial_path']}")

        except Exception as exc:
            print(f"  FAILED node {node_idx}: {type(exc).__name__}: {exc}")

            all_rows.append(
                {
                    "dataset": "IBM AMLSim",
                    "setting": "previous_reproduction",
                    "model": model_name,
                    "explainer": "GNNExplainer",
                    "center_node_idx": int(node_idx),
                    "rank": int(rank),
                    "split": "test",
                    "true_label": int(true_label),
                    "original_pred_label": int(pred_label),
                    "original_fraud_probability": float(fraud_probability),
                    "subgraph_pred_label": -1,
                    "subgraph_fraud_probability": np.nan,
                    "used_hops": -1,
                    "subgraph_num_nodes": -1,
                    "subgraph_num_edges": -1,
                    "local_center_node_idx": -1,
                    "edge_rank": -1,
                    "edge_pos": -1,
                    "edge_mask": 0.0,
                    "local_src": -1,
                    "local_dst": -1,
                    "src": -1,
                    "dst": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            pd.DataFrame(all_rows).to_csv(paths["partial_path"], index=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(paths["results_path"], index=False)

    summary_df = summarise_results(results_df)
    summary_df.to_csv(paths["summary_path"], index=False)

    print("\n" + "=" * 100)
    print("GNNExplainer completed")
    print("=" * 100)

    print(f"Saved results: {paths['results_path']}")
    print(f"Saved summary: {paths['summary_path']}")

    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()