from pathlib import Path
import sys
import argparse
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.explain import SubgraphX

from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.utils import k_hop_subgraph


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

# Important for AMLSim:
# 2-hop SubgraphX failed earlier because AMLSim local subgraphs are too large.
# Therefore we use 1-hop for all AMLSim models.
DEFAULT_HOPS = 1
FALLBACK_HOPS = 1
MAX_SUBGRAPH_EDGES = 10000

TOP_K_EDGES = 20

SUBGRAPHX_NUM_HOPS = 1
NUM_ROLLOUTS = 3
NUM_CHILDREN = 3
NODE_MIN = 3
SHAPLEY_STEPS = 3


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
# PyG Models
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
# DGL wrapper
# =============================================================================

class DGLTargetNodeAsGraphModel(torch.nn.Module):
    """
    DGL SubgraphX in this installed version explains graph-level predictions.

    Our task is node classification. Therefore, for each target node, we wrap the
    PyG node model as a graph model that returns the logits of the target local
    node as a single graph-level prediction.
    """

    def __init__(self, pyg_model, target_local_node_idx: int):
        super().__init__()

        self.pyg_model = pyg_model
        self.target_local_node_idx = int(target_local_node_idx)

    def forward(self, graph, feat):
        src, dst = graph.edges()

        edge_index = torch.stack([src, dst], dim=0).long()
        edge_index = edge_index.to(feat.device)

        logits = self.pyg_model(feat, edge_index)

        target_logits = logits[self.target_local_node_idx].unsqueeze(0)

        return target_logits


# =============================================================================
# Paths / loading
# =============================================================================

def get_paths(model_name: str):
    output_dir = PROJECT_ROOT / "outputs" / "explainers" / f"amlsim_{model_name}"

    checkpoint_path = output_dir / f"shared_amlsim_{model_name}_model.pt"
    nodes_path = output_dir / f"shared_amlsim_{model_name}_explanation_nodes.csv"

    results_path = output_dir / f"dgl_subgraphx_shared_amlsim_{model_name}_1hop_results.csv"
    partial_path = output_dir / f"dgl_subgraphx_shared_amlsim_{model_name}_1hop_partial.csv"
    summary_path = output_dir / f"dgl_subgraphx_shared_amlsim_{model_name}_1hop_summary.csv"

    return {
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "nodes_path": nodes_path,
        "results_path": results_path,
        "partial_path": partial_path,
        "summary_path": summary_path,
    }


def load_data():
    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    # DGL environment is CPU-based in your setup, so keep data on CPU.
    data = data.cpu()

    return data


def load_shared_model(model_name: str, data, checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            f"Run this first:\n"
            f"python scripts\\prepare_shared_amlsim_model_explainer_setup.py --model {model_name}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(
        model_name=model_name,
        in_channels=data.num_features,
        checkpoint=checkpoint,
    ).cpu()

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


# =============================================================================
# Helpers
# =============================================================================

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

    return subset.cpu(), sub_edge_index.cpu(), mapping.cpu()


def get_subgraph_with_fallback(data, node_idx: int):
    subset, sub_edge_index, mapping = make_local_subgraph(
        data=data,
        node_idx=node_idx,
        requested_hops=DEFAULT_HOPS,
    )

    used_hops = DEFAULT_HOPS

    if int(sub_edge_index.size(1)) > MAX_SUBGRAPH_EDGES:
        print(
            f"  1-hop subgraph too large "
            f"({int(sub_edge_index.size(1))} edges). "
            f"Keeping it because fallback is also 1-hop."
        )

    return subset, sub_edge_index, mapping, used_hops


def pyg_to_dgl_graph(sub_edge_index, num_nodes: int):
    src = sub_edge_index[0].long()
    dst = sub_edge_index[1].long()

    graph = dgl.graph(
        (src, dst),
        num_nodes=int(num_nodes),
    )

    return graph


def map_selected_nodes_to_edges(selected_local_nodes, sub_edge_index):
    """
    Convert SubgraphX selected nodes into induced local edge positions.

    SubgraphX returns important nodes/coefficient coalition. To make the output
    compatible with our edge-based faithfulness evaluator, we convert the selected
    node set into the induced edges between those nodes.
    """

    selected_set = set(int(n) for n in selected_local_nodes)

    selected_edge_positions = []

    for edge_pos in range(int(sub_edge_index.size(1))):
        src = int(sub_edge_index[0, edge_pos].item())
        dst = int(sub_edge_index[1, edge_pos].item())

        if src in selected_set and dst in selected_set:
            selected_edge_positions.append(edge_pos)

    return selected_edge_positions


def normalise_subgraphx_output(result):
    """
    DGL SubgraphX versions may return a tensor/list of selected nodes.
    This helper converts it to a Python list of local node ids.
    """

    if result is None:
        return []

    if isinstance(result, torch.Tensor):
        return [int(x) for x in result.detach().cpu().view(-1).tolist()]

    if isinstance(result, np.ndarray):
        return [int(x) for x in result.reshape(-1).tolist()]

    if isinstance(result, list):
        out = []
        for item in result:
            if isinstance(item, torch.Tensor):
                out.extend([int(x) for x in item.detach().cpu().view(-1).tolist()])
            elif isinstance(item, (list, tuple, np.ndarray)):
                out.extend([int(x) for x in np.array(item).reshape(-1).tolist()])
            else:
                out.append(int(item))
        return out

    if isinstance(result, tuple):
        out = []
        for item in result:
            if isinstance(item, torch.Tensor):
                out.extend([int(x) for x in item.detach().cpu().view(-1).tolist()])
            elif isinstance(item, (list, tuple, np.ndarray)):
                out.extend([int(x) for x in np.array(item).reshape(-1).tolist()])
            else:
                try:
                    out.append(int(item))
                except Exception:
                    pass
        return out

    try:
        return [int(result)]
    except Exception:
        return []


def explain_one_node(
    model_name: str,
    pyg_model,
    data,
    node_idx: int,
    rank: int,
    true_label: int,
    original_pred_label: int,
    original_fraud_probability: float,
):
    subset, sub_edge_index, mapping, used_hops = get_subgraph_with_fallback(
        data=data,
        node_idx=node_idx,
    )

    x_sub = data.x[subset].cpu()
    local_node_idx = int(mapping.item())

    subgraph_pred_label, subgraph_fraud_probability = get_prediction(
        model=pyg_model,
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

    dgl_graph = pyg_to_dgl_graph(
        sub_edge_index=sub_edge_index,
        num_nodes=int(subset.numel()),
    )

    graph_model = DGLTargetNodeAsGraphModel(
        pyg_model=pyg_model,
        target_local_node_idx=local_node_idx,
    )

    graph_model.eval()

    explainer = SubgraphX(
        graph_model,
        num_hops=SUBGRAPHX_NUM_HOPS,
        num_rollouts=NUM_ROLLOUTS,
        node_min=NODE_MIN,
        shapley_steps=SHAPLEY_STEPS,
        num_child=NUM_CHILDREN,
    )

    selected_nodes_raw = explainer.explain_graph(
        dgl_graph,
        x_sub,
        target_class=int(subgraph_pred_label),
    )

    selected_local_nodes = normalise_subgraphx_output(selected_nodes_raw)

    # Always include the target local node if SubgraphX returns an empty set.
    if len(selected_local_nodes) == 0:
        selected_local_nodes = [local_node_idx]

    selected_local_nodes = sorted(set(int(n) for n in selected_local_nodes))

    selected_edge_positions = map_selected_nodes_to_edges(
        selected_local_nodes=selected_local_nodes,
        sub_edge_index=sub_edge_index,
    )

    # If the induced subgraph is too large, keep only the first TOP_K_EDGES.
    selected_edge_positions = selected_edge_positions[:TOP_K_EDGES]

    local_edges = sub_edge_index.detach().cpu()
    subset_cpu = subset.detach().cpu()

    rows = []

    if len(selected_edge_positions) == 0:
        rows.append(
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction",
                "model": model_name,
                "explainer": "DGL_SubgraphX_1hop",
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
                "selected_subgraph_nodes": int(len(selected_local_nodes)),
                "selected_local_nodes": " ".join(str(n) for n in selected_local_nodes),
                "edge_rank": -1,
                "edge_pos": -1,
                "edge_mask": 1.0,
                "local_src": -1,
                "local_dst": -1,
                "src": -1,
                "dst": -1,
            }
        )

        return rows

    for edge_rank, edge_pos in enumerate(selected_edge_positions, start=1):
        local_src = int(local_edges[0, edge_pos].item())
        local_dst = int(local_edges[1, edge_pos].item())

        global_src = int(subset_cpu[local_src].item())
        global_dst = int(subset_cpu[local_dst].item())

        rows.append(
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction",
                "model": model_name,
                "explainer": "DGL_SubgraphX_1hop",
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
                "selected_subgraph_nodes": int(len(selected_local_nodes)),
                "selected_local_nodes": " ".join(str(n) for n in selected_local_nodes),
                "edge_rank": int(edge_rank),
                "edge_pos": int(edge_pos),
                "edge_mask": 1.0,
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
                    "dataset": "IBM AMLSim",
                    "setting": "previous_reproduction",
                    "model": results_df["model"].iloc[0] if len(results_df) else "unknown",
                    "explainer": "DGL_SubgraphX_1hop",
                    "explained_nodes": 0,
                    "total_rows": int(len(results_df)),
                    "mean_used_hops": np.nan,
                    "mean_subgraph_nodes": np.nan,
                    "mean_subgraph_edges": np.nan,
                    "mean_selected_subgraph_nodes": np.nan,
                    "mean_edge_mask": np.nan,
                    "max_edge_mask": np.nan,
                    "min_edge_mask": np.nan,
                }
            ]
        )

    summary = {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "model": valid_df["model"].iloc[0],
        "explainer": "DGL_SubgraphX_1hop",
        "explained_nodes": int(valid_df["center_node_idx"].nunique()),
        "total_rows": int(len(valid_df)),
        "mean_used_hops": float(valid_df.groupby("center_node_idx")["used_hops"].first().mean()),
        "mean_subgraph_nodes": float(valid_df.groupby("center_node_idx")["subgraph_num_nodes"].first().mean()),
        "mean_subgraph_edges": float(valid_df.groupby("center_node_idx")["subgraph_num_edges"].first().mean()),
        "mean_selected_subgraph_nodes": float(valid_df.groupby("center_node_idx")["selected_subgraph_nodes"].first().mean()),
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
        help="AMLSim model to explain with DGL SubgraphX.",
    )

    args = parser.parse_args()
    model_name = args.model.lower()

    seed_everything(SEED)

    paths = get_paths(model_name)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Running DGL SubgraphX 1-hop on shared AMLSim {model_name.upper()} model")
    print("=" * 100)
    print("Using device: CPU")
    print("Note: DGL SubgraphX is run through the separate .venv-dgl environment.")

    print("\nLoading AMLSim graph...")
    data = load_data()

    print("\nLoading shared checkpoint...")
    model, checkpoint = load_shared_model(
        model_name=model_name,
        data=data,
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
                pyg_model=model,
                data=data,
                node_idx=node_idx,
                rank=rank,
                true_label=true_label,
                original_pred_label=pred_label,
                original_fraud_probability=fraud_probability,
            )

            all_rows.extend(rows)

            partial_df = pd.DataFrame(all_rows)
            partial_df.to_csv(paths["partial_path"], index=False)

            valid_rows = [r for r in rows if int(r["edge_pos"]) >= 0]

            if valid_rows:
                selected_nodes = valid_rows[0].get("selected_subgraph_nodes", np.nan)
                print(
                    f"  status=ok | "
                    f"selected_nodes={selected_nodes} | "
                    f"explanation_edges={len(valid_rows)}"
                )
            else:
                print("  status=ok_empty | explanation_edges=0")

            print(f"  Saved partial results: {paths['partial_path']}")

        except Exception as exc:
            print(f"  FAILED node {node_idx}: {type(exc).__name__}: {exc}")

            all_rows.append(
                {
                    "dataset": "IBM AMLSim",
                    "setting": "previous_reproduction",
                    "model": model_name,
                    "explainer": "DGL_SubgraphX_1hop",
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
                    "selected_subgraph_nodes": -1,
                    "selected_local_nodes": "",
                    "edge_rank": -1,
                    "edge_pos": -1,
                    "edge_mask": 1.0,
                    "local_src": -1,
                    "local_dst": -1,
                    "src": -1,
                    "dst": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            pd.DataFrame(all_rows).to_csv(paths["partial_path"], index=False)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(paths["results_path"], index=False)

    summary_df = summarise_results(results_df)
    summary_df.to_csv(paths["summary_path"], index=False)

    print("\n" + "=" * 100)
    print("DGL SubgraphX completed")
    print("=" * 100)

    print(f"Saved results: {paths['results_path']}")
    print(f"Saved summary: {paths['summary_path']}")

    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()