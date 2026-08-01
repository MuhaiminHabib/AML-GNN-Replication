from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import dgl

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph

from dgl.nn.pytorch.explain import SubgraphX


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import build_ibm_amlsim_graph


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage"

CHECKPOINT_PATH = BASE_DIR / "shared_amlsim_graphsage_model.pt"
NODE_LIST_PATH = BASE_DIR / "shared_amlsim_graphsage_explanation_nodes.csv"

OUTPUT_PATH = BASE_DIR / "dgl_subgraphx_shared_amlsim_graphsage_1hop_results.csv"
PARTIAL_PATH = BASE_DIR / "dgl_subgraphx_shared_amlsim_graphsage_1hop_partial.csv"


# =============================================================================
# Settings must match shared AMLSim GraphSAGE checkpoint
# =============================================================================

SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

HIDDEN_DIM = 64
DROPOUT = 0.5

FRAUD_LABEL = 1

TOP_K_EDGES = 20

# =============================================================================
# Important AMLSim SubgraphX adjustment
# =============================================================================
# SubgraphX failed on AMLSim 2-hop subgraphs because the MCTS search recursion
# became too deep. Therefore, this script tests SubgraphX on 1-hop ego-subgraphs.
#
# This keeps:
#   - same shared checkpoint
#   - same original node list
#   - same model
#   - same dataset setting
#
# Only the local explanation search space is reduced from 2-hop to 1-hop.
# =============================================================================

DEFAULT_HOPS = 1
FALLBACK_HOPS = 1
MAX_SUBGRAPH_EDGES = 10_000

# Smaller MCTS settings for AMLSim diagnostic run.
SUBGRAPHX_NUM_HOPS = 1
SUBGRAPHX_NUM_ROLLOUTS = 3
SUBGRAPHX_NUM_CHILD = 3
SUBGRAPHX_NODE_MIN = 3
SUBGRAPHX_SHAPLEY_STEPS = 3


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# PyG GraphSAGE model
# =============================================================================

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


class DGLTargetNodeAsGraphModel(torch.nn.Module):
    """
    DGL SubgraphX is designed for graph classification.

    Our task is node classification.

    Adaptation:
        For each local ego-subgraph, we make the target node's logits behave
        like graph-level logits.

    DGL sees:
        model(graph, feat) -> shape [1, num_classes]

    Internally:
        PyG model gives logits for all local nodes.
        We return logits only for the center target node.
    """

    def __init__(self, pyg_model, target_local_node_idx: int):
        super().__init__()
        self.pyg_model = pyg_model
        self.target_local_node_idx = int(target_local_node_idx)

    def forward(self, graph, feat):
        src, dst = graph.edges()

        edge_index = torch.stack(
            [
                src.to(feat.device),
                dst.to(feat.device),
            ],
            dim=0,
        )

        node_logits = self.pyg_model(feat, edge_index)
        target_logits = node_logits[self.target_local_node_idx].unsqueeze(0)

        return target_logits


def load_shared_model(data, device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=checkpoint.get("hidden_dim", HIDDEN_DIM),
        out_channels=2,
        dropout=checkpoint.get("dropout", DROPOUT),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


# =============================================================================
# Subgraph utilities
# =============================================================================

def build_local_subgraph(data, node_idx: int, device):
    """
    Build a 1-hop ego-subgraph around the explanation node.

    Returns:
        subset: global node IDs included in local subgraph
        sub_edge_index: relabelled local edge_index
        mapping: local index of the target node
        used_hops: number of hops used
    """

    for hops in [DEFAULT_HOPS, FALLBACK_HOPS]:
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
            flow="source_to_target",
        )

        if sub_edge_index.size(1) <= MAX_SUBGRAPH_EDGES:
            return (
                subset.to(device),
                sub_edge_index.to(device),
                mapping.to(device),
                hops,
            )

    return (
        subset.to(device),
        sub_edge_index.to(device),
        mapping.to(device),
        FALLBACK_HOPS,
    )


def pyg_subgraph_to_dgl(sub_edge_index, num_nodes: int, device):
    src = sub_edge_index[0].detach().cpu()
    dst = sub_edge_index[1].detach().cpu()

    graph = dgl.graph(
        (src, dst),
        num_nodes=num_nodes,
    )

    graph = graph.to(device)

    return graph


@torch.no_grad()
def get_prediction(model, x, edge_index, local_node_idx: int):
    model.eval()

    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[local_node_idx].argmax().detach().cpu().item())
    fraud_probability = float(probs[local_node_idx, FRAUD_LABEL].detach().cpu().item())

    return pred_label, fraud_probability


def normalise_subgraphx_nodes(raw_output, local_node_idx: int):
    """
    DGL SubgraphX explain_graph returns selected local node IDs.

    This function handles tensors, arrays, lists, tuples, sets, and dictionaries.
    """

    if raw_output is None:
        return [int(local_node_idx)]

    if isinstance(raw_output, torch.Tensor):
        nodes = raw_output.detach().cpu().view(-1).tolist()
        nodes = [int(x) for x in nodes]
        return sorted(set(nodes)) if nodes else [int(local_node_idx)]

    if isinstance(raw_output, np.ndarray):
        nodes = raw_output.reshape(-1).tolist()
        nodes = [int(x) for x in nodes]
        return sorted(set(nodes)) if nodes else [int(local_node_idx)]

    if isinstance(raw_output, dict):
        for key in [
            "nodes",
            "node_ids",
            "coalition",
            "subgraph_nodes",
            "selected_nodes",
        ]:
            if key in raw_output:
                return normalise_subgraphx_nodes(raw_output[key], local_node_idx)

        return [int(local_node_idx)]

    if isinstance(raw_output, (list, tuple, set)):
        nodes = []

        for item in raw_output:
            if isinstance(item, torch.Tensor):
                item_nodes = item.detach().cpu().view(-1).tolist()
                nodes.extend([int(x) for x in item_nodes])

            elif isinstance(item, np.ndarray):
                item_nodes = item.reshape(-1).tolist()
                nodes.extend([int(x) for x in item_nodes])

            elif isinstance(item, (list, tuple, set)):
                for sub_item in item:
                    if isinstance(sub_item, (int, np.integer, float)):
                        nodes.append(int(sub_item))

            elif isinstance(item, (int, np.integer)):
                nodes.append(int(item))

            elif isinstance(item, float):
                nodes.append(int(item))

        nodes = sorted(set(nodes))

        return nodes if nodes else [int(local_node_idx)]

    return [int(local_node_idx)]


def induced_edges_from_selected_nodes(
    selected_local_nodes,
    sub_edge_index_cpu,
    subset_cpu,
    center_node_idx,
    local_node_idx,
    used_hops,
    pred_label,
    fraud_probability,
):
    """
    SubgraphX explains using an important node coalition.

    For the common faithfulness evaluator, we convert the selected node coalition
    into induced edges. Each induced edge gets edge_mask=1.0.
    """

    selected_set = set(int(x) for x in selected_local_nodes)

    if int(local_node_idx) not in selected_set:
        selected_set.add(int(local_node_idx))

    rows = []

    num_edges = int(sub_edge_index_cpu.size(1))

    for edge_pos in range(num_edges):
        local_src = int(sub_edge_index_cpu[0, edge_pos].item())
        local_dst = int(sub_edge_index_cpu[1, edge_pos].item())

        if local_src in selected_set and local_dst in selected_set:
            global_src = int(subset_cpu[local_src].item())
            global_dst = int(subset_cpu[local_dst].item())

            rows.append(
                {
                    "dataset": "IBM AMLSim",
                    "setting": "previous_reproduction",
                    "model": "GraphSAGE",
                    "explainer": "DGL_SubgraphX",
                    "center_node_idx": int(center_node_idx),
                    "center_node_local_idx": int(local_node_idx),
                    "used_hops": int(used_hops),
                    "subgraph_num_nodes": int(subset_cpu.numel()),
                    "subgraph_num_edges": int(num_edges),
                    "selected_subgraph_nodes": int(len(selected_set)),
                    "pred_label": int(pred_label),
                    "fraud_probability": float(fraud_probability),
                    "edge_pos": int(edge_pos),
                    "src": int(global_src),
                    "dst": int(global_dst),
                    "edge_mask": 1.0,
                }
            )

    rows = rows[:TOP_K_EDGES]

    for rank, row in enumerate(rows, start=1):
        row["edge_rank"] = int(rank)

    return rows


def explain_one_node(
    pyg_model,
    data,
    global_node_idx: int,
    device,
):
    subset, sub_edge_index, mapping, used_hops = build_local_subgraph(
        data=data,
        node_idx=global_node_idx,
        device=device,
    )

    x_sub = data.x[subset].to(device)
    sub_edge_index = sub_edge_index.to(device)
    local_node_idx = int(mapping.item())

    pred_label, fraud_probability = get_prediction(
        model=pyg_model,
        x=x_sub,
        edge_index=sub_edge_index,
        local_node_idx=local_node_idx,
    )

    dgl_graph = pyg_subgraph_to_dgl(
        sub_edge_index=sub_edge_index,
        num_nodes=int(x_sub.size(0)),
        device=device,
    )

    dgl_graph_model = DGLTargetNodeAsGraphModel(
        pyg_model=pyg_model,
        target_local_node_idx=local_node_idx,
    ).to(device)

    dgl_graph_model.eval()

    explainer = SubgraphX(
        dgl_graph_model,
        num_hops=SUBGRAPHX_NUM_HOPS,
        num_rollouts=SUBGRAPHX_NUM_ROLLOUTS,
        num_child=SUBGRAPHX_NUM_CHILD,
        node_min=SUBGRAPHX_NODE_MIN,
        shapley_steps=SUBGRAPHX_SHAPLEY_STEPS,
        log=False,
    )

    try:
        raw_nodes = explainer.explain_graph(
            dgl_graph,
            x_sub,
            target_class=pred_label,
        )
    except RecursionError as exc:
        print(f"SubgraphX recursion failed for node {global_node_idx}: {exc}")

        subset_cpu = subset.detach().cpu()
        sub_edge_index_cpu = sub_edge_index.detach().cpu()

        return [
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction_1hop",
                "model": "GraphSAGE",
                "explainer": "DGL_SubgraphX",
                "center_node_idx": int(global_node_idx),
                "center_node_local_idx": int(local_node_idx),
                "used_hops": int(used_hops),
                "subgraph_num_nodes": int(subset_cpu.numel()),
                "subgraph_num_edges": int(sub_edge_index_cpu.size(1)),
                "selected_subgraph_nodes": 0,
                "pred_label": int(pred_label),
                "fraud_probability": float(fraud_probability),
                "edge_rank": 0,
                "edge_pos": -1,
                "src": -1,
                "dst": -1,
                "edge_mask": 0.0,
                "status": "recursion_failed",
            }
        ]

    print(f"Raw SubgraphX output type: {type(raw_nodes)}")

    selected_local_nodes = normalise_subgraphx_nodes(
        raw_output=raw_nodes,
        local_node_idx=local_node_idx,
    )

    selected_local_nodes = [
        int(n)
        for n in selected_local_nodes
        if 0 <= int(n) < int(x_sub.size(0))
    ]

    if not selected_local_nodes:
        selected_local_nodes = [int(local_node_idx)]

    sub_edge_index_cpu = sub_edge_index.detach().cpu()
    subset_cpu = subset.detach().cpu()

    rows = induced_edges_from_selected_nodes(
        selected_local_nodes=selected_local_nodes,
        sub_edge_index_cpu=sub_edge_index_cpu,
        subset_cpu=subset_cpu,
        center_node_idx=global_node_idx,
        local_node_idx=local_node_idx,
        used_hops=used_hops,
        pred_label=pred_label,
        fraud_probability=fraud_probability,
    )

    if not rows:
        rows = [
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction_1hop",
                "model": "GraphSAGE",
                "explainer": "DGL_SubgraphX",
                "center_node_idx": int(global_node_idx),
                "center_node_local_idx": int(local_node_idx),
                "used_hops": int(used_hops),
                "subgraph_num_nodes": int(subset_cpu.numel()),
                "subgraph_num_edges": int(sub_edge_index_cpu.size(1)),
                "selected_subgraph_nodes": int(len(selected_local_nodes)),
                "pred_label": int(pred_label),
                "fraud_probability": float(fraud_probability),
                "edge_rank": 0,
                "edge_pos": -1,
                "src": -1,
                "dst": -1,
                "edge_mask": 0.0,
                "status": "no_induced_edges",
            }
        ]
    else:
        for row in rows:
            row["setting"] = "previous_reproduction_1hop"
            row["status"] = "ok"

    return rows


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("Running DGL SubgraphX on shared AMLSim GraphSAGE setup")
    print("=" * 100)
    print("Mode: 1-hop diagnostic SubgraphX run")
    print("=" * 100)

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Node list:  {NODE_LIST_PATH}")
    print(f"Local hops: {DEFAULT_HOPS}")
    print(f"SubgraphX num_hops: {SUBGRAPHX_NUM_HOPS}")
    print(f"SubgraphX rollouts: {SUBGRAPHX_NUM_ROLLOUTS}")
    print(f"SubgraphX num_child: {SUBGRAPHX_NUM_CHILD}")
    print(f"SubgraphX shapley_steps: {SUBGRAPHX_SHAPLEY_STEPS}")

    if not NODE_LIST_PATH.exists():
        raise FileNotFoundError(f"Missing node list: {NODE_LIST_PATH}")

    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    data = data.to(device)

    pyg_model, checkpoint = load_shared_model(data, device)
    pyg_model.eval()

    node_df = pd.read_csv(NODE_LIST_PATH)
    node_ids = node_df["node_idx"].astype(int).tolist()

    all_rows = []

    for i, node_idx in enumerate(node_ids, start=1):
        print("\n" + "-" * 100)
        print(f"[{i}/{len(node_ids)}] Explaining AMLSim node {node_idx} with DGL SubgraphX")
        print("-" * 100)

        try:
            rows = explain_one_node(
                pyg_model=pyg_model,
                data=data,
                global_node_idx=int(node_idx),
                device=device,
            )
        except Exception as exc:
            print(f"Failed node {node_idx}: {type(exc).__name__}: {exc}")

            rows = [
                {
                    "dataset": "IBM AMLSim",
                    "setting": "previous_reproduction_1hop",
                    "model": "GraphSAGE",
                    "explainer": "DGL_SubgraphX",
                    "center_node_idx": int(node_idx),
                    "center_node_local_idx": -1,
                    "used_hops": DEFAULT_HOPS,
                    "subgraph_num_nodes": -1,
                    "subgraph_num_edges": -1,
                    "selected_subgraph_nodes": 0,
                    "pred_label": -1,
                    "fraud_probability": np.nan,
                    "edge_rank": 0,
                    "edge_pos": -1,
                    "src": -1,
                    "dst": -1,
                    "edge_mask": 0.0,
                    "status": f"failed_{type(exc).__name__}",
                }
            ]

        all_rows.extend(rows)

        pd.DataFrame(all_rows).to_csv(PARTIAL_PATH, index=False)

        first = rows[0]

        valid_edges = [
            row for row in rows
            if int(row["src"]) >= 0 and int(row["dst"]) >= 0
        ]

        print(
            f"Done node {node_idx} | "
            f"status={first.get('status', 'unknown')} | "
            f"subgraph_nodes={first['subgraph_num_nodes']} | "
            f"subgraph_edges={first['subgraph_num_edges']} | "
            f"used_hops={first['used_hops']} | "
            f"selected_nodes={first['selected_subgraph_nodes']} | "
            f"explanation_edges={len(valid_edges)}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("DGL SubgraphX AMLSim GraphSAGE 1-hop diagnostic completed")
    print("=" * 100)
    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Saved partial to: {PARTIAL_PATH}")

    if result_df.empty:
        print("No rows were produced.")
        return

    status_summary = (
        result_df
        .groupby("status", as_index=False)
        .agg(
            rows=("center_node_idx", "count"),
            nodes=("center_node_idx", "nunique"),
        )
    )

    print("\nStatus summary:")
    print(status_summary.to_string(index=False))

    valid_df = result_df[
        (result_df["src"] >= 0)
        & (result_df["dst"] >= 0)
    ].copy()

    summary = (
        result_df
        .groupby(["dataset", "setting", "model", "explainer"], as_index=False)
        .agg(
            explained_nodes=("center_node_idx", "nunique"),
            total_rows=("edge_rank", "count"),
            mean_subgraph_nodes=("subgraph_num_nodes", "mean"),
            mean_subgraph_edges=("subgraph_num_edges", "mean"),
            mean_selected_subgraph_nodes=("selected_subgraph_nodes", "mean"),
            mean_edge_mask=("edge_mask", "mean"),
            max_edge_mask=("edge_mask", "max"),
        )
    )

    print("\nSummary including empty/failed explanation rows:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not valid_df.empty:
        valid_summary = (
            valid_df
            .groupby(["dataset", "setting", "model", "explainer"], as_index=False)
            .agg(
                explained_nodes=("center_node_idx", "nunique"),
                total_valid_edges=("edge_rank", "count"),
                mean_subgraph_nodes=("subgraph_num_nodes", "mean"),
                mean_subgraph_edges=("subgraph_num_edges", "mean"),
                mean_selected_subgraph_nodes=("selected_subgraph_nodes", "mean"),
            )
        )

        print("\nSummary valid explanation edges only:")
        print(valid_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\nNo valid induced SubgraphX explanation edges were produced.")


if __name__ == "__main__":
    main()