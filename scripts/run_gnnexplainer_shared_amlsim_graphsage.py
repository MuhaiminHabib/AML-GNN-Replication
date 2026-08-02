from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import build_ibm_amlsim_graph


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage"

CHECKPOINT_PATH = BASE_DIR / "shared_amlsim_graphsage_model.pt"
NODE_LIST_PATH = BASE_DIR / "shared_amlsim_graphsage_explanation_nodes.csv"

OUTPUT_PATH = BASE_DIR / "gnnexplainer_shared_amlsim_graphsage_results.csv"
PARTIAL_PATH = BASE_DIR / "gnnexplainer_shared_amlsim_graphsage_partial.csv"


# =============================================================================
# Settings must match shared checkpoint setup
# =============================================================================

SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

HIDDEN_DIM = 64
DROPOUT = 0.5

FRAUD_LABEL = 1

EXPLAINER_EPOCHS = 200
TOP_K_EDGES = 20

# AMLSim graph is large, so use local ego-subgraphs.
# If a 2-hop subgraph is too large, the script falls back to 1-hop.
DEFAULT_HOPS = 2
FALLBACK_HOPS = 1
MAX_SUBGRAPH_EDGES = 100_000


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model
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


def build_local_subgraph(data, node_idx: int, device):
    """
    Build an ego-subgraph around the explanation node.

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
            return subset.to(device), sub_edge_index.to(device), mapping.to(device), hops

    # If even fallback is large, still return the fallback subgraph.
    return subset.to(device), sub_edge_index.to(device), mapping.to(device), FALLBACK_HOPS


def make_explainer(model):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EXPLAINER_EPOCHS),
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


@torch.no_grad()
def get_prediction(model, x, edge_index, local_node_idx: int):
    model.eval()

    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[local_node_idx].argmax().detach().cpu().item())
    fraud_probability = float(probs[local_node_idx, FRAUD_LABEL].detach().cpu().item())

    return pred_label, fraud_probability


def explain_one_node(
    explainer,
    model,
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
    local_node_idx = int(mapping.item())

    pred_label, fraud_probability = get_prediction(
        model=model,
        x=x_sub,
        edge_index=sub_edge_index,
        local_node_idx=local_node_idx,
    )

    explanation = explainer(
        x=x_sub,
        edge_index=sub_edge_index,
        index=local_node_idx,
    )

    edge_mask = explanation.edge_mask

    if edge_mask is None:
        raise RuntimeError(f"GNNExplainer returned no edge mask for node {global_node_idx}")

    edge_mask = edge_mask.detach().cpu()
    sub_edge_index_cpu = sub_edge_index.detach().cpu()
    subset_cpu = subset.detach().cpu()

    num_edges = int(sub_edge_index_cpu.size(1))

    if num_edges == 0:
        return []

    k = min(TOP_K_EDGES, num_edges)

    top_values, top_indices = torch.topk(edge_mask, k=k)

    rows = []

    for rank, edge_pos in enumerate(top_indices.tolist(), start=1):
        local_src = int(sub_edge_index_cpu[0, edge_pos].item())
        local_dst = int(sub_edge_index_cpu[1, edge_pos].item())

        global_src = int(subset_cpu[local_src].item())
        global_dst = int(subset_cpu[local_dst].item())

        rows.append(
            {
                "dataset": "IBM AMLSim",
                "setting": "previous_reproduction",
                "model": "GraphSAGE",
                "explainer": "GNNExplainer",
                "center_node_idx": int(global_node_idx),
                "center_node_local_idx": int(local_node_idx),
                "used_hops": int(used_hops),
                "subgraph_num_nodes": int(subset_cpu.numel()),
                "subgraph_num_edges": int(num_edges),
                "pred_label": int(pred_label),
                "fraud_probability": float(fraud_probability),
                "edge_rank": int(rank),
                "edge_pos": int(edge_pos),
                "src": int(global_src),
                "dst": int(global_dst),
                "edge_mask": float(top_values[rank - 1].item()),
            }
        )

    return rows


def main():
    print("=" * 100)
    print("Running GNNExplainer on shared AMLSim GraphSAGE setup")
    print("=" * 100)

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Node list:  {NODE_LIST_PATH}")

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

    model, checkpoint = load_shared_model(data, device)

    node_df = pd.read_csv(NODE_LIST_PATH)
    node_ids = node_df["node_idx"].astype(int).tolist()

    explainer = make_explainer(model)

    all_rows = []

    for i, node_idx in enumerate(node_ids, start=1):
        print("\n" + "-" * 100)
        print(f"[{i}/{len(node_ids)}] Explaining AMLSim node {node_idx}")
        print("-" * 100)

        rows = explain_one_node(
            explainer=explainer,
            model=model,
            data=data,
            global_node_idx=node_idx,
            device=device,
        )

        all_rows.extend(rows)

        pd.DataFrame(all_rows).to_csv(PARTIAL_PATH, index=False)

        if rows:
            print(
                f"Done node {node_idx} | "
                f"subgraph_nodes={rows[0]['subgraph_num_nodes']} | "
                f"subgraph_edges={rows[0]['subgraph_num_edges']} | "
                f"used_hops={rows[0]['used_hops']} | "
                f"top_edges={len(rows)}"
            )
        else:
            print(f"Node {node_idx} had no explanation edges.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("GNNExplainer AMLSim GraphSAGE completed")
    print("=" * 100)
    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Saved partial to: {PARTIAL_PATH}")

    if not result_df.empty:
        summary = (
            result_df
            .groupby(["dataset", "setting", "model", "explainer"], as_index=False)
            .agg(
                explained_nodes=("center_node_idx", "nunique"),
                total_rows=("edge_rank", "count"),
                mean_subgraph_nodes=("subgraph_num_nodes", "mean"),
                mean_subgraph_edges=("subgraph_num_edges", "mean"),
                mean_edge_mask=("edge_mask", "mean"),
                max_edge_mask=("edge_mask", "max"),
            )
        )

        print("\nSummary:")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()