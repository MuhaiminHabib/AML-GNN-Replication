from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import build_ibm_amlsim_graph


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage"

CHECKPOINT_PATH = BASE_DIR / "shared_amlsim_graphsage_model.pt"

EXPLAINER_FILES = {
    "GNNExplainer": BASE_DIR / "gnnexplainer_shared_amlsim_graphsage_results.csv",
    "PGExplainer": BASE_DIR / "pgexplainer_shared_amlsim_graphsage_results.csv",
    "DGL_SubgraphX_1hop": BASE_DIR / "dgl_subgraphx_shared_amlsim_graphsage_1hop_results.csv",
}

OUTPUT_DETAIL_PATH = BASE_DIR / "shared_amlsim_graphsage_explainer_faithfulness_detail.csv"
OUTPUT_SUMMARY_PATH = BASE_DIR / "shared_amlsim_graphsage_explainer_faithfulness_summary.csv"


# =============================================================================
# Dataset / model settings
# =============================================================================

SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

HIDDEN_DIM = 64
DROPOUT = 0.5

FRAUD_LABEL = 1

# Evaluate different top-k budgets.
TOP_K_LIST = [5, 10, 20]


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


# =============================================================================
# Local subgraph and prediction helpers
# =============================================================================

def build_local_subgraph(data, node_idx: int, hops: int, device):
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        flow="source_to_target",
    )

    return (
        subset.to(device),
        sub_edge_index.to(device),
        mapping.to(device),
    )


@torch.no_grad()
def predict_target(model, x_sub, sub_edge_index, local_node_idx: int):
    model.eval()

    logits = model(x_sub, sub_edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[local_node_idx].argmax().detach().cpu().item())
    fraud_probability = float(probs[local_node_idx, FRAUD_LABEL].detach().cpu().item())

    return pred_label, fraud_probability


def make_edge_index_after_deletion(sub_edge_index, selected_edge_positions):
    """
    Remove selected explanation edges from the local subgraph.
    """

    num_edges = int(sub_edge_index.size(1))

    selected_edge_positions = [
        int(pos)
        for pos in selected_edge_positions
        if 0 <= int(pos) < num_edges
    ]

    if len(selected_edge_positions) == 0:
        return sub_edge_index

    keep_mask = torch.ones(num_edges, dtype=torch.bool, device=sub_edge_index.device)
    keep_mask[selected_edge_positions] = False

    deleted_edge_index = sub_edge_index[:, keep_mask]

    return deleted_edge_index


def make_edge_index_after_insertion(sub_edge_index, selected_edge_positions):
    """
    Keep only selected explanation edges.

    If no valid explanation edges exist, return an empty edge_index.
    """

    num_edges = int(sub_edge_index.size(1))

    selected_edge_positions = [
        int(pos)
        for pos in selected_edge_positions
        if 0 <= int(pos) < num_edges
    ]

    selected_edge_positions = sorted(set(selected_edge_positions))

    if len(selected_edge_positions) == 0:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=sub_edge_index.device,
        )

    keep_positions = torch.tensor(
        selected_edge_positions,
        dtype=torch.long,
        device=sub_edge_index.device,
    )

    inserted_edge_index = sub_edge_index[:, keep_positions]

    return inserted_edge_index


def get_selected_edge_positions(node_expl_df, top_k: int):
    """
    Select top-k explanation edge positions.

    The explanation CSVs already contain edge_rank. We use that first.
    We ignore empty SubgraphX rows where edge_pos is -1.
    """

    df = node_expl_df.copy()

    if "edge_pos" not in df.columns:
        raise ValueError("Explanation file must contain edge_pos column.")

    df = df[df["edge_pos"].astype(int) >= 0].copy()

    if df.empty:
        return []

    if "edge_rank" in df.columns:
        df = df.sort_values(["edge_rank", "edge_pos"], ascending=[True, True])
    elif "edge_mask" in df.columns:
        df = df.sort_values("edge_mask", ascending=False)
    else:
        df = df.sort_values("edge_pos", ascending=True)

    edge_positions = df["edge_pos"].astype(int).tolist()

    # Keep order but remove duplicate edge positions.
    unique_positions = []
    seen = set()

    for pos in edge_positions:
        if pos not in seen:
            unique_positions.append(pos)
            seen.add(pos)

        if len(unique_positions) >= top_k:
            break

    return unique_positions


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_one_node(
    model,
    data,
    explainer_name: str,
    node_idx: int,
    node_expl_df: pd.DataFrame,
    top_k: int,
    device,
):
    """
    Evaluate deletion and insertion for one target node and one explainer.
    """

    if "used_hops" in node_expl_df.columns:
        used_hops = int(node_expl_df["used_hops"].dropna().iloc[0])
    else:
        used_hops = 2

    subset, sub_edge_index, mapping = build_local_subgraph(
        data=data,
        node_idx=int(node_idx),
        hops=used_hops,
        device=device,
    )

    x_sub = data.x[subset].to(device)
    local_node_idx = int(mapping.item())

    original_pred, original_fraud_prob = predict_target(
        model=model,
        x_sub=x_sub,
        sub_edge_index=sub_edge_index,
        local_node_idx=local_node_idx,
    )

    selected_edge_positions = get_selected_edge_positions(
        node_expl_df=node_expl_df,
        top_k=top_k,
    )

    deletion_edge_index = make_edge_index_after_deletion(
        sub_edge_index=sub_edge_index,
        selected_edge_positions=selected_edge_positions,
    )

    deletion_pred, deletion_fraud_prob = predict_target(
        model=model,
        x_sub=x_sub,
        sub_edge_index=deletion_edge_index,
        local_node_idx=local_node_idx,
    )

    insertion_edge_index = make_edge_index_after_insertion(
        sub_edge_index=sub_edge_index,
        selected_edge_positions=selected_edge_positions,
    )

    insertion_pred, insertion_fraud_prob = predict_target(
        model=model,
        x_sub=x_sub,
        sub_edge_index=insertion_edge_index,
        local_node_idx=local_node_idx,
    )

    num_subgraph_edges = int(sub_edge_index.size(1))
    num_selected_edges = int(len(selected_edge_positions))

    if num_subgraph_edges > 0:
        sparsity = 1.0 - (num_selected_edges / num_subgraph_edges)
    else:
        sparsity = np.nan

    deletion_drop = float(original_fraud_prob - deletion_fraud_prob)
    deletion_flip = int(deletion_pred != original_pred)

    insertion_preservation = int(insertion_pred == original_pred)

    return {
        "dataset": "IBM AMLSim",
        "setting": "previous_reproduction",
        "model": "GraphSAGE",
        "explainer": explainer_name,
        "center_node_idx": int(node_idx),
        "used_hops": int(used_hops),
        "top_k": int(top_k),
        "subgraph_num_nodes": int(subset.numel()),
        "subgraph_num_edges": int(num_subgraph_edges),
        "selected_edges": int(num_selected_edges),
        "sparsity": float(sparsity) if not pd.isna(sparsity) else np.nan,
        "original_pred": int(original_pred),
        "original_fraud_prob": float(original_fraud_prob),
        "deletion_pred": int(deletion_pred),
        "deletion_fraud_prob": float(deletion_fraud_prob),
        "deletion_drop": float(deletion_drop),
        "deletion_flip": int(deletion_flip),
        "insertion_pred": int(insertion_pred),
        "insertion_fraud_prob": float(insertion_fraud_prob),
        "insertion_preservation": int(insertion_preservation),
    }


def load_explainer_results():
    loaded = {}

    for explainer_name, path in EXPLAINER_FILES.items():
        if not path.exists():
            print(f"WARNING: Missing {explainer_name} file: {path}")
            continue

        df = pd.read_csv(path)

        if "center_node_idx" not in df.columns:
            raise ValueError(f"{path} does not contain center_node_idx column.")

        loaded[explainer_name] = df

        print(
            f"Loaded {explainer_name}: "
            f"{len(df)} rows, "
            f"{df['center_node_idx'].nunique()} nodes "
            f"from {path.name}"
        )

    if not loaded:
        raise RuntimeError("No explainer result files were found.")

    return loaded


def build_summary(detail_df: pd.DataFrame):
    summary = (
        detail_df
        .groupby(["dataset", "setting", "model", "explainer", "top_k"], as_index=False)
        .agg(
            explained_nodes=("center_node_idx", "nunique"),
            mean_used_hops=("used_hops", "mean"),
            mean_subgraph_nodes=("subgraph_num_nodes", "mean"),
            mean_subgraph_edges=("subgraph_num_edges", "mean"),
            mean_selected_edges=("selected_edges", "mean"),
            mean_sparsity=("sparsity", "mean"),
            mean_original_fraud_prob=("original_fraud_prob", "mean"),
            mean_deletion_fraud_prob=("deletion_fraud_prob", "mean"),
            mean_deletion_drop=("deletion_drop", "mean"),
            deletion_flip_rate=("deletion_flip", "mean"),
            mean_insertion_fraud_prob=("insertion_fraud_prob", "mean"),
            insertion_preservation_rate=("insertion_preservation", "mean"),
        )
    )

    return summary


def main():
    print("=" * 100)
    print("Evaluating AMLSim GraphSAGE explainer faithfulness")
    print("=" * 100)

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading AMLSim graph...")
    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    data = data.to(device)

    print("\nLoading shared GraphSAGE checkpoint...")
    model, checkpoint = load_shared_model(data, device)
    model.eval()

    print("\nLoading explainer CSV files...")
    explainer_results = load_explainer_results()

    all_rows = []

    for explainer_name, expl_df in explainer_results.items():
        print("\n" + "-" * 100)
        print(f"Evaluating {explainer_name}")
        print("-" * 100)

        node_ids = sorted(expl_df["center_node_idx"].astype(int).unique().tolist())

        for node_i, node_idx in enumerate(node_ids, start=1):
            node_expl_df = expl_df[
                expl_df["center_node_idx"].astype(int) == int(node_idx)
            ].copy()

            print(f"[{node_i}/{len(node_ids)}] Node {node_idx}")

            for top_k in TOP_K_LIST:
                try:
                    row = evaluate_one_node(
                        model=model,
                        data=data,
                        explainer_name=explainer_name,
                        node_idx=int(node_idx),
                        node_expl_df=node_expl_df,
                        top_k=int(top_k),
                        device=device,
                    )

                    all_rows.append(row)

                    print(
                        f"  top_k={top_k:>2} | "
                        f"selected={row['selected_edges']:>2} | "
                        f"orig={row['original_fraud_prob']:.6f} | "
                        f"del={row['deletion_fraud_prob']:.6f} | "
                        f"drop={row['deletion_drop']:.6f} | "
                        f"ins={row['insertion_fraud_prob']:.6f} | "
                        f"preserve={row['insertion_preservation']}"
                    )

                except Exception as exc:
                    print(
                        f"  FAILED top_k={top_k} for node {node_idx}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    all_rows.append(
                        {
                            "dataset": "IBM AMLSim",
                            "setting": "previous_reproduction",
                            "model": "GraphSAGE",
                            "explainer": explainer_name,
                            "center_node_idx": int(node_idx),
                            "used_hops": np.nan,
                            "top_k": int(top_k),
                            "subgraph_num_nodes": np.nan,
                            "subgraph_num_edges": np.nan,
                            "selected_edges": 0,
                            "sparsity": np.nan,
                            "original_pred": -1,
                            "original_fraud_prob": np.nan,
                            "deletion_pred": -1,
                            "deletion_fraud_prob": np.nan,
                            "deletion_drop": np.nan,
                            "deletion_flip": np.nan,
                            "insertion_pred": -1,
                            "insertion_fraud_prob": np.nan,
                            "insertion_preservation": np.nan,
                        }
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(OUTPUT_DETAIL_PATH, index=False)

    summary_df = build_summary(detail_df)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("\n" + "=" * 100)
    print("Faithfulness evaluation completed")
    print("=" * 100)

    print(f"Saved detail results to:  {OUTPUT_DETAIL_PATH}")
    print(f"Saved summary results to: {OUTPUT_SUMMARY_PATH}")

    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nInterpretation guide:")
    print("Higher deletion_drop means removing the explanation edges hurts the model more.")
    print("Higher deletion_flip_rate means removing the explanation edges often changes the prediction.")
    print("Higher insertion_fraud_prob means the explanation edges alone preserve fraud confidence better.")
    print("Higher insertion_preservation_rate means the explanation edges alone preserve the original prediction more often.")
    print("Higher sparsity means the explanation is smaller relative to the local subgraph.")


if __name__ == "__main__":
    main()