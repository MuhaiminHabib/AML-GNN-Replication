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

TOP_K_LIST = [5, 10, 20]


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


EXPLAINER_FILE_TEMPLATES = {
    "GNNExplainer": "gnnexplainer_shared_amlsim_{model}_results.csv",
    "PGExplainer": "pgexplainer_shared_amlsim_{model}_results.csv",
    "DGL_SubgraphX_1hop": "dgl_subgraphx_shared_amlsim_{model}_1hop_results.csv",
}


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainers" / "amlsim_all_models"
OUTPUT_DETAIL_PATH = OUTPUT_DIR / "all_amlsim_models_explainer_faithfulness_detail.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "all_amlsim_models_explainer_faithfulness_summary.csv"


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
# Loading helpers
# =============================================================================

def load_data(device):
    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=SEED,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        include_fraud_tx_count_features=INCLUDE_FRAUD_TX_COUNT_FEATURES,
    )

    return data.to(device)


def get_model_dir(model_name: str):
    return PROJECT_ROOT / "outputs" / "explainers" / f"amlsim_{model_name}"


def get_checkpoint_path(model_name: str):
    return get_model_dir(model_name) / f"shared_amlsim_{model_name}_model.pt"


def load_shared_model(model_name: str, data, device):
    checkpoint_path = get_checkpoint_path(model_name)

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


def get_explainer_path(model_name: str, explainer_name: str):
    model_dir = get_model_dir(model_name)

    filename = EXPLAINER_FILE_TEMPLATES[explainer_name].format(model=model_name)

    return model_dir / filename


def load_explainer_results(model_name: str):
    loaded = {}

    for explainer_name in EXPLAINER_FILE_TEMPLATES:
        path = get_explainer_path(model_name, explainer_name)

        if not path.exists():
            print(f"WARNING: Missing {explainer_name} file for {model_name}: {path}")
            continue

        df = pd.read_csv(path)

        if "center_node_idx" not in df.columns:
            raise ValueError(f"{path} is missing center_node_idx column.")

        loaded[explainer_name] = df

        print(
            f"Loaded {model_name} / {explainer_name}: "
            f"{len(df)} rows, "
            f"{df['center_node_idx'].nunique()} nodes"
        )

    return loaded


# =============================================================================
# Prediction and graph perturbation
# =============================================================================

@torch.no_grad()
def predict_target(model, x_sub, sub_edge_index, local_node_idx: int):
    model.eval()

    logits = model(x_sub, sub_edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[local_node_idx].argmax().detach().cpu().item())
    fraud_probability = float(probs[local_node_idx, FRAUD_LABEL].detach().cpu().item())

    return pred_label, fraud_probability


def build_local_subgraph(data, node_idx: int, hops: int, device):
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=int(node_idx),
        num_hops=int(hops),
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


def get_used_hops(node_expl_df: pd.DataFrame):
    if "used_hops" in node_expl_df.columns:
        values = node_expl_df["used_hops"].dropna().astype(int).tolist()
        values = [v for v in values if v > 0]

        if values:
            return int(values[0])

    return 2


def get_selected_edge_positions(node_expl_df: pd.DataFrame, top_k: int):
    if "edge_pos" not in node_expl_df.columns:
        raise ValueError("Explanation result must contain edge_pos column.")

    df = node_expl_df.copy()
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

    unique_positions = []
    seen = set()

    for pos in edge_positions:
        if pos not in seen:
            unique_positions.append(pos)
            seen.add(pos)

        if len(unique_positions) >= top_k:
            break

    return unique_positions


def make_edge_index_after_deletion(sub_edge_index, selected_edge_positions):
    num_edges = int(sub_edge_index.size(1))

    valid_positions = [
        int(pos)
        for pos in selected_edge_positions
        if 0 <= int(pos) < num_edges
    ]

    if len(valid_positions) == 0:
        return sub_edge_index

    keep_mask = torch.ones(num_edges, dtype=torch.bool, device=sub_edge_index.device)
    keep_mask[valid_positions] = False

    return sub_edge_index[:, keep_mask]


def make_edge_index_after_insertion(sub_edge_index, selected_edge_positions):
    num_edges = int(sub_edge_index.size(1))

    valid_positions = sorted(
        set(
            int(pos)
            for pos in selected_edge_positions
            if 0 <= int(pos) < num_edges
        )
    )

    if len(valid_positions) == 0:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=sub_edge_index.device,
        )

    keep_positions = torch.tensor(
        valid_positions,
        dtype=torch.long,
        device=sub_edge_index.device,
    )

    return sub_edge_index[:, keep_positions]


# =============================================================================
# Faithfulness evaluation
# =============================================================================

def evaluate_one_node(
    model_name: str,
    explainer_name: str,
    model,
    data,
    node_idx: int,
    node_expl_df: pd.DataFrame,
    top_k: int,
    device,
):
    used_hops = get_used_hops(node_expl_df)

    subset, sub_edge_index, mapping = build_local_subgraph(
        data=data,
        node_idx=int(node_idx),
        hops=int(used_hops),
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
        "model": model_name,
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


def evaluate_model(model_name: str, data, device):
    print("\n" + "=" * 100)
    print(f"Evaluating AMLSim {model_name.upper()} explainer faithfulness")
    print("=" * 100)

    print("\nLoading shared model...")
    model, checkpoint = load_shared_model(
        model_name=model_name,
        data=data,
        device=device,
    )

    print(f"Loaded checkpoint: {get_checkpoint_path(model_name)}")

    print("\nLoading explainer outputs...")
    explainer_results = load_explainer_results(model_name)

    if not explainer_results:
        print(f"No explainer results found for model={model_name}. Skipping.")
        return []

    rows = []

    for explainer_name, expl_df in explainer_results.items():
        print("\n" + "-" * 100)
        print(f"Model={model_name} | Explainer={explainer_name}")
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
                        model_name=model_name,
                        explainer_name=explainer_name,
                        model=model,
                        data=data,
                        node_idx=int(node_idx),
                        node_expl_df=node_expl_df,
                        top_k=int(top_k),
                        device=device,
                    )

                    rows.append(row)

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

                    rows.append(
                        {
                            "dataset": "IBM AMLSim",
                            "setting": "previous_reproduction",
                            "model": model_name,
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
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return rows


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gcn", "graphsage", "gatv2", "all"],
        default="all",
        help="Model to evaluate. Default: all.",
    )

    args = parser.parse_args()

    if args.model == "all":
        model_names = ["gcn", "graphsage", "gatv2"]
    else:
        model_names = [args.model]

    seed_everything(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("Evaluating all AMLSim model explainer faithfulness")
    print("=" * 100)
    print(f"Using device: {device}")

    print("\nLoading AMLSim graph...")
    data = load_data(device=device)

    all_rows = []

    for model_name in model_names:
        rows = evaluate_model(
            model_name=model_name,
            data=data,
            device=device,
        )

        all_rows.extend(rows)

        if all_rows:
            partial_df = pd.DataFrame(all_rows)
            partial_path = OUTPUT_DIR / "all_amlsim_models_explainer_faithfulness_partial.csv"
            partial_df.to_csv(partial_path, index=False)
            print(f"\nSaved partial combined results: {partial_path}")

    if not all_rows:
        raise RuntimeError("No faithfulness rows were produced.")

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(OUTPUT_DETAIL_PATH, index=False)

    summary_df = build_summary(detail_df)
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("\n" + "=" * 100)
    print("AMLSim all-model faithfulness evaluation completed")
    print("=" * 100)

    print(f"Saved detail results to:  {OUTPUT_DETAIL_PATH}")
    print(f"Saved summary results to: {OUTPUT_SUMMARY_PATH}")

    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nInterpretation guide:")
    print("Higher deletion_drop means removing the explanation edges hurts the model more.")
    print("Higher deletion_flip_rate means removing the explanation edges often changes the prediction.")
    print("Higher insertion_fraud_prob means explanation edges alone preserve fraud confidence better.")
    print("Higher insertion_preservation_rate means explanation edges alone preserve the original prediction more often.")
    print("Higher sparsity means the explanation is smaller relative to the local subgraph.")


if __name__ == "__main__":
    main()