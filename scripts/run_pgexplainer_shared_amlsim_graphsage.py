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
from torch_geometric.explain.algorithm import PGExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim_prev_reproduction import build_ibm_amlsim_graph


BASE_DIR = PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage"

CHECKPOINT_PATH = BASE_DIR / "shared_amlsim_graphsage_model.pt"

# Use moderate-confidence nodes for PGExplainer fixing attempt.
NODE_LIST_PATH = BASE_DIR / "shared_amlsim_graphsage_explanation_nodes_moderate.csv"

OUTPUT_PATH = BASE_DIR / "pgexplainer_shared_amlsim_graphsage_moderate_results.csv"
PARTIAL_PATH = BASE_DIR / "pgexplainer_shared_amlsim_graphsage_moderate_partial.csv"
TRAINING_LOG_PATH = BASE_DIR / "pgexplainer_shared_amlsim_graphsage_moderate_training_log.csv"


SEED = 42

VAL_SIZE = 0.15
TEST_SIZE = 0.20
INCLUDE_FRAUD_TX_COUNT_FEATURES = False

HIDDEN_DIM = 64
DROPOUT = 0.5

FRAUD_LABEL = 1

PGEXPLAINER_EPOCHS = 50
PGEXPLAINER_LR = 0.003

TOP_K_EDGES = 20

DEFAULT_HOPS = 1
FALLBACK_HOPS = 1
MAX_SUBGRAPH_EDGES = 100_000

MAX_TRAIN_NODES = 60

TRAIN_MIN_PROB = 0.55
TRAIN_MAX_PROB = 0.98
TRAIN_TARGET_PROB = 0.80


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def get_prediction(model, x, edge_index, local_node_idx: int):
    model.eval()

    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=1)

    pred_label = int(logits[local_node_idx].argmax().detach().cpu().item())
    fraud_probability = float(probs[local_node_idx, FRAUD_LABEL].detach().cpu().item())

    return pred_label, fraud_probability


@torch.no_grad()
def select_pgexplainer_training_nodes(model, data, max_nodes: int):
    model.eval()

    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, FRAUD_LABEL]
    preds = logits.argmax(dim=1)

    correct_fraud_train_mask = (
        data.train_mask
        & (data.y == FRAUD_LABEL)
        & (preds == data.y)
    )

    candidate_nodes = torch.where(correct_fraud_train_mask)[0]

    if candidate_nodes.numel() == 0:
        raise RuntimeError("No correctly predicted fraud training nodes found for PGExplainer.")

    candidate_probs = probs[candidate_nodes]

    moderate_mask = (
        (candidate_probs >= TRAIN_MIN_PROB)
        & (candidate_probs <= TRAIN_MAX_PROB)
    )

    moderate_nodes = candidate_nodes[moderate_mask]
    moderate_probs = probs[moderate_nodes]

    if moderate_nodes.numel() >= max_nodes:
        distance = torch.abs(moderate_probs - TRAIN_TARGET_PROB)
        sorted_order = torch.argsort(distance, descending=False)
        selected = moderate_nodes[sorted_order][:max_nodes]
        rule = "moderate_probability_correct_fraud_train_nodes"
    else:
        distance = torch.abs(candidate_probs - TRAIN_TARGET_PROB)
        sorted_order = torch.argsort(distance, descending=False)
        selected = candidate_nodes[sorted_order][:max_nodes]
        rule = "fallback_closest_probability_correct_fraud_train_nodes"

    print(f"PGExplainer training candidate nodes: {candidate_nodes.numel()}")
    print(f"Moderate training nodes in range: {moderate_nodes.numel()}")
    print(f"Training selection rule: {rule}")

    selected_probs = probs[selected].detach().cpu().numpy()

    print(
        f"Selected train probability range: "
        f"min={selected_probs.min():.8f}, "
        f"mean={selected_probs.mean():.8f}, "
        f"max={selected_probs.max():.8f}"
    )

    return selected.detach().cpu().tolist()


def make_pgexplainer(model, device):
    algorithm = PGExplainer(
        epochs=PGEXPLAINER_EPOCHS,
        lr=PGEXPLAINER_LR,
    )

    algorithm = algorithm.to(device)

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    return explainer


def train_pgexplainer(explainer, model, data, train_node_ids, device):
    training_rows = []

    for epoch in range(PGEXPLAINER_EPOCHS):
        epoch_losses = []
        skipped = 0

        print(f"\nPGExplainer epoch {epoch + 1}/{PGEXPLAINER_EPOCHS}")

        for node_idx in train_node_ids:
            subset, sub_edge_index, mapping, used_hops = build_local_subgraph(
                data=data,
                node_idx=int(node_idx),
                device=device,
            )

            x_sub = data.x[subset].to(device)
            sub_edge_index = sub_edge_index.to(device)
            local_node_idx = int(mapping.item())

            with torch.no_grad():
                logits = model(x_sub, sub_edge_index)
                target = logits.argmax(dim=1).to(device)

            try:
                loss = explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=x_sub,
                    edge_index=sub_edge_index,
                    target=target,
                    index=local_node_idx,
                )

                if loss is None:
                    skipped += 1
                    continue

                epoch_losses.append(float(loss))

            except Exception as exc:
                skipped += 1
                print(f"Skipped train node {node_idx}: {type(exc).__name__}: {exc}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan

        row = {
            "epoch": epoch,
            "train_nodes": len(train_node_ids),
            "used_nodes": len(epoch_losses),
            "skipped_nodes": skipped,
            "mean_loss": mean_loss,
        }

        training_rows.append(row)

        print(
            f"Epoch {epoch + 1} summary | "
            f"used={row['used_nodes']} | "
            f"skipped={row['skipped_nodes']} | "
            f"mean_loss={row['mean_loss']}"
        )

        pd.DataFrame(training_rows).to_csv(TRAINING_LOG_PATH, index=False)

    return pd.DataFrame(training_rows)


def explain_one_node(explainer, model, data, global_node_idx: int, device):
    subset, sub_edge_index, mapping, used_hops = build_local_subgraph(
        data=data,
        node_idx=global_node_idx,
        device=device,
    )

    x_sub = data.x[subset].to(device)
    sub_edge_index = sub_edge_index.to(device)
    local_node_idx = int(mapping.item())

    pred_label, fraud_probability = get_prediction(
        model=model,
        x=x_sub,
        edge_index=sub_edge_index,
        local_node_idx=local_node_idx,
    )

    with torch.no_grad():
        logits = model(x_sub, sub_edge_index)
        target = logits.argmax(dim=1).to(device)

    explanation = explainer(
        x=x_sub,
        edge_index=sub_edge_index,
        target=target,
        index=local_node_idx,
    )

    edge_mask = explanation.edge_mask

    if edge_mask is None:
        raise RuntimeError(f"PGExplainer returned no edge mask for node {global_node_idx}")

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
                "setting": "previous_reproduction_moderate_nodes",
                "model": "GraphSAGE",
                "explainer": "PGExplainer",
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
    print("Running fixed PGExplainer on shared AMLSim GraphSAGE setup")
    print("=" * 100)

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Node list:  {NODE_LIST_PATH}")
    print(f"PGExplainer epochs: {PGEXPLAINER_EPOCHS}")
    print(f"PGExplainer hops: {DEFAULT_HOPS}")
    print(f"PGExplainer training nodes: {MAX_TRAIN_NODES}")

    if not NODE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Missing moderate node list: {NODE_LIST_PATH}. "
            f"Run scripts/select_moderate_amlsim_graphsage_explanation_nodes.py first."
        )

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

    print("\nExplanation node probabilities from selected CSV:")
    print(node_df[["rank", "node_idx", "fraud_probability"]].to_string(index=False))

    pg_train_nodes = select_pgexplainer_training_nodes(
        model=model,
        data=data,
        max_nodes=MAX_TRAIN_NODES,
    )

    print(f"\nSelected {len(pg_train_nodes)} PGExplainer training nodes.")
    print(pg_train_nodes[:20])

    explainer = make_pgexplainer(model, device)

    train_pgexplainer(
        explainer=explainer,
        model=model,
        data=data,
        train_node_ids=pg_train_nodes,
        device=device,
    )

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
                f"top_edges={len(rows)} | "
                f"max_edge_mask={max(row['edge_mask'] for row in rows):.8f}"
            )
        else:
            print(f"Node {node_idx} had no explanation edges.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("Fixed PGExplainer AMLSim GraphSAGE completed")
    print("=" * 100)
    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Saved partial to: {PARTIAL_PATH}")
    print(f"Saved training log to: {TRAINING_LOG_PATH}")

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
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

        print("\nExact edge mask stats:")
        print(result_df["edge_mask"].describe().to_string())
        print(f"Nonzero masks: {(result_df['edge_mask'] > 0).sum()} / {len(result_df)}")
        print(f"Max mask: {result_df['edge_mask'].max()}")


if __name__ == "__main__":
    main()