from pathlib import Path
import sys
import copy
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


OUTPUT_DIR = Path("outputs/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = OUTPUT_DIR / "dgl_subgraphx_elliptic_results.csv"
OUTPUT_FILE = OUTPUT_DIR / "dgl_subgraphx_elliptic_faithfulness.csv"
SUMMARY_FILE = OUTPUT_DIR / "dgl_subgraphx_elliptic_faithfulness_summary.csv"

SEED = 42
MODEL_NAME = "graphsage"

HIDDEN_CHANNELS = 110
EPOCHS = 1000
PATIENCE = 100
LR = 0.009
WEIGHT_DECAY = 5e-4

ILLICIT_LABEL = 0
LICIT_LABEL = 1


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    return build_marasi_elliptic_data(
        feature_setting="tx+agg",
        seed=SEED,
    )


@torch.no_grad()
def evaluate_model(model, data, mask):
    model.eval()

    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = pred[mask].detach().cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "illicit_precision": precision_score(
            y_true,
            y_pred,
            pos_label=ILLICIT_LABEL,
            zero_division=0,
        ),
        "illicit_recall": recall_score(
            y_true,
            y_pred,
            pos_label=ILLICIT_LABEL,
            zero_division=0,
        ),
        "illicit_f1": f1_score(
            y_true,
            y_pred,
            pos_label=ILLICIT_LABEL,
            zero_division=0,
        ),
    }


def train_model(model_name: str, data, device):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    model = build_marasi_model(
        model_name=model_name,
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        loss.backward()
        optimizer.step()

        val_metrics = evaluate_model(model, data, data.val_mask)
        val_f1 = val_metrics["illicit_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"{model_name} | epoch={epoch:04d} | "
                f"loss={loss.item():.5f} | "
                f"val_illicit_f1={val_f1:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_model(model, data, data.train_mask)
    val_metrics = evaluate_model(model, data, data.val_mask)
    test_metrics = evaluate_model(model, data, data.test_mask)

    print("\nFinal metrics")
    print("-" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Train illicit F1: {train_metrics['illicit_f1']:.4f}")
    print(f"Val illicit F1:   {val_metrics['illicit_f1']:.4f}")
    print(f"Test illicit F1:  {test_metrics['illicit_f1']:.4f}")

    return model, {
        "best_epoch": best_epoch,
        "train_illicit_f1": train_metrics["illicit_f1"],
        "val_illicit_f1": val_metrics["illicit_f1"],
        "test_illicit_f1": test_metrics["illicit_f1"],
    }


@torch.no_grad()
def predict_node(model, data, node_id: int, edge_index=None):
    model.eval()

    if edge_index is None:
        edge_index = data.edge_index

    logits = model(data.x, edge_index)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)

    return {
        "logits": logits[node_id].detach().cpu(),
        "probs": probs[node_id].detach().cpu(),
        "pred": int(pred[node_id].detach().cpu()),
        "prob_illicit": float(probs[node_id, ILLICIT_LABEL].detach().cpu()),
    }


def remove_explanation_edges_from_full_graph(data, explanation_edge_pairs):
    if len(explanation_edge_pairs) == 0:
        return data.edge_index

    remove_pairs = set((int(src), int(dst)) for src, dst in explanation_edge_pairs)

    edge_index_cpu = data.edge_index.detach().cpu()

    keep_mask = []

    for i in range(edge_index_cpu.size(1)):
        src = int(edge_index_cpu[0, i])
        dst = int(edge_index_cpu[1, i])

        keep_mask.append((src, dst) not in remove_pairs)

    keep_mask = torch.tensor(keep_mask, dtype=torch.bool, device=data.edge_index.device)

    return data.edge_index[:, keep_mask]


def keep_only_explanation_edges(data, explanation_edge_pairs):
    if len(explanation_edge_pairs) == 0:
        # Empty edge graph.
        return torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device)

    keep_pairs = set((int(src), int(dst)) for src, dst in explanation_edge_pairs)

    edge_index_cpu = data.edge_index.detach().cpu()

    keep_mask = []

    for i in range(edge_index_cpu.size(1)):
        src = int(edge_index_cpu[0, i])
        dst = int(edge_index_cpu[1, i])

        keep_mask.append((src, dst) in keep_pairs)

    keep_mask = torch.tensor(keep_mask, dtype=torch.bool, device=data.edge_index.device)

    return data.edge_index[:, keep_mask]


def evaluate_explanation_row(model, data, row):
    node_id = int(row["node_id"])

    explanation_edge_pairs = json.loads(row["explanation_original_edge_pairs"])
    explanation_nodes = json.loads(row["explanation_original_nodes"])

    original = predict_node(
        model=model,
        data=data,
        node_id=node_id,
        edge_index=data.edge_index,
    )

    deletion_edge_index = remove_explanation_edges_from_full_graph(
        data=data,
        explanation_edge_pairs=explanation_edge_pairs,
    )

    deletion = predict_node(
        model=model,
        data=data,
        node_id=node_id,
        edge_index=deletion_edge_index,
    )

    insertion_edge_index = keep_only_explanation_edges(
        data=data,
        explanation_edge_pairs=explanation_edge_pairs,
    )

    insertion = predict_node(
        model=model,
        data=data,
        node_id=node_id,
        edge_index=insertion_edge_index,
    )

    deletion_drop = original["prob_illicit"] - deletion["prob_illicit"]
    insertion_gap = original["prob_illicit"] - insertion["prob_illicit"]

    deletion_label_flip = int(deletion["pred"] != original["pred"])
    insertion_preservation = int(insertion["pred"] == original["pred"])

    return {
        "dataset": row["dataset"],
        "feature_setting": row["feature_setting"],
        "explainer": row["explainer"],
        "model": row["model"],
        "node_id": node_id,
        "true_label": int(row["true_label"]),
        "original_pred_label": int(original["pred"]),
        "original_prob_illicit": float(original["prob_illicit"]),
        "saved_pred_prob_illicit": float(row["pred_prob_illicit"]),
        "wrapped_pred_label": int(row["wrapped_pred_label"]),
        "wrapped_matches_full_prediction": bool(row["wrapped_matches_full_prediction"]),
        "deletion_pred_label": int(deletion["pred"]),
        "deletion_prob_illicit": float(deletion["prob_illicit"]),
        "deletion_drop": float(deletion_drop),
        "deletion_label_flip": deletion_label_flip,
        "insertion_pred_label": int(insertion["pred"]),
        "insertion_prob_illicit": float(insertion["prob_illicit"]),
        "insertion_gap": float(insertion_gap),
        "insertion_preservation": insertion_preservation,
        "num_ego_nodes": int(row["num_ego_nodes"]),
        "num_ego_edges": int(row["num_ego_edges"]),
        "num_explanation_nodes": int(row["num_explanation_nodes"]),
        "num_explanation_edges": int(row["num_explanation_edges"]),
        "sparsity_nodes": float(row["sparsity_nodes"]),
        "sparsity_edges": float(row["sparsity_edges"]),
        "explanation_original_nodes": json.dumps(explanation_nodes),
        "explanation_original_edge_pairs": json.dumps(explanation_edge_pairs),
    }


def make_summary(results_df):
    summary = (
        results_df.groupby(["explainer", "model"])
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
            sparsity_nodes_mean=("sparsity_nodes", "mean"),
            sparsity_edges_mean=("sparsity_edges", "mean"),
            num_explanation_nodes_mean=("num_explanation_nodes", "mean"),
            num_explanation_edges_mean=("num_explanation_edges", "mean"),
            wrapped_match_rate=("wrapped_matches_full_prediction", "mean"),
        )
        .reset_index()
    )

    return summary


def main():
    set_seed(SEED)

    device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Reading SubgraphX explanations from: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    explanation_df = pd.read_csv(INPUT_FILE)

    print("\nLoaded explanation rows:")
    print(len(explanation_df))

    if "wrapped_matches_full_prediction" in explanation_df.columns:
        explanation_df = explanation_df[
            explanation_df["wrapped_matches_full_prediction"] == True
        ].copy()

    print("\nRows kept for faithfulness evaluation:")
    print(len(explanation_df))

    if len(explanation_df) == 0:
        raise RuntimeError("No valid SubgraphX rows left after filtering.")

    data = load_data().to(device)

    model, model_metrics = train_model(
        model_name=MODEL_NAME,
        data=data,
        device=device,
    )

    rows = []

    for i, row in explanation_df.iterrows():
        print("\n" + "-" * 80)
        print(f"Evaluating SubgraphX faithfulness for node_id={int(row['node_id'])}")
        print("-" * 80)

        result = evaluate_explanation_row(
            model=model,
            data=data,
            row=row,
        )

        rows.append(result)

        print(
            f"node={result['node_id']} | "
            f"original_prob={result['original_prob_illicit']:.6f} | "
            f"deletion_prob={result['deletion_prob_illicit']:.6f} | "
            f"deletion_drop={result['deletion_drop']:.6f} | "
            f"insertion_prob={result['insertion_prob_illicit']:.6f} | "
            f"insertion_preservation={result['insertion_preservation']}"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_FILE, index=False)

    summary_df = make_summary(results_df)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nDGL SubgraphX faithfulness summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print(f"\nSaved faithfulness results to: {OUTPUT_FILE}")
    print(f"Saved faithfulness summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()