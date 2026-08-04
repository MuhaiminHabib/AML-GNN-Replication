from pathlib import Path
import argparse
import ast
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ELLIPTIC_FAITHFULNESS_DETAIL = (
    PROJECT_ROOT / "outputs" / "explainers" / "all_models_explainer_faithfulness.csv"
)

AMLSIM_FAITHFULNESS_DETAIL = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "amlsim_all_models"
    / "all_amlsim_models_explainer_faithfulness_detail.csv"
)


EXPLAINER_DISPLAY = {
    "gnnexplainer": "GNNExplainer",
    "pgexplainer": "PGExplainer",
    "subgraphx": "SubgraphX",
}


MODEL_DISPLAY = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
}


def get_result_file(dataset: str, model: str, explainer: str) -> Path:
    dataset = dataset.lower()
    model = model.lower()
    explainer = explainer.lower()

    if dataset == "elliptic":
        if explainer == "gnnexplainer":
            return PROJECT_ROOT / "outputs" / "explainers" / f"gnnexplainer_shared_{model}_results.csv"

        if explainer == "pgexplainer":
            return PROJECT_ROOT / "outputs" / "explainers" / f"pgexplainer_shared_{model}_results.csv"

        if explainer == "subgraphx":
            return PROJECT_ROOT / "outputs" / "explainers" / f"dgl_subgraphx_shared_{model}_results.csv"

    if dataset == "amlsim":
        base = PROJECT_ROOT / "outputs" / "explainers" / f"amlsim_{model}"

        if explainer == "gnnexplainer":
            return base / f"gnnexplainer_shared_amlsim_{model}_results.csv"

        if explainer == "pgexplainer":
            return base / f"pgexplainer_shared_amlsim_{model}_results.csv"

        if explainer == "subgraphx":
            return base / f"dgl_subgraphx_shared_amlsim_{model}_1hop_results.csv"

    raise ValueError(f"Unknown dataset/model/explainer combination: {dataset}, {model}, {explainer}")


def safe_literal(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value:
        return []

    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def normalise_pair(pair):
    """
    Convert [src, dst] or (src, dst) into integer tuple.
    """

    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None

    try:
        return int(pair[0]), int(pair[1])
    except Exception:
        return None


def pick_best_case(dataset: str, model: str, explainer: str):
    """
    Automatically select the most visually useful node.

    We choose the node with highest deletion_drop because that means
    the explanation edges actually mattered to the model prediction.
    """

    dataset = dataset.lower()
    model = model.lower()
    explainer = explainer.lower()

    if dataset == "elliptic":
        path = ELLIPTIC_FAITHFULNESS_DETAIL

        if not path.exists():
            raise FileNotFoundError(f"Missing Elliptic faithfulness file: {path}")

        df = pd.read_csv(path)

        explainer_name = EXPLAINER_DISPLAY[explainer]

        df = df[
            (df["model"].str.lower() == model)
            & (df["explainer"] == explainer_name)
        ].copy()

        if df.empty:
            raise RuntimeError(f"No Elliptic faithfulness rows found for {model} / {explainer_name}")

        df["deletion_drop"] = pd.to_numeric(df["deletion_drop"], errors="coerce")
        df = df.sort_values("deletion_drop", ascending=False)

        return int(df.iloc[0]["node_id"])

    if dataset == "amlsim":
        path = AMLSIM_FAITHFULNESS_DETAIL

        if not path.exists():
            raise FileNotFoundError(f"Missing AMLSim faithfulness file: {path}")

        df = pd.read_csv(path)

        explainer_name = EXPLAINER_DISPLAY[explainer]
        if explainer == "subgraphx":
            # AMLSim SubgraphX may be stored as SubgraphX or DGL_SubgraphX_1hop.
            df = df[
                (df["model"].str.lower() == model)
                & (df["explainer"].str.lower().str.contains("subgraphx"))
            ].copy()
        else:
            df = df[
                (df["model"].str.lower() == model)
                & (df["explainer"] == explainer_name)
            ].copy()

        if "top_k" in df.columns:
            df = df[pd.to_numeric(df["top_k"], errors="coerce") == 20].copy()

        if df.empty:
            raise RuntimeError(f"No AMLSim faithfulness rows found for {model} / {explainer_name}")

        df["deletion_drop"] = pd.to_numeric(df["deletion_drop"], errors="coerce")
        df = df.sort_values("deletion_drop", ascending=False)

        return int(df.iloc[0]["center_node_idx"])

    raise ValueError(f"Unknown dataset: {dataset}")


def extract_edges_from_amlsim(df: pd.DataFrame, node_id: int, top_k: int):
    df = df.copy()

    df = df[pd.to_numeric(df["center_node_idx"], errors="coerce") == int(node_id)].copy()

    if df.empty:
        raise RuntimeError(f"No AMLSim explanation rows found for node {node_id}")

    if "edge_rank" in df.columns:
        df["edge_rank"] = pd.to_numeric(df["edge_rank"], errors="coerce")
        df = df.sort_values("edge_rank", ascending=True)
    elif "edge_mask" in df.columns:
        df["edge_mask"] = pd.to_numeric(df["edge_mask"], errors="coerce")
        df = df.sort_values("edge_mask", ascending=False)

    df = df.head(top_k).copy()

    edges = []
    scores = []

    for _, row in df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        score = float(row["edge_mask"]) if "edge_mask" in df.columns and pd.notna(row["edge_mask"]) else 1.0

        edges.append((src, dst))
        scores.append(score)

    meta = {
        "dataset": "AMLSim",
        "model": str(df.iloc[0]["model"]) if "model" in df.columns else "",
        "explainer": str(df.iloc[0]["explainer"]) if "explainer" in df.columns else "",
        "true_label": df.iloc[0]["true_label"] if "true_label" in df.columns else np.nan,
        "pred_label": df.iloc[0]["original_pred_label"] if "original_pred_label" in df.columns else np.nan,
        "prob": df.iloc[0]["original_fraud_probability"] if "original_fraud_probability" in df.columns else np.nan,
        "used_hops": df.iloc[0]["used_hops"] if "used_hops" in df.columns else np.nan,
        "subgraph_num_nodes": df.iloc[0]["subgraph_num_nodes"] if "subgraph_num_nodes" in df.columns else np.nan,
        "subgraph_num_edges": df.iloc[0]["subgraph_num_edges"] if "subgraph_num_edges" in df.columns else np.nan,
    }

    return edges, scores, meta


def extract_edges_from_elliptic(df: pd.DataFrame, node_id: int, top_k: int):
    df = df.copy()

    df = df[pd.to_numeric(df["node_id"], errors="coerce") == int(node_id)].copy()

    if df.empty:
        raise RuntimeError(f"No Elliptic explanation row found for node {node_id}")

    row = df.iloc[0]

    if "explanation_original_edge_pairs" in df.columns:
        raw_edges = safe_literal(row["explanation_original_edge_pairs"])
    else:
        raw_edges = []

    edges = []

    for pair in raw_edges:
        edge = normalise_pair(pair)
        if edge is not None:
            edges.append(edge)

    edges = edges[:top_k]

    scores = []

    if "top_edge_scores" in df.columns:
        raw_scores = safe_literal(row["top_edge_scores"])

        for score in raw_scores[: len(edges)]:
            try:
                scores.append(float(score))
            except Exception:
                scores.append(1.0)

    if len(scores) != len(edges):
        # SubgraphX often has only selected edge pairs, not continuous masks.
        scores = [1.0 for _ in edges]

    meta = {
        "dataset": "Elliptic",
        "model": str(row["model"]) if "model" in df.columns else "",
        "explainer": str(row["explainer"]) if "explainer" in df.columns else "",
        "true_label": row["true_label"] if "true_label" in df.columns else np.nan,
        "pred_label": row["pred_label"] if "pred_label" in df.columns else np.nan,
        "prob": row["pred_prob_illicit"] if "pred_prob_illicit" in df.columns else np.nan,
        "used_hops": row["ego_hops"] if "ego_hops" in df.columns else np.nan,
        "subgraph_num_nodes": row["num_graph_nodes"] if "num_graph_nodes" in df.columns else np.nan,
        "subgraph_num_edges": row["num_graph_edges"] if "num_graph_edges" in df.columns else np.nan,
    }

    return edges, scores, meta


def load_explanation_edges(dataset: str, model: str, explainer: str, node_id: int, top_k: int):
    path = get_result_file(dataset, model, explainer)

    if not path.exists():
        raise FileNotFoundError(f"Missing explanation result file:\n{path}")

    df = pd.read_csv(path)

    if dataset.lower() == "amlsim":
        edges, scores, meta = extract_edges_from_amlsim(df, node_id, top_k)
    elif dataset.lower() == "elliptic":
        edges, scores, meta = extract_edges_from_elliptic(df, node_id, top_k)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not edges:
        raise RuntimeError(f"No explanation edges found for {dataset} / {model} / {explainer} / node {node_id}")

    return path, edges, scores, meta


def normalise_widths(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return []

    if np.all(np.isnan(scores)):
        return [2.0] * len(scores)

    scores = np.nan_to_num(scores, nan=0.0)

    min_score = float(scores.min())
    max_score = float(scores.max())

    if math.isclose(min_score, max_score):
        return [3.0] * len(scores)

    widths = 1.5 + 5.5 * ((scores - min_score) / (max_score - min_score))
    return widths.tolist()


def shorten_node_label(node_id, target_node):
    node_id = int(node_id)

    if node_id == int(target_node):
        return f"TARGET\n{node_id}"

    return str(node_id)


def draw_explanation_graph(
    dataset: str,
    model: str,
    explainer: str,
    node_id: int,
    edges,
    scores,
    meta,
    output_path: Path,
):
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    if int(node_id) not in graph.nodes:
        graph.add_node(int(node_id))

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    # Layout
    if node_count <= 2:
        pos = nx.spring_layout(graph, seed=42)
    else:
        pos = nx.spring_layout(graph, seed=42, k=1.2 / math.sqrt(max(node_count, 1)), iterations=200)

    # Node styling
    node_sizes = []
    node_colors = []
    node_border_colors = []

    for n in graph.nodes:
        if int(n) == int(node_id):
            node_sizes.append(1450)
            node_colors.append("#d62728")
            node_border_colors.append("#7f0000")
        else:
            degree = graph.degree[n]
            node_sizes.append(420 + min(degree, 5) * 80)
            node_colors.append("#d9e8fb")
            node_border_colors.append("#5b7fa6")

    edge_widths = normalise_widths(scores)

    plt.figure(figsize=(11, 8.5))
    ax = plt.gca()
    ax.set_title(
        f"{dataset.upper()} analyst explanation view\n"
        f"{MODEL_DISPLAY.get(model.lower(), model)} + {EXPLAINER_DISPLAY.get(explainer.lower(), explainer)} | "
        f"Target node: {node_id}",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_border_colors,
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edges,
        width=edge_widths,
        edge_color="#333333",
        arrows=True,
        arrowsize=15,
        alpha=0.78,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    # Labels: show target and the most connected neighbours.
    degrees = dict(graph.degree())
    top_label_nodes = set(
        [int(node_id)]
        + [
            int(n)
            for n, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    )

    labels = {
        n: shorten_node_label(n, node_id)
        for n in graph.nodes
        if int(n) in top_label_nodes
    }

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    # Edge rank labels for top few edges.
    edge_labels = {}

    for i, edge in enumerate(edges[:8], start=1):
        edge_labels[edge] = f"#{i}"

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.55,
        ax=ax,
    )

    prob = meta.get("prob", np.nan)
    prob_text = "unknown" if pd.isna(prob) else f"{float(prob):.4f}"

    true_label = meta.get("true_label", np.nan)
    pred_label = meta.get("pred_label", np.nan)
    used_hops = meta.get("used_hops", np.nan)
    subgraph_num_nodes = meta.get("subgraph_num_nodes", np.nan)
    subgraph_num_edges = meta.get("subgraph_num_edges", np.nan)

    info_text = (
        f"Prediction score: {prob_text}\n"
        f"True label: {true_label} | Predicted label: {pred_label}\n"
        f"Explanation edges shown: {edge_count}\n"
        f"Nodes in visual: {node_count}\n"
        f"Original local subgraph: {subgraph_num_nodes} nodes, {subgraph_num_edges} edges\n"
        f"Hops used: {used_hops}\n\n"
        f"How to read: thicker arrows are more important explanation edges.\n"
        f"The red node is the suspicious account/transaction being investigated."
    )

    ax.text(
        0.01,
        0.01,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_edge_table(dataset, model, explainer, node_id, edges, scores, output_path: Path):
    rows = []

    for rank, ((src, dst), score) in enumerate(zip(edges, scores), start=1):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "explainer": explainer,
                "target_node": node_id,
                "edge_rank": rank,
                "src": src,
                "dst": dst,
                "importance_score": score,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def plot_one_case(dataset: str, model: str, explainer: str, node_id=None, top_k=20):
    dataset = dataset.lower()
    model = model.lower()
    explainer = explainer.lower()

    if node_id is None:
        node_id = pick_best_case(dataset, model, explainer)
        print(f"Auto-selected node for {dataset} / {model} / {explainer}: {node_id}")

    result_file, edges, scores, meta = load_explanation_edges(
        dataset=dataset,
        model=model,
        explainer=explainer,
        node_id=int(node_id),
        top_k=top_k,
    )

    safe_name = f"{dataset}_{model}_{explainer}_node_{int(node_id)}_top{top_k}"
    output_png = OUTPUT_DIR / f"{safe_name}.png"
    output_csv = OUTPUT_DIR / f"{safe_name}_edges.csv"

    draw_explanation_graph(
        dataset=dataset,
        model=model,
        explainer=explainer,
        node_id=int(node_id),
        edges=edges,
        scores=scores,
        meta=meta,
        output_path=output_png,
    )

    save_edge_table(
        dataset=dataset,
        model=model,
        explainer=explainer,
        node_id=int(node_id),
        edges=edges,
        scores=scores,
        output_path=output_csv,
    )

    print("\nCreated local explanation visual:")
    print(f"Dataset:   {dataset}")
    print(f"Model:     {model}")
    print(f"Explainer: {explainer}")
    print(f"Node:      {node_id}")
    print(f"Source:    {result_file.relative_to(PROJECT_ROOT)}")
    print(f"PNG:       {output_png}")
    print(f"CSV:       {output_csv}")

    return output_png, output_csv


def run_demo_cases(top_k: int):
    """
    Create a small set of professor-ready example visuals.
    """

    demo_cases = [
        ("elliptic", "gcn", "gnnexplainer"),
        ("elliptic", "gcn", "subgraphx"),
        ("elliptic", "gatv2", "gnnexplainer"),
        ("amlsim", "gatv2", "gnnexplainer"),
        ("amlsim", "graphsage", "gnnexplainer"),
        ("amlsim", "gcn", "gnnexplainer"),
    ]

    created = []

    for dataset, model, explainer in demo_cases:
        try:
            png, csv = plot_one_case(
                dataset=dataset,
                model=model,
                explainer=explainer,
                node_id=None,
                top_k=top_k,
            )
            created.append((png, csv))
        except Exception as e:
            print("\nWARNING: Could not create demo case:")
            print(f"  {dataset} / {model} / {explainer}")
            print(f"  Reason: {e}")

    print("\n" + "=" * 100)
    print("Demo visuals created")
    print("=" * 100)

    for png, csv in created:
        print(png)
        print(csv)

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Plot analyst-facing local explanation graph."
    )

    parser.add_argument(
        "--dataset",
        choices=["elliptic", "amlsim"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--model",
        choices=["gcn", "graphsage", "gatv2"],
        help="Model name.",
    )

    parser.add_argument(
        "--explainer",
        choices=["gnnexplainer", "pgexplainer", "subgraphx"],
        help="Explainer name.",
    )

    parser.add_argument(
        "--node",
        type=int,
        default=None,
        help="Target node/account ID. If omitted, script auto-selects the best deletion-drop case.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of explanation edges to show.",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a small set of demo visuals automatically.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Analyst local explanation graph visualizer")
    print("=" * 100)

    if args.demo:
        run_demo_cases(top_k=args.top_k)
        return

    if not args.dataset or not args.model or not args.explainer:
        raise SystemExit(
            "Please provide --dataset, --model, and --explainer, or use --demo."
        )

    plot_one_case(
        dataset=args.dataset,
        model=args.model,
        explainer=args.explainer,
        node_id=args.node,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()