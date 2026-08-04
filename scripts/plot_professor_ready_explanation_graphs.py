from pathlib import Path
import math
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "graphs"
ALERT_TABLE = PROJECT_ROOT / "outputs" / "analyst_view" / "analyst_alert_queue_all.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "presentation_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_DISPLAY = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
}

EXPLAINER_DISPLAY = {
    "gnnexplainer": "GNNExplainer",
    "pgexplainer": "PGExplainer",
    "subgraphx": "SubgraphX",
}


def load_alert_table():
    if not ALERT_TABLE.exists():
        print(f"WARNING: Alert table not found: {ALERT_TABLE}")
        return pd.DataFrame()

    df = pd.read_csv(ALERT_TABLE)

    df["Dataset_lower"] = df["Dataset"].astype(str).str.lower()
    df["Model_lower"] = df["Model"].astype(str).str.lower()

    return df


def get_alert_info(alert_df, dataset, model, target_node):
    if alert_df.empty:
        return {}

    dataset = dataset.lower()
    model = model.lower()

    temp = alert_df[
        (alert_df["Dataset_lower"] == dataset)
        & (alert_df["Model_lower"].str.contains(model))
        & (pd.to_numeric(alert_df["Node ID"], errors="coerce") == int(target_node))
    ].copy()

    if temp.empty:
        return {}

    row = temp.iloc[0]

    return {
        "prob": row.get("Suspicion probability", np.nan),
        "risk_band": row.get("Risk band", "Unknown"),
        "true_raw": row.get("True label raw", np.nan),
        "pred_raw": row.get("Predicted label raw", np.nan),
        "true_suspicious": row.get("True suspicious?", np.nan),
        "pred_suspicious": row.get("Predicted suspicious?", np.nan),
        "action": row.get("Recommended analyst action", "Review manually"),
        "suspicious_class": row.get("Suspicious class name", "Suspicious"),
    }


def clean_string(value):
    return str(value).strip().lower()


def aggregate_edges(edge_df):
    df = edge_df.copy()

    df["src"] = pd.to_numeric(df["src"], errors="coerce")
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df["importance_score"] = pd.to_numeric(df["importance_score"], errors="coerce").fillna(0.0)
    df["edge_rank"] = pd.to_numeric(df["edge_rank"], errors="coerce")

    df = df.dropna(subset=["src", "dst"])
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    # Remove zero-importance clutter when there are positive scores.
    positive_df = df[df["importance_score"] > 1e-8].copy()
    if not positive_df.empty:
        df = positive_df

    grouped = (
        df.groupby(["src", "dst"], as_index=False)
        .agg(
            min_rank=("edge_rank", "min"),
            max_importance=("importance_score", "max"),
            mean_importance=("importance_score", "mean"),
            repeated_edges=("importance_score", "size"),
        )
    )

    grouped = grouped.sort_values(
        ["max_importance", "repeated_edges"],
        ascending=[False, False],
    ).reset_index(drop=True)

    grouped["display_rank"] = np.arange(1, len(grouped) + 1)

    return grouped


def keep_target_component(edge_df, target_node):
    graph = nx.Graph()

    for _, row in edge_df.iterrows():
        graph.add_edge(int(row["src"]), int(row["dst"]))

    target_node = int(target_node)

    if target_node not in graph.nodes:
        return edge_df, False

    component_nodes = nx.node_connected_component(graph, target_node)

    filtered = edge_df[
        edge_df["src"].isin(component_nodes)
        & edge_df["dst"].isin(component_nodes)
    ].copy()

    return filtered, True


def radial_layout_from_target(graph, target_node):
    """
    Clean analyst layout:
    target node in the centre,
    1-hop nodes around it,
    2-hop nodes in an outer ring.
    """

    target_node = int(target_node)
    undirected = graph.to_undirected()

    if target_node not in undirected.nodes:
        return nx.spring_layout(graph, seed=42, iterations=200)

    distances = nx.single_source_shortest_path_length(undirected, target_node)

    rings = {}

    for node in graph.nodes:
        dist = distances.get(node, 99)
        rings.setdefault(dist, []).append(node)

    pos = {target_node: np.array([0.0, 0.0])}

    for dist, nodes in rings.items():
        if dist == 0:
            continue

        nodes = sorted(nodes)
        radius = 1.4 + (dist - 1) * 1.25

        if dist >= 99:
            radius = 3.5

        count = len(nodes)

        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / max(count, 1)

            # Small rotation so diagrams do not always align exactly the same.
            angle += math.pi / 8

            pos[node] = np.array(
                [
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                ]
            )

    return pos


def normalise_widths(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return []

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if math.isclose(min_score, max_score):
        return [4.0 for _ in scores]

    return (2.0 + 6.0 * ((scores - min_score) / (max_score - min_score))).tolist()


def draw_graph(edge_df, metadata, alert_info, output_png):
    dataset = metadata["dataset"]
    model = metadata["model"]
    explainer = metadata["explainer"]
    target_node = int(metadata["target_node"])

    graph = nx.DiGraph()

    for _, row in edge_df.iterrows():
        graph.add_edge(
            int(row["src"]),
            int(row["dst"]),
            importance=float(row["max_importance"]),
            repeated_edges=int(row["repeated_edges"]),
            rank=int(row["display_rank"]),
        )

    if target_node not in graph.nodes:
        graph.add_node(target_node)

    pos = radial_layout_from_target(graph, target_node)

    fig, ax = plt.subplots(figsize=(11, 8))

    title = (
        f"{dataset.upper()} analyst explanation view\n"
        f"{MODEL_DISPLAY.get(model, model)} + {EXPLAINER_DISPLAY.get(explainer, explainer)} "
        f"| Target node: {target_node}"
    )

    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)

    node_colors = []
    node_sizes = []
    edge_colors = []

    for node in graph.nodes:
        if int(node) == target_node:
            node_colors.append("#d62728")
            edge_colors.append("#7f0000")
            node_sizes.append(1800)
        else:
            node_colors.append("#d8ecff")
            edge_colors.append("#4c78a8")
            node_sizes.append(800)

    edges = list(graph.edges())
    scores = [graph.edges[e]["importance"] for e in edges]
    widths = normalise_widths(scores)

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edges,
        width=widths,
        edge_color="#333333",
        arrows=True,
        arrowsize=22,
        alpha=0.80,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=edge_colors,
        node_size=node_sizes,
        linewidths=2.0,
        ax=ax,
    )

    labels = {}
    for node in graph.nodes:
        if int(node) == target_node:
            labels[node] = f"TARGET\n{node}"
        else:
            labels[node] = str(node)

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
        ax=ax,
    )

    edge_labels = {}

    for src, dst in edges:
        data = graph.edges[(src, dst)]
        rank = data["rank"]
        repeated = data["repeated_edges"]

        if repeated > 1:
            edge_labels[(src, dst)] = f"#{rank}\n×{repeated}"
        else:
            edge_labels[(src, dst)] = f"#{rank}"

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.55,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70),
        ax=ax,
    )

    prob = alert_info.get("prob", np.nan)
    prob_text = "Unknown" if pd.isna(prob) else f"{float(prob):.4f}"

    risk_band = alert_info.get("risk_band", "Unknown")
    true_raw = alert_info.get("true_raw", "Unknown")
    pred_raw = alert_info.get("pred_raw", "Unknown")
    pred_suspicious = alert_info.get("pred_suspicious", "Unknown")
    action = alert_info.get("action", "Review manually")
    suspicious_class = alert_info.get("suspicious_class", "Suspicious")

    total_raw_edges = int(metadata["raw_edge_rows"])
    unique_edges = len(edge_df)
    removed_note = metadata.get("removed_note", "")

    info_text = (
        f"Analyst case summary\n"
        f"Suspicious class: {suspicious_class}\n"
        f"Model suspicion score: {prob_text}\n"
        f"Risk band: {risk_band}\n"
        f"True label: {true_raw} | Predicted label: {pred_raw}\n"
        f"Predicted suspicious?: {pred_suspicious}\n"
        f"Recommended action: {action}\n\n"
        f"Explanation summary\n"
        f"Raw explanation edge rows: {total_raw_edges}\n"
        f"Unique visual links after collapsing duplicates: {unique_edges}\n"
        f"{removed_note}\n\n"
        f"How to read\n"
        f"Red node = flagged account/transaction.\n"
        f"Arrow thickness = explanation importance.\n"
        f"×N means repeated transaction links collapsed into one visual edge."
    )

    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.95,
        ),
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close()


def parse_metadata_from_df(edge_df, fallback_path):
    row = edge_df.iloc[0]

    dataset = clean_string(row.get("dataset", "unknown"))
    model = clean_string(row.get("model", "unknown"))
    explainer = clean_string(row.get("explainer", "unknown"))
    target_node = int(row.get("target_node"))

    return {
        "dataset": dataset,
        "model": model,
        "explainer": explainer,
        "target_node": target_node,
        "source_file": fallback_path.name,
    }


def process_edge_file(path, alert_df):
    print("=" * 100)
    print(f"Processing: {path.relative_to(PROJECT_ROOT)}")

    edge_df = pd.read_csv(path)

    if edge_df.empty:
        print("Skipping empty edge file.")
        return None

    metadata = parse_metadata_from_df(edge_df, path)
    metadata["raw_edge_rows"] = len(edge_df)

    aggregated_df = aggregate_edges(edge_df)

    before_component = len(aggregated_df)

    component_df, target_found = keep_target_component(
        aggregated_df,
        metadata["target_node"],
    )

    if target_found and not component_df.empty:
        aggregated_df = component_df.copy()
        removed = before_component - len(aggregated_df)
        metadata["removed_note"] = (
            f"Disconnected non-target links removed: {removed}"
            if removed > 0
            else "No disconnected explanation links removed."
        )
    else:
        metadata["removed_note"] = (
            "Target node was not connected to the selected explanation links."
        )

    if aggregated_df.empty:
        print("No edges remained after filtering.")
        return None

    alert_info = get_alert_info(
        alert_df=alert_df,
        dataset=metadata["dataset"],
        model=metadata["model"],
        target_node=metadata["target_node"],
    )

    safe_name = (
        f"{metadata['dataset']}_{metadata['model']}_{metadata['explainer']}"
        f"_node_{metadata['target_node']}_presentation"
    )

    output_png = OUTPUT_DIR / f"{safe_name}.png"
    output_csv = OUTPUT_DIR / f"{safe_name}_aggregated_edges.csv"

    draw_graph(
        edge_df=aggregated_df,
        metadata=metadata,
        alert_info=alert_info,
        output_png=output_png,
    )

    aggregated_df.to_csv(output_csv, index=False)

    print(f"Saved PNG: {output_png}")
    print(f"Saved CSV: {output_csv}")

    return output_png, output_csv


def main():
    print("=" * 100)
    print("Creating professor-ready analyst explanation graphs")
    print("=" * 100)

    alert_df = load_alert_table()

    edge_files = sorted(INPUT_DIR.glob("*_edges.csv"))

    if not edge_files:
        raise RuntimeError(
            f"No edge CSV files found in {INPUT_DIR}. "
            f"Run plot_local_explanation_graph.py first."
        )

    created = []

    for path in edge_files:
        try:
            result = process_edge_file(path, alert_df)
            if result is not None:
                created.append(result)
        except Exception as e:
            print(f"WARNING: Could not process {path.name}")
            print(f"Reason: {e}")

    print("\n" + "=" * 100)
    print("Professor-ready graphs created")
    print("=" * 100)

    for png, csv in created:
        print(png)
        print(csv)

    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()