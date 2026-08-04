from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EDGE_INPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "graphs"
ALERT_TABLE_PATH = PROJECT_ROOT / "outputs" / "analyst_view" / "analyst_alert_queue_all.csv"
FINAL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "final_comparison"
    / "elliptic_vs_amlsim_explainer_comparison.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "final_case_cards"
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


SELECTED_CASES = [
    {
        "dataset": "amlsim",
        "model": "gatv2",
        "explainer": "gnnexplainer",
        "target_node": 1751,
        "case_type": "main_amlsim_demo",
        "slide_message": (
            "Best AMLSim analyst example: the model gives a critical fraud score, "
            "and GNNExplainer highlights a compact set of transaction links connected to the target account."
        ),
    },
    {
        "dataset": "elliptic",
        "model": "gcn",
        "explainer": "gnnexplainer",
        "target_node": 18018,
        "case_type": "main_elliptic_demo",
        "slide_message": (
            "Strong Elliptic example: the model flags an illicit transaction, "
            "and the explanation forms a clear connected local structure around the target node."
        ),
    },
    {
        "dataset": "amlsim",
        "model": "graphsage",
        "explainer": "gnnexplainer",
        "target_node": 4842,
        "case_type": "limitation_case",
        "slide_message": (
            "Limitation case: GraphSAGE performs strongly as a classifier, "
            "but many explanation edges were disconnected from the target and had to be removed."
        ),
    },
]


def load_csv_if_exists(path: Path):
    if not path.exists():
        print(f"WARNING: Missing file: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def normalise_text(value):
    return str(value).strip().lower()


def display_dataset(dataset):
    dataset = normalise_text(dataset)

    if dataset == "amlsim":
        return "AMLSim"

    if dataset == "elliptic":
        return "Elliptic"

    return str(dataset)


def display_model(model):
    return MODEL_DISPLAY.get(normalise_text(model), str(model))


def display_explainer(explainer):
    return EXPLAINER_DISPLAY.get(normalise_text(explainer), str(explainer))


def find_edge_file(dataset, model, explainer, target_node):
    pattern = f"{dataset}_{model}_{explainer}_node_{target_node}_top20_edges.csv"
    path = EDGE_INPUT_DIR / pattern

    if path.exists():
        return path

    candidates = list(
        EDGE_INPUT_DIR.glob(
            f"{dataset}_{model}_{explainer}_node_{target_node}_*_edges.csv"
        )
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find edge file for {dataset}/{model}/{explainer}/node {target_node}"
    )


def aggregate_edges(edge_df):
    df = edge_df.copy()

    df["src"] = pd.to_numeric(df["src"], errors="coerce")
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df["importance_score"] = pd.to_numeric(
        df["importance_score"],
        errors="coerce",
    ).fillna(0.0)
    df["edge_rank"] = pd.to_numeric(df["edge_rank"], errors="coerce")

    df = df.dropna(subset=["src", "dst"])
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

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
        ["max_importance", "repeated_edges", "min_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    grouped["display_rank"] = np.arange(1, len(grouped) + 1)

    return grouped


def keep_target_component(edge_df, target_node):
    graph = nx.Graph()

    for _, row in edge_df.iterrows():
        graph.add_edge(int(row["src"]), int(row["dst"]))

    target_node = int(target_node)

    if target_node not in graph.nodes:
        return edge_df.copy(), False, 0

    component_nodes = nx.node_connected_component(graph, target_node)

    filtered = edge_df[
        edge_df["src"].isin(component_nodes)
        & edge_df["dst"].isin(component_nodes)
    ].copy()

    removed_count = len(edge_df) - len(filtered)

    return filtered, True, removed_count


def radial_layout(graph, target_node):
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
        radius = 1.45 + (dist - 1) * 1.25
        count = len(nodes)

        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / max(count, 1)
            angle += math.pi / 7
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

    min_score = float(scores.min())
    max_score = float(scores.max())

    if math.isclose(min_score, max_score):
        return [4.5 for _ in scores]

    widths = 2.0 + 7.0 * ((scores - min_score) / (max_score - min_score))

    return widths.tolist()


def get_alert_info(alert_df, dataset, model, target_node):
    if alert_df.empty:
        return {}

    dataset_display = display_dataset(dataset)
    model_display = display_model(model)

    df = alert_df.copy()
    df["node_numeric"] = pd.to_numeric(df["Node ID"], errors="coerce")

    matched = df[
        (df["Dataset"].astype(str).str.lower() == dataset_display.lower())
        & (df["Model"].astype(str).str.lower() == model_display.lower())
        & (df["node_numeric"] == int(target_node))
    ].copy()

    if matched.empty:
        return {}

    return matched.iloc[0].to_dict()


def get_faithfulness_info(final_df, dataset, model, explainer):
    if final_df.empty:
        return {}

    dataset_display = display_dataset(dataset)
    model_display = display_model(model)
    explainer_display = display_explainer(explainer)

    df = final_df.copy()

    matched = df[
        (df["Dataset"].astype(str).str.lower() == dataset_display.lower())
        & (df["Model"].astype(str).str.lower() == model_display.lower())
        & (df["Explainer"].astype(str).str.lower() == explainer_display.lower())
    ].copy()

    if matched.empty:
        return {}

    return matched.iloc[0].to_dict()


def safe_float_text(value, digits=4, default="Unknown"):
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def bool_to_text(value):
    if isinstance(value, bool):
        return "True" if value else "False"

    text = str(value)

    if text.lower() in ["true", "1", "1.0"]:
        return "True"

    if text.lower() in ["false", "0", "0.0"]:
        return "False"

    return text


def draw_case_card(case, edge_df, alert_info, faithfulness_info, output_png):
    dataset = case["dataset"]
    model = case["model"]
    explainer = case["explainer"]
    target_node = int(case["target_node"])

    dataset_display = display_dataset(dataset)
    model_display = display_model(model)
    explainer_display = display_explainer(explainer)

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

    pos = radial_layout(graph, target_node)

    fig = plt.figure(figsize=(17, 9.2))

    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.35, 1.25],
        wspace=0.08,
    )

    ax_graph = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        f"{dataset_display} analyst case card: {model_display} + {explainer_display}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    # ------------------------------------------------------------------
    # Left panel: local explanation graph
    # ------------------------------------------------------------------
    edges = list(graph.edges())
    scores = [graph.edges[e]["importance"] for e in edges]
    widths = normalise_widths(scores)

    node_colors = []
    node_edges = []
    node_sizes = []

    for node in graph.nodes:
        if int(node) == target_node:
            node_colors.append("#d62728")
            node_edges.append("#7f0000")
            node_sizes.append(1900)
        else:
            node_colors.append("#d8ecff")
            node_edges.append("#4c78a8")
            node_sizes.append(850)

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edges,
        width=widths,
        edge_color="#333333",
        arrows=True,
        arrowsize=22,
        alpha=0.78,
        connectionstyle="arc3,rad=0.08",
        ax=ax_graph,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=node_edges,
        node_size=node_sizes,
        linewidths=2.0,
        ax=ax_graph,
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
        ax=ax_graph,
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
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        ax=ax_graph,
    )

    ax_graph.set_title(
        f"Local explanation graph for target node {target_node}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax_graph.axis("off")

    # ------------------------------------------------------------------
    # Right panel: analyst explanation summary
    # ------------------------------------------------------------------
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    suspicion_score = safe_float_text(
        alert_info.get("Suspicion probability", np.nan),
        digits=4,
    )

    risk_band = alert_info.get("Risk band", "Unknown")
    action = alert_info.get("Recommended analyst action", "Review manually")
    suspicious_class = alert_info.get("Suspicious class name", "Suspicious")
    true_label = alert_info.get("True label raw", "Unknown")
    pred_label = alert_info.get("Predicted label raw", "Unknown")
    pred_suspicious = bool_to_text(alert_info.get("Predicted suspicious?", "Unknown"))

    deletion_drop = safe_float_text(
        faithfulness_info.get("Deletion drop", np.nan),
        digits=4,
    )

    flip_rate = safe_float_text(
        faithfulness_info.get("Flip rate", np.nan),
        digits=4,
    )

    insertion_preservation = safe_float_text(
        faithfulness_info.get("Insertion preservation", np.nan),
        digits=4,
    )

    sparsity = safe_float_text(
        faithfulness_info.get("Sparsity", np.nan),
        digits=4,
    )

    interpretation = faithfulness_info.get(
        "Interpretation",
        "Faithfulness result not available.",
    )

    raw_edges = int(case.get("raw_edge_rows", len(edge_df)))
    unique_edges = len(edge_df)
    disconnected_removed = int(case.get("disconnected_removed", 0))

    y = 0.96

    def section_header(text, y_pos):
        ax_info.text(
            0.02,
            y_pos,
            text,
            fontsize=13.5,
            fontweight="bold",
            transform=ax_info.transAxes,
            verticalalignment="top",
        )
        return y_pos - 0.060

    def line(label, value, y_pos):
        ax_info.text(
            0.04,
            y_pos,
            f"{label}:",
            fontsize=10.2,
            fontweight="bold",
            transform=ax_info.transAxes,
            verticalalignment="top",
        )

        ax_info.text(
            0.62,
            y_pos,
            str(value),
            fontsize=10.2,
            transform=ax_info.transAxes,
            verticalalignment="top",
        )

        return y_pos - 0.041

    def text_box(text, y_pos, height=0.13):
        ax_info.text(
            0.04,
            y_pos,
            text,
            fontsize=10.1,
            transform=ax_info.transAxes,
            verticalalignment="top",
            wrap=True,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#f7f7f7",
                edgecolor="#d0d0d0",
                alpha=1.0,
            ),
        )
        return y_pos - height

    y = section_header("Analyst case summary", y)
    y = line("Suspicious class", suspicious_class, y)
    y = line("Target node", target_node, y)
    y = line("Model score", suspicion_score, y)
    y = line("Risk band", risk_band, y)
    y = line("True label", true_label, y)
    y = line("Predicted label", pred_label, y)
    y = line("Pred. suspicious?", pred_suspicious, y)
    y = line("Action", action, y)

    y -= 0.030
    y = section_header("Explanation graph summary", y)
    y = line("Raw edge rows", raw_edges, y)
    y = line("Unique visual links", unique_edges, y)
    y = line("Disconnected removed", disconnected_removed, y)

    y = text_box(
        "Red node = flagged account/transaction. "
        "Thicker arrows = more important explanation links. "
        "×N means repeated transaction links were collapsed into one visual edge.",
        y,
        height=0.150,
    )

    y -= 0.020
    y = section_header("Faithfulness summary", y)
    y = line("Deletion drop", deletion_drop, y)
    y = line("Flip rate", flip_rate, y)
    y = line("Insertion preserve.", insertion_preservation, y)
    y = line("Sparsity", sparsity, y)

    y = text_box(
        str(interpretation),
        y,
        height=0.130,
    )

    y -= 0.020
    y = section_header("Researcher-as-analyst message", y)

    y = text_box(
        case["slide_message"],
        y,
        height=0.160,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def create_case_card(case, alert_df, final_df):
    edge_file = find_edge_file(
        dataset=case["dataset"],
        model=case["model"],
        explainer=case["explainer"],
        target_node=case["target_node"],
    )

    print("=" * 100)
    print(f"Creating final case card from: {edge_file.relative_to(PROJECT_ROOT)}")

    raw_edge_df = pd.read_csv(edge_file)

    aggregated = aggregate_edges(raw_edge_df)

    component_df, target_found, disconnected_removed = keep_target_component(
        aggregated,
        case["target_node"],
    )

    if target_found and not component_df.empty:
        final_edges = component_df.copy()
    else:
        final_edges = aggregated.copy()

    case["raw_edge_rows"] = len(raw_edge_df)
    case["disconnected_removed"] = disconnected_removed

    alert_info = get_alert_info(
        alert_df=alert_df,
        dataset=case["dataset"],
        model=case["model"],
        target_node=case["target_node"],
    )

    faithfulness_info = get_faithfulness_info(
        final_df=final_df,
        dataset=case["dataset"],
        model=case["model"],
        explainer=case["explainer"],
    )

    output_name = (
        f"{case['dataset']}_{case['model']}_{case['explainer']}"
        f"_node_{case['target_node']}_{case['case_type']}.png"
    )

    output_png = OUTPUT_DIR / output_name

    draw_case_card(
        case=case,
        edge_df=final_edges,
        alert_info=alert_info,
        faithfulness_info=faithfulness_info,
        output_png=output_png,
    )

    output_edges = OUTPUT_DIR / output_name.replace(".png", "_edges_used.csv")
    final_edges.to_csv(output_edges, index=False)

    print(f"Saved case card: {output_png}")
    print(f"Saved used edges: {output_edges}")

    return output_png, output_edges


def main():
    print("=" * 100)
    print("Creating final analyst case cards")
    print("=" * 100)

    alert_df = load_csv_if_exists(ALERT_TABLE_PATH)
    final_df = load_csv_if_exists(FINAL_COMPARISON_PATH)

    created = []

    for case in SELECTED_CASES:
        try:
            result = create_case_card(
                case=case.copy(),
                alert_df=alert_df,
                final_df=final_df,
            )
            created.append(result)
        except Exception as e:
            print(f"WARNING: Could not create case card for {case}")
            print(f"Reason: {e}")

    print("\n" + "=" * 100)
    print("Final analyst case cards created")
    print("=" * 100)

    for png, csv in created:
        print(png)
        print(csv)

    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()