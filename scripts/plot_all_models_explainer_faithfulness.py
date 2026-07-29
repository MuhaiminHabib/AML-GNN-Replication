from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/explainers")
PLOT_DIR = OUTPUT_DIR / "plots_all_models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "all_models_explainer_faithfulness_summary.csv"
DETAIL_FILE = OUTPUT_DIR / "all_models_explainer_faithfulness.csv"

SUMMARY_MD_FILE = OUTPUT_DIR / "all_models_explainer_faithfulness_summary.md"

DELETION_DROP_PLOT = PLOT_DIR / "all_models_deletion_drop.png"
DELETION_FLIP_PLOT = PLOT_DIR / "all_models_deletion_flip_rate.png"
INSERTION_PRESERVATION_PLOT = PLOT_DIR / "all_models_insertion_preservation.png"
EXPLANATION_SIZE_PLOT = PLOT_DIR / "all_models_explanation_size.png"
COMBINED_PLOT = PLOT_DIR / "all_models_explainer_faithfulness_combined.png"


MODEL_ORDER = ["gcn", "graphsage", "gatv2"]
MODEL_LABELS = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
}

EXPLAINER_ORDER = ["GNNExplainer", "PGExplainer", "DGL_SubgraphX"]
EXPLAINER_LABELS = {
    "GNNExplainer": "GNNExplainer",
    "PGExplainer": "PGExplainer",
    "DGL_SubgraphX": "SubgraphX",
}


def load_data():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_FILE}")

    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"Missing detail file: {DETAIL_FILE}")

    summary_df = pd.read_csv(SUMMARY_FILE)
    detail_df = pd.read_csv(DETAIL_FILE)

    return summary_df, detail_df


def prepare_summary(summary_df):
    df = summary_df.copy()

    df["model_display"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["explainer_display"] = df["explainer"].map(EXPLAINER_LABELS).fillna(df["explainer"])

    df["model_display"] = pd.Categorical(
        df["model_display"],
        categories=[MODEL_LABELS[m] for m in MODEL_ORDER],
        ordered=True,
    )

    df["explainer_display"] = pd.Categorical(
        df["explainer_display"],
        categories=[EXPLAINER_LABELS[e] for e in EXPLAINER_ORDER],
        ordered=True,
    )

    return df.sort_values(["model_display", "explainer_display"])


def save_markdown_summary(summary_df):
    table_df = summary_df[
        [
            "model_display",
            "explainer_display",
            "nodes_evaluated",
            "deletion_drop_mean",
            "deletion_label_flip_rate",
            "insertion_prob_mean",
            "insertion_preservation_rate",
            "sparsity_edges_mean",
            "num_explanation_edges_mean",
            "num_explanation_nodes_mean",
        ]
    ].copy()

    table_df = table_df.rename(
        columns={
            "model_display": "Model",
            "explainer_display": "Explainer",
            "nodes_evaluated": "Nodes",
            "deletion_drop_mean": "Deletion drop mean",
            "deletion_label_flip_rate": "Deletion flip rate",
            "insertion_prob_mean": "Insertion prob mean",
            "insertion_preservation_rate": "Insertion preservation",
            "sparsity_edges_mean": "Edge sparsity",
            "num_explanation_edges_mean": "Avg explanation edges",
            "num_explanation_nodes_mean": "Avg explanation nodes",
        }
    )

    numeric_cols = [
        "Deletion drop mean",
        "Deletion flip rate",
        "Insertion prob mean",
        "Insertion preservation",
        "Edge sparsity",
        "Avg explanation edges",
        "Avg explanation nodes",
    ]

    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")

    markdown = table_df.to_markdown(index=False)

    with open(SUMMARY_MD_FILE, "w", encoding="utf-8") as f:
        f.write("# All Models Explainer Faithfulness Summary\n\n")
        f.write(markdown)
        f.write("\n")

    print(f"Saved markdown summary to: {SUMMARY_MD_FILE}")


def plot_grouped_bar(summary_df, metric_col, ylabel, title, output_path):
    models = [MODEL_LABELS[m] for m in MODEL_ORDER]
    explainers = [EXPLAINER_LABELS[e] for e in EXPLAINER_ORDER]

    x = list(range(len(models)))
    width = 0.22

    plt.figure(figsize=(10, 6))

    for i, explainer in enumerate(explainers):
        values = []

        for model in models:
            row = summary_df[
                (summary_df["model_display"].astype(str) == model)
                & (summary_df["explainer_display"].astype(str) == explainer)
            ]

            if len(row) == 0:
                values.append(0.0)
            else:
                values.append(float(row.iloc[0][metric_col]))

        positions = [v + (i - 1) * width for v in x]

        plt.bar(
            positions,
            values,
            width=width,
            label=explainer,
        )

    plt.xticks(ticks=x, labels=models)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


def plot_combined(summary_df):
    metrics = [
        ("deletion_drop_mean", "Deletion drop"),
        ("deletion_label_flip_rate", "Deletion flip"),
        ("insertion_preservation_rate", "Insertion preservation"),
        ("num_explanation_edges_mean", "Avg edges"),
    ]

    labels = []

    for _, row in summary_df.iterrows():
        labels.append(
            f"{str(row['model_display'])}\n{str(row['explainer_display'])}"
        )

    x = list(range(len(labels)))
    width = 0.18

    plt.figure(figsize=(14, 7))

    for i, (metric_col, metric_label) in enumerate(metrics):
        positions = [v + (i - 1.5) * width for v in x]
        values = summary_df[metric_col].tolist()

        plt.bar(
            positions,
            values,
            width=width,
            label=metric_label,
        )

    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.xlabel("Model and explainer")
    plt.ylabel("Metric value")
    plt.title("All Models Explainer Faithfulness Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMBINED_PLOT, dpi=300)
    plt.close()

    print(f"Saved combined plot to: {COMBINED_PLOT}")


def main():
    summary_df, detail_df = load_data()
    summary_df = prepare_summary(summary_df)

    print("\nAll-model explainer faithfulness summary")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    save_markdown_summary(summary_df)

    plot_grouped_bar(
        summary_df=summary_df,
        metric_col="deletion_drop_mean",
        ylabel="Mean deletion drop",
        title="Deletion Drop by Model and Explainer",
        output_path=DELETION_DROP_PLOT,
    )

    plot_grouped_bar(
        summary_df=summary_df,
        metric_col="deletion_label_flip_rate",
        ylabel="Deletion label flip rate",
        title="Deletion Label Flip Rate by Model and Explainer",
        output_path=DELETION_FLIP_PLOT,
    )

    plot_grouped_bar(
        summary_df=summary_df,
        metric_col="insertion_preservation_rate",
        ylabel="Insertion preservation rate",
        title="Insertion Preservation by Model and Explainer",
        output_path=INSERTION_PRESERVATION_PLOT,
    )

    plot_grouped_bar(
        summary_df=summary_df,
        metric_col="num_explanation_edges_mean",
        ylabel="Average explanation edges",
        title="Explanation Size by Model and Explainer",
        output_path=EXPLANATION_SIZE_PLOT,
    )

    plot_combined(summary_df)

    print("\nDone.")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()