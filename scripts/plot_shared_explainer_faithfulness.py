from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/explainers")
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "shared_explainer_faithfulness_summary.csv"
DETAIL_FILE = OUTPUT_DIR / "shared_explainer_faithfulness.csv"

SUMMARY_PLOT_FILE = PLOT_DIR / "shared_explainer_faithfulness_summary.png"
DELETION_PLOT_FILE = PLOT_DIR / "shared_explainer_deletion_drop.png"
INSERTION_PLOT_FILE = PLOT_DIR / "shared_explainer_insertion_preservation.png"
FLIP_PLOT_FILE = PLOT_DIR / "shared_explainer_deletion_flip_rate.png"
EDGE_COUNT_PLOT_FILE = PLOT_DIR / "shared_explainer_edge_count.png"

SUMMARY_TABLE_MD = OUTPUT_DIR / "shared_explainer_faithfulness_summary.md"


def load_data():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_FILE}")

    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"Missing detail file: {DETAIL_FILE}")

    summary_df = pd.read_csv(SUMMARY_FILE)
    detail_df = pd.read_csv(DETAIL_FILE)

    return summary_df, detail_df


def normalise_explainer_names(df):
    name_map = {
        "GNNExplainer": "GNNExplainer",
        "PGExplainer": "PGExplainer",
        "DGL_SubgraphX": "SubgraphX",
        "DGL_SubgraphX_large": "SubgraphX-large",
    }

    df = df.copy()
    df["explainer_display"] = df["explainer"].map(name_map).fillna(df["explainer"])
    df["explainer_display"] = df["explainer_display"].astype(str)

    order = [
        "GNNExplainer",
        "PGExplainer",
        "SubgraphX",
        "SubgraphX-large",
    ]

    df["explainer_display"] = pd.Categorical(
        df["explainer_display"],
        categories=order,
        ordered=True,
    )

    return df.sort_values("explainer_display")


def save_markdown_summary(summary_df):
    cols = [
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

    table_df = summary_df[cols].copy()

    table_df = table_df.rename(
        columns={
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

    with open(SUMMARY_TABLE_MD, "w", encoding="utf-8") as f:
        f.write("# Shared Explainer Faithfulness Summary\n\n")
        f.write(markdown)
        f.write("\n")

    print(f"Saved markdown summary to: {SUMMARY_TABLE_MD}")


def plot_bar(summary_df, column, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary_df["explainer_display"].astype(str),
        summary_df[column],
    )

    plt.xlabel("Explainer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


def plot_combined_summary(summary_df):
    metrics = [
        ("deletion_drop_mean", "Deletion drop"),
        ("deletion_label_flip_rate", "Deletion flip rate"),
        ("insertion_preservation_rate", "Insertion preservation"),
        ("num_explanation_edges_mean", "Avg edges"),
    ]

    x = range(len(summary_df))
    width = 0.18

    plt.figure(figsize=(10, 6))

    for i, (column, label) in enumerate(metrics):
        positions = [v + (i - 1.5) * width for v in x]
        values = summary_df[column].tolist()

        plt.bar(
            positions,
            values,
            width=width,
            label=label,
        )

    plt.xticks(
        ticks=list(x),
        labels=summary_df["explainer_display"].astype(str).tolist(),
    )

    plt.xlabel("Explainer")
    plt.ylabel("Metric value")
    plt.title("Shared Explainer Faithfulness Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_FILE, dpi=300)
    plt.close()

    print(f"Saved combined summary plot to: {SUMMARY_PLOT_FILE}")


def main():
    summary_df, detail_df = load_data()
    summary_df = normalise_explainer_names(summary_df)

    print("\nShared explainer faithfulness summary")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    save_markdown_summary(summary_df)

    plot_bar(
        summary_df=summary_df,
        column="deletion_drop_mean",
        ylabel="Mean deletion drop",
        title="Deletion Faithfulness by Explainer",
        output_path=DELETION_PLOT_FILE,
    )

    plot_bar(
        summary_df=summary_df,
        column="insertion_preservation_rate",
        ylabel="Insertion preservation rate",
        title="Insertion Preservation by Explainer",
        output_path=INSERTION_PLOT_FILE,
    )

    plot_bar(
        summary_df=summary_df,
        column="deletion_label_flip_rate",
        ylabel="Deletion label flip rate",
        title="Deletion Label Flip Rate by Explainer",
        output_path=FLIP_PLOT_FILE,
    )

    plot_bar(
        summary_df=summary_df,
        column="num_explanation_edges_mean",
        ylabel="Average explanation edges",
        title="Explanation Size by Explainer",
        output_path=EDGE_COUNT_PLOT_FILE,
    )

    plot_combined_summary(summary_df)

    print("\nDone.")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()