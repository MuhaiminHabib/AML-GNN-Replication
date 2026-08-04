from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "amlsim_all_models"
    / "all_amlsim_models_explainer_faithfulness_summary.csv"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "amlsim_all_models"
    / "plots"
)

COMBINED_PLOT_PATH = PLOT_DIR / "all_amlsim_models_explainer_faithfulness_combined.png"
DELETION_DROP_PLOT_PATH = PLOT_DIR / "all_amlsim_models_deletion_drop.png"
DELETION_FLIP_PLOT_PATH = PLOT_DIR / "all_amlsim_models_deletion_flip_rate.png"
INSERTION_PRESERVATION_PLOT_PATH = PLOT_DIR / "all_amlsim_models_insertion_preservation.png"
EXPLANATION_SIZE_PLOT_PATH = PLOT_DIR / "all_amlsim_models_explanation_size.png"


MODEL_ORDER = ["gcn", "graphsage", "gatv2"]

EXPLAINER_ORDER = [
    "GNNExplainer",
    "PGExplainer",
    "DGL_SubgraphX_1hop",
]

EXPLAINER_DISPLAY = {
    "GNNExplainer": "GNNExplainer",
    "PGExplainer": "PGExplainer",
    "DGL_SubgraphX_1hop": "SubgraphX (1-hop)",
}

MODEL_DISPLAY = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
}


def load_summary():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_PATH}\n\n"
            f"Run this first:\n"
            f"python scripts\\evaluate_all_amlsim_models_explainer_faithfulness.py"
        )

    df = pd.read_csv(INPUT_PATH)

    required_cols = {
        "model",
        "explainer",
        "top_k",
        "mean_deletion_drop",
        "deletion_flip_rate",
        "mean_insertion_fraud_prob",
        "insertion_preservation_rate",
        "mean_selected_edges",
        "mean_sparsity",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Summary file is missing columns: {sorted(missing)}")

    df = df.copy()

    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    df["explainer_display"] = df["explainer"].map(EXPLAINER_DISPLAY).fillna(df["explainer"])

    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["explainer"] = pd.Categorical(df["explainer"], categories=EXPLAINER_ORDER, ordered=True)

    df = df.sort_values(["model", "explainer", "top_k"]).reset_index(drop=True)

    return df


def make_label(row):
    return f"{row['model_display']}\n{row['explainer_display']}\nK={int(row['top_k'])}"


def plot_metric_bar(df, metric, ylabel, title, output_path):
    plot_df = df.copy()
    plot_df["label"] = plot_df.apply(make_label, axis=1)

    plt.figure(figsize=(18, 7))
    plt.bar(plot_df["label"], plot_df[metric])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_combined(df):
    top20_df = df[df["top_k"] == 20].copy()

    if top20_df.empty:
        raise RuntimeError("No top_k=20 rows found in summary file.")

    top20_df["label"] = top20_df.apply(
        lambda row: f"{row['model_display']}\n{row['explainer_display']}",
        axis=1,
    )

    metrics = [
        ("mean_deletion_drop", "Mean deletion drop"),
        ("deletion_flip_rate", "Deletion flip rate"),
        ("mean_insertion_fraud_prob", "Mean insertion fraud probability"),
        ("insertion_preservation_rate", "Insertion preservation rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(top20_df["label"], top20_df[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel(metric)

    fig.suptitle("AMLSim all-model explainer faithfulness summary at Top-K = 20", fontsize=16)
    fig.tight_layout()
    fig.savefig(COMBINED_PLOT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved: {COMBINED_PLOT_PATH}")


def save_markdown_summary(df):
    top20_df = df[df["top_k"] == 20].copy()

    out_path = (
        PROJECT_ROOT
        / "outputs"
        / "explainers"
        / "amlsim_all_models"
        / "all_amlsim_models_explainer_faithfulness_summary.md"
    )

    cols = [
        "model_display",
        "explainer_display",
        "top_k",
        "explained_nodes",
        "mean_deletion_drop",
        "deletion_flip_rate",
        "mean_insertion_fraud_prob",
        "insertion_preservation_rate",
        "mean_sparsity",
    ]

    existing_cols = [col for col in cols if col in top20_df.columns]

    md_df = top20_df[existing_cols].copy()

    rename_map = {
        "model_display": "Model",
        "explainer_display": "Explainer",
        "top_k": "Top-K",
        "explained_nodes": "Nodes",
        "mean_deletion_drop": "Deletion drop",
        "deletion_flip_rate": "Flip rate",
        "mean_insertion_fraud_prob": "Insertion fraud prob.",
        "insertion_preservation_rate": "Insertion preservation",
        "mean_sparsity": "Sparsity",
    }

    md_df = md_df.rename(columns=rename_map)

    markdown = "# AMLSim all-model explainer faithfulness summary\n\n"
    markdown += "Top-K = 20 results.\n\n"
    markdown += md_df.to_markdown(index=False, floatfmt=".6f")
    markdown += "\n"

    out_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {out_path}")


def main():
    print("=" * 100)
    print("Plotting AMLSim all-model explainer faithfulness results")
    print("=" * 100)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_summary()

    print("\nLoaded summary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    plot_metric_bar(
        df=df,
        metric="mean_deletion_drop",
        ylabel="Mean deletion drop",
        title="AMLSim explanation necessity: deletion drop",
        output_path=DELETION_DROP_PLOT_PATH,
    )

    plot_metric_bar(
        df=df,
        metric="deletion_flip_rate",
        ylabel="Deletion flip rate",
        title="AMLSim explanation necessity: deletion flip rate",
        output_path=DELETION_FLIP_PLOT_PATH,
    )

    plot_metric_bar(
        df=df,
        metric="insertion_preservation_rate",
        ylabel="Insertion preservation rate",
        title="AMLSim explanation sufficiency: insertion preservation",
        output_path=INSERTION_PRESERVATION_PLOT_PATH,
    )

    plot_metric_bar(
        df=df,
        metric="mean_selected_edges",
        ylabel="Mean selected edges",
        title="AMLSim explanation size",
        output_path=EXPLANATION_SIZE_PLOT_PATH,
    )

    plot_combined(df)

    save_markdown_summary(df)

    print("\nDone.")


if __name__ == "__main__":
    main()