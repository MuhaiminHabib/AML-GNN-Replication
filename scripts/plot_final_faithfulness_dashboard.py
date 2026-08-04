from pathlib import Path
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "final_comparison"
    / "elliptic_vs_amlsim_explainer_comparison.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "faithfulness_dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "final_faithfulness_dashboard.png"
OUTPUT_CSV = OUTPUT_DIR / "final_faithfulness_dashboard_source.csv"
OUTPUT_MD = OUTPUT_DIR / "final_faithfulness_dashboard_summary.md"


METRIC_CONFIG = [
    {
        "column": "Deletion drop",
        "title": "Necessity: deletion drop",
        "subtitle": "Higher means the highlighted edges were important.",
        "x_label": "Mean probability drop after removing explanation edges",
    },
    {
        "column": "Flip rate",
        "title": "Necessity: deletion flip rate",
        "subtitle": "Higher means removing explanation edges changed the prediction.",
        "x_label": "Prediction flip rate",
    },
    {
        "column": "Insertion preservation",
        "title": "Sufficiency: insertion preservation",
        "subtitle": "Higher means the explanation alone preserves the prediction.",
        "x_label": "Prediction preservation rate",
    },
    {
        "column": "Sparsity",
        "title": "Human usability: sparsity",
        "subtitle": "Higher means fewer edges are shown to the analyst.",
        "x_label": "Sparsity score",
    },
]


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing final comparison file:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = [
        "Dataset",
        "Model",
        "Explainer",
        "Deletion drop",
        "Flip rate",
        "Insertion preservation",
        "Sparsity",
        "Interpretation",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns in final comparison CSV: {missing}")

    for col in ["Deletion drop", "Flip rate", "Insertion preservation", "Sparsity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Combination"] = df["Model"] + " + " + df["Explainer"]

    return df


def shorten_label(label, max_len=34):
    label = str(label)

    if len(label) <= max_len:
        return label

    return label[: max_len - 3] + "..."


def make_dataset_dashboard(df, dataset_name, output_path):
    dataset_df = df[df["Dataset"] == dataset_name].copy()

    if dataset_df.empty:
        print(f"WARNING: No rows found for dataset {dataset_name}")
        return

    dataset_df = dataset_df.sort_values(
        ["Model", "Explainer"],
        ascending=[True, True],
    ).reset_index(drop=True)

    labels = [shorten_label(x) for x in dataset_df["Combination"].tolist()]
    y_positions = np.arange(len(dataset_df))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        constrained_layout=True,
    )

    axes = axes.flatten()

    fig.suptitle(
        f"{dataset_name} faithfulness dashboard",
        fontsize=20,
        fontweight="bold",
    )

    for ax, metric in zip(axes, METRIC_CONFIG):
        col = metric["column"]
        values = dataset_df[col].fillna(0.0).values

        ax.barh(y_positions, values)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()

        ax.set_xlim(0, max(1.0, float(np.nanmax(values)) * 1.10 if len(values) else 1.0))
        ax.set_xlabel(metric["x_label"], fontsize=10)
        ax.set_title(metric["title"], fontsize=13, fontweight="bold")

        for i, value in enumerate(values):
            ax.text(
                value + 0.015,
                i,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )

        ax.grid(axis="x", linestyle="--", alpha=0.35)

        ax.text(
            0.0,
            -0.16,
            metric["subtitle"],
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.80,
        )

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_combined_dashboard(df):
    df = df.copy()

    df["Display label"] = (
        df["Dataset"]
        + " | "
        + df["Model"]
        + " + "
        + df["Explainer"]
    )

    # Sort by deletion drop because this is the main faithfulness signal.
    df = df.sort_values("Deletion drop", ascending=True).reset_index(drop=True)

    labels = [shorten_label(x, max_len=45) for x in df["Display label"].tolist()]
    y_positions = np.arange(len(df))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 10),
        constrained_layout=True,
    )

    fig.suptitle(
        "Final explainer faithfulness comparison: Elliptic vs AMLSim",
        fontsize=20,
        fontweight="bold",
    )

    metrics = [
        {
            "column": "Deletion drop",
            "title": "Deletion drop",
            "x_label": "Higher = explanation edges matter more",
        },
        {
            "column": "Flip rate",
            "title": "Prediction flip rate",
            "x_label": "Higher = prediction changes after removal",
        },
        {
            "column": "Insertion preservation",
            "title": "Insertion preservation",
            "x_label": "Higher = explanation alone is sufficient",
        },
    ]

    for ax, metric in zip(axes, metrics):
        col = metric["column"]
        values = df[col].fillna(0.0).values

        ax.barh(y_positions, values)
        ax.set_yticks(y_positions)

        if col == "Deletion drop":
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_xlim(0, max(1.0, float(np.nanmax(values)) * 1.10 if len(values) else 1.0))
        ax.set_title(metric["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel(metric["x_label"], fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        for i, value in enumerate(values):
            ax.text(
                value + 0.015,
                i,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )

    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_summary(df):
    md = "# Final faithfulness dashboard summary\n\n"

    md += "This dashboard explains whether the highlighted explanation edges are actually important to the model prediction.\n\n"

    md += "## How to read the metrics\n\n"
    md += "- **Deletion drop**: how much the suspicious probability drops after removing the explanation edges. Higher is better.\n"
    md += "- **Flip rate**: how often the model changes its prediction after removing the explanation edges. Higher is better.\n"
    md += "- **Insertion preservation**: whether the explanation edges alone can preserve the original prediction. Higher is better.\n"
    md += "- **Sparsity**: how compact the explanation is. Higher usually means fewer edges for the analyst to inspect.\n\n"

    best_deletion = df.sort_values("Deletion drop", ascending=False).head(5).copy()

    md += "## Top explanations by deletion drop\n\n"
    cols = [
        "Dataset",
        "Model",
        "Explainer",
        "Deletion drop",
        "Flip rate",
        "Insertion preservation",
        "Sparsity",
        "Interpretation",
    ]

    md += best_deletion[cols].to_markdown(index=False, floatfmt=".4f")
    md += "\n\n"

    md += "## Suggested professor explanation\n\n"
    md += textwrap.dedent(
        """
        The faithfulness dashboard shows that the explanations are not accepted blindly. 
        We test whether the highlighted edges are actually necessary and sufficient for the model prediction. 
        On Elliptic, GCN with GNNExplainer and PGExplainer gives the strongest deletion effect, meaning the selected edges are highly influential. 
        On AMLSim, the strongest result is GATv2 with GNNExplainer, while GraphSAGE shows weak deletion effect despite strong classification performance. 
        This supports the key research insight that high predictive performance does not automatically guarantee faithful explanations.
        """
    ).strip()

    md += "\n"

    return md


def main():
    print("=" * 100)
    print("Creating final faithfulness dashboard")
    print("=" * 100)

    df = load_data()
    df.to_csv(OUTPUT_CSV, index=False)

    make_combined_dashboard(df)

    make_dataset_dashboard(
        df,
        dataset_name="Elliptic",
        output_path=OUTPUT_DIR / "elliptic_faithfulness_dashboard.png",
    )

    make_dataset_dashboard(
        df,
        dataset_name="AMLSim",
        output_path=OUTPUT_DIR / "amlsim_faithfulness_dashboard.png",
    )

    summary = build_summary(df)
    OUTPUT_MD.write_text(summary, encoding="utf-8")

    print("\nSaved dashboard files:")
    print(f"Combined dashboard: {OUTPUT_PNG}")
    print(f"Elliptic dashboard: {OUTPUT_DIR / 'elliptic_faithfulness_dashboard.png'}")
    print(f"AMLSim dashboard:   {OUTPUT_DIR / 'amlsim_faithfulness_dashboard.png'}")
    print(f"Source CSV:         {OUTPUT_CSV}")
    print(f"Summary MD:         {OUTPUT_MD}")

    print("\nTop explanations by deletion drop:")
    print(
        df.sort_values("Deletion drop", ascending=False)
        [
            [
                "Dataset",
                "Model",
                "Explainer",
                "Deletion drop",
                "Flip rate",
                "Insertion preservation",
                "Sparsity",
            ]
        ]
        .head(8)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()