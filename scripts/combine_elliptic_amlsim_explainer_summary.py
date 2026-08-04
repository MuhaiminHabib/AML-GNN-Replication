from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ELLIPTIC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "all_models_explainer_faithfulness_summary.csv"
)

AMLSIM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explainers"
    / "amlsim_all_models"
    / "all_amlsim_models_explainer_faithfulness_summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainers" / "final_comparison"
OUTPUT_CSV = OUTPUT_DIR / "elliptic_vs_amlsim_explainer_comparison.csv"
OUTPUT_MD = OUTPUT_DIR / "elliptic_vs_amlsim_explainer_comparison.md"


MODEL_DISPLAY = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
}

EXPLAINER_DISPLAY = {
    "GNNExplainer": "GNNExplainer",
    "PGExplainer": "PGExplainer",
    "DGL_SubgraphX": "SubgraphX",
    "DGL_SubgraphX_1hop": "SubgraphX (1-hop)",
    "SubgraphX": "SubgraphX",
}


def load_summary(path: Path, dataset_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file:\n{path}")

    df = pd.read_csv(path)
    df = df.copy()
    df["dataset_clean"] = dataset_name

    return df


def copy_first_available(df: pd.DataFrame, target_col: str, source_cols):
    """
    Create target_col from the first available source column.
    If target_col already exists, keep it.
    """

    df = df.copy()

    if target_col in df.columns:
        return df

    for source_col in source_cols:
        if source_col in df.columns:
            df[target_col] = df[source_col]
            return df

    df[target_col] = np.nan

    return df


def normalise_summary(df: pd.DataFrame, dataset_name: str):
    """
    Normalise Elliptic and AMLSim summary files into the same schema.

    Elliptic summary currently has columns like:
      nodes_evaluated
      deletion_drop_mean
      deletion_label_flip_rate
      insertion_prob_mean
      insertion_preservation_rate
      sparsity_edges_mean

    AMLSim summary has columns like:
      explained_nodes
      mean_deletion_drop
      deletion_flip_rate
      mean_insertion_fraud_prob
      insertion_preservation_rate
      mean_sparsity
    """

    df = df.copy()

    df = copy_first_available(
        df,
        "model",
        ["model", "Model", "experiment_name"],
    )

    df = copy_first_available(
        df,
        "explainer",
        ["explainer", "Explainer", "method"],
    )

    df = copy_first_available(
        df,
        "top_k",
        ["top_k", "Top-K", "k"],
    )

    df = copy_first_available(
        df,
        "explained_nodes",
        ["explained_nodes", "nodes_evaluated", "Nodes", "num_nodes"],
    )

    df = copy_first_available(
        df,
        "mean_deletion_drop",
        ["mean_deletion_drop", "deletion_drop_mean", "deletion_drop", "Deletion drop"],
    )

    df = copy_first_available(
        df,
        "deletion_flip_rate",
        ["deletion_flip_rate", "deletion_label_flip_rate", "flip_rate", "Deletion flip rate"],
    )

    df = copy_first_available(
        df,
        "mean_insertion_fraud_prob",
        ["mean_insertion_fraud_prob", "insertion_prob_mean", "insertion_prob", "Insertion fraud prob."],
    )

    df = copy_first_available(
        df,
        "insertion_preservation_rate",
        ["insertion_preservation_rate", "preservation", "Insertion preservation"],
    )

    df = copy_first_available(
        df,
        "mean_sparsity",
        ["mean_sparsity", "sparsity_edges_mean", "sparsity", "Sparsity"],
    )

    # Elliptic old all-model summary has no top_k column.
    # It was produced from the selected final explanation edges, effectively top-k=20.
    df["top_k"] = pd.to_numeric(df["top_k"], errors="coerce").fillna(20)

    numeric_cols = [
        "top_k",
        "explained_nodes",
        "mean_deletion_drop",
        "deletion_flip_rate",
        "mean_insertion_fraud_prob",
        "insertion_preservation_rate",
        "mean_sparsity",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dataset_clean"] = dataset_name

    return df


def display_model_name(model):
    model = str(model).lower()

    if model in MODEL_DISPLAY:
        return MODEL_DISPLAY[model]

    if "graphsage" in model:
        return "GraphSAGE"

    if "gatv2" in model or "gat" in model:
        return "GATv2"

    if "gcn" in model:
        return "GCN"

    return str(model)


def display_explainer_name(explainer):
    explainer = str(explainer)

    if explainer in EXPLAINER_DISPLAY:
        return EXPLAINER_DISPLAY[explainer]

    lower = explainer.lower()

    if "gnnexplainer" in lower:
        return "GNNExplainer"

    if "pgexplainer" in lower:
        return "PGExplainer"

    if "subgraphx" in lower:
        if "1hop" in lower or "1-hop" in lower:
            return "SubgraphX (1-hop)"
        return "SubgraphX"

    return explainer


def add_interpretation(row):
    dataset = row["Dataset"]
    model = row["Model"]
    explainer = row["Explainer"]

    deletion_drop = row["Deletion drop"]
    flip_rate = row["Flip rate"]
    insertion_preservation = row["Insertion preservation"]

    notes = []

    if pd.notna(deletion_drop) and deletion_drop >= 0.20:
        notes.append("strong deletion effect")
    elif pd.notna(deletion_drop) and deletion_drop >= 0.05:
        notes.append("moderate deletion effect")
    elif pd.notna(deletion_drop):
        notes.append("weak deletion effect")

    if pd.notna(flip_rate) and flip_rate >= 0.30:
        notes.append("prediction flips observed")
    elif pd.notna(flip_rate) and flip_rate > 0:
        notes.append("some prediction flips")
    elif pd.notna(flip_rate):
        notes.append("no prediction flips")

    if pd.notna(insertion_preservation) and insertion_preservation >= 0.80:
        notes.append("high sufficiency")
    elif pd.notna(insertion_preservation) and insertion_preservation >= 0.50:
        notes.append("moderate sufficiency")
    elif pd.notna(insertion_preservation):
        notes.append("low sufficiency")

    if dataset == "AMLSim" and explainer == "PGExplainer":
        notes.append("PGExplainer masks collapsed")

    if dataset == "AMLSim" and "SubgraphX" in explainer:
        notes.append("1-hop reduced setting")

    if dataset == "AMLSim" and model in ["GraphSAGE", "GATv2"]:
        notes.append("saturated fraud probabilities likely")

    return "; ".join(notes)


def main():
    print("=" * 100)
    print("Combining Elliptic and AMLSim explainer faithfulness summaries")
    print("=" * 100)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    elliptic_df = load_summary(ELLIPTIC_PATH, "Elliptic")
    amlsim_df = load_summary(AMLSIM_PATH, "AMLSim")

    print("\nLoaded files:")
    print(f"Elliptic: {ELLIPTIC_PATH} | rows={len(elliptic_df)}")
    print(f"AMLSim:   {AMLSIM_PATH} | rows={len(amlsim_df)}")

    elliptic_df = normalise_summary(elliptic_df, "Elliptic")
    amlsim_df = normalise_summary(amlsim_df, "AMLSim")

    combined_df = pd.concat([elliptic_df, amlsim_df], ignore_index=True)

    combined_df = combined_df[pd.to_numeric(combined_df["top_k"], errors="coerce") == 20].copy()

    if combined_df.empty:
        raise RuntimeError("No rows remained after filtering to Top-K = 20.")

    combined_df["Dataset"] = combined_df["dataset_clean"]
    combined_df["Model"] = combined_df["model"].apply(display_model_name)
    combined_df["Explainer"] = combined_df["explainer"].apply(display_explainer_name)

    final_df = combined_df[
        [
            "Dataset",
            "Model",
            "Explainer",
            "top_k",
            "explained_nodes",
            "mean_deletion_drop",
            "deletion_flip_rate",
            "mean_insertion_fraud_prob",
            "insertion_preservation_rate",
            "mean_sparsity",
        ]
    ].copy()

    final_df = final_df.rename(
        columns={
            "top_k": "Top-K",
            "explained_nodes": "Nodes",
            "mean_deletion_drop": "Deletion drop",
            "deletion_flip_rate": "Flip rate",
            "mean_insertion_fraud_prob": "Insertion fraud prob.",
            "insertion_preservation_rate": "Insertion preservation",
            "mean_sparsity": "Sparsity",
        }
    )

    final_df["Top-K"] = final_df["Top-K"].astype("Int64")
    final_df["Nodes"] = final_df["Nodes"].astype("Int64")

    final_df["Interpretation"] = final_df.apply(add_interpretation, axis=1)

    dataset_order = {"Elliptic": 0, "AMLSim": 1}
    model_order = {"GCN": 0, "GraphSAGE": 1, "GATv2": 2}
    explainer_order = {
        "GNNExplainer": 0,
        "PGExplainer": 1,
        "SubgraphX": 2,
        "SubgraphX (1-hop)": 2,
    }

    final_df["_dataset_order"] = final_df["Dataset"].map(dataset_order).fillna(99)
    final_df["_model_order"] = final_df["Model"].map(model_order).fillna(99)
    final_df["_explainer_order"] = final_df["Explainer"].map(explainer_order).fillna(99)

    final_df = (
        final_df
        .sort_values(["_dataset_order", "_model_order", "_explainer_order"])
        .drop(columns=["_dataset_order", "_model_order", "_explainer_order"])
        .reset_index(drop=True)
    )

    final_df.to_csv(OUTPUT_CSV, index=False)

    markdown = "# Elliptic vs AMLSim explainer faithfulness comparison\n\n"
    markdown += "Main reporting table using Top-K = 20 explanation edges.\n\n"
    markdown += final_df.to_markdown(index=False, floatfmt=".6f")
    markdown += "\n"

    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print("\nCombined comparison table:")
    print(final_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved files:")
    print(f"CSV:      {OUTPUT_CSV}")
    print(f"Markdown: {OUTPUT_MD}")

    print("\nDataset/model/explainer counts:")
    print(
        final_df
        .groupby(["Dataset", "Model", "Explainer"])
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()