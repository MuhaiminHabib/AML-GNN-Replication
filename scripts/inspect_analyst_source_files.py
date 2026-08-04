from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    PROJECT_ROOT / "outputs" / "explainers" / "shared_gcn_explanation_nodes.csv",
    PROJECT_ROOT / "outputs" / "explainers" / "shared_graphsage_explanation_nodes.csv",
    PROJECT_ROOT / "outputs" / "explainers" / "shared_gatv2_explanation_nodes.csv",

    PROJECT_ROOT / "outputs" / "explainers" / "all_models_explainer_faithfulness.csv",
    PROJECT_ROOT / "outputs" / "explainers" / "all_models_explainer_faithfulness_summary.csv",

    PROJECT_ROOT / "outputs" / "explainers" / "amlsim_gcn" / "shared_amlsim_gcn_explanation_nodes.csv",
    PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage" / "shared_amlsim_graphsage_explanation_nodes.csv",
    PROJECT_ROOT / "outputs" / "explainers" / "amlsim_gatv2" / "shared_amlsim_gatv2_explanation_nodes.csv",

    PROJECT_ROOT / "outputs" / "explainers" / "amlsim_all_models" / "all_amlsim_models_explainer_faithfulness_detail.csv",
    PROJECT_ROOT / "outputs" / "explainers" / "amlsim_all_models" / "all_amlsim_models_explainer_faithfulness_summary.csv",
]


def inspect_file(path: Path):
    print("=" * 120)
    print(path)

    if not path.exists():
        print("MISSING")
        return

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nHead:")
    print(df.head(5).to_string(index=False))

    possible_prob_cols = [
        col for col in df.columns
        if "prob" in col.lower()
        or "score" in col.lower()
        or "confidence" in col.lower()
        or "logit" in col.lower()
    ]

    possible_label_cols = [
        col for col in df.columns
        if "label" in col.lower()
        or "pred" in col.lower()
        or "true" in col.lower()
        or col.lower() in ["y", "y_true", "y_pred"]
    ]

    possible_node_cols = [
        col for col in df.columns
        if "node" in col.lower()
        or "idx" in col.lower()
        or "account" in col.lower()
    ]

    print("\nPossible node columns:", possible_node_cols)
    print("Possible label columns:", possible_label_cols)
    print("Possible probability/score columns:", possible_prob_cols)
    print()


def main():
    for path in FILES:
        inspect_file(path)


if __name__ == "__main__":
    main()