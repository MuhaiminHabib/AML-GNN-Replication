from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATTERNS = [
    "outputs/explainers/*gnnexplainer*results.csv",
    "outputs/explainers/*pgexplainer*results.csv",
    "outputs/explainers/*subgraphx*results.csv",

    "outputs/explainers/amlsim_gcn/*gnnexplainer*results.csv",
    "outputs/explainers/amlsim_gcn/*pgexplainer*results.csv",
    "outputs/explainers/amlsim_gcn/*subgraphx*results.csv",

    "outputs/explainers/amlsim_graphsage/*gnnexplainer*results.csv",
    "outputs/explainers/amlsim_graphsage/*pgexplainer*results.csv",
    "outputs/explainers/amlsim_graphsage/*subgraphx*results.csv",

    "outputs/explainers/amlsim_gatv2/*gnnexplainer*results.csv",
    "outputs/explainers/amlsim_gatv2/*pgexplainer*results.csv",
    "outputs/explainers/amlsim_gatv2/*subgraphx*results.csv",

    "outputs/explainers/all_models_explainer_faithfulness.csv",
    "outputs/explainers/amlsim_all_models/all_amlsim_models_explainer_faithfulness_detail.csv",
]


def inspect_file(path: Path):
    print("=" * 120)
    print(path.relative_to(PROJECT_ROOT))

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read file: {e}")
        return

    print(f"Rows: {len(df)}")
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nHead:")
    print(df.head(8).to_string(index=False))

    node_cols = [
        col for col in df.columns
        if "node" in col.lower()
        or "target" in col.lower()
        or "center" in col.lower()
        or "src" in col.lower()
        or "dst" in col.lower()
        or "source" in col.lower()
        or "destination" in col.lower()
    ]

    edge_cols = [
        col for col in df.columns
        if "edge" in col.lower()
        or "src" in col.lower()
        or "dst" in col.lower()
        or "source" in col.lower()
        or "destination" in col.lower()
    ]

    score_cols = [
        col for col in df.columns
        if "mask" in col.lower()
        or "score" in col.lower()
        or "importance" in col.lower()
        or "weight" in col.lower()
        or "prob" in col.lower()
        or "drop" in col.lower()
    ]

    print("\nPossible node columns:")
    print(node_cols)

    print("\nPossible edge columns:")
    print(edge_cols)

    print("\nPossible score / importance columns:")
    print(score_cols)

    print()


def main():
    files = []

    for pattern in CANDIDATE_PATTERNS:
        files.extend(PROJECT_ROOT.glob(pattern))

    files = sorted(set(files))

    print("=" * 120)
    print("Inspecting explanation files for local graph visualization")
    print("=" * 120)
    print(f"Found {len(files)} candidate files.")

    if not files:
        print("No files found. Check output paths.")
        return

    for path in files:
        inspect_file(path)


if __name__ == "__main__":
    main()