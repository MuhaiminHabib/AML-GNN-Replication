from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARASI_REPO = PROJECT_ROOT / "data" / "external_repos" / "aml-elliptic-gnn"

INPUT_FILES = [
    MARASI_REPO / "marasi_5seed_txagg_results.csv"
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_RESULTS_CSV = OUTPUT_DIR / "marasi_5seed_txagg_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "marasi_5seed_txagg_summary.csv"
REPORT_MD = (
    PROJECT_ROOT
    / "experiments"
    / "aml_replication"
    / "marasi_2024_elliptic_gnn"
    / "FIVE_SEED_RESULTS.md"
)

PAPER_F1 = {
    "GCN": 0.616,
    "GAT": 0.766,
    "GraphSAGE": 0.889,
    "ChebNet": 0.910,
    "GATv2": 0.881,
}

MODEL_ORDER = ["GCN", "GAT", "GraphSAGE", "ChebNet", "GATv2"]


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main():
    missing_files = [path for path in INPUT_FILES if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(str(p) for p in missing_files)
        )

    frames = [pd.read_csv(path) for path in INPUT_FILES]
    df = pd.concat(frames, ignore_index=True)

    required_cols = {
        "dataset",
        "feature_setting",
        "seed",
        "model",
        "precision",
        "recall",
        "f1",
        "f1_micro_avg",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[df["feature_setting"] == "tx+agg"].copy()

    # Remove duplicates just in case a seed/model was run twice.
    df = df.drop_duplicates(
        subset=["dataset", "feature_setting", "seed", "model"],
        keep="last",
    )

    expected_seeds = {42, 43, 44, 45, 46}
    found_seeds = set(df["seed"].astype(int).unique())

    if found_seeds != expected_seeds:
        print("WARNING: Seed set is not exactly as expected.")
        print(f"Expected: {sorted(expected_seeds)}")
        print(f"Found:    {sorted(found_seeds)}")

    counts = df.groupby("model")["seed"].nunique()
    print("\nRuns per model:")
    print(counts.to_string())

    summary = (
        df.groupby(["dataset", "feature_setting", "model"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            f1_micro_mean=("f1_micro_avg", "mean"),
            f1_micro_std=("f1_micro_avg", "std"),
        )
    )

    summary["paper_f1"] = summary["model"].map(PAPER_F1)
    summary["f1_difference_vs_paper"] = summary["f1_mean"] - summary["paper_f1"]
    summary["f1_mean_std"] = summary.apply(
        lambda row: format_mean_std(row["f1_mean"], row["f1_std"]),
        axis=1,
    )

    summary["model_order"] = summary["model"].apply(
        lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )
    summary = summary.sort_values("model_order").drop(columns=["model_order"])

    df["model_order"] = df["model"].apply(
        lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
    )
    df = df.sort_values(["model_order", "seed"]).drop(columns=["model_order"])

    df.to_csv(COMBINED_RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nFive-seed summary:")
    print(
        summary[
            [
                "dataset",
                "model",
                "runs",
                "paper_f1",
                "f1_mean",
                "f1_std",
                "f1_difference_vs_paper",
                "f1_mean_std",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    lines = [
        "# Marasi & Ferretti 2024 Five-Seed Replication Results",
        "",
        "## Summary",
        "",
        "| Dataset | Model | Runs | Paper F1 | Reproduced F1 mean ± std | Difference vs paper |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | {int(row['runs'])} | "
            f"{row['paper_f1']:.4f} | {row['f1_mean_std']} | "
            f"{row['f1_difference_vs_paper']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Dataset: Elliptic",
            "- Feature setting: tx+agg",
            "- Seeds: 42, 43, 44, 45, 46",
            "- Metric: illicit-class F1",
            "- Values are reported as mean ± standard deviation across five runs.",
        ]
    )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved combined results to: {COMBINED_RESULTS_CSV}")
    print(f"Saved summary to: {SUMMARY_CSV}")
    print(f"Saved report to: {REPORT_MD}")


if __name__ == "__main__":
    main()