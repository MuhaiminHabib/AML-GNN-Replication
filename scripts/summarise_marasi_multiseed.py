from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_REPO = PROJECT_ROOT / "data" / "external_repos" / "aml-elliptic-gnn"

INPUT_CANDIDATES = [
    EXTERNAL_REPO / "marasi_multiseed_txagg_results.csv",
    EXTERNAL_REPO / "marasi_multiseed_txagg_partial_results.csv",
    PROJECT_ROOT / "outputs" / "results" / "marasi_multiseed_txagg_results.csv",
    PROJECT_ROOT / "outputs" / "results" / "marasi_multiseed_txagg_partial_results.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "marasi_multiseed_txagg_summary.csv"
REPORT_MD = PROJECT_ROOT / "experiments" / "aml_replication" / "marasi_2024_elliptic_gnn" / "MULTISEED_RESULTS.md"


PAPER_F1 = {
    "GCN": 0.616,
    "GAT": 0.766,
    "GraphSAGE": 0.889,
    "ChebNet": 0.910,
    "GATv2": 0.881,
}


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a Marasi multi-seed results file. Searched:\n"
        f"{searched}"
    )


def format_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.4f} ± NA"
    return f"{mean:.4f} ± {std:.4f}"


def main():
    input_path = find_input_file()

    print(f"Reading results from: {input_path}")

    df = pd.read_csv(input_path)

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

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only tx+agg because this is the paper-comparable setting.
    df = df[df["feature_setting"] == "tx+agg"].copy()

    if df.empty:
        raise ValueError("No tx+agg rows found in results file.")

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

    model_order = ["GCN", "GAT", "GraphSAGE", "ChebNet", "GATv2"]
    summary["model_order"] = summary["model"].apply(
        lambda x: model_order.index(x) if x in model_order else 999
    )
    summary = summary.sort_values("model_order").drop(columns=["model_order"])

    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nMulti-seed summary:")
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

    report_lines = [
        "# Marasi & Ferretti 2024 Multi-Seed Replication Results",
        "",
        "## Source",
        "",
        f"Input file: `{input_path}`",
        "",
        "## Summary",
        "",
        "| Dataset | Model | Runs | Paper F1 | Reproduced F1 mean ± std | Difference vs paper |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        report_lines.append(
            f"| {row['dataset']} | {row['model']} | {int(row['runs'])} | "
            f"{row['paper_f1']:.4f} | {row['f1_mean_std']} | "
            f"{row['f1_difference_vs_paper']:+.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Dataset: Elliptic",
            "- Feature setting: tx+agg",
            "- Reported values are illicit-class F1 scores.",
            "- The multi-seed result uses different random seeds and reports mean ± standard deviation.",
        ]
    )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nSaved summary CSV to: {SUMMARY_CSV}")
    print(f"Saved markdown report to: {REPORT_MD}")


if __name__ == "__main__":
    main()