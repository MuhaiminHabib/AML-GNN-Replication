import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def mean_std(series):
    mean = series.mean()
    std = series.std()
    return f"{mean:.4f} ± {std:.4f}"


def extract_mean(value):
    """
    Extract the mean value from strings like:
    '0.6054 ± 0.0101'
    """
    match = re.match(r"([0-9.]+)", str(value))
    if match:
        return float(match.group(1))
    return 0.0


def main():
    input_path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / "multiseed"
        / "elliptic_multiseed_results.csv"
    )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / "multiseed"
        / "elliptic_multiseed_summary.csv"
    )

    df = pd.read_csv(input_path)

    # Backward compatibility for old result files
    if "direction_aware" not in df.columns:
        df["direction_aware"] = False

    group_cols = [
        "experiment_name",
        "dataset",
        "model",
        "normalize",
        "undirected",
        "direction_aware",
    ]

    metric_cols = [
        "accuracy",
        "illicit_precision",
        "illicit_recall",
        "illicit_f1",
        "roc_auc",
        "pr_auc",
    ]

    summary_rows = []

    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for metric in metric_cols:
            row[metric] = mean_std(group[metric])

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df["sort_f1"] = summary_df["illicit_f1"].apply(extract_mean)
    summary_df = summary_df.sort_values("sort_f1", ascending=False)
    summary_df = summary_df.drop(columns=["sort_f1"])

    print("\nMulti-seed summary:")
    print(summary_df.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print("\nSaved summary to:")
    print(output_path)


if __name__ == "__main__":
    main()