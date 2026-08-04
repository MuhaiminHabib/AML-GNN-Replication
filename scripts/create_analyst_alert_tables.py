from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ALL = OUTPUT_DIR / "analyst_alert_queue_all.csv"
OUTPUT_ELLIPTIC = OUTPUT_DIR / "elliptic_alert_queue.csv"
OUTPUT_AMLSIM = OUTPUT_DIR / "amlsim_alert_queue.csv"
OUTPUT_MD = OUTPUT_DIR / "analyst_alert_queue_summary.md"


MODEL_DISPLAY = {
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gatv2": "GATv2",
}


ALERT_NODE_FILES = [
    {
        "dataset": "Elliptic",
        "model": "gcn",
        "suspicious_class_label": 0,
        "suspicious_class_name": "Illicit",
        "path": PROJECT_ROOT / "outputs" / "explainers" / "shared_gcn_explanation_nodes.csv",
    },
    {
        "dataset": "Elliptic",
        "model": "graphsage",
        "suspicious_class_label": 0,
        "suspicious_class_name": "Illicit",
        "path": PROJECT_ROOT / "outputs" / "explainers" / "shared_graphsage_explanation_nodes.csv",
    },
    {
        "dataset": "Elliptic",
        "model": "gatv2",
        "suspicious_class_label": 0,
        "suspicious_class_name": "Illicit",
        "path": PROJECT_ROOT / "outputs" / "explainers" / "shared_gatv2_explanation_nodes.csv",
    },
    {
        "dataset": "AMLSim",
        "model": "gcn",
        "suspicious_class_label": 1,
        "suspicious_class_name": "Fraud",
        "path": PROJECT_ROOT / "outputs" / "explainers" / "amlsim_gcn" / "shared_amlsim_gcn_explanation_nodes.csv",
    },
    {
        "dataset": "AMLSim",
        "model": "graphsage",
        "suspicious_class_label": 1,
        "suspicious_class_name": "Fraud",
        "path": PROJECT_ROOT / "outputs" / "explainers" / "amlsim_graphsage" / "shared_amlsim_graphsage_explanation_nodes.csv",
    },
    {
        "dataset": "AMLSim",
        "model": "gatv2",
        "suspicious_class_label": 1,
        "suspicious_class_name": "Fraud",
        "path": PROJECT_ROOT / "outputs" / "explainers" / "amlsim_gatv2" / "shared_amlsim_gatv2_explanation_nodes.csv",
    },
]


def first_existing_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_series_or_default(df: pd.DataFrame, candidates, default=np.nan):
    col = first_existing_column(df, candidates)

    if col is None:
        return pd.Series([default] * len(df), index=df.index)

    return df[col]


def risk_band(prob):
    if pd.isna(prob):
        return "Unknown"

    prob = float(prob)

    if prob >= 0.95:
        return "Critical"
    if prob >= 0.80:
        return "High"
    if prob >= 0.60:
        return "Medium"

    return "Review"


def analyst_action(prob, predicted_is_suspicious):
    if pd.isna(prob):
        return "Review manually"

    prob = float(prob)

    if predicted_is_suspicious and prob >= 0.95:
        return "Escalate immediately"
    if predicted_is_suspicious and prob >= 0.80:
        return "Investigate with explanation"
    if predicted_is_suspicious:
        return "Monitor / secondary review"

    return "No suspicious alert"


def normalise_alert_file(
    dataset: str,
    model: str,
    path: Path,
    suspicious_class_label: int,
    suspicious_class_name: str,
):
    if not path.exists():
        print(f"WARNING: Missing node file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        print(f"WARNING: Empty node file: {path}")
        return pd.DataFrame()

    model_display = MODEL_DISPLAY.get(model, model)

    rank_series = get_series_or_default(
        df,
        ["rank", "Rank", "alert_rank"],
        default=np.nan,
    )

    node_series = get_series_or_default(
        df,
        ["node_idx", "center_node_idx", "target_node_idx", "node_id", "Node ID"],
        default=np.nan,
    )

    split_series = get_series_or_default(
        df,
        ["split", "Split"],
        default="test",
    )

    true_label_series = get_series_or_default(
        df,
        ["true_label", "y_true", "label", "True label"],
        default=np.nan,
    )

    pred_label_series = get_series_or_default(
        df,
        ["pred_label", "predicted_label", "y_pred", "prediction", "Predicted label"],
        default=np.nan,
    )

    # Important:
    # Elliptic uses pred_prob_illicit.
    # AMLSim uses fraud_probability.
    prob_series = get_series_or_default(
        df,
        [
            "fraud_probability",
            "pred_prob_illicit",
            "illicit_probability",
            "fraud_prob",
            "illicit_prob",
            "prediction_probability",
            "pred_probability",
            "probability",
            "original_prob_illicit",
            "original_fraud_prob",
        ],
        default=np.nan,
    )

    out = pd.DataFrame(
        {
            "Dataset": dataset,
            "Model": model_display,
            "Alert rank": rank_series,
            "Node ID": node_series,
            "Split": split_series,
            "Suspicious class label": suspicious_class_label,
            "Suspicious class name": suspicious_class_name,
            "True label raw": true_label_series,
            "Predicted label raw": pred_label_series,
            "Suspicion probability": prob_series,
            "Source file": str(path.relative_to(PROJECT_ROOT)),
        }
    )

    out["Alert rank"] = pd.to_numeric(out["Alert rank"], errors="coerce")
    out["Node ID"] = pd.to_numeric(out["Node ID"], errors="coerce")
    out["True label raw"] = pd.to_numeric(out["True label raw"], errors="coerce")
    out["Predicted label raw"] = pd.to_numeric(out["Predicted label raw"], errors="coerce")
    out["Suspicion probability"] = pd.to_numeric(out["Suspicion probability"], errors="coerce")

    # If no rank exists, rank by suspicion probability inside dataset/model.
    if out["Alert rank"].isna().all():
        out = out.sort_values("Suspicion probability", ascending=False).reset_index(drop=True)
        out["Alert rank"] = np.arange(1, len(out) + 1)

    out["True suspicious?"] = out["True label raw"] == suspicious_class_label
    out["Predicted suspicious?"] = out["Predicted label raw"] == suspicious_class_label

    out["Risk band"] = out["Suspicion probability"].apply(risk_band)

    out["Recommended analyst action"] = out.apply(
        lambda row: analyst_action(
            prob=row["Suspicion probability"],
            predicted_is_suspicious=bool(row["Predicted suspicious?"]),
        ),
        axis=1,
    )

    out["Case explanation view"] = out.apply(
        lambda row: (
            f"{row['Dataset']} / {row['Model']} / node {int(row['Node ID'])}"
            if pd.notna(row["Node ID"])
            else f"{row['Dataset']} / {row['Model']} / node unknown"
        ),
        axis=1,
    )

    return out


def build_markdown_summary(all_df: pd.DataFrame):
    md = "# Analyst alert queue summary\n\n"

    md += (
        "This table represents the first analyst-facing view. "
        "It lists the nodes/accounts selected for explanation, their model suspicion score, "
        "risk band, and suggested investigation action.\n\n"
    )

    md += "## Class-label note\n\n"
    md += "- For **Elliptic**, suspicious/illicit class is represented as label **0** in this project output.\n"
    md += "- For **AMLSim**, suspicious/fraud class is represented as label **1**.\n"
    md += "- Therefore, the table uses dataset-specific suspicious-class mapping instead of assuming fraud is always label 1.\n\n"

    if all_df.empty:
        md += "No alert rows were generated.\n"
        return md

    top_df = all_df.copy()

    top_df = top_df.sort_values(
        ["Dataset", "Model", "Alert rank"],
        ascending=[True, True, True],
    )

    display_cols = [
        "Dataset",
        "Model",
        "Alert rank",
        "Node ID",
        "Suspicious class name",
        "True label raw",
        "Predicted label raw",
        "True suspicious?",
        "Predicted suspicious?",
        "Suspicion probability",
        "Risk band",
        "Recommended analyst action",
    ]

    md += "## Full alert queue\n\n"
    md += top_df[display_cols].to_markdown(index=False, floatfmt=".6f")
    md += "\n\n"

    md += "## Alert count by dataset, model, and risk band\n\n"

    count_df = (
        all_df
        .groupby(["Dataset", "Model", "Risk band"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )

    md += count_df.to_markdown(index=False)
    md += "\n\n"

    md += "## How to read this\n\n"
    md += "- **Suspicion probability** is the model's fraud/illicit confidence for the selected node.\n"
    md += "- **Risk band** converts the probability into an analyst-friendly priority label.\n"
    md += "- **Predicted suspicious?** tells whether the model predicted the suspicious class for that dataset.\n"
    md += "- **Recommended analyst action** shows how the case could be handled in a bank investigation workflow.\n"
    md += "- The next step after this table is to open the local graph explanation for a selected case.\n"

    return md


def main():
    print("=" * 100)
    print("Creating analyst alert queue tables")
    print("=" * 100)

    all_alerts = []

    for item in ALERT_NODE_FILES:
        dataset = item["dataset"]
        model = item["model"]
        path = item["path"]
        suspicious_class_label = item["suspicious_class_label"]
        suspicious_class_name = item["suspicious_class_name"]

        print(f"\nLoading {dataset} / {model}: {path}")

        alert_df = normalise_alert_file(
            dataset=dataset,
            model=model,
            path=path,
            suspicious_class_label=suspicious_class_label,
            suspicious_class_name=suspicious_class_name,
        )

        if not alert_df.empty:
            print(f"  Loaded {len(alert_df)} alert rows.")
            all_alerts.append(alert_df)
        else:
            print("  No rows loaded.")

    if not all_alerts:
        raise RuntimeError("No alert files were loaded. Check the explanation node CSV paths.")

    all_df = pd.concat(all_alerts, ignore_index=True)

    all_df = all_df.sort_values(
        ["Dataset", "Model", "Alert rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    elliptic_df = all_df[all_df["Dataset"] == "Elliptic"].copy()
    amlsim_df = all_df[all_df["Dataset"] == "AMLSim"].copy()

    all_df.to_csv(OUTPUT_ALL, index=False)
    elliptic_df.to_csv(OUTPUT_ELLIPTIC, index=False)
    amlsim_df.to_csv(OUTPUT_AMLSIM, index=False)

    markdown = build_markdown_summary(all_df)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print("\nSaved files:")
    print(f"All alerts:      {OUTPUT_ALL}")
    print(f"Elliptic alerts: {OUTPUT_ELLIPTIC}")
    print(f"AMLSim alerts:   {OUTPUT_AMLSIM}")
    print(f"Markdown:        {OUTPUT_MD}")

    print("\nPreview:")
    preview_cols = [
        "Dataset",
        "Model",
        "Alert rank",
        "Node ID",
        "Suspicious class name",
        "True label raw",
        "Predicted label raw",
        "True suspicious?",
        "Predicted suspicious?",
        "Suspicion probability",
        "Risk band",
        "Recommended analyst action",
    ]

    print(all_df[preview_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()