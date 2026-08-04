from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analyst_view" / "professor_visual_pack"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_COPY = [
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "analyst_alert_queue_summary.md",
        "target": "01_analyst_alert_queue_summary.md",
        "description": "Analyst alert queue showing suspicious nodes/accounts, risk band, and recommended action.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "final_case_cards" / "amlsim_gatv2_gnnexplainer_node_1751_main_amlsim_demo.png",
        "target": "02_amlsim_gatv2_gnnexplainer_main_case.png",
        "description": "Main AMLSim analyst case card showing GATv2 + GNNExplainer on a critical fraud account.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "final_case_cards" / "elliptic_gcn_gnnexplainer_node_18018_main_elliptic_demo.png",
        "target": "03_elliptic_gcn_gnnexplainer_main_case.png",
        "description": "Main Elliptic analyst case card showing GCN + GNNExplainer on an illicit transaction.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "final_case_cards" / "amlsim_graphsage_gnnexplainer_node_4842_limitation_case.png",
        "target": "04_amlsim_graphsage_limitation_case.png",
        "description": "Limitation case showing that strong classifier performance does not guarantee clean explanations.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "faithfulness_dashboard" / "amlsim_faithfulness_dashboard.png",
        "target": "05_amlsim_faithfulness_dashboard.png",
        "description": "AMLSim faithfulness dashboard comparing model and explainer combinations.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "faithfulness_dashboard" / "elliptic_faithfulness_dashboard.png",
        "target": "06_elliptic_faithfulness_dashboard.png",
        "description": "Elliptic faithfulness dashboard comparing model and explainer combinations.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "faithfulness_dashboard" / "final_faithfulness_dashboard.png",
        "target": "07_final_elliptic_vs_amlsim_faithfulness_dashboard.png",
        "description": "Final cross-dataset faithfulness comparison.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "explainers" / "final_comparison" / "elliptic_vs_amlsim_explainer_comparison.md",
        "target": "08_final_comparison_table.md",
        "description": "Final comparison table across datasets, models, explainers, and faithfulness metrics.",
    },
    {
        "source": PROJECT_ROOT / "outputs" / "analyst_view" / "faithfulness_dashboard" / "final_faithfulness_dashboard_summary.md",
        "target": "09_faithfulness_dashboard_summary.md",
        "description": "Short written explanation of the faithfulness dashboard.",
    },
]


def copy_files():
    copied = []

    for item in FILES_TO_COPY:
        source = item["source"]
        target = OUTPUT_DIR / item["target"]

        if not source.exists():
            print(f"WARNING: Missing file: {source}")
            continue

        shutil.copy2(source, target)
        copied.append(
            {
                "target": target,
                "description": item["description"],
            }
        )

        print(f"Copied: {target}")

    return copied


def create_story_file(copied):
    story_path = OUTPUT_DIR / "00_professor_visual_story.md"

    md = "# Professor visual story: researcher in the shoes of a bank analyst\n\n"

    md += "This visual pack explains how the AML GNN detection and explanation framework can be understood from the point of view of a bank analyst.\n\n"

    md += "## Core story\n\n"
    md += (
        "The GNN model first flags suspicious accounts or transactions. "
        "The analyst then opens a local graph explanation to see which neighbouring transaction links influenced the model. "
        "Finally, faithfulness metrics test whether those highlighted links actually matter to the model prediction.\n\n"
    )

    md += "In short:\n\n"

    md += "```text\n"
    md += "Transaction graph\n"
    md += "      ↓\n"
    md += "GNN fraud prediction\n"
    md += "      ↓\n"
    md += "Suspicious case appears in analyst queue\n"
    md += "      ↓\n"
    md += "Explainer highlights important local edges/subgraph\n"
    md += "      ↓\n"
    md += "Faithfulness metrics test whether the explanation matters\n"
    md += "      ↓\n"
    md += "Analyst decides: escalate, monitor, or dismiss\n"
    md += "```\n\n"

    md += "## Visual sequence for professor discussion\n\n"

    md += "### 1. Analyst alert queue\n\n"
    md += (
        "Start by showing the alert queue. This represents the first screen a bank analyst would see. "
        "It lists suspicious nodes/accounts, model suspicion score, risk band, and recommended analyst action.\n\n"
    )

    md += "Use file: `01_analyst_alert_queue_summary.md`\n\n"

    md += "### 2. Main AMLSim analyst case\n\n"
    md += (
        "Show the AMLSim GATv2 + GNNExplainer case. "
        "This is the best AMLSim visual because the account has a critical fraud score and the explanation forms a compact connected set of transaction links.\n\n"
    )

    md += "Use file: `02_amlsim_gatv2_gnnexplainer_main_case.png`\n\n"

    md += "Talking point:\n\n"
    md += (
        "> From the analyst's perspective, the model says this account is highly suspicious, "
        "and the explainer shows which transaction links contributed to that suspicion.\n\n"
    )

    md += "### 3. Main Elliptic analyst case\n\n"
    md += (
        "Show the Elliptic GCN + GNNExplainer case. "
        "This is the strongest Elliptic visual because the explanation is connected to the target transaction and has strong faithfulness scores.\n\n"
    )

    md += "Use file: `03_elliptic_gcn_gnnexplainer_main_case.png`\n\n"

    md += "Talking point:\n\n"
    md += (
        "> This example shows a clearer edge-dependent explanation. "
        "When explanation edges are removed, the model prediction is strongly affected.\n\n"
    )

    md += "### 4. Limitation case\n\n"
    md += (
        "Show the AMLSim GraphSAGE + GNNExplainer limitation case. "
        "This is useful because GraphSAGE is a strong classifier on AMLSim, but its explanation does not show strong deletion faithfulness.\n\n"
    )

    md += "Use file: `04_amlsim_graphsage_limitation_case.png`\n\n"

    md += "Talking point:\n\n"
    md += (
        "> This supports the main research insight: high predictive performance does not guarantee faithful explanations.\n\n"
    )

    md += "### 5. Faithfulness dashboards\n\n"
    md += (
        "Use the faithfulness dashboards to explain how the explanations are evaluated. "
        "The dashboard compares deletion drop, flip rate, insertion preservation, and sparsity.\n\n"
    )

    md += "Use files:\n\n"
    md += "- `05_amlsim_faithfulness_dashboard.png`\n"
    md += "- `06_elliptic_faithfulness_dashboard.png`\n"
    md += "- `07_final_elliptic_vs_amlsim_faithfulness_dashboard.png`\n\n"

    md += "Talking point:\n\n"
    md += (
        "> We do not trust explanations blindly. "
        "We test whether removing the highlighted edges reduces the fraud probability or changes the prediction.\n\n"
    )

    md += "## Main conclusions\n\n"

    md += "1. **Elliptic gives cleaner faithfulness behaviour**, especially for GCN + GNNExplainer.\n"
    md += "2. **AMLSim works as a second AML dataset**, but many explanations are weaker because predictions are saturated and node features are highly influential.\n"
    md += "3. **AMLSim GATv2 + GNNExplainer is the strongest AMLSim explanation result**.\n"
    md += "4. **GraphSAGE performs strongly as a classifier on AMLSim, but explanation faithfulness is weak**.\n"
    md += "5. **The key research insight is that model accuracy and explanation faithfulness are not the same thing**.\n\n"

    md += "## Files included\n\n"

    for item in copied:
        rel = item["target"].relative_to(OUTPUT_DIR)
        md += f"- `{rel}` — {item['description']}\n"

    story_path.write_text(md, encoding="utf-8")

    print(f"\nCreated story file: {story_path}")


def main():
    print("=" * 100)
    print("Creating professor visual pack")
    print("=" * 100)

    copied = copy_files()
    create_story_file(copied)

    print("\nProfessor visual pack created at:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()