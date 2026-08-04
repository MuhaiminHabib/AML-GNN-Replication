# Professor visual story: researcher in the shoes of a bank analyst

This visual pack explains how the AML GNN detection and explanation framework can be understood from the point of view of a bank analyst.

## Core story

The GNN model first flags suspicious accounts or transactions. The analyst then opens a local graph explanation to see which neighbouring transaction links influenced the model. Finally, faithfulness metrics test whether those highlighted links actually matter to the model prediction.

In short:

```text
Transaction graph
      ↓
GNN fraud prediction
      ↓
Suspicious case appears in analyst queue
      ↓
Explainer highlights important local edges/subgraph
      ↓
Faithfulness metrics test whether the explanation matters
      ↓
Analyst decides: escalate, monitor, or dismiss
```

## Visual sequence for professor discussion

### 1. Analyst alert queue

Start by showing the alert queue. This represents the first screen a bank analyst would see. It lists suspicious nodes/accounts, model suspicion score, risk band, and recommended analyst action.

Use file: `01_analyst_alert_queue_summary.md`

### 2. Main AMLSim analyst case

Show the AMLSim GATv2 + GNNExplainer case. This is the best AMLSim visual because the account has a critical fraud score and the explanation forms a compact connected set of transaction links.

Use file: `02_amlsim_gatv2_gnnexplainer_main_case.png`

Talking point:

> From the analyst's perspective, the model says this account is highly suspicious, and the explainer shows which transaction links contributed to that suspicion.

### 3. Main Elliptic analyst case

Show the Elliptic GCN + GNNExplainer case. This is the strongest Elliptic visual because the explanation is connected to the target transaction and has strong faithfulness scores.

Use file: `03_elliptic_gcn_gnnexplainer_main_case.png`

Talking point:

> This example shows a clearer edge-dependent explanation. When explanation edges are removed, the model prediction is strongly affected.

### 4. Limitation case

Show the AMLSim GraphSAGE + GNNExplainer limitation case. This is useful because GraphSAGE is a strong classifier on AMLSim, but its explanation does not show strong deletion faithfulness.

Use file: `04_amlsim_graphsage_limitation_case.png`

Talking point:

> This supports the main research insight: high predictive performance does not guarantee faithful explanations.

### 5. Faithfulness dashboards

Use the faithfulness dashboards to explain how the explanations are evaluated. The dashboard compares deletion drop, flip rate, insertion preservation, and sparsity.

Use files:

- `05_amlsim_faithfulness_dashboard.png`
- `06_elliptic_faithfulness_dashboard.png`
- `07_final_elliptic_vs_amlsim_faithfulness_dashboard.png`

Talking point:

> We do not trust explanations blindly. We test whether removing the highlighted edges reduces the fraud probability or changes the prediction.

## Main conclusions

1. **Elliptic gives cleaner faithfulness behaviour**, especially for GCN + GNNExplainer.
2. **AMLSim works as a second AML dataset**, but many explanations are weaker because predictions are saturated and node features are highly influential.
3. **AMLSim GATv2 + GNNExplainer is the strongest AMLSim explanation result**.
4. **GraphSAGE performs strongly as a classifier on AMLSim, but explanation faithfulness is weak**.
5. **The key research insight is that model accuracy and explanation faithfulness are not the same thing**.

## Files included

- `01_analyst_alert_queue_summary.md` — Analyst alert queue showing suspicious nodes/accounts, risk band, and recommended action.
- `02_amlsim_gatv2_gnnexplainer_main_case.png` — Main AMLSim analyst case card showing GATv2 + GNNExplainer on a critical fraud account.
- `03_elliptic_gcn_gnnexplainer_main_case.png` — Main Elliptic analyst case card showing GCN + GNNExplainer on an illicit transaction.
- `04_amlsim_graphsage_limitation_case.png` — Limitation case showing that strong classifier performance does not guarantee clean explanations.
- `05_amlsim_faithfulness_dashboard.png` — AMLSim faithfulness dashboard comparing model and explainer combinations.
- `06_elliptic_faithfulness_dashboard.png` — Elliptic faithfulness dashboard comparing model and explainer combinations.
- `07_final_elliptic_vs_amlsim_faithfulness_dashboard.png` — Final cross-dataset faithfulness comparison.
- `08_final_comparison_table.md` — Final comparison table across datasets, models, explainers, and faithfulness metrics.
- `09_faithfulness_dashboard_summary.md` — Short written explanation of the faithfulness dashboard.
