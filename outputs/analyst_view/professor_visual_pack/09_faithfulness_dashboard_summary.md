# Final faithfulness dashboard summary

This dashboard explains whether the highlighted explanation edges are actually important to the model prediction.

## How to read the metrics

- **Deletion drop**: how much the suspicious probability drops after removing the explanation edges. Higher is better.
- **Flip rate**: how often the model changes its prediction after removing the explanation edges. Higher is better.
- **Insertion preservation**: whether the explanation edges alone can preserve the original prediction. Higher is better.
- **Sparsity**: how compact the explanation is. Higher usually means fewer edges for the analyst to inspect.

## Top explanations by deletion drop

| Dataset   | Model   | Explainer    |   Deletion drop |   Flip rate |   Insertion preservation |   Sparsity | Interpretation                                                                                                |
|:----------|:--------|:-------------|----------------:|------------:|-------------------------:|-----------:|:--------------------------------------------------------------------------------------------------------------|
| Elliptic  | GCN     | GNNExplainer |          0.6722 |      1.0000 |                   1.0000 |     0.9995 | strong deletion effect; prediction flips observed; high sufficiency                                           |
| Elliptic  | GCN     | PGExplainer  |          0.6722 |      1.0000 |                   1.0000 |     0.9995 | strong deletion effect; prediction flips observed; high sufficiency                                           |
| Elliptic  | GCN     | SubgraphX    |          0.2962 |      0.3333 |                   0.6667 |     0.9999 | strong deletion effect; prediction flips observed; moderate sufficiency                                       |
| AMLSim    | GATv2   | GNNExplainer |          0.2771 |      0.3000 |                   0.5000 |     0.9872 | strong deletion effect; prediction flips observed; moderate sufficiency; saturated fraud probabilities likely |
| Elliptic  | GATv2   | PGExplainer  |          0.1892 |      0.2000 |                   1.0000 |     0.9995 | moderate deletion effect; some prediction flips; high sufficiency                                             |

## Suggested professor explanation

The faithfulness dashboard shows that the explanations are not accepted blindly. 
We test whether the highlighted edges are actually necessary and sufficient for the model prediction. 
On Elliptic, GCN with GNNExplainer and PGExplainer gives the strongest deletion effect, meaning the selected edges are highly influential. 
On AMLSim, the strongest result is GATv2 with GNNExplainer, while GraphSAGE shows weak deletion effect despite strong classification performance. 
This supports the key research insight that high predictive performance does not automatically guarantee faithful explanations.
