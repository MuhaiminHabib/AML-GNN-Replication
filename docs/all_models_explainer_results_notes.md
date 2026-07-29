\# All Models Explainer Faithfulness Results



\## Experimental setup



A shared checkpoint was trained separately for each backbone model: GCN, GraphSAGE, and GATv2. For each model, correctly predicted illicit test nodes were selected as fixed explanation targets. The same target nodes for that model were then used across GNNExplainer, PGExplainer, and SubgraphX.



GNNExplainer and PGExplainer were implemented using PyTorch Geometric. SubgraphX was implemented using the official DGL implementation through a PyG-to-DGL ego-graph bridge. All explanation outputs were mapped back to the original Elliptic graph and evaluated using the same deletion and insertion faithfulness metrics.



\## Key finding



GNNExplainer and PGExplainer produced the strongest faithfulness results overall when top-20 explanation edges were used. SubgraphX produced more compact explanations, but its deletion-based necessity was usually weaker because it returned much smaller explanation subgraphs.



\## Model-level observation



GATv2 achieved the strongest predictive performance among the shared checkpoints. However, the strongest deletion faithfulness result appeared for GCN with GNNExplainer and PGExplainer. This means that stronger classification performance does not automatically imply stronger deletion-based explanation faithfulness.



\## Interpretation



The results suggest that explanation faithfulness depends on both the explainer and the model backbone. GNNExplainer and PGExplainer were consistently sufficient because their insertion preservation rate was 1.00 across all three models. SubgraphX was more compact, but its smaller explanations often removed fewer graph edges and therefore produced weaker deletion effects.

