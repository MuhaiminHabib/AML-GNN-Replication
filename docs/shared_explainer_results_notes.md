\# Shared Explainer Faithfulness Results



\## Experimental setting



To compare the three explanation methods fairly, the experiment used a shared GraphSAGE checkpoint trained on the Elliptic dataset with the tx+agg feature setting. The same seven correctly predicted illicit test nodes were used as explanation targets for all explainers.



GNNExplainer and PGExplainer were applied using PyTorch Geometric. SubgraphX was applied using the official DGL implementation through a PyG-to-DGL ego-graph bridge. All explanation outputs were mapped back to the original Elliptic graph and evaluated using the same deletion and insertion faithfulness procedure.



\## Summary



Under the shared setting, GNNExplainer and PGExplainer produced very similar results. PGExplainer achieved a slightly higher mean deletion drop than GNNExplainer, but the difference was small. Both explainers achieved a deletion label-flip rate of 0.286 and an insertion preservation rate of 1.000.



SubgraphX produced much smaller explanations after mapping its returned node subsets back to original Elliptic edges. Its insertion preservation rate was 0.857, but its mean deletion drop was negative and its deletion label-flip rate was 0.000. This suggests that, in this setting, SubgraphX explanations were compact and sometimes sufficient, but not strongly necessary under deletion-based faithfulness evaluation.



\## Key results



| Explainer | Nodes | Mean deletion drop | Deletion flip rate | Mean insertion probability | Insertion preservation | Avg explanation edges |

|---|---:|---:|---:|---:|---:|---:|

| GNNExplainer | 7 | 0.1241 | 0.2857 | 0.8455 | 1.0000 | 20.0000 |

| PGExplainer | 7 | 0.1266 | 0.2857 | 0.8463 | 1.0000 | 20.0000 |

| SubgraphX | 7 | -0.0091 | 0.0000 | 0.8010 | 0.8571 | 0.8571 |



\## Interpretation



The shared-node experiment shows that GNNExplainer and PGExplainer provide the strongest faithfulness results in this Elliptic GraphSAGE setting. Their explanations are sufficient because the insertion preservation rate is 1.000, and partially necessary because deleting the selected edges flips the model prediction for 28.6% of the explained nodes.



PGExplainer is marginally stronger than GNNExplainer on mean deletion drop, but the difference is too small to claim a clear winner. SubgraphX gives much smaller explanations, which may make it attractive for compactness, but its deletion results are weak in this current implementation.

