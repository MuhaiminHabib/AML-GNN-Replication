# All Models Explainer Faithfulness Summary

| Model     | Explainer    |   Nodes |   Deletion drop mean |   Deletion flip rate |   Insertion prob mean |   Insertion preservation |   Edge sparsity |   Avg explanation edges |   Avg explanation nodes |
|:----------|:-------------|--------:|---------------------:|---------------------:|----------------------:|-------------------------:|----------------:|------------------------:|------------------------:|
| GCN       | GNNExplainer |       6 |               0.6722 |               1      |                0.8224 |                   1      |          0.9995 |                 20      |                 34.3333 |
| GCN       | PGExplainer  |       6 |               0.6722 |               1      |                0.8224 |                   1      |          0.9995 |                 20      |                 34.3333 |
| GCN       | SubgraphX    |       6 |               0.2962 |               0.3333 |                0.6432 |                   0.6667 |          0.9999 |                  1.8333 |                  2.8333 |
| GraphSAGE | GNNExplainer |       7 |               0.1241 |               0.2857 |                0.8455 |                   1      |          0.9995 |                 20      |                 27.8571 |
| GraphSAGE | PGExplainer  |       7 |               0.1266 |               0.2857 |                0.8463 |                   1      |          0.9995 |                 20      |                 27.8571 |
| GraphSAGE | SubgraphX    |       7 |              -0.0091 |               0      |                0.801  |                   0.8571 |          1      |                  0.8571 |                  1.4286 |
| GATv2     | GNNExplainer |      10 |               0.1892 |               0.2    |                0.9321 |                   1      |          0.9995 |                 20      |                 29.9    |
| GATv2     | PGExplainer  |      10 |               0.1892 |               0.2    |                0.9321 |                   1      |          0.9995 |                 20      |                 29.9    |
| GATv2     | SubgraphX    |      10 |               0.1777 |               0.3    |                0.9416 |                   1      |          1      |                  1.5    |                  2.3    |
