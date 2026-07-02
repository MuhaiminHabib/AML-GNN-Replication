\# Marasi \& Ferretti 2024 Replication Notes



Repository: https://github.com/simonemarasi/aml-elliptic-gnn  

Commit inspected: b02c6e6be51becf2147ce372002fef8998f17b21



This repository is used as the main public-code replication target for GCN, GAT, GraphSAGE, ChebNet, and GATv2 on the Elliptic dataset.



The repository implements:

\- GCNConv

\- GATConv

\- SAGEConv

\- ChebConv

\- GATv2Conv



Config:

\- hidden\_units: 110

\- hidden\_units\_noAgg: 64

\- epochs: 13000

\- learning rate: 9e-3

\- weight\_decay: 5e-4



Important observation:

The paper states a 65/15/20 split while maintaining label proportions, but the public code uses PyG RandomNodeSplit(num\_val=0.15, num\_test=0.2). Therefore, I will first run the repository as released, then reproduce the setup inside my own framework.

