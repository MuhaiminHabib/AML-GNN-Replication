# Marasi & Ferretti 2024 Multi-Seed Replication Results

## Source

Input file: `D:\habib dissertation\Projects\AML-GNN-Replication\data\external_repos\aml-elliptic-gnn\marasi_multiseed_txagg_results.csv`

## Summary

| Dataset | Model | Runs | Paper F1 | Reproduced F1 mean ± std | Difference vs paper |
|---|---|---:|---:|---:|---:|
| Elliptic | GCN | 3 | 0.6160 | 0.6047 ± 0.0101 | -0.0113 |
| Elliptic | GAT | 3 | 0.7660 | 0.7700 ± 0.0122 | +0.0040 |
| Elliptic | GraphSAGE | 3 | 0.8890 | 0.8750 ± 0.0106 | -0.0140 |
| Elliptic | ChebNet | 3 | 0.9100 | 0.8983 ± 0.0075 | -0.0117 |
| Elliptic | GATv2 | 3 | 0.8810 | 0.8793 ± 0.0078 | -0.0017 |

## Notes

- Dataset: Elliptic
- Feature setting: tx+agg
- Reported values are illicit-class F1 scores.
- The multi-seed result uses different random seeds and reports mean ± standard deviation.