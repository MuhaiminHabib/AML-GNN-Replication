# Marasi & Ferretti 2024 Five-Seed Replication Results

## Summary

| Dataset | Model | Runs | Paper F1 | Reproduced F1 mean ± std | Difference vs paper |
|---|---|---:|---:|---:|---:|
| Elliptic | GCN | 5 | 0.6160 | 0.6080 ± 0.0119 | -0.0080 |
| Elliptic | GAT | 5 | 0.7660 | 0.7690 ± 0.0115 | +0.0030 |
| Elliptic | GraphSAGE | 5 | 0.8890 | 0.8762 ± 0.0075 | -0.0128 |
| Elliptic | ChebNet | 5 | 0.9100 | 0.8974 ± 0.0053 | -0.0126 |
| Elliptic | GATv2 | 5 | 0.8810 | 0.8688 ± 0.0048 | -0.0122 |

## Notes

- Dataset: Elliptic
- Feature setting: tx+agg
- Seeds: 42, 43, 44, 45, 46
- Metric: illicit-class F1
- Values are reported as mean ± standard deviation across five runs.