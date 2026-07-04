# Marasi & Ferretti 2024 Public-Code Replication Results

## Run status

The public repository was run successfully on the Elliptic dataset using CUDA.

Repository inspected:

- Repository: https://github.com/simonemarasi/aml-elliptic-gnn
- Commit inspected: b02c6e6be51becf2147ce372002fef8998f17b21

## Result summary

The repository trains two settings for each model:

- `tx`: transaction/local features only
- `tx+agg`: transaction/local + aggregate features

The `tx` results are close to the repository README values.

The `tx+agg` results are close to the main published paper values.

## Reproduced results

| Model | Feature setting | Precision | Recall | F1 | Micro-F1 |
|---|---|---:|---:|---:|---:|
| GCN | tx | 0.8450 | 0.4480 | 0.5860 | 0.9400 |
| GCN | tx+agg | 0.7930 | 0.5030 | 0.6160 | 0.9400 |
| GAT | tx | 0.8250 | 0.6610 | 0.7340 | 0.9540 |
| GAT | tx+agg | 0.8240 | 0.7260 | 0.7720 | 0.9590 |
| GraphSAGE | tx | 0.9510 | 0.7660 | 0.8490 | 0.9740 |
| GraphSAGE | tx+agg | 0.9400 | 0.8440 | 0.8900 | 0.9800 |
| ChebNet | tx | 0.9600 | 0.7650 | 0.8520 | 0.9750 |
| ChebNet | tx+agg | 0.9540 | 0.8640 | 0.9070 | 0.9830 |
| GATv2 | tx | 0.8840 | 0.7970 | 0.8380 | 0.9710 |
| GATv2 | tx+agg | 0.8870 | 0.8680 | 0.8770 | 0.9770 |

## Main replication conclusion

The `tx+agg` setting successfully reproduces the main Marasi & Ferretti baseline results.

The reproduced F1 scores are:

| Model | Reproduced F1 |
|---|---:|
| GCN | 0.6160 |
| GAT | 0.7720 |
| GraphSAGE | 0.8900 |
| ChebNet | 0.9070 |
| GATv2 | 0.8770 |

These values are close to the reported paper values and can be treated as a successful baseline replication.

## Notes

The public repository originally failed under pandas 2.x because `DataFrame.append()` has been removed. A local compatibility fix was applied using `pd.concat()`.

A pull request was opened to the original repository:

- PR title: Fix pandas 2 compatibility by replacing DataFrame.append