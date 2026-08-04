# Analyst alert queue summary

This table represents the first analyst-facing view. It lists the nodes/accounts selected for explanation, their model suspicion score, risk band, and suggested investigation action.

## Class-label note

- For **Elliptic**, suspicious/illicit class is represented as label **0** in this project output.
- For **AMLSim**, suspicious/fraud class is represented as label **1**.
- Therefore, the table uses dataset-specific suspicious-class mapping instead of assuming fraud is always label 1.

## Full alert queue

| Dataset   | Model     |   Alert rank |   Node ID | Suspicious class name   |   True label raw |   Predicted label raw | True suspicious?   | Predicted suspicious?   |   Suspicion probability | Risk band   | Recommended analyst action   |
|:----------|:----------|-------------:|----------:|:------------------------|-----------------:|----------------------:|:-------------------|:------------------------|------------------------:|:------------|:-----------------------------|
| AMLSim    | GATv2     |            1 |      1751 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            2 |      9440 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            3 |      2627 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            4 |      2454 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            5 |      7768 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            6 |      7615 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            7 |      8126 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            8 |      6969 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |            9 |      4651 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GATv2     |           10 |       914 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GCN       |            1 |      9999 | Fraud                   |                1 |                     1 | True               | True                    |                0.970542 | Critical    | Escalate immediately         |
| AMLSim    | GCN       |            2 |      9994 | Fraud                   |                1 |                     1 | True               | True                    |                0.960524 | Critical    | Escalate immediately         |
| AMLSim    | GCN       |            3 |      9981 | Fraud                   |                1 |                     1 | True               | True                    |                0.933286 | High        | Investigate with explanation |
| AMLSim    | GCN       |            4 |      9986 | Fraud                   |                1 |                     1 | True               | True                    |                0.931793 | High        | Investigate with explanation |
| AMLSim    | GCN       |            5 |      9971 | Fraud                   |                1 |                     1 | True               | True                    |                0.930659 | High        | Investigate with explanation |
| AMLSim    | GCN       |            6 |      9972 | Fraud                   |                1 |                     1 | True               | True                    |                0.927167 | High        | Investigate with explanation |
| AMLSim    | GCN       |            7 |      9685 | Fraud                   |                1 |                     1 | True               | True                    |                0.924698 | High        | Investigate with explanation |
| AMLSim    | GCN       |            8 |      4309 | Fraud                   |                1 |                     1 | True               | True                    |                0.919851 | High        | Investigate with explanation |
| AMLSim    | GCN       |            9 |      9201 | Fraud                   |                1 |                     1 | True               | True                    |                0.898769 | High        | Investigate with explanation |
| AMLSim    | GCN       |           10 |      9976 | Fraud                   |                1 |                     1 | True               | True                    |                0.896515 | High        | Investigate with explanation |
| AMLSim    | GraphSAGE |            1 |      9432 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            2 |      9985 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            3 |      9986 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            4 |      5998 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            5 |      5824 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            6 |      9854 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            7 |      5919 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            8 |      7847 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |            9 |      9994 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| AMLSim    | GraphSAGE |           10 |      4842 | Fraud                   |                1 |                     1 | True               | True                    |                1.000000 | Critical    | Escalate immediately         |
| Elliptic  | GATv2     |            1 |     29912 | Illicit                 |                0 |                     0 | True               | True                    |                0.999269 | Critical    | Escalate immediately         |
| Elliptic  | GATv2     |            2 |     18903 | Illicit                 |                0 |                     0 | True               | True                    |                0.986931 | Critical    | Escalate immediately         |
| Elliptic  | GATv2     |            3 |     19539 | Illicit                 |                0 |                     0 | True               | True                    |                0.979951 | Critical    | Escalate immediately         |
| Elliptic  | GATv2     |            4 |     18018 | Illicit                 |                0 |                     0 | True               | True                    |                0.972135 | Critical    | Escalate immediately         |
| Elliptic  | GATv2     |            5 |     21007 | Illicit                 |                0 |                     0 | True               | True                    |                0.961161 | Critical    | Escalate immediately         |
| Elliptic  | GATv2     |            6 |     24361 | Illicit                 |                0 |                     0 | True               | True                    |                0.941957 | High        | Investigate with explanation |
| Elliptic  | GATv2     |            7 |     17047 | Illicit                 |                0 |                     0 | True               | True                    |                0.918374 | High        | Investigate with explanation |
| Elliptic  | GATv2     |            8 |     24239 | Illicit                 |                0 |                     0 | True               | True                    |                0.880244 | High        | Investigate with explanation |
| Elliptic  | GATv2     |            9 |     18323 | Illicit                 |                0 |                     0 | True               | True                    |                0.851462 | High        | Investigate with explanation |
| Elliptic  | GATv2     |           10 |     17494 | Illicit                 |                0 |                     0 | True               | True                    |                0.829636 | High        | Investigate with explanation |
| Elliptic  | GCN       |            1 |     18018 | Illicit                 |                0 |                     0 | True               | True                    |                0.986804 | Critical    | Escalate immediately         |
| Elliptic  | GCN       |            2 |     25447 | Illicit                 |                0 |                     0 | True               | True                    |                0.892636 | High        | Investigate with explanation |
| Elliptic  | GCN       |            3 |     18323 | Illicit                 |                0 |                     0 | True               | True                    |                0.892485 | High        | Investigate with explanation |
| Elliptic  | GCN       |            4 |     17494 | Illicit                 |                0 |                     0 | True               | True                    |                0.745116 | Medium      | Monitor / secondary review   |
| Elliptic  | GCN       |            5 |     21518 | Illicit                 |                0 |                     0 | True               | True                    |                0.716764 | Medium      | Monitor / secondary review   |
| Elliptic  | GCN       |            6 |     18076 | Illicit                 |                0 |                     0 | True               | True                    |                0.693277 | Medium      | Monitor / secondary review   |
| Elliptic  | GraphSAGE |            1 |     29912 | Illicit                 |                0 |                     0 | True               | True                    |                0.984986 | Critical    | Escalate immediately         |
| Elliptic  | GraphSAGE |            2 |     19539 | Illicit                 |                0 |                     0 | True               | True                    |                0.941880 | High        | Investigate with explanation |
| Elliptic  | GraphSAGE |            3 |     18903 | Illicit                 |                0 |                     0 | True               | True                    |                0.939176 | High        | Investigate with explanation |
| Elliptic  | GraphSAGE |            4 |     18323 | Illicit                 |                0 |                     0 | True               | True                    |                0.848049 | High        | Investigate with explanation |
| Elliptic  | GraphSAGE |            5 |     17047 | Illicit                 |                0 |                     0 | True               | True                    |                0.843714 | High        | Investigate with explanation |
| Elliptic  | GraphSAGE |            6 |     24239 | Illicit                 |                0 |                     0 | True               | True                    |                0.781879 | Medium      | Monitor / secondary review   |
| Elliptic  | GraphSAGE |            7 |     25447 | Illicit                 |                0 |                     0 | True               | True                    |                0.585732 | Review      | Monitor / secondary review   |

## Alert count by dataset, model, and risk band

| Dataset   | Model     | Risk band   |   Count |
|:----------|:----------|:------------|--------:|
| AMLSim    | GATv2     | Critical    |      10 |
| AMLSim    | GCN       | Critical    |       2 |
| AMLSim    | GCN       | High        |       8 |
| AMLSim    | GraphSAGE | Critical    |      10 |
| Elliptic  | GATv2     | Critical    |       5 |
| Elliptic  | GATv2     | High        |       5 |
| Elliptic  | GCN       | Critical    |       1 |
| Elliptic  | GCN       | High        |       2 |
| Elliptic  | GCN       | Medium      |       3 |
| Elliptic  | GraphSAGE | Critical    |       1 |
| Elliptic  | GraphSAGE | High        |       4 |
| Elliptic  | GraphSAGE | Medium      |       1 |
| Elliptic  | GraphSAGE | Review      |       1 |

## How to read this

- **Suspicion probability** is the model's fraud/illicit confidence for the selected node.
- **Risk band** converts the probability into an analyst-friendly priority label.
- **Predicted suspicious?** tells whether the model predicted the suspicious class for that dataset.
- **Recommended analyst action** shows how the case could be handled in a bank investigation workflow.
- The next step after this table is to open the local graph explanation for a selected case.
