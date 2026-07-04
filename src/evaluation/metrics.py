from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_binary_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    positive_label should be the label representing the class of interest.

    For Weber-style setup:
        illicit = 1

    For Marasi-style setup:
        illicit = 0
    """
    if mask.sum().item() == 0:
        return {
            "accuracy": float("nan"),
            "illicit_precision": float("nan"),
            "illicit_recall": float("nan"),
            "illicit_f1": float("nan"),
            "micro_f1": float("nan"),
        }

    y_true = y[mask].detach().cpu().numpy()
    y_pred = logits[mask].argmax(dim=1).detach().cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "illicit_precision": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "illicit_recall": recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "illicit_f1": f1_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "micro_f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())