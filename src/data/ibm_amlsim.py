from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data


def _one_hot_encode(df: pd.DataFrame, columns):
    if not columns:
        return np.empty((len(df), 0), dtype=np.float32)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[columns])
    return encoded.astype(np.float32)


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.maximum(values, 0.0)
    return np.log1p(values)


def build_ibm_amlsim_graph(
    data_dir: str | Path = "data/raw/ibm_amlsim",
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.20,
) -> Data:
    """
    Build a full IBM AMLSim account-level graph.

    Nodes:
        Accounts from accounts.csv.

    Edges:
        Directed transactions from sender account to receiver account.

    Label:
        accounts.csv -> IS_FRAUD
        normal = 0, fraud = 1

    Important:
        This loader intentionally does NOT use transaction IS_FRAUD or ALERT_ID
        as node features, because those would introduce label leakage.
    """

    data_dir = Path(data_dir)

    accounts_path = data_dir / "accounts.csv"
    transactions_path = data_dir / "transactions.csv"

    accounts = pd.read_csv(accounts_path)
    transactions = pd.read_csv(transactions_path)

    accounts = accounts.sort_values("ACCOUNT_ID").reset_index(drop=True)

    account_ids = accounts["ACCOUNT_ID"].to_numpy()
    account_to_idx = {account_id: idx for idx, account_id in enumerate(account_ids)}

    # Target label only. Do not include this in features.
    y_np = accounts["IS_FRAUD"].astype(bool).astype(int).to_numpy()

    # Account-level numeric features.
    numeric_features = accounts[["INIT_BALANCE", "TX_BEHAVIOR_ID"]].copy()

    # Account-level categorical features.
    categorical_x = _one_hot_encode(accounts, ["COUNTRY", "ACCOUNT_TYPE"])

    # Transaction-derived features.
    # Do NOT use IS_FRAUD or ALERT_ID here.
    outgoing_stats = (
        transactions
        .groupby("SENDER_ACCOUNT_ID")
        .agg(
            out_tx_count=("TX_ID", "count"),
            out_amount_sum=("TX_AMOUNT", "sum"),
            out_amount_mean=("TX_AMOUNT", "mean"),
            out_amount_std=("TX_AMOUNT", "std"),
            out_amount_min=("TX_AMOUNT", "min"),
            out_amount_max=("TX_AMOUNT", "max"),
            out_first_time=("TIMESTAMP", "min"),
            out_last_time=("TIMESTAMP", "max"),
            out_unique_receivers=("RECEIVER_ACCOUNT_ID", "nunique"),
        )
    )

    incoming_stats = (
        transactions
        .groupby("RECEIVER_ACCOUNT_ID")
        .agg(
            in_tx_count=("TX_ID", "count"),
            in_amount_sum=("TX_AMOUNT", "sum"),
            in_amount_mean=("TX_AMOUNT", "mean"),
            in_amount_std=("TX_AMOUNT", "std"),
            in_amount_min=("TX_AMOUNT", "min"),
            in_amount_max=("TX_AMOUNT", "max"),
            in_first_time=("TIMESTAMP", "min"),
            in_last_time=("TIMESTAMP", "max"),
            in_unique_senders=("SENDER_ACCOUNT_ID", "nunique"),
        )
    )

    stats = pd.DataFrame({"ACCOUNT_ID": account_ids}).set_index("ACCOUNT_ID")
    stats = stats.join(outgoing_stats, how="left").join(incoming_stats, how="left")
    stats = stats.fillna(0.0)

    # Extra structural ratios.
    stats["total_tx_count"] = stats["out_tx_count"] + stats["in_tx_count"]
    stats["total_amount_sum"] = stats["out_amount_sum"] + stats["in_amount_sum"]
    stats["out_in_count_ratio"] = stats["out_tx_count"] / (stats["in_tx_count"] + 1.0)
    stats["in_out_count_ratio"] = stats["in_tx_count"] / (stats["out_tx_count"] + 1.0)

    # Log-transform skewed count/amount features.
    skewed_cols = [
        "out_tx_count",
        "out_amount_sum",
        "out_amount_mean",
        "out_amount_std",
        "out_amount_min",
        "out_amount_max",
        "out_unique_receivers",
        "in_tx_count",
        "in_amount_sum",
        "in_amount_mean",
        "in_amount_std",
        "in_amount_min",
        "in_amount_max",
        "in_unique_senders",
        "total_tx_count",
        "total_amount_sum",
    ]

    for col in skewed_cols:
        stats[col] = _safe_log1p(stats[col].to_numpy())

    # Scale numeric features.
    scaler = StandardScaler()
    account_numeric_x = scaler.fit_transform(numeric_features).astype(np.float32)

    stats_scaler = StandardScaler()
    stats_x = stats_scaler.fit_transform(stats.to_numpy()).astype(np.float32)

    x_np = np.concatenate([account_numeric_x, categorical_x, stats_x], axis=1)

    # Full directed transaction graph.
    src = transactions["SENDER_ACCOUNT_ID"].map(account_to_idx)
    dst = transactions["RECEIVER_ACCOUNT_ID"].map(account_to_idx)

    valid_edges = src.notna() & dst.notna()
    src = src[valid_edges].astype(int).to_numpy()
    dst = dst[valid_edges].astype(int).to_numpy()

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.long)

    num_nodes = len(accounts)
    indices = np.arange(num_nodes)

    # Stratified node split.
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y_np,
    )

    adjusted_val_size = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=y_np[train_val_idx],
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.dataset_name = "IBM AMLSim"
    data.task_name = "account_node_classification"
    data.label_meaning = "normal=0, fraud=1"

    return data


def describe_ibm_amlsim_data(data: Data) -> dict:
    def mask_count(mask):
        return int(mask.sum().item())

    def fraud_count(mask):
        return int((data.y[mask] == 1).sum().item())

    def normal_count(mask):
        return int((data.y[mask] == 0).sum().item())

    return {
        "dataset": "IBM AMLSim",
        "task": "account node classification",
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_features,
        "num_fraud_nodes": int((data.y == 1).sum().item()),
        "num_normal_nodes": int((data.y == 0).sum().item()),
        "train_samples": mask_count(data.train_mask),
        "train_fraud": fraud_count(data.train_mask),
        "train_normal": normal_count(data.train_mask),
        "val_samples": mask_count(data.val_mask),
        "val_fraud": fraud_count(data.val_mask),
        "val_normal": normal_count(data.val_mask),
        "test_samples": mask_count(data.test_mask),
        "test_fraud": fraud_count(data.test_mask),
        "test_normal": normal_count(data.test_mask),
    }