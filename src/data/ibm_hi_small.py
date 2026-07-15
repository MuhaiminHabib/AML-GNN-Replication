from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data


def _one_hot(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return enc.fit_transform(df[cols].astype(str))


def _safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(x.fillna(0).clip(lower=0))


def build_ibm_hi_small_graph(
    data_dir: str | Path = "data/raw/ibm_transactions_aml",
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.20,
) -> Data:
    data_dir = Path(data_dir)
    csv_path = data_dir / "HI-Small_Trans.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    print(f"Loading IBM HI-Small from {csv_path} ...")
    df = pd.read_csv(csv_path)

    required_cols = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.rename(
        columns={
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "Account": "sender_account",
            "To Bank": "to_bank",
            "Account.1": "receiver_account",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )

    # ------------------------------------------------------------------
    # Nodes = all unique accounts appearing as sender or receiver.
    # ------------------------------------------------------------------
    sender_accounts = df["sender_account"].astype(str)
    receiver_accounts = df["receiver_account"].astype(str)

    accounts = pd.Index(pd.concat([sender_accounts, receiver_accounts]).unique())
    account_to_idx = {account: i for i, account in enumerate(accounts)}
    num_nodes = len(accounts)

    print(f"Number of account nodes: {num_nodes:,}")
    print(f"Number of transaction edges: {len(df):,}")

    src = sender_accounts.map(account_to_idx).to_numpy()
    dst = receiver_accounts.map(account_to_idx).to_numpy()

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # ------------------------------------------------------------------
    # Node label:
    # An account is laundering-related if it appears in at least one
    # laundering transaction as sender or receiver.
    # ------------------------------------------------------------------
    y_np = np.zeros(num_nodes, dtype=np.int64)

    laundering_df = df[df["is_laundering"] == 1]
    laundering_accounts = pd.Index(
        pd.concat(
            [
                laundering_df["sender_account"].astype(str),
                laundering_df["receiver_account"].astype(str),
            ]
        ).unique()
    )

    laundering_idx = [
        account_to_idx[a]
        for a in laundering_accounts
        if a in account_to_idx
    ]
    y_np[laundering_idx] = 1

    # ------------------------------------------------------------------
    # Timestamp features.
    # ------------------------------------------------------------------
    timestamp = pd.to_datetime(df["timestamp"])
    df["timestamp_seconds"] = timestamp.astype("int64") // 10**9
    df["hour"] = timestamp.dt.hour
    df["dayofweek"] = timestamp.dt.dayofweek

    # ------------------------------------------------------------------
    # Transaction statistics by sender and receiver.
    # Do NOT use is_laundering as a feature.
    # ------------------------------------------------------------------
    sender_stats = df.groupby("sender_account").agg(
        out_tx_count=("amount_paid", "count"),
        out_paid_sum=("amount_paid", "sum"),
        out_paid_mean=("amount_paid", "mean"),
        out_paid_std=("amount_paid", "std"),
        out_paid_min=("amount_paid", "min"),
        out_paid_max=("amount_paid", "max"),
        out_received_sum=("amount_received", "sum"),
        out_first_time=("timestamp_seconds", "min"),
        out_last_time=("timestamp_seconds", "max"),
        out_unique_receivers=("receiver_account", "nunique"),
        out_unique_to_banks=("to_bank", "nunique"),
    )

    receiver_stats = df.groupby("receiver_account").agg(
        in_tx_count=("amount_received", "count"),
        in_received_sum=("amount_received", "sum"),
        in_received_mean=("amount_received", "mean"),
        in_received_std=("amount_received", "std"),
        in_received_min=("amount_received", "min"),
        in_received_max=("amount_received", "max"),
        in_paid_sum=("amount_paid", "sum"),
        in_first_time=("timestamp_seconds", "min"),
        in_last_time=("timestamp_seconds", "max"),
        in_unique_senders=("sender_account", "nunique"),
        in_unique_from_banks=("from_bank", "nunique"),
    )

    # Only low-cardinality categorical features.
    # Do NOT one-hot encode bank IDs because there are thousands of them.
    sender_cat = df.groupby("sender_account").agg(
        sender_payment_currency=("payment_currency", lambda x: x.mode().iloc[0]),
        sender_payment_format=("payment_format", lambda x: x.mode().iloc[0]),
    )

    receiver_cat = df.groupby("receiver_account").agg(
        receiver_currency=("receiving_currency", lambda x: x.mode().iloc[0]),
    )

    node_df = pd.DataFrame(index=accounts)
    node_df.index.name = "account"

    node_df = node_df.join(sender_stats, how="left")
    node_df = node_df.join(receiver_stats, how="left")
    node_df = node_df.join(sender_cat, how="left")
    node_df = node_df.join(receiver_cat, how="left")

    numeric_cols = [
        c for c in node_df.columns
        if node_df[c].dtype.kind in {"i", "u", "f"}
    ]

    node_df[numeric_cols] = node_df[numeric_cols].fillna(0)

    skewed_cols = [
        c for c in numeric_cols
        if any(key in c for key in ["count", "sum", "mean", "std", "min", "max", "unique"])
    ]

    for col in skewed_cols:
        node_df[col] = _safe_log1p(node_df[col])

    node_df["total_tx_count"] = node_df["out_tx_count"] + node_df["in_tx_count"]
    node_df["total_amount_sum"] = node_df["out_paid_sum"] + node_df["in_received_sum"]
    node_df["out_in_count_ratio"] = node_df["out_tx_count"] / (node_df["in_tx_count"] + 1.0)
    node_df["in_out_count_ratio"] = node_df["in_tx_count"] / (node_df["out_tx_count"] + 1.0)

    numeric_cols = [
        c for c in node_df.columns
        if node_df[c].dtype.kind in {"i", "u", "f"}
    ]

    scaler = StandardScaler()
    numeric_x = scaler.fit_transform(node_df[numeric_cols].fillna(0))

    categorical_cols = [
        "sender_payment_currency",
        "sender_payment_format",
        "receiver_currency",
    ]

    for col in categorical_cols:
        node_df[col] = node_df[col].fillna("missing").astype(str)

    cat_x = _one_hot(node_df, categorical_cols)

    x_np = np.hstack([numeric_x, cat_x]).astype(np.float32)

    # ------------------------------------------------------------------
    # Stratified train/val/test split over account labels.
    # ------------------------------------------------------------------
    all_idx = np.arange(num_nodes)

    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_np,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
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
        x=torch.tensor(x_np, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_np, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.dataset_name = "IBM HI-Small"
    data.task_name = "account_node_classification"
    data.num_transactions = len(df)
    data.num_laundering_transactions = int(df["is_laundering"].sum())

    return data


def describe_ibm_hi_small_data(data: Data) -> dict:
    y = data.y.cpu()

    def count(mask, label):
        return int(((y == label) & mask.cpu()).sum())

    return {
        "dataset": "IBM HI-Small",
        "task": "account node classification",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
        "num_transactions": int(data.num_transactions),
        "num_laundering_transactions": int(data.num_laundering_transactions),
        "num_laundering_accounts": int((y == 1).sum()),
        "num_normal_accounts": int((y == 0).sum()),
        "train_samples": int(data.train_mask.sum()),
        "train_laundering": count(data.train_mask, 1),
        "train_normal": count(data.train_mask, 0),
        "val_samples": int(data.val_mask.sum()),
        "val_laundering": count(data.val_mask, 1),
        "val_normal": count(data.val_mask, 0),
        "test_samples": int(data.test_mask.sum()),
        "test_laundering": count(data.test_mask, 1),
        "test_normal": count(data.test_mask, 0),
    }