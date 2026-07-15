from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def _one_hot(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return enc.fit_transform(df[cols].astype(str))


def _safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(x.fillna(0).clip(lower=0))


def build_saml_d_graph(
    data_dir: str | Path = "data/raw/saml_d",
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.20,
    coalesce_edges: bool = True,
) -> Data:
    data_dir = Path(data_dir)
    csv_path = data_dir / "SAML-D.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    print(f"Loading SAML-D from {csv_path} ...")
    df = pd.read_csv(csv_path)

    required_cols = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
        "Laundering_type",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sender_accounts = df["Sender_account"].astype(str)
    receiver_accounts = df["Receiver_account"].astype(str)

    accounts = pd.Index(pd.concat([sender_accounts, receiver_accounts]).unique())
    account_to_idx = {account: i for i, account in enumerate(accounts)}
    num_nodes = len(accounts)

    print(f"Number of account nodes: {num_nodes:,}")
    print(f"Number of raw transaction edges: {len(df):,}")

    src = sender_accounts.map(account_to_idx).to_numpy()
    dst = receiver_accounts.map(account_to_idx).to_numpy()

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    if coalesce_edges:
        before_edges = edge_index.size(1)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        after_edges = edge_index.size(1)
        print(f"Coalesced edges: {before_edges:,} -> {after_edges:,}")
    else:
        print(f"Using raw edges: {edge_index.size(1):,}")

    # ------------------------------------------------------------------
    # Node label:
    # An account is positive if it appears as sender or receiver in at
    # least one laundering transaction.
    # ------------------------------------------------------------------
    y_np = np.zeros(num_nodes, dtype=np.int64)

    laundering_df = df[df["Is_laundering"] == 1]
    laundering_accounts = pd.Index(
        pd.concat(
            [
                laundering_df["Sender_account"].astype(str),
                laundering_df["Receiver_account"].astype(str),
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
    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )

    df["timestamp_seconds"] = timestamp.astype("int64") // 10**9
    df["hour"] = timestamp.dt.hour.fillna(0).astype(int)
    df["dayofweek"] = timestamp.dt.dayofweek.fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Transaction statistics by sender and receiver.
    # Do NOT use Is_laundering or Laundering_type as features.
    # ------------------------------------------------------------------
    sender_stats = df.groupby("Sender_account").agg(
        out_tx_count=("Amount", "count"),
        out_amount_sum=("Amount", "sum"),
        out_amount_mean=("Amount", "mean"),
        out_amount_std=("Amount", "std"),
        out_amount_min=("Amount", "min"),
        out_amount_max=("Amount", "max"),
        out_first_time=("timestamp_seconds", "min"),
        out_last_time=("timestamp_seconds", "max"),
        out_unique_receivers=("Receiver_account", "nunique"),
        out_unique_receiver_locations=("Receiver_bank_location", "nunique"),
    )

    receiver_stats = df.groupby("Receiver_account").agg(
        in_tx_count=("Amount", "count"),
        in_amount_sum=("Amount", "sum"),
        in_amount_mean=("Amount", "mean"),
        in_amount_std=("Amount", "std"),
        in_amount_min=("Amount", "min"),
        in_amount_max=("Amount", "max"),
        in_first_time=("timestamp_seconds", "min"),
        in_last_time=("timestamp_seconds", "max"),
        in_unique_senders=("Sender_account", "nunique"),
        in_unique_sender_locations=("Sender_bank_location", "nunique"),
    )

    sender_cat = df.groupby("Sender_account").agg(
        sender_payment_currency=("Payment_currency", lambda x: x.mode().iloc[0]),
        sender_received_currency=("Received_currency", lambda x: x.mode().iloc[0]),
        sender_payment_type=("Payment_type", lambda x: x.mode().iloc[0]),
        sender_bank_location=("Sender_bank_location", lambda x: x.mode().iloc[0]),
    )

    receiver_cat = df.groupby("Receiver_account").agg(
        receiver_payment_currency=("Payment_currency", lambda x: x.mode().iloc[0]),
        receiver_received_currency=("Received_currency", lambda x: x.mode().iloc[0]),
        receiver_payment_type=("Payment_type", lambda x: x.mode().iloc[0]),
        receiver_bank_location=("Receiver_bank_location", lambda x: x.mode().iloc[0]),
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
        if any(
            key in c
            for key in [
                "count",
                "amount",
                "sum",
                "mean",
                "std",
                "min",
                "max",
                "unique",
            ]
        )
    ]

    for col in skewed_cols:
        node_df[col] = _safe_log1p(node_df[col])

    node_df["total_tx_count"] = node_df["out_tx_count"] + node_df["in_tx_count"]
    node_df["total_amount_sum"] = node_df["out_amount_sum"] + node_df["in_amount_sum"]
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
        "sender_received_currency",
        "sender_payment_type",
        "sender_bank_location",
        "receiver_payment_currency",
        "receiver_received_currency",
        "receiver_payment_type",
        "receiver_bank_location",
    ]

    for col in categorical_cols:
        node_df[col] = node_df[col].fillna("missing").astype(str)

    cat_x = _one_hot(node_df, categorical_cols)

    x_np = np.hstack([numeric_x, cat_x]).astype(np.float32)

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

    data.dataset_name = "SAML-D"
    data.task_name = "account_node_classification"
    data.num_transactions = len(df)
    data.num_laundering_transactions = int(df["Is_laundering"].sum())
    data.coalesced_edges = bool(coalesce_edges)

    return data


def describe_saml_d_data(data: Data) -> dict:
    y = data.y.cpu()

    def count(mask, label):
        return int(((y == label) & mask.cpu()).sum())

    return {
        "dataset": "SAML-D",
        "task": "account node classification",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
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