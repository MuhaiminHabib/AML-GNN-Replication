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


def build_ibm_amlsim_graph(
    data_dir: str | Path = "data/raw/ibm_amlsim",
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.20,
    include_fraud_tx_count_features: bool = False,
) -> Data:
    """
    Previous-style IBM AMLSim account-level node classification graph.

    Graph design:
        node = account
        edge = transaction from sender account to receiver account
        label = account-level IS_FRAUD from accounts.csv

    Split:
        stratified 65/15/20 over account labels
        train = 6500, val = 1500, test = 2000 for 10,000 accounts

    Main recovered previous-code behaviour:
        - uses account static features
        - uses transaction aggregation features
        - uses StandardScaler for numeric features
        - uses OneHotEncoder for categorical features
        - does NOT use transactions.IS_FRAUD as edge_attr
        - does NOT use ALERT_ID as feature
        - optional fraud transaction count features can be enabled for reproduction tests
    """

    data_dir = Path(data_dir)

    accounts_path = data_dir / "accounts.csv"
    transactions_path = data_dir / "transactions.csv"
    alerts_path = data_dir / "alerts.csv"

    if not accounts_path.exists():
        raise FileNotFoundError(f"Could not find {accounts_path}")

    if not transactions_path.exists():
        raise FileNotFoundError(f"Could not find {transactions_path}")

    if not alerts_path.exists():
        raise FileNotFoundError(f"Could not find {alerts_path}")

    print(f"Loading accounts from: {accounts_path}")
    print(f"Loading transactions from: {transactions_path}")

    accounts = pd.read_csv(accounts_path)
    transactions = pd.read_csv(transactions_path)

    required_account_cols = [
        "ACCOUNT_ID",
        "CUSTOMER_ID",
        "INIT_BALANCE",
        "COUNTRY",
        "ACCOUNT_TYPE",
        "IS_FRAUD",
        "TX_BEHAVIOR_ID",
    ]

    required_transaction_cols = [
        "TX_ID",
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
        "TX_TYPE",
        "TX_AMOUNT",
        "TIMESTAMP",
        "IS_FRAUD",
        "ALERT_ID",
    ]

    missing_accounts = [
        col for col in required_account_cols
        if col not in accounts.columns
    ]

    missing_transactions = [
        col for col in required_transaction_cols
        if col not in transactions.columns
    ]

    if missing_accounts:
        raise ValueError(f"Missing account columns: {missing_accounts}")

    if missing_transactions:
        raise ValueError(f"Missing transaction columns: {missing_transactions}")

    accounts = accounts.copy()
    transactions = transactions.copy()

    accounts["ACCOUNT_ID"] = accounts["ACCOUNT_ID"].astype(str)
    transactions["SENDER_ACCOUNT_ID"] = transactions["SENDER_ACCOUNT_ID"].astype(str)
    transactions["RECEIVER_ACCOUNT_ID"] = transactions["RECEIVER_ACCOUNT_ID"].astype(str)

    transactions["TX_AMOUNT"] = transactions["TX_AMOUNT"].astype(float)
    transactions["IS_FRAUD"] = transactions["IS_FRAUD"].astype(bool)

    # ------------------------------------------------------------------
    # Nodes = accounts from accounts.csv
    # ------------------------------------------------------------------
    account_ids = accounts["ACCOUNT_ID"].tolist()
    account_to_idx = {
        account_id: idx
        for idx, account_id in enumerate(account_ids)
    }

    num_nodes = len(account_ids)

    print(f"Number of account nodes: {num_nodes:,}")
    print(f"Number of transaction rows before filtering: {len(transactions):,}")

    # Keep only transactions where both accounts exist in accounts.csv.
    transactions = transactions[
        transactions["SENDER_ACCOUNT_ID"].isin(account_to_idx)
        & transactions["RECEIVER_ACCOUNT_ID"].isin(account_to_idx)
    ].copy()

    print(f"Number of transaction edges after filtering: {len(transactions):,}")

    src = transactions["SENDER_ACCOUNT_ID"].map(account_to_idx).to_numpy()
    dst = transactions["RECEIVER_ACCOUNT_ID"].map(account_to_idx).to_numpy()

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
    )

    # ------------------------------------------------------------------
    # Labels: account-level fraud label from accounts.csv
    # ------------------------------------------------------------------
    y_np = accounts["IS_FRAUD"].astype(bool).astype(np.int64).to_numpy()

    # ------------------------------------------------------------------
    # Timestamp features
    # ------------------------------------------------------------------
    # Keep robust handling because AMLSim TIMESTAMP can be integer-like.
    if np.issubdtype(transactions["TIMESTAMP"].dtype, np.number):
        transactions["timestamp_seconds"] = transactions["TIMESTAMP"].astype(float)
        transactions["hour"] = 0
        transactions["dayofweek"] = 0
    else:
        timestamp = pd.to_datetime(transactions["TIMESTAMP"], errors="coerce")
        transactions["timestamp_seconds"] = timestamp.astype("int64") // 10**9
        transactions["hour"] = timestamp.dt.hour.fillna(0).astype(int)
        transactions["dayofweek"] = timestamp.dt.dayofweek.fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Transaction-derived account statistics
    # Do not use ALERT_ID as feature.
    # Do not use transactions.IS_FRAUD directly as edge feature.
    # ------------------------------------------------------------------
    sender_stats = transactions.groupby("SENDER_ACCOUNT_ID").agg(
        out_tx_count=("TX_AMOUNT", "count"),
        out_amount_sum=("TX_AMOUNT", "sum"),
        out_amount_mean=("TX_AMOUNT", "mean"),
        out_amount_std=("TX_AMOUNT", "std"),
        out_amount_min=("TX_AMOUNT", "min"),
        out_amount_max=("TX_AMOUNT", "max"),
        out_first_time=("timestamp_seconds", "min"),
        out_last_time=("timestamp_seconds", "max"),
        out_unique_receivers=("RECEIVER_ACCOUNT_ID", "nunique"),
        out_unique_tx_types=("TX_TYPE", "nunique"),
    )

    receiver_stats = transactions.groupby("RECEIVER_ACCOUNT_ID").agg(
        in_tx_count=("TX_AMOUNT", "count"),
        in_amount_sum=("TX_AMOUNT", "sum"),
        in_amount_mean=("TX_AMOUNT", "mean"),
        in_amount_std=("TX_AMOUNT", "std"),
        in_amount_min=("TX_AMOUNT", "min"),
        in_amount_max=("TX_AMOUNT", "max"),
        in_first_time=("timestamp_seconds", "min"),
        in_last_time=("timestamp_seconds", "max"),
        in_unique_senders=("SENDER_ACCOUNT_ID", "nunique"),
        in_unique_tx_types=("TX_TYPE", "nunique"),
    )

    sender_cat = transactions.groupby("SENDER_ACCOUNT_ID").agg(
        sender_dominant_tx_type=("TX_TYPE", lambda x: x.mode().iloc[0]),
    )

    receiver_cat = transactions.groupby("RECEIVER_ACCOUNT_ID").agg(
        receiver_dominant_tx_type=("TX_TYPE", lambda x: x.mode().iloc[0]),
    )

    # ------------------------------------------------------------------
    # Optional fraud transaction count features for previous-result testing
    # ------------------------------------------------------------------
    if include_fraud_tx_count_features:
        fraud_transactions = transactions[transactions["IS_FRAUD"]].copy()

        fraud_sender_stats = fraud_transactions.groupby("SENDER_ACCOUNT_ID").agg(
            fraud_tx_sent_count=("TX_AMOUNT", "count"),
        )

        fraud_receiver_stats = fraud_transactions.groupby("RECEIVER_ACCOUNT_ID").agg(
            fraud_tx_received_count=("TX_AMOUNT", "count"),
        )
    else:
        fraud_sender_stats = None
        fraud_receiver_stats = None

    # ------------------------------------------------------------------
    # Account static features
    # ------------------------------------------------------------------
    node_df = accounts.set_index("ACCOUNT_ID").copy()

    # Do not keep label as feature.
    node_df = node_df.drop(columns=["IS_FRAUD"])

    node_df = node_df.join(sender_stats, how="left")
    node_df = node_df.join(receiver_stats, how="left")
    node_df = node_df.join(sender_cat, how="left")
    node_df = node_df.join(receiver_cat, how="left")

    if include_fraud_tx_count_features:
        node_df = node_df.join(fraud_sender_stats, how="left")
        node_df = node_df.join(fraud_receiver_stats, how="left")

    numeric_cols = [
        col for col in node_df.columns
        if node_df[col].dtype.kind in {"i", "u", "f", "b"}
    ]

    node_df[numeric_cols] = node_df[numeric_cols].fillna(0)

    skewed_cols = [
        col for col in numeric_cols
        if any(
            key in col.lower()
            for key in [
                "balance",
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

    # Extra ratio features
    node_df["total_tx_count"] = node_df["out_tx_count"] + node_df["in_tx_count"]
    node_df["total_amount_sum"] = node_df["out_amount_sum"] + node_df["in_amount_sum"]
    node_df["out_in_count_ratio"] = node_df["out_tx_count"] / (node_df["in_tx_count"] + 1.0)
    node_df["in_out_count_ratio"] = node_df["in_tx_count"] / (node_df["out_tx_count"] + 1.0)

    if include_fraud_tx_count_features:
        node_df["total_fraud_tx_count"] = (
            node_df["fraud_tx_sent_count"] + node_df["fraud_tx_received_count"]
        )

    numeric_cols = [
        col for col in node_df.columns
        if node_df[col].dtype.kind in {"i", "u", "f", "b"}
    ]

    scaler = StandardScaler()
    numeric_x = scaler.fit_transform(node_df[numeric_cols].fillna(0))

    categorical_cols = [
        "COUNTRY",
        "ACCOUNT_TYPE",
        "sender_dominant_tx_type",
        "receiver_dominant_tx_type",
    ]

    for col in categorical_cols:
        if col not in node_df.columns:
            node_df[col] = "missing"

        node_df[col] = node_df[col].fillna("missing").astype(str)

    cat_x = _one_hot(node_df, categorical_cols)

    x_np = np.hstack([numeric_x, cat_x]).astype(np.float32)

    # ------------------------------------------------------------------
    # Stratified train/val/test split over account labels
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
        x=torch.tensor(x_np, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_np, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.dataset_name = "IBM AMLSim"
    data.task_name = "account_node_classification"
    data.num_transactions = int(len(transactions))
    data.num_fraud_transactions = int(transactions["IS_FRAUD"].astype(bool).sum())
    data.include_fraud_tx_count_features = bool(include_fraud_tx_count_features)
    data.feature_names = list(numeric_cols) + list(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        .fit(node_df[categorical_cols].astype(str))
        .get_feature_names_out(categorical_cols)
    )

    return data


def describe_ibm_amlsim_data(data: Data) -> dict:
    y = data.y.cpu()

    def count(mask, label):
        return int(((y == label) & mask.cpu()).sum())

    return {
        "dataset": "IBM AMLSim",
        "task": "account node classification",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
        "num_node_features": int(data.x.size(1)),
        "num_transactions": int(data.num_transactions),
        "num_fraud_transactions": int(data.num_fraud_transactions),
        "include_fraud_tx_count_features": bool(data.include_fraud_tx_count_features),
        "num_fraud_nodes": int((y == 1).sum()),
        "num_normal_nodes": int((y == 0).sum()),
        "train_samples": int(data.train_mask.sum()),
        "train_fraud": count(data.train_mask, 1),
        "train_normal": count(data.train_mask, 0),
        "val_samples": int(data.val_mask.sum()),
        "val_fraud": count(data.val_mask, 1),
        "val_normal": count(data.val_mask, 0),
        "test_samples": int(data.test_mask.sum()),
        "test_fraud": count(data.test_mask, 1),
        "test_normal": count(data.test_mask, 0),
    }
