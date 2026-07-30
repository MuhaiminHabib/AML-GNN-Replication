from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


FRAUD_LABEL = 1
NORMAL_LABEL = 0


def _standardise_numeric(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    std = series.std()

    if std == 0 or np.isnan(std):
        return series * 0.0

    return (series - series.mean()) / std


def build_ibm_amlsim_data(
    data_dir: str | Path = "data/raw/ibm_amlsim",
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    temporal_split: bool = False,
    label_source: str = "accounts",
    include_fraud_tx_features: bool = False,
) -> Data:
    """
    Build a PyTorch Geometric account-level graph from IBM AMLSim-style CSV files.

    Expected files:
        accounts.csv
        transactions.csv

    Node:
        ACCOUNT_ID

    Edge:
        SENDER_ACCOUNT_ID -> RECEIVER_ACCOUNT_ID

    Main recommended label setting:
        label_source="accounts"

    Label options:
        label_source="accounts":
            Use accounts.IS_FRAUD directly.

        label_source="transactions":
            Mark an account as fraud if it appears in at least one fraudulent
            transaction.

        label_source="combined":
            Mark an account as fraud if either accounts.IS_FRAUD is True or the
            account appears in at least one fraudulent transaction.

    Feature leakage warning:
        include_fraud_tx_features=False by default.

        Do not use fraud_tx_sent_count or fraud_tx_received_count in the final
        experiment because they are derived from transactions.IS_FRAUD and may
        leak fraud-label information into the model.
    """

    data_dir = Path(data_dir)

    accounts_path = data_dir / "accounts.csv"
    transactions_path = data_dir / "transactions.csv"

    if not accounts_path.exists():
        raise FileNotFoundError(f"Missing accounts file: {accounts_path}")

    if not transactions_path.exists():
        raise FileNotFoundError(f"Missing transactions file: {transactions_path}")

    accounts = pd.read_csv(accounts_path)
    tx = pd.read_csv(transactions_path)

    required_account_cols = {
        "ACCOUNT_ID",
        "INIT_BALANCE",
        "COUNTRY",
        "ACCOUNT_TYPE",
        "IS_FRAUD",
        "TX_BEHAVIOR_ID",
    }

    required_tx_cols = {
        "SENDER_ACCOUNT_ID",
        "RECEIVER_ACCOUNT_ID",
        "TX_AMOUNT",
        "TIMESTAMP",
        "IS_FRAUD",
    }

    missing_accounts = required_account_cols - set(accounts.columns)
    missing_tx = required_tx_cols - set(tx.columns)

    if missing_accounts:
        raise ValueError(f"accounts.csv missing columns: {sorted(missing_accounts)}")

    if missing_tx:
        raise ValueError(f"transactions.csv missing columns: {sorted(missing_tx)}")

    accounts = accounts.copy()
    tx = tx.copy()

    label_source = label_source.lower()
    valid_label_sources = {"accounts", "transactions", "combined"}

    if label_source not in valid_label_sources:
        raise ValueError(
            f"Invalid label_source={label_source}. "
            f"Expected one of: {sorted(valid_label_sources)}"
        )

    accounts["ACCOUNT_ID"] = accounts["ACCOUNT_ID"].astype(int)
    tx["SENDER_ACCOUNT_ID"] = tx["SENDER_ACCOUNT_ID"].astype(int)
    tx["RECEIVER_ACCOUNT_ID"] = tx["RECEIVER_ACCOUNT_ID"].astype(int)

    accounts = accounts.sort_values("ACCOUNT_ID").reset_index(drop=True)

    account_ids = accounts["ACCOUNT_ID"].tolist()

    account_to_idx = {
        account_id: idx
        for idx, account_id in enumerate(account_ids)
    }

    # Keep only transactions where both sender and receiver exist in accounts.csv.
    tx = tx[
        tx["SENDER_ACCOUNT_ID"].isin(account_to_idx)
        & tx["RECEIVER_ACCOUNT_ID"].isin(account_to_idx)
    ].copy()

    tx["src_idx"] = tx["SENDER_ACCOUNT_ID"].map(account_to_idx).astype(int)
    tx["dst_idx"] = tx["RECEIVER_ACCOUNT_ID"].map(account_to_idx).astype(int)
    tx["TX_AMOUNT"] = tx["TX_AMOUNT"].astype(float)
    tx["TIMESTAMP"] = tx["TIMESTAMP"].astype(int)
    tx["IS_FRAUD"] = tx["IS_FRAUD"].astype(bool)

    num_accounts = len(accounts)

    # -------------------------------------------------------------------------
    # Transaction aggregation features at account/node level
    # -------------------------------------------------------------------------
    out_degree = (
        tx.groupby("src_idx")
        .size()
        .reindex(range(num_accounts), fill_value=0)
    )

    in_degree = (
        tx.groupby("dst_idx")
        .size()
        .reindex(range(num_accounts), fill_value=0)
    )

    total_sent = (
        tx.groupby("src_idx")["TX_AMOUNT"]
        .sum()
        .reindex(range(num_accounts), fill_value=0.0)
    )

    total_received = (
        tx.groupby("dst_idx")["TX_AMOUNT"]
        .sum()
        .reindex(range(num_accounts), fill_value=0.0)
    )

    mean_sent = (
        tx.groupby("src_idx")["TX_AMOUNT"]
        .mean()
        .reindex(range(num_accounts), fill_value=0.0)
    )

    mean_received = (
        tx.groupby("dst_idx")["TX_AMOUNT"]
        .mean()
        .reindex(range(num_accounts), fill_value=0.0)
    )

    agg_features_dict = {
        "out_degree": out_degree.values,
        "in_degree": in_degree.values,
        "total_sent_amount": total_sent.values,
        "total_received_amount": total_received.values,
        "mean_sent_amount": mean_sent.values,
        "mean_received_amount": mean_received.values,
    }

    # -------------------------------------------------------------------------
    # Optional leakage-prone fraud transaction features
    # -------------------------------------------------------------------------
    # These are disabled by default because they are derived from tx.IS_FRAUD.
    # Keep them only for debugging or ablation, not for the main experiment.
    # -------------------------------------------------------------------------
    if include_fraud_tx_features:
        fraud_tx = tx[tx["IS_FRAUD"]]

        fraud_sent_count = (
            fraud_tx.groupby("src_idx")
            .size()
            .reindex(range(num_accounts), fill_value=0)
        )

        fraud_received_count = (
            fraud_tx.groupby("dst_idx")
            .size()
            .reindex(range(num_accounts), fill_value=0)
        )

        agg_features_dict["fraud_tx_sent_count"] = fraud_sent_count.values
        agg_features_dict["fraud_tx_received_count"] = fraud_received_count.values

    agg_features = pd.DataFrame(agg_features_dict)

    # -------------------------------------------------------------------------
    # Account base features
    # -------------------------------------------------------------------------
    base_features = pd.DataFrame(index=accounts.index)

    base_features["init_balance"] = accounts["INIT_BALANCE"].astype(float)
    base_features["tx_behavior_id"] = accounts["TX_BEHAVIOR_ID"].astype(float)

    country_dummies = pd.get_dummies(
        accounts["COUNTRY"].astype(str),
        prefix="country",
        dtype=float,
    )

    account_type_dummies = pd.get_dummies(
        accounts["ACCOUNT_TYPE"].astype(str),
        prefix="account_type",
        dtype=float,
    )

    features = pd.concat(
        [
            base_features,
            country_dummies,
            account_type_dummies,
            agg_features,
        ],
        axis=1,
    )

    for col in features.columns:
        features[col] = _standardise_numeric(features[col])

    x = torch.tensor(features.values, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    # accounts label:
    #   accounts.IS_FRAUD gives account-level fraud labels directly.
    #
    # transaction label:
    #   an account is marked fraud if it appears in at least one fraudulent
    #   transaction as sender or receiver.
    #
    # combined label:
    #   fraud if either account-level label or transaction-derived label is fraud.
    # -------------------------------------------------------------------------
    account_label_np = accounts["IS_FRAUD"].astype(bool).astype(int).values

    tx_fraud = tx[tx["IS_FRAUD"]].copy()

    transaction_label_np = np.zeros(num_accounts, dtype=int)

    if len(tx_fraud) > 0:
        fraud_src = tx_fraud["src_idx"].astype(int).values
        fraud_dst = tx_fraud["dst_idx"].astype(int).values

        transaction_label_np[fraud_src] = 1
        transaction_label_np[fraud_dst] = 1

    if label_source == "accounts":
        y_np = account_label_np
    elif label_source == "transactions":
        y_np = transaction_label_np
    elif label_source == "combined":
        y_np = np.maximum(account_label_np, transaction_label_np)
    else:
        raise ValueError(f"Unsupported label_source: {label_source}")

    y = torch.tensor(y_np, dtype=torch.long)

    # -------------------------------------------------------------------------
    # Edge index and edge attributes
    # -------------------------------------------------------------------------
    edge_index = torch.tensor(
        np.vstack(
            [
                tx["src_idx"].values,
                tx["dst_idx"].values,
            ]
        ),
        dtype=torch.long,
    )

    edge_attr_df = pd.DataFrame(
        {
            "tx_amount": tx["TX_AMOUNT"].values,
            "timestamp": tx["TIMESTAMP"].values,
        }
    )

    # Do not include tx.IS_FRAUD as an edge feature in the final setup.
    # It is a transaction label and can leak fraud information.

    edge_attr_df["tx_amount"] = _standardise_numeric(edge_attr_df["tx_amount"])
    edge_attr_df["timestamp"] = _standardise_numeric(edge_attr_df["timestamp"])

    edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Split masks
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    indices = np.arange(num_accounts)

    if temporal_split:
        # Account-level temporal proxy:
        # accounts are ordered by first transaction timestamp.
        first_out = tx.groupby("src_idx")["TIMESTAMP"].min()
        first_in = tx.groupby("dst_idx")["TIMESTAMP"].min()

        first_seen = pd.concat([first_out, first_in], axis=1).min(axis=1)

        fallback_time = int(tx["TIMESTAMP"].max()) + 1

        first_seen = first_seen.reindex(
            range(num_accounts),
            fill_value=fallback_time,
        )

        sorted_indices = first_seen.sort_values().index.to_numpy()
    else:
        sorted_indices = indices.copy()
        rng.shuffle(sorted_indices)

    train_end = int(train_ratio * num_accounts)
    val_end = int((train_ratio + val_ratio) * num_accounts)

    train_idx = sorted_indices[:train_end]
    val_idx = sorted_indices[train_end:val_end]
    test_idx = sorted_indices[val_end:]

    train_mask = torch.zeros(num_accounts, dtype=torch.bool)
    val_mask = torch.zeros(num_accounts, dtype=torch.bool)
    test_mask = torch.zeros(num_accounts, dtype=torch.bool)

    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.account_ids = torch.tensor(account_ids, dtype=torch.long)
    data.feature_names = list(features.columns)

    data.num_transactions = int(tx.shape[0])
    data.num_fraud_transactions = int(tx["IS_FRAUD"].sum())

    data.num_fraud_accounts = int(y.sum())
    data.num_account_label_fraud_accounts = int(account_label_np.sum())
    data.num_transaction_label_fraud_accounts = int(transaction_label_np.sum())

    data.label_source = label_source
    data.temporal_split = bool(temporal_split)
    data.include_fraud_tx_features = bool(include_fraud_tx_features)

    return data


# Backward-compatible alias for older scripts.
# Your older code imports build_ibm_amlsim_graph, so we keep this name working.
def build_ibm_amlsim_graph(
    data_dir: str | Path = "data/raw/ibm_amlsim",
    seed: int = 42,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    temporal_split: bool = False,
    label_source: str = "accounts",
    include_fraud_tx_features: bool = False,
) -> Data:
    return build_ibm_amlsim_data(
        data_dir=data_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        temporal_split=temporal_split,
        label_source=label_source,
        include_fraud_tx_features=include_fraud_tx_features,
    )


def describe_ibm_amlsim_data(data: Data) -> dict:
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_node_features": int(data.num_features),
        "num_edge_features": int(data.edge_attr.size(1)) if data.edge_attr is not None else 0,
        "num_features": int(data.num_features),
        "num_transactions": int(data.num_transactions),
        "num_fraud_transactions": int(data.num_fraud_transactions),
        "num_fraud_accounts": int(data.num_fraud_accounts),
        "num_account_label_fraud_accounts": int(data.num_account_label_fraud_accounts),
        "num_transaction_label_fraud_accounts": int(data.num_transaction_label_fraud_accounts),
        "label_source": str(data.label_source),
        "include_fraud_tx_features": bool(data.include_fraud_tx_features),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
        "train_samples": int(data.train_mask.sum()),
        "val_samples": int(data.val_mask.sum()),
        "test_samples": int(data.test_mask.sum()),
        "train_fraud_accounts": int(data.y[data.train_mask].sum()),
        "val_fraud_accounts": int(data.y[data.val_mask].sum()),
        "test_fraud_accounts": int(data.y[data.test_mask].sum()),
        "train_fraud": int(data.y[data.train_mask].sum()),
        "val_fraud": int(data.y[data.val_mask].sum()),
        "test_fraud": int(data.y[data.test_mask].sum()),
        "temporal_split": bool(data.temporal_split),
    }