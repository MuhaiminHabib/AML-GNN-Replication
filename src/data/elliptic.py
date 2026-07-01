from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


@dataclass
class EllipticPaths:
    root: Path

    @property
    def features_path(self) -> Path:
        return self.root / "elliptic_txs_features.csv"

    @property
    def classes_path(self) -> Path:
        return self.root / "elliptic_txs_classes.csv"

    @property
    def edges_path(self) -> Path:
        return self.root / "elliptic_txs_edgelist.csv"


def _check_files(paths: EllipticPaths) -> None:
    missing = [
        str(path)
        for path in [paths.features_path, paths.classes_path, paths.edges_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Elliptic dataset files:\n" + "\n".join(missing)
        )


def load_elliptic_raw(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Elliptic CSV files.

    Expected files:
    - elliptic_txs_features.csv
    - elliptic_txs_classes.csv
    - elliptic_txs_edgelist.csv
    """
    paths = EllipticPaths(Path(data_dir))
    _check_files(paths)

    features = pd.read_csv(paths.features_path, header=None)
    classes = pd.read_csv(paths.classes_path)
    edges = pd.read_csv(paths.edges_path)

    # Assign feature column names.
    # Elliptic feature file normally has:
    # column 0 = txId
    # column 1 = time_step
    # remaining columns = anonymised transaction/graph features
    num_cols = features.shape[1]
    feature_names = ["txId", "time_step"] + [f"f_{i}" for i in range(num_cols - 2)]
    features.columns = feature_names

    return features, classes, edges


def build_elliptic_pyg_data(
    data_dir: str | Path = "data/raw/elliptic",
    make_undirected: bool = False,
    include_time_as_feature: bool = False,
) -> Data:
    """
    Build a PyTorch Geometric Data object for the Elliptic dataset.

    Labels:
    - illicit = 1
    - licit = 0
    - unknown = -1

    Unknown labels are kept in the graph but excluded from supervised masks.
    """
    features, classes, edges = load_elliptic_raw(data_dir)

    required_class_cols = {"txId", "class"}
    if not required_class_cols.issubset(set(classes.columns)):
        raise ValueError(
            f"Classes file must contain columns {required_class_cols}, "
            f"but got {list(classes.columns)}"
        )

    required_edge_cols = {"txId1", "txId2"}
    if not required_edge_cols.issubset(set(edges.columns)):
        raise ValueError(
            f"Edge file must contain columns {required_edge_cols}, "
            f"but got {list(edges.columns)}"
        )

    # Merge labels into feature table.
    df = features.merge(classes, on="txId", how="left")

    if df["class"].isna().any():
        raise ValueError("Some transactions in features file have no class entry.")

    # Map labels.
    # Elliptic convention:
    # class 1 = illicit
    # class 2 = licit
    # unknown = unknown
    label_map = {
        "1": 1,
        "2": 0,
        "unknown": -1,
        1: 1,
        2: 0,
    }

    df["label"] = df["class"].map(label_map)

    if df["label"].isna().any():
        bad_values = df.loc[df["label"].isna(), "class"].unique()
        raise ValueError(f"Unexpected class values found: {bad_values}")

    df["label"] = df["label"].astype(int)

    # Build node index.
    tx_ids = df["txId"].tolist()
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    # Filter edges to transactions available in the feature table.
    edge_df = edges[
        edges["txId1"].isin(tx_to_idx) & edges["txId2"].isin(tx_to_idx)
    ].copy()

    src = edge_df["txId1"].map(tx_to_idx).to_numpy()
    dst = edge_df["txId2"].map(tx_to_idx).to_numpy()

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    if make_undirected:
        edge_index = to_undirected(edge_index)

    # Build features.
    if include_time_as_feature:
        x_cols = ["time_step"] + [col for col in df.columns if col.startswith("f_")]
    else:
        x_cols = [col for col in df.columns if col.startswith("f_")]

    x = torch.tensor(df[x_cols].to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor(df["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    time_step = torch.tensor(df["time_step"].to_numpy(dtype=np.int64), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=time_step,
    )

    data.tx_ids = tx_ids
    data.num_labelled_nodes = int((y >= 0).sum().item())
    data.num_illicit_nodes = int((y == 1).sum().item())
    data.num_licit_nodes = int((y == 0).sum().item())
    data.num_unknown_nodes = int((y == -1).sum().item())
    data.make_undirected = make_undirected
    data.include_time_as_feature = include_time_as_feature

    return data


def add_weber_2019_masks(data: Data) -> Data:
    """
    Weber et al. 2019 split:
    - Train: time steps 1 to 34
    - Test: time steps 35 to 49
    - Unknown labels excluded from supervised training/testing
    """
    labelled = data.y >= 0

    train_mask = labelled & (data.time_step >= 1) & (data.time_step <= 34)
    test_mask = labelled & (data.time_step >= 35) & (data.time_step <= 49)

    data.train_mask = train_mask
    data.val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    data.test_mask = test_mask

    return data


def add_marasi_2024_masks(
    data: Data,
    seed: int = 42,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
) -> Data:
    """
    Marasi & Ferretti 2024-style split:
    - 65/15/20 train/validation/test
    - label proportions maintained
    - only labelled nodes are split
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    labelled_idx = torch.where(data.y >= 0)[0].cpu().numpy()
    labelled_y = data.y[labelled_idx].cpu().numpy()

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        labelled_idx,
        labelled_y,
        train_size=train_ratio,
        stratify=labelled_y,
        random_state=seed,
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_y,
        train_size=relative_val_ratio,
        stratify=temp_y,
        random_state=seed,
    )

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def describe_data(data: Data) -> dict:
    """
    Return a simple dataset summary.
    """
    summary = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "num_labelled_nodes": int((data.y >= 0).sum().item()),
        "num_illicit_nodes": int((data.y == 1).sum().item()),
        "num_licit_nodes": int((data.y == 0).sum().item()),
        "num_unknown_nodes": int((data.y == -1).sum().item()),
        "min_time_step": int(data.time_step.min().item()),
        "max_time_step": int(data.time_step.max().item()),
    }

    if hasattr(data, "train_mask"):
        summary.update(
            {
                "train_nodes": int(data.train_mask.sum().item()),
                "val_nodes": int(data.val_mask.sum().item()),
                "test_nodes": int(data.test_mask.sum().item()),
                "train_illicit": int(((data.y == 1) & data.train_mask).sum().item()),
                "train_licit": int(((data.y == 0) & data.train_mask).sum().item()),
                "val_illicit": int(((data.y == 1) & data.val_mask).sum().item()),
                "val_licit": int(((data.y == 0) & data.val_mask).sum().item()),
                "test_illicit": int(((data.y == 1) & data.test_mask).sum().item()),
                "test_licit": int(((data.y == 0) & data.test_mask).sum().item()),
            }
        )

    return summary