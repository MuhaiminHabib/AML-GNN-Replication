from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit


def _load_raw_elliptic(data_dir: str | Path):
    data_dir = Path(data_dir)

    features_path = data_dir / "elliptic_txs_features.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not classes_path.exists():
        raise FileNotFoundError(classes_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    feature_cols = ["txId", "time_step"] + [
        f"f_{i}" for i in range(features.shape[1] - 2)
    ]
    features.columns = feature_cols

    return features, classes, edges


def build_marasi_elliptic_data(
    data_dir: str | Path = "data/raw/elliptic",
    feature_setting: str = "tx+agg",
    seed: int = 42,
) -> Data:
    """
    Build Elliptic graph using the Marasi/Ferretti public-code style.

    Important:
    - Unknown-labelled transactions are removed.
    - Labels follow Marasi convention:
        illicit = 0
        licit   = 1
    - feature_setting:
        "tx"     = transaction/local features only
        "tx+agg" = transaction/local + aggregate features
    - Split uses PyG RandomNodeSplit(num_val=0.15, num_test=0.2),
      matching the public repository.
    """
    if feature_setting not in {"tx", "tx+agg"}:
        raise ValueError("feature_setting must be either 'tx' or 'tx+agg'")

    torch.manual_seed(seed)

    features, classes, edges = _load_raw_elliptic(data_dir)

    required_class_cols = {"txId", "class"}
    required_edge_cols = {"txId1", "txId2"}

    if not required_class_cols.issubset(classes.columns):
        raise ValueError(f"Class file must contain {required_class_cols}")

    if not required_edge_cols.issubset(edges.columns):
        raise ValueError(f"Edge file must contain {required_edge_cols}")

    # Keep only known labels.
    labelled_classes = classes[classes["class"].astype(str) != "unknown"].copy()

    # Marasi-style label convention:
    # illicit = 0, licit = 1
    label_map = {
        "1": 0,
        "2": 1,
        1: 0,
        2: 1,
    }
    labelled_classes["label"] = labelled_classes["class"].map(label_map)

    if labelled_classes["label"].isna().any():
        bad_values = labelled_classes[labelled_classes["label"].isna()]["class"].unique()
        raise ValueError(f"Unexpected class values: {bad_values}")

    df = features.merge(
        labelled_classes[["txId", "label"]],
        on="txId",
        how="inner",
    ).copy()

    # Keep labelled transaction IDs only.
    known_tx_ids = set(df["txId"].tolist())

    edges = edges[
        edges["txId1"].isin(known_tx_ids) & edges["txId2"].isin(known_tx_ids)
    ].copy()

    # Map transaction IDs to contiguous node indices.
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(df["txId"].tolist())}

    edges["src"] = edges["txId1"].map(tx_to_idx)
    edges["dst"] = edges["txId2"].map(tx_to_idx)

    edge_index = torch.tensor(
        [edges["src"].values, edges["dst"].values],
        dtype=torch.long,
    )

    # Elliptic has 165 features:
    # first 93 are transaction/local features,
    # last 72 are aggregate features.
    all_feature_cols = [c for c in df.columns if c.startswith("f_")]

    if len(all_feature_cols) != 165:
        raise ValueError(f"Expected 165 features, found {len(all_feature_cols)}")

    if feature_setting == "tx":
        selected_feature_cols = all_feature_cols[:93]
    else:
        selected_feature_cols = all_feature_cols

    x = torch.tensor(
        df[selected_feature_cols].values,
        dtype=torch.float,
    )

    y = torch.tensor(
        df["label"].values,
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    data.tx_ids = df["txId"].tolist()
    data.time_step = torch.tensor(df["time_step"].values, dtype=torch.long)
    data.feature_setting = feature_setting
    data.label_convention = "Marasi: illicit=0, licit=1"

    splitter = RandomNodeSplit(num_val=0.15, num_test=0.20)
    data = splitter(data)

    return data


def describe_marasi_data(data: Data) -> Dict[str, int | str]:
    summary = {
        "feature_setting": data.feature_setting,
        "label_convention": data.label_convention,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_node_features),
        "num_illicit_nodes_label_0": int((data.y == 0).sum().item()),
        "num_licit_nodes_label_1": int((data.y == 1).sum().item()),
    }

    for split_name in ["train", "val", "test"]:
        mask = getattr(data, f"{split_name}_mask")
        y_split = data.y[mask]

        summary[f"{split_name}_nodes"] = int(mask.sum().item())
        summary[f"{split_name}_illicit_label_0"] = int((y_split == 0).sum().item())
        summary[f"{split_name}_licit_label_1"] = int((y_split == 1).sum().item())

    return summary