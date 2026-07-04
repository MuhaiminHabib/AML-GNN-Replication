from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.marasi_elliptic import build_marasi_elliptic_data
from src.models.marasi_models import build_marasi_model


def main():
    data = build_marasi_elliptic_data(
        data_dir="data/raw/elliptic",
        feature_setting="tx+agg",
        seed=42,
    )

    model_names = ["gcn", "gat", "graphsage", "chebnet", "gatv2"]

    print("Checking Marasi model forward passes...")

    for model_name in model_names:
        model = build_marasi_model(
            model_name=model_name,
            in_channels=data.num_node_features,
            hidden_channels=110,
            out_channels=2,
        )

        out = model(data.x, data.edge_index)

        print(
            f"{model_name}: output shape = {tuple(out.shape)}"
        )

        assert out.shape[0] == data.num_nodes
        assert out.shape[1] == 2

    print("All Marasi model checks passed.")


if __name__ == "__main__":
    main()