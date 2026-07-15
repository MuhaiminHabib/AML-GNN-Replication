from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_hi_small import build_ibm_hi_small_graph, describe_ibm_hi_small_data


def main():
    data = build_ibm_hi_small_graph()
    summary = describe_ibm_hi_small_data(data)

    print("\nIBM HI-Small graph summary")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nTensor shapes")
    print("=" * 80)
    print("x:", data.x.shape)
    print("edge_index:", data.edge_index.shape)
    print("y:", data.y.shape)
    print("train_mask:", int(data.train_mask.sum()))
    print("val_mask:", int(data.val_mask.sum()))
    print("test_mask:", int(data.test_mask.sum()))


if __name__ == "__main__":
    main()