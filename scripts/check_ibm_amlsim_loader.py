from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim import build_ibm_amlsim_graph, describe_ibm_amlsim_data


def main():
    data = build_ibm_amlsim_graph(
        data_dir=PROJECT_ROOT / "data" / "raw" / "ibm_amlsim",
        seed=42,
    )

    summary = describe_ibm_amlsim_data(data)

    print("\nIBM AMLSim graph summary")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDevice / tensor checks")
    print("=" * 60)
    print(f"x shape: {data.x.shape}")
    print(f"edge_index shape: {data.edge_index.shape}")
    print(f"y shape: {data.y.shape}")
    print(f"train_mask sum: {int(data.train_mask.sum())}")
    print(f"val_mask sum: {int(data.val_mask.sum())}")
    print(f"test_mask sum: {int(data.test_mask.sum())}")


if __name__ == "__main__":
    main()