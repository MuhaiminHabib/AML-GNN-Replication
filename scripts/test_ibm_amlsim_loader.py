from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibm_amlsim import build_ibm_amlsim_data, describe_ibm_amlsim_data


def run_one(label_source: str):
    data = build_ibm_amlsim_data(
        data_dir="data/raw/ibm_amlsim",
        seed=42,
        temporal_split=False,
        label_source=label_source,
    )

    summary = describe_ibm_amlsim_data(data)

    print("\n" + "=" * 100)
    print(f"IBM AMLSim PyG graph summary | label_source={label_source}")
    print("=" * 100)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nData object:")
    print(data)

    print("\nFirst 20 feature names:")
    for name in data.feature_names[:20]:
        print(f"- {name}")


def main():
    for label_source in ["accounts", "transactions", "combined"]:
        run_one(label_source)


if __name__ == "__main__":
    main()