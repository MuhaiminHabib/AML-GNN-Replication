from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.marasi_elliptic import (
    build_marasi_elliptic_data,
    describe_marasi_data,
)


def print_summary(title, summary):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")


def main():
    print("Checking Marasi-style Elliptic loader...")

    data_tx = build_marasi_elliptic_data(
        data_dir="data/raw/elliptic",
        feature_setting="tx",
        seed=42,
    )
    print_summary("Marasi Elliptic - tx only", describe_marasi_data(data_tx))

    data_tx_agg = build_marasi_elliptic_data(
        data_dir="data/raw/elliptic",
        feature_setting="tx+agg",
        seed=42,
    )
    print_summary("Marasi Elliptic - tx+agg", describe_marasi_data(data_tx_agg))


if __name__ == "__main__":
    main()