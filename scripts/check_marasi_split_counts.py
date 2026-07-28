from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARASI_REPO = PROJECT_ROOT / "data" / "external_repos" / "aml-elliptic-gnn"

sys.path.insert(0, str(MARASI_REPO))

from loader import load_data, data_to_pyg


def count_mask(mask):
    return int(mask.sum().item())


def count_labels(data, mask):
    y = data.y[mask]
    illicit = int((y == 0).sum().item())
    licit = int((y == 1).sum().item())
    return illicit, licit


def main():
    data_path = PROJECT_ROOT / "data" / "raw" / "elliptic"

    features, edges = load_data(str(data_path))
    data = data_to_pyg(features, edges)

    train_n = count_mask(data.train_mask)
    val_n = count_mask(data.val_mask)
    test_n = count_mask(data.test_mask)

    train_illicit, train_licit = count_labels(data, data.train_mask)
    val_illicit, val_licit = count_labels(data, data.val_mask)
    test_illicit, test_licit = count_labels(data, data.test_mask)

    print("\nMarasi Elliptic tx+agg split counts")
    print("=" * 60)
    print(f"Train samples:      {train_n} | illicit: {train_illicit} | licit: {train_licit}")
    print(f"Validation samples: {val_n}  | illicit: {val_illicit}  | licit: {val_licit}")
    print(f"Test samples:       {test_n}  | illicit: {test_illicit}  | licit: {test_licit}")
    print(f"Total labelled:     {train_n + val_n + test_n}")


if __name__ == "__main__":
    main()