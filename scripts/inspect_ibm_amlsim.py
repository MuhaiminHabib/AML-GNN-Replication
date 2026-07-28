from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ibm_amlsim"

FILES = {
    "accounts": DATA_DIR / "accounts.csv",
    "alerts": DATA_DIR / "alerts.csv",
    "transactions": DATA_DIR / "transactions.csv",
}


def inspect_file(name, path):
    print("\n" + "=" * 80)
    print(f"{name.upper()} FILE")
    print("=" * 80)
    print(f"Path: {path}")

    if not path.exists():
        print("Missing file.")
        return

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nPossible label-like columns:")
    label_keywords = ["label", "target", "class", "fraud", "launder", "sar", "alert", "is_", "suspicious"]
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in label_keywords):
            print(f"\nColumn: {col}")
            print(df[col].value_counts(dropna=False).head(20).to_string())


def main():
    print(f"Inspecting IBM AMLSim dataset from: {DATA_DIR}")

    for name, path in FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()