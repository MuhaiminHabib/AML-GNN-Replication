from pathlib import Path
import pandas as pd

csv_path = Path("data/raw/saml_d/SAML-D.csv")

print("Reading first 5 rows...")
df_head = pd.read_csv(csv_path, nrows=5)

print("\nColumns:")
for i, col in enumerate(df_head.columns):
    print(f"{i}: {col}")

print("\nFirst 5 rows:")
print(df_head)

print("\nReading selected summary information...")
df = pd.read_csv(csv_path)

print("\nShape:")
print(df.shape)

print("\nDtypes:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nPossible label columns:")
for col in df.columns:
    col_lower = col.lower()
    if any(x in col_lower for x in ["label", "fraud", "launder", "suspicious", "target", "typology"]):
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(30))

print("\nUnique counts:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")