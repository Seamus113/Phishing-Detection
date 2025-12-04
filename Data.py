import pandas as pd
from typing import Tuple
import os


def load_phishing_dataset(csv_path: str = "C:/Users/styu0/OneDrive/Desktop/25 full/privacy/project/dataset.csv") -> Tuple[pd.DataFrame, pd.Series]:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("Original columns:", df.columns.tolist())
    print("Original shape:", df.shape)

    # remove unnecessary columns 
    drop_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "index" or col_lower.startswith("unnamed"):
            drop_cols.append(col)

    if drop_cols:
        print("Removing columns:", drop_cols)
        df = df.drop(columns=drop_cols, errors="ignore")

    # check label column
    if "Result" not in df.columns:
        raise ValueError("Label column 'Result' not found in dataset.")

    # extract features
    y_raw = df["Result"]
    X = df.drop(columns=["Result"])

    # 1 -> 1, -1 -> 0
    y = (y_raw == 1).astype(int)

    print("Processed shape:")
    print("  X:", X.shape)
    print("  y distribution:", y.value_counts().to_dict())

    return X, y
