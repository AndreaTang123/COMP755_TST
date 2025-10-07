"""
Preprocess the Credit Card Fraud dataset in the same clean format as the census script.
Outputs a self-contained folder you can pass to evaluation scripts, e.g.:

    python3 creditcard_preprocess.py \
        --out-dir ./processed_creditcard \
        --test-size 0.5 \
        --seed 42

Generated artifacts in --out-dir:
  - X.csv, y.csv: full processed feature matrix/labels
  - train_X.csv, train_y.csv, test_X.csv, test_y.csv: stratified split
  - fraud_vs_normal_A.csv, fraud_vs_normal_B.csv: class-balanced two-sample pair
  - timeslice_A.csv, timeslice_B.csv: early vs late time-based two-sample pair
  - scaler.pkl: fitted StandardScaler (sklearn)
  - meta.json: basic dataset statistics and generation parameters

Notes:
  - Drops `Time` and standardizes all remaining numeric columns (Amount + V1..V28).
  - Uses stratified split by Class to preserve fraud ratio.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import kagglehub
from kagglehub import KaggleDatasetAdapter


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Credit Card Fraud dataset")
    p.add_argument("--out-dir", type=str, default="./processed_creditcard", help="Output directory for processed files")
    p.add_argument("--test-size", type=float, default=0.5, help="Test split size for train/test (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--time-frac", type=float, default=0.5, help="Fraction of earliest records to form timeslice_A (0-1)")
    return p.parse_args()


def ensure_outdir(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def basic_info(df: pd.DataFrame) -> dict:
    info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "null_cells": int(df.isnull().sum().sum()),
        "class_counts": df["Class"].value_counts().to_dict(),
    }
    return info


def load_creditcard_kaggle() -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(dataset_path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    return df


def load_and_clean(raw_path: str) -> tuple[pd.DataFrame, pd.Series]:
   
    df = load_creditcard_kaggle()
    # Separate features/labels
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    # Keep a copy of original Time for time-based slicing, then drop from features
    time_col = None
    if "Time" in X.columns:
        time_col = X["Time"].copy()
        X = X.drop(columns=["Time"])  # common practice for this dataset

    # Standardize numerical columns (Amount + V1..V28)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, time_col, scaler


def save_full(out_dir: str, X: pd.DataFrame, y: pd.Series):
    X.to_csv(os.path.join(out_dir, "X.csv"), index=False)
    y.to_csv(os.path.join(out_dir, "y.csv"), index=False)


def save_train_test(out_dir: str, X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train.to_csv(os.path.join(out_dir, "train_X.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "train_y.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "test_X.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "test_y.csv"), index=False)


def save_two_sample_class_balanced(out_dir: str, X: pd.DataFrame, y: pd.Series, seed: int):
    """Create matched-size fraud vs normal samples for two-sample tests.
    A = all fraud samples; B = randomly sampled normal with same count.
    """
    rng = np.random.default_rng(seed)
    fraud_idx = y[y == 1].index
    normal_idx = y[y == 0].index

    n_fraud = len(fraud_idx)
    if n_fraud == 0:
        return  # nothing to do
    normal_sample = rng.choice(normal_idx, size=n_fraud, replace=False)

    A = X.loc[fraud_idx]
    B = X.loc[normal_sample]

    A.to_csv(os.path.join(out_dir, "fraud_vs_normal_A.csv"), index=False)
    B.to_csv(os.path.join(out_dir, "fraud_vs_normal_B.csv"), index=False)


def save_two_sample_timeslice(out_dir: str, X: pd.DataFrame, y: pd.Series, time_col: pd.Series | None, time_frac: float):
    """Create early vs late time-based two-sample sets.
    We use the original (unscaled) Time column to split by chronological order.
    """
    if time_col is None:
        return

    # Order by time, split by fraction
    order = time_col.sort_values().index
    cutoff = int(len(order) * time_frac)
    early_idx = order[:cutoff]
    late_idx = order[cutoff:]

    X.loc[early_idx].to_csv(os.path.join(out_dir, "timeslice_A.csv"), index=False)
    X.loc[late_idx].to_csv(os.path.join(out_dir, "timeslice_B.csv"), index=False)


def save_meta(out_dir: str, raw_info: dict, args, scaler):
    meta = {
        "raw_info": raw_info,
        "params": {
            "test_size": args.test_size,
            "seed": args.seed,
            "time_frac": args.time_frac,
            "dropped_cols": ["Time"],
            "scaler": "StandardScaler",
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))


def main():
    args = parse_args()
    ensure_outdir(args.out_dir)

    raw_df = load_creditcard_kaggle()
    raw_info = basic_info(raw_df)

    X, y, time_col, scaler = load_and_clean("")

    # Save full, split, and two-sample variants
    save_full(args.out_dir, X, y)
    save_train_test(args.out_dir, X, y, args.test_size, args.seed)
    save_two_sample_class_balanced(args.out_dir, X, y, args.seed)
    save_two_sample_timeslice(args.out_dir, X, y, time_col, args.time_frac)
    save_meta(args.out_dir, raw_info, args, scaler)

    print("=== Credit Card Fraud data preprocessing done ===")
    print(f"Rows: {raw_info['n_rows']}, Cols: {raw_info['n_cols']}, Null cells: {raw_info['null_cells']}")
    print(f"Class counts: {raw_info['class_counts']}")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()