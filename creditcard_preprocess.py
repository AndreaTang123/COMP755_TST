"""
Preprocess the Credit Card Fraud dataset in the same clean format as the census script,
with the option to **not save anything** and only operate in-memory.

Usage examples:
    # Pure in-memory run (no files saved; prints summary only)
    python3 creditcard_preprocess.py \
        --test-size 0.5 \
        --seed 42 \
        --save-format none

    # Save small, compressed files only (recommended for GitHub)
    python3 creditcard_preprocess.py \
        --out-dir ./processed_creditcard \
        --test-size 0.5 \
        --seed 42 \
        --save-format npz

    # (Legacy) Save CSV artifacts (may exceed GitHub limits)
    python3 creditcard_preprocess.py \
        --out-dir ./processed_creditcard \
        --test-size 0.5 \
        --seed 42 \
        --save-format csv

Generated artifacts when saving:
  - For CSV: X.csv, y.csv, train_X.csv, train_y.csv, test_X.csv, test_y.csv
  - Two-sample CSVs (only when save-format in {csv, both}):
      fraud_vs_normal_A.csv, fraud_vs_normal_B.csv, timeslice_A.csv, timeslice_B.csv
  - For NPZ: data.npz, train_data.npz, test_data.npz (compressed)
  - scaler.pkl, meta.json (skipped when --save-format none)

Notes:
  - Drops `Time` and standardizes all remaining numeric columns (Amount + V1..V28).
  - Uses stratified split by Class to preserve fraud ratio.
  - Loads directly from Kaggle via kagglehub; no need to store a local CSV.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import kagglehub


def load_creditcard_kaggle() -> pd.DataFrame:
    """Download & load the Kaggle dataset into a DataFrame (no local CSV required)."""
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(dataset_path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    return df


def basic_info(df: pd.DataFrame) -> dict:
    info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "null_cells": int(df.isnull().sum().sum()),
        "class_counts": df["Class"].value_counts().to_dict(),
    }
    return info


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Return standardized features X, labels y, original Time column, and the scaler."""
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    time_col = None
    if "Time" in X.columns:
        time_col = X["Time"].copy()
        X = X.drop(columns=["Time"])  # common practice for this dataset

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return X, y, time_col, scaler



def get_processed(test_size: float = 0.5, seed: int = 42, time_frac: float = 0.5):
    """Convenience function to use from notebooks / other modules.
    Returns a dict containing in-memory artifacts (no file writes):
      {
        'X', 'y', 'time_col', 'scaler',
        'X_train', 'X_test', 'y_train', 'y_test'
      }
    """
    raw_df = load_creditcard_kaggle()
    X, y, time_col, scaler = preprocess(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return {
        "X": X, "y": y, "time_col": time_col, "scaler": scaler,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    }


def save_full_csv(out_dir: str, X: pd.DataFrame, y: pd.Series):
    X.to_csv(os.path.join(out_dir, "X.csv"), index=False)
    y.to_csv(os.path.join(out_dir, "y.csv"), index=False)


def save_train_test_csv(out_dir: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    X_train.to_csv(os.path.join(out_dir, "train_X.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "train_y.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "test_X.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "test_Y.csv"), index=False)


def save_two_sample_csvs(out_dir: str, X: pd.DataFrame, y: pd.Series, time_col: pd.Series | None, seed: int, time_frac: float):
    rng = np.random.default_rng(seed)
    fraud_idx = y[y == 1].index
    normal_idx = y[y == 0].index

    if len(fraud_idx) > 0:
        n_fraud = len(fraud_idx)
        normal_sample = rng.choice(normal_idx, size=n_fraud, replace=False)
        A = X.loc[fraud_idx]
        B = X.loc[normal_sample]
        A.to_csv(os.path.join(out_dir, "fraud_vs_normal_A.csv"), index=False)
        B.to_csv(os.path.join(out_dir, "fraud_vs_normal_B.csv"), index=False)

    if time_col is not None:
        order = time_col.sort_values().index
        cutoff = int(len(order) * time_frac)
        early_idx = order[:cutoff]
        late_idx = order[cutoff:]
        X.loc[early_idx].to_csv(os.path.join(out_dir, "timeslice_A.csv"), index=False)
        X.loc[late_idx].to_csv(os.path.join(out_dir, "timeslice_B.csv"), index=False)


def save_full_npz(out_dir: str, X: pd.DataFrame, y: pd.Series):
    np.savez_compressed(
        os.path.join(out_dir, "data.npz"),
        X=X.values.astype(np.float32),
        y=y.values.astype(np.int64),
        feature_names=np.array(X.columns, dtype=object),
    )


def save_train_test_npz(out_dir: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    np.savez_compressed(
        os.path.join(out_dir, "train_data.npz"),
        X=X_train.values.astype(np.float32),
        y=y_train.values.astype(np.int64),
        feature_names=np.array(X_train.columns, dtype=object),
    )
    np.savez_compressed(
        os.path.join(out_dir, "test_data.npz"),
        X=X_test.values.astype(np.float32),
        y=y_test.values.astype(np.int64),
        feature_names=np.array(X_test.columns, dtype=object),
    )


def save_meta(out_dir: str, raw_info: dict, args, scaler: StandardScaler):
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



def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Credit Card Fraud dataset")
    p.add_argument("--out-dir", type=str, default="./processed_creditcard", help="Output directory for processed files")
    p.add_argument("--test-size", type=float, default=0.5, help="Test split size for train/test (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--time-frac", type=float, default=0.5, help="Fraction of earliest records to form timeslice_A (0-1)")
    p.add_argument("--save-format", choices=["none", "csv", "npz", "both"], default="npz",
                   help="What to save: none (in-memory only), csv, npz, or both")
    return p.parse_args()


def ensure_outdir(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    # Load and preprocess (in-memory)
    raw_df = load_creditcard_kaggle()
    raw_info = basic_info(raw_df)
    X, y, time_col, scaler = preprocess(raw_df)

    # Split once (in-memory)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Optionally save to disk
    if args.save_format != "none":
        ensure_outdir(args.out_dir)
        if args.save_format in ("csv", "both"):
            save_full_csv(args.out_dir, X, y)
            save_train_test_csv(args.out_dir, X_train, y_train, X_test, y_test)
            # Two-sample CSVs can get large; write only when explicitly choosing CSV/both
            save_two_sample_csvs(args.out_dir, X, y, time_col, args.seed, args.time_frac)
        if args.save_format in ("npz", "both"):
            save_full_npz(args.out_dir, X, y)
            save_train_test_npz(args.out_dir, X_train, y_train, X_test, y_test)
            # Keep scaler + meta small and helpful
            save_meta(args.out_dir, raw_info, args, scaler)

    # Always print a concise summary
    print("=== Credit Card Fraud preprocessing (in-memory) ===")
    print(f"Rows: {raw_info['n_rows']}, Cols: {raw_info['n_cols']}, Null cells: {raw_info['null_cells']}")
    print(f"Class counts: {raw_info['class_counts']}")
    print(f"Train/Test: {len(X_train)}/{len(X_test)}  (test_size={args.test_size})")
    print(f"Save format: {args.save_format}")
    if args.save_format != "none":
        print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()