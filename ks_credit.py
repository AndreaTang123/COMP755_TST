#!/usr/bin/env python3
"""
ks_credit.py
Univariate KS evaluation tailored for the Credit Card Fraud dataset.
Supports in-memory loading via creditcard_preprocess.get_processed (recommended),
or loading from saved NPZ/CSV artifacts.

Usage examples:
  # In-memory, fraud vs normal, default params
  python3 ks_credit.py --source inmemory --mode fraud_vs_normal --out-dir ./ks_credit_results

  # In-memory, time-slice comparison (early vs late)
  python3 ks_credit.py --source inmemory --mode timeslice --out-dir ./ks_credit_results

  # From NPZ
  python3 ks_credit.py --source npz --npz ./processed_creditcard/data.npz --mode fraud_vs_normal

  # From CSV directory
  python3 ks_credit.py --source csv --data-dir ./processed_creditcard --mode fraud_vs_normal

This script mirrors ks_eval.py but bakes in credit-card specific sampling helpers.
It reports Type I error (null splits) and power (alternative splits) using KS tests
aggregated across features via Fisher and Bonferroni.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, combine_pvalues
import matplotlib.pyplot as plt

# ---------- Data loading ----------

def load_inmemory(test_size: float, seed: int):
    """Load preprocessed credit-card data in-memory using get_processed()."""
    from creditcard_preprocess import get_processed
    bundle = get_processed(test_size=test_size, seed=seed)
    X = bundle["X"]; y = bundle["y"]
    time_col = bundle.get("time_col", None)
    feat_names = list(X.columns)
    return X.values, y.values.astype(int), feat_names, time_col


def load_npz(npz_path: str):
    arr = np.load(npz_path, allow_pickle=True)
    X = arr["X"]; y = arr["y"].astype(int)
    feat_names = [str(x) for x in arr["feature_names"]]
    # NPZ doesn't carry Time; timeslice mode won't work from plain data.npz
    time_col = None
    return X, y, feat_names, time_col


def load_csv_dir(data_dir: str):
    data_dir = Path(data_dir)
    X_df = pd.read_csv(data_dir / "X.csv")
    # y could be either single-column CSV or a column named 'label'/'Class'
    y_df = pd.read_csv(data_dir / "y.csv")
    if "label" in y_df.columns:
        y_series = y_df["label"]
    elif "Class" in y_df.columns:
        y_series = y_df["Class"]
    else:
        # first column fallback
        y_series = y_df.iloc[:, 0]
    X = X_df.values
    y = y_series.values.astype(int)
    feat_names = list(X_df.columns)
    # CSV path also lacks Time unless timeslice_A/B are present separately
    time_col = None
    return X, y, feat_names, time_col

# ---------- Sampling helpers ----------

def random_split_same_dist(X: np.ndarray, n_per_group: int, rng: np.random.Generator):
    """Null: A and B are sampled from the SAME distribution."""
    if 2 * n_per_group > X.shape[0]:
        raise ValueError("n_per_group too large for random split.")
    idx = rng.permutation(X.shape[0])
    return X[idx[:n_per_group], :], X[idx[n_per_group:2*n_per_group], :]


def class_split(X: np.ndarray, y: np.ndarray, n_per_group: int, rng: np.random.Generator,
                class_a: int = 1, class_b: int = 0):
    """Alternative: A=class_a, B=class_b with downsampling to equal size."""
    a_idx = np.where(y == class_a)[0]
    b_idx = np.where(y == class_b)[0]
    if len(a_idx) == 0 or len(b_idx) == 0:
        raise ValueError("Selected classes are empty.")
    n = min(n_per_group, len(a_idx), len(b_idx))
    A = X[rng.choice(a_idx, size=n, replace=False), :]
    B = X[rng.choice(b_idx, size=n, replace=False), :]
    return A, B


def timeslice_split_inmemory(X_df: pd.DataFrame, time_col: pd.Series, frac: float = 0.5):
    """Alternative: early vs late using original Time order (requires in-memory DF)."""
    order = time_col.sort_values().index
    cut = int(len(order) * frac)
    early_idx = order[:cut]
    late_idx = order[cut:]
    A = X_df.loc[early_idx].values
    B = X_df.loc[late_idx].values
    return A, B

# ---------- KS aggregation ----------

def ks_aggregate(A: np.ndarray, B: np.ndarray):
    """Aggregate per-feature KS into two combined p-values: Fisher & Bonferroni."""
    pvals = [ks_2samp(A[:, j], B[:, j]).pvalue for j in range(A.shape[1])]
    # Fisher's method (combines evidence across dimensions)
    _, p_fisher = combine_pvalues(pvals, method='fisher')
    # Bonferroni on the minimum p
    p_bonf = min(1.0, np.min(pvals) * len(pvals))
    return float(p_fisher), float(p_bonf)

# ---------- Plot helpers ----------

def bar_chart(labels, values, title, out_path):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Rate")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="KS two-sample evaluation for Credit Card dataset")
    ap.add_argument("--source", choices=["inmemory", "npz", "csv"], default="inmemory",
                    help="Where to load data from.")
    ap.add_argument("--npz", type=str, default=None, help="Path to data.npz when --source=npz")
    ap.add_argument("--data-dir", type=str, default=None, help="Directory containing X.csv/y.csv when --source=csv")

    ap.add_argument("--mode", choices=["fraud_vs_normal", "timeslice", "train_vs_test"], default="fraud_vs_normal",
                    help="Which alternative to test for power. Null is always same-distribution split.")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--n-per-group", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--test-size", type=float, default=0.5, help="Only used for inmemory loading")
    ap.add_argument("--time-frac", type=float, default=0.5, help="Early-vs-late split fraction (timeslice mode)")
    ap.add_argument("--out-dir", type=str, default="./ks_credit_results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load
    if args.source == "inmemory":
        Xy = load_inmemory(test_size=args.test_size, seed=args.seed)
    elif args.source == "npz":
        if not args.npz:
            raise ValueError("--npz is required when --source=npz")
        Xy = load_npz(args.npz)
    else:
        if not args.data_dir:
            raise ValueError("--data-dir is required when --source=csv")
        Xy = load_csv_dir(args.data_dir)

    X, y, feat_names, time_col = Xy

    # --- Null (Type I): random split from the same distribution ---
    null_reject_fisher, null_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = random_split_same_dist(X, args.n_per_group, rng)
        p_f, p_b = ks_aggregate(A, B)
        null_reject_fisher.append(p_f < args.alpha)
        null_reject_bonf.append(p_b < args.alpha)

    # --- Alternative (Power): based on requested mode ---
    alt_reject_fisher, alt_reject_bonf = [], []
    for _ in range(args.n_trials):
        if args.mode == "fraud_vs_normal":
            A, B = class_split(X, y, args.n_per_group, rng, class_a=1, class_b=0)
        elif args.mode == "train_vs_test":
            # random split simulating different shards (still same dist in expectation, so power low)
            # Keep for symmetry; you can adapt to use your persistent train/test if desired.
            A, B = random_split_same_dist(X, args.n_per_group, rng)
        else:  # timeslice
            if args.source != "inmemory" or time_col is None:
                raise ValueError("timeslice mode requires --source inmemory (it needs original Time column).")
            # Build A/B once per trial by jittering the fraction slightly to avoid degenerate repeats
            jitter = rng.uniform(-0.02, 0.02)
            frac = float(np.clip(args.time_frac + jitter, 0.1, 0.9))
            # Reconstruct DF to use index selection by time order
            X_df = pd.DataFrame(X, columns=feat_names)
            A, B = timeslice_split_inmemory(X_df, time_col, frac=frac)
            # Optionally subsample to n_per_group for consistency
            n = min(args.n_per_group, len(A), len(B))
            A = A[rng.choice(len(A), size=n, replace=False), :]
            B = B[rng.choice(len(B), size=n, replace=False), :]

        p_f, p_b = ks_aggregate(A, B)
        alt_reject_fisher.append(p_f < args.alpha)
        alt_reject_bonf.append(p_b < args.alpha)

    # --- Summarize ---
    summary = pd.DataFrame([
        {
            "alpha": args.alpha,
            "n_trials": args.n_trials,
            "n_per_group": args.n_per_group,
            "typeI_fisher": float(np.mean(null_reject_fisher)),
            "typeI_bonf": float(np.mean(null_reject_bonf)),
            "power_fisher": float(np.mean(alt_reject_fisher)),
            "power_bonf": float(np.mean(alt_reject_bonf)),
            "mode": args.mode,
            "source": args.source,
        }
    ])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "results.csv", index=False)

    # Plots
    def _bar(labels, values, title, fname):
        plt.figure()
        plt.bar(labels, values)
        plt.title(title)
        plt.ylabel("Rate")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()

    _bar(["KS (Fisher)", "KS (Bonf.)"],
         [summary["typeI_fisher"].iloc[0], summary["typeI_bonf"].iloc[0]],
         f"KS Type I Error (alpha={args.alpha})", "typeI_bar.png")

    _bar(["KS (Fisher)", "KS (Bonf.)"],
         [summary["power_fisher"].iloc[0], summary["power_bonf"].iloc[0]],
         f"KS Power ({args.mode})", "power_bar.png")

    print("=== KS (credit) evaluation done ===")
    print("Saved:", (out_dir / "results.csv").resolve())


if __name__ == "__main__":
    main()
