#!/usr/bin/env python3
"""
ks_eval.py
Run univariate KS tests across features for two-sample testing on preprocessed datasets.
See header for usage examples.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, combine_pvalues
import matplotlib.pyplot as plt

def load_data(data_dir: str | None, npz_path: str | None):
    if npz_path is not None:
        arr = np.load(npz_path, allow_pickle=True)
        X = arr["X"]; y = arr["y"]
        feat_names = [str(x) for x in arr["feature_names"]]
        return X, y, feat_names
    elif data_dir is not None:
        data_dir = Path(data_dir)
        X_df = pd.read_csv(data_dir / "X.csv")
        y_df = pd.read_csv(data_dir / "y.csv")
        X = X_df.values
        y = y_df["label"].values
        feat_names = list(X_df.columns)
        return X, y, feat_names
    else:
        raise ValueError("Provide either --data-dir or --npz.")

def random_split_same_dist(X, n_per_group, rng):
    idx = rng.permutation(X.shape[0])
    return X[idx[:n_per_group], :], X[idx[n_per_group:2*n_per_group], :]

def class_split(X, y, class_a, class_b, n_per_group, rng):
    a_idx = np.where(y == class_a)[0]
    b_idx = np.where(y == class_b)[0]
    if len(a_idx) < n_per_group or len(b_idx) < n_per_group:
        raise ValueError("Not enough samples in selected classes for requested n_per_group.")
    A = rng.choice(a_idx, size=n_per_group, replace=False)
    B = rng.choice(b_idx, size=n_per_group, replace=False)
    return X[A, :], X[B, :]

def ks_aggregate(A, B):
    pvals = [ks_2samp(A[:, j], B[:, j]).pvalue for j in range(A.shape[1])]
    _, p_fisher = combine_pvalues(pvals, method='fisher')
    p_bonf = min(1.0, np.min(pvals) * len(pvals))
    return float(p_fisher), float(p_bonf)

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

def main():
    parser = argparse.ArgumentParser(description="KS two-sample evaluation with Monte Carlo.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-per-group", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--class-a", type=int, default=0)
    parser.add_argument("--class-b", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", type=str, default="./ks_results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y, feat_names = load_data(args.data_dir, args.npz)

    null_reject_fisher, null_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = random_split_same_dist(X, args.n_per_group, rng)
        p_f, p_b = ks_aggregate(A, B)
        null_reject_fisher.append(p_f < args.alpha)
        null_reject_bonf.append(p_b < args.alpha)

    alt_reject_fisher, alt_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = class_split(X, y, args.class_a, args.class_b, args.n_per_group, rng)
        p_f, p_b = ks_aggregate(A, B)
        alt_reject_fisher.append(p_f < args.alpha)
        alt_reject_bonf.append(p_b < args.alpha)

    summary = pd.DataFrame([{
        "alpha": args.alpha,
        "n_trials": args.n_trials,
        "n_per_group": args.n_per_group,
        "typeI_fisher": float(np.mean(null_reject_fisher)),
        "typeI_bonf": float(np.mean(null_reject_bonf)),
        "power_fisher": float(np.mean(alt_reject_fisher)),
        "power_bonf": float(np.mean(alt_reject_bonf)),
        "class_a": args.class_a,
        "class_b": args.class_b
    }])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "results.csv", index=False)

    bar_chart(["KS (Fisher)", "KS (Bonferroni)"],
              [summary["typeI_fisher"].iloc[0], summary["typeI_bonf"].iloc[0]],
              f"KS Type I Error (alpha={args.alpha})",
              out_dir / "typeI_bar.png")
    bar_chart(["KS (Fisher)", "KS (Bonferroni)"],
              [summary["power_fisher"].iloc[0], summary["power_bonf"].iloc[0]],
              f"KS Power (class {args.class_a} vs {args.class_b})",
              out_dir / "power_bar.png")

    print("=== KS evaluation done ===")
    print("Saved:", (out_dir / "results.csv").resolve())

if __name__ == "__main__":
    main()
