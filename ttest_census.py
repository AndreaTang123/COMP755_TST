#!/usr/bin/env python3
"""
ttest_eval.py
Run Welch's t-tests across features for two-sample testing on preprocessed datasets.
See header for usage examples.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, combine_pvalues
import matplotlib.pyplot as plt

def load_data(data_dir: str | None, npz_path: str | None):
    if npz_path is not None:
        p = Path(npz_path)
        if p.is_file():
            arr = np.load(p, allow_pickle=True)
            X = arr["X"]; y = arr["y"]
            feat_names = [str(x) for x in arr["feature_names"]]
            return X, y, feat_names
        elif p.is_dir():
            from scipy import sparse
            X_sp = p / "X_sparse.npz"
            y_npy = p / "y.npy"
            fn_npy = p / "feature_names.npy"
            if X_sp.exists() and y_npy.exists() and fn_npy.exists():
                X = sparse.load_npz(X_sp).toarray()
                y = np.load(y_npy)
                feat_names = np.load(fn_npy, allow_pickle=True).tolist()
                feat_names = [str(x) for x in feat_names]
                return X, y, feat_names
            else:
                raise ValueError(f"Directory '{p}' does not contain expected files: X_sparse.npz, y.npy, feature_names.npy")
        else:
            raise ValueError(f"--npz path '{p}' not found.")
    elif data_dir is not None:
        data_dir = Path(data_dir)
        X_df = pd.read_csv(data_dir / 'X.csv')
        y_df = pd.read_csv(data_dir / 'y.csv')
        X = X_df.values
        y = y_df['label'].values
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
def is_binary_column(col: np.ndarray) -> bool:
    vals = np.unique(col[~np.isnan(col)])
    return len(vals) <= 2 and set(vals).issubset({0.0, 1.0})

def safe_t_pvalues(A: np.ndarray, B: np.ndarray):
    pvals, tstats, means0, means1 = [], [], [], []
    for j in range(A.shape[1]):
        a = A[:, j]; b = B[:, j]
        # Drop NaNs
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if a.size < 2 or b.size < 2:
            pvals.append(1.0); tstats.append(0.0); means0.append(np.nan); means1.append(np.nan); continue
        # If either side has zero variance, ttest_ind returns nan; handle it.
        if np.var(a) == 0 or np.var(b) == 0:
            # If means equal, no signal; else treat as very small variance difference -> use large t via Welch? Safer: set p=1 when no variance.
            pvals.append(1.0); tstats.append(0.0); means0.append(np.mean(a)); means1.append(np.mean(b)); continue
        res = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        pvals.append(float(res.pvalue))
        tstats.append(float(res.statistic))
        means0.append(float(np.mean(a))); means1.append(float(np.mean(b)))
    return np.array(pvals), np.array(tstats), np.array(means0), np.array(means1)

def t_aggregate(A, B):
    pvals, tstats, m0, m1 = safe_t_pvalues(A, B)
    # Replace NaNs with 1.0 (non-significant) to avoid crashing aggregators
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    # Guard: if all pvals are 1 (no informative features), return neutral results
    if pvals.size == 0:
        return 1.0, 1.0, pvals, tstats, m0, m1
    _, p_fisher = combine_pvalues(pvals, method='fisher')
    p_bonf = float(min(1.0, np.nanmin(pvals) * len(pvals)))
    return float(p_fisher), float(p_bonf), pvals, tstats, m0, m1

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
    parser = argparse.ArgumentParser(description="Welch's t-test two-sample evaluation with Monte Carlo.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--npz", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-per-group", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--class-a", type=int, default=0)
    parser.add_argument("--class-b", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out-dir", type=str, default="./ttest_results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y, feat_names = load_data(args.data_dir, args.npz)
    col_var = np.var(X, axis=0)
    keep = col_var > 0.0
    if not np.all(keep):
        X = X[:, keep]
        feat_names = [f for f, k in zip(feat_names, keep) if k]

    null_reject_fisher, null_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = random_split_same_dist(X, args.n_per_group, rng)
        p_f, p_b, _, _, _, _ = t_aggregate(A, B)
        null_reject_fisher.append(p_f < args.alpha)
        null_reject_bonf.append(p_b < args.alpha)

    alt_reject_fisher, alt_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = class_split(X, y, args.class_a, args.class_b, args.n_per_group, rng)
        p_f, p_b, pvals, tstats, m0, m1 = t_aggregate(A, B)
        alt_reject_fisher.append(p_f < args.alpha)
        alt_reject_bonf.append(p_b < args.alpha)

    if _ == 0:
            diag = pd.DataFrame({
                'feature': feat_names,
                't_stat': tstats,
                'p_value': pvals,
                'mean_classA': m0,
                'mean_classB': m1
            })
            diag.to_csv(Path(args.out_dir) / 't_feature_diagnostics.csv', index=False)

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

    bar_chart(["t (Fisher)", "t (Bonferroni)"],
              [summary["typeI_fisher"].iloc[0], summary["typeI_bonf"].iloc[0]],
              f"t-test Type I Error (alpha={args.alpha})",
              out_dir / "typeI_bar.png")
    bar_chart(["t (Fisher)", "t (Bonferroni)"],
              [summary["power_fisher"].iloc[0], summary["power_bonf"].iloc[0]],
              f"t-test Power (class {args.class_a} vs {args.class_b})",
              out_dir / "power_bar.png")

    print("=== t-test evaluation done ===")
    print("Saved:", (out_dir / "results.csv").resolve())

if __name__ == "__main__":
    main()
