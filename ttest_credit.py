"""
Welch's t-test two-sample evaluation tailored for the Credit Card Fraud dataset.
Supports in-memory loading via creditcard_preprocess.get_processed (recommended),
or loading from saved NPZ/CSV artifacts.

Usage examples:
  # In-memory, fraud vs normal
  python3 ttest_credit.py --source inmemory --mode fraud_vs_normal --out-dir ./ttest_credit_results

  # In-memory, time-slice comparison (early vs late)
  python3 ttest_credit.py --source inmemory --mode timeslice --out-dir ./ttest_credit_results

  # From NPZ
  python3 ttest_credit.py --source npz --npz ./processed_creditcard/data.npz --mode fraud_vs_normal

  # From CSV directory
  python3 ttest_credit.py --source csv --data-dir ./processed_creditcard --mode fraud_vs_normal
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, combine_pvalues
import matplotlib.pyplot as plt

# ---------------- Data loading ----------------

def load_inmemory(test_size: float, seed: int):
    """Load preprocessed credit-card data in-memory using get_processed()."""
    from creditcard_preprocess import get_processed
    bundle = get_processed(test_size=test_size, seed=seed)
    X_df = bundle["X"]; y = bundle["y"].values.astype(int)
    time_col = bundle.get("time_col", None)
    feat_names = list(X_df.columns)
    return X_df.values, y, feat_names, time_col


def load_npz(npz_path: str):
    arr = np.load(npz_path, allow_pickle=True)
    X = arr["X"]; y = arr["y"].astype(int)
    feat_names = [str(x) for x in arr["feature_names"]]
    time_col = None
    return X, y, feat_names, time_col


def load_csv_dir(data_dir: str):
    data_dir = Path(data_dir)
    X_df = pd.read_csv(data_dir / "X.csv")
    y_df = pd.read_csv(data_dir / "y.csv")
    if "label" in y_df.columns:
        y_series = y_df["label"]
    elif "Class" in y_df.columns:
        y_series = y_df["Class"]
    else:
        y_series = y_df.iloc[:, 0]
    X = X_df.values
    y = y_series.values.astype(int)
    feat_names = list(X_df.columns)
    time_col = None
    return X, y, feat_names, time_col

# ---------------- Sampling helpers ----------------

def random_split_same_dist(X: np.ndarray, n_per_group: int, rng: np.random.Generator):
    if 2 * n_per_group > X.shape[0]:
        raise ValueError("n_per_group too large for random split.")
    idx = rng.permutation(X.shape[0])
    return X[idx[:n_per_group], :], X[idx[n_per_group:2*n_per_group], :]


def class_split(X: np.ndarray, y: np.ndarray, n_per_group: int, rng: np.random.Generator,
                class_a: int = 1, class_b: int = 0):
    a_idx = np.where(y == class_a)[0]
    b_idx = np.where(y == class_b)[0]
    if len(a_idx) == 0 or len(b_idx) == 0:
        raise ValueError("Selected classes are empty.")
    n = min(n_per_group, len(a_idx), len(b_idx))
    A = X[rng.choice(a_idx, size=n, replace=False), :]
    B = X[rng.choice(b_idx, size=n, replace=False), :]
    return A, B


def timeslice_split_inmemory(X: np.ndarray, feat_names: list[str], time_col: pd.Series, frac: float = 0.5,
                             rng: np.random.Generator | None = None, n_per_group: int | None = None):
    """Early vs late using original Time order (requires inmemory with time_col). Optionally subsample to n_per_group."""
    order = time_col.sort_values().index
    cut = int(len(order) * frac)
    early_idx = order[:cut]
    late_idx = order[cut:]
    A = X[early_idx, :]
    B = X[late_idx, :]
    if n_per_group is not None:
        n = min(n_per_group, len(A), len(B))
        r = rng or np.random.default_rng(0)
        A = A[r.choice(len(A), size=n, replace=False), :]
        B = B[r.choice(len(B), size=n, replace=False), :]
    return A, B

# ---------------- t-test utils ----------------

def safe_t_pvalues(A: np.ndarray, B: np.ndarray):
    pvals, tstats, mA, mB = [], [], [], []
    for j in range(A.shape[1]):
        a = A[:, j]; b = B[:, j]
        # drop NaNs
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if a.size < 2 or b.size < 2:
            pvals.append(1.0); tstats.append(0.0); mA.append(np.nan); mB.append(np.nan); continue
        # handle zero variance: ttest_ind would return nan -> mark as non-significant
        if np.var(a) == 0 or np.var(b) == 0:
            pvals.append(1.0); tstats.append(0.0); mA.append(float(np.mean(a))); mB.append(float(np.mean(b))); continue
        res = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        pvals.append(float(res.pvalue))
        tstats.append(float(res.statistic))
        mA.append(float(np.mean(a))); mB.append(float(np.mean(b)))
    return np.array(pvals), np.array(tstats), np.array(mA), np.array(mB)


def t_aggregate(A: np.ndarray, B: np.ndarray):
    pvals, tstats, mA, mB = safe_t_pvalues(A, B)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    if pvals.size == 0:
        return 1.0, 1.0, pvals, tstats, mA, mB
    # Fisher combine
    _, p_fisher = combine_pvalues(pvals, method='fisher')
    # Bonferroni on min p
    p_bonf = float(min(1.0, np.nanmin(pvals) * len(pvals)))
    return float(p_fisher), float(p_bonf), pvals, tstats, mA, mB

# ---------------- Plot helper ----------------

def bar_chart(labels, values, title, out_path):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Rate")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Welch t-test two-sample evaluation for Credit Card dataset")
    ap.add_argument("--source", choices=["inmemory", "npz", "csv"], default="inmemory")
    ap.add_argument("--npz", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)

    ap.add_argument("--mode", choices=["fraud_vs_normal", "timeslice", "train_vs_test"], default="fraud_vs_normal")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--n-per-group", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--test-size", type=float, default=0.5)
    ap.add_argument("--time-frac", type=float, default=0.5)
    ap.add_argument("--out-dir", type=str, default="./ttest_credit_results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load
    if args.source == "inmemory":
        X, y, feat_names, time_col = load_inmemory(args.test_size, args.seed)
    elif args.source == "npz":
        if not args.npz:
            raise ValueError("--npz is required when --source=npz")
        X, y, feat_names, time_col = load_npz(args.npz)
    else:
        if not args.data_dir:
            raise ValueError("--data-dir is required when --source=csv")
        X, y, feat_names, time_col = load_csv_dir(args.data_dir)

    # Remove zero-variance columns to avoid degenerate t-tests
    col_var = np.var(X, axis=0)
    keep = col_var > 0.0
    if not np.all(keep):
        X = X[:, keep]
        feat_names = [f for f, k in zip(feat_names, keep) if k]

    # --- Null (Type I): same distribution ---
    null_reject_fisher, null_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = random_split_same_dist(X, args.n_per_group, rng)
        p_f, p_b, *_ = t_aggregate(A, B)
        null_reject_fisher.append(p_f < args.alpha)
        null_reject_bonf.append(p_b < args.alpha)

    # --- Alternative (Power): by mode ---
    alt_reject_fisher, alt_reject_bonf = [], []
    diag_saved = False
    for _ in range(args.n_trials):
        if args.mode == "fraud_vs_normal":
            A, B = class_split(X, y, args.n_per_group, rng, class_a=1, class_b=0)
        elif args.mode == "train_vs_test":
            A, B = random_split_same_dist(X, args.n_per_group, rng)
        else:  # timeslice
            if args.source != "inmemory" or time_col is None:
                raise ValueError("timeslice mode requires --source inmemory (needs original Time column).")
            A, B = timeslice_split_inmemory(X, feat_names, time_col, frac=args.time_frac, rng=rng, n_per_group=args.n_per_group)

        p_f, p_b, pvals, tstats, mA, mB = t_aggregate(A, B)
        alt_reject_fisher.append(p_f < args.alpha)
        alt_reject_bonf.append(p_b < args.alpha)

        # Save per-feature diagnostics for the first trial
        if not diag_saved:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            diag = pd.DataFrame({
                'feature': feat_names,
                't_stat': tstats,
                'p_value': pvals,
                'mean_A': mA,
                'mean_B': mB
            })
            diag.to_csv(out_dir / 't_feature_diagnostics.csv', index=False)
            diag_saved = True

    # --- Summarize ---
    summary = pd.DataFrame([{
        "alpha": args.alpha,
        "n_trials": args.n_trials,
        "n_per_group": args.n_per_group,
        "typeI_fisher": float(np.mean(null_reject_fisher)),
        "typeI_bonf": float(np.mean(null_reject_bonf)),
        "power_fisher": float(np.mean(alt_reject_fisher)),
        "power_bonf": float(np.mean(alt_reject_bonf)),
        "mode": args.mode,
        "source": args.source,
    }])

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

    _bar(["t (Fisher)", "t (Bonf.)"],
         [summary["typeI_fisher"].iloc[0], summary["typeI_bonf"].iloc[0]],
         f"t-test Type I Error (alpha={args.alpha})", "typeI_bar.png")

    _bar(["t (Fisher)", "t (Bonf.)"],
         [summary["power_fisher"].iloc[0], summary["power_bonf"].iloc[0]],
         f"t-test Power ({args.mode})", "power_bar.png")

    print("=== t-test (credit) evaluation done ===")
    print("Saved:", (out_dir / "results.csv").resolve())


if __name__ == "__main__":
    main()
