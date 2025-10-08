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
from typing import List, Tuple

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

def ks_per_feature(A, B) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-feature KS statistics and p-values.
    D: array of KS statistics for each feature
    p: array of p-values for each feature
    """
    D = np.empty(A.shape[1], dtype=float)
    p = np.empty(A.shape[1], dtype=float)
    for j in range(A.shape[1]):
        res = ks_2samp(A[:, j], B[:, j])
        D[j] = res.statistic
        p[j] = res.pvalue
    return D, p

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
    parser.add_argument("--per-feature", action="store_true",
                        help="If set, record per-feature KS stats across trials and save plots/CSV.")
    parser.add_argument("--per-feature-metric", choices=["D", "p", "reject"], default="D",
                        help="Which metric to visualize for per-feature plot: KS statistic (D), p-value (p), or reject rate (reject).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y, feat_names = load_data(args.data_dir, args.npz)

    null_reject_fisher, null_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = random_split_same_dist(X, args.n_per_group, rng)
        p_f, p_b = ks_aggregate(A, B)
        null_reject_fisher.append(p_f < args.alpha)
        null_reject_bonf.append(p_b < args.alpha)

    num_features = X.shape[1]
    if args.per_feature:
        alt_sum_D = np.zeros(num_features, dtype=float)
        alt_sum_p = np.zeros(num_features, dtype=float)
        alt_reject_counts = np.zeros(num_features, dtype=float)

    alt_reject_fisher, alt_reject_bonf = [], []
    for _ in range(args.n_trials):
        A, B = class_split(X, y, args.class_a, args.class_b, args.n_per_group, rng)
        # Per-feature stats (for visualization)
        if args.per_feature:
            D_vec, p_vec = ks_per_feature(A, B)
            alt_sum_D += D_vec
            alt_sum_p += p_vec
            alt_reject_counts += (p_vec < args.alpha).astype(float)
            # For the aggregate decisions we still use all per-feature p-values:
            _, p_fisher = combine_pvalues(p_vec, method='fisher')
            p_bonf = min(1.0, float(np.min(p_vec)) * len(p_vec))
            p_f, p_b = float(p_fisher), float(p_bonf)
        else:
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

    if args.per_feature:
        # Averages across trials
        mean_D = alt_sum_D / args.n_trials
        mean_p = alt_sum_p / args.n_trials
        reject_rate = alt_reject_counts / args.n_trials
        per_feat_df = pd.DataFrame({
            "feature": feat_names,
            "mean_D": mean_D,
            "mean_p": mean_p,
            "reject_rate": reject_rate
        })
        per_feat_df.to_csv(out_dir / "per_feature_alt.csv", index=False)

        # Choose metric to visualize
        if args.per_feature_metric == "D":
            vec = mean_D
            vtitle = "Per-feature mean KS statistic (ALT)"
            fname_hm = "per_feature_alt_D_heatmap.png"
            fname_bar = "per_feature_alt_D_bar.png"
        elif args.per_feature_metric == "p":
            vec = mean_p
            vtitle = "Per-feature mean p-value (ALT)"
            fname_hm = "per_feature_alt_p_heatmap.png"
            fname_bar = "per_feature_alt_p_bar.png"
        else:
            vec = reject_rate
            vtitle = "Per-feature reject rate (ALT)"
            fname_hm = "per_feature_alt_reject_heatmap.png"
            fname_bar = "per_feature_alt_reject_bar.png"

        # Heatmap (1 x d)
        plt.figure(figsize=(max(6, len(vec)*0.25), 2.2))
        plt.imshow(vec.reshape(1, -1), aspect="auto")
        plt.yticks([0], ["ALT"])
        plt.xticks(range(len(feat_names)), feat_names, rotation=60, ha="right", fontsize=8)
        plt.title(vtitle)
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.savefig(out_dir / fname_hm, dpi=160)
        plt.close()

        # Bar plot
        plt.figure(figsize=(max(6, len(vec)*0.35), 3.0))
        plt.bar(range(len(vec)), vec)
        plt.xticks(range(len(feat_names)), feat_names, rotation=60, ha="right", fontsize=8)
        plt.title(vtitle)
        plt.ylabel(args.per_feature_metric)
        plt.tight_layout()
        plt.savefig(out_dir / fname_bar, dpi=160)
        plt.close()

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
