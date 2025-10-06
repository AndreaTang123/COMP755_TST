#!/usr/bin/env python3
"""
wine_preprocess.py

Preprocess the UCI Wine dataset for two-sample testing and other ML experiments.

Features:
- Load from ucimlrepo (id=109) or sklearn fallback
- Clean column names (fix OD280/OD315 naming)
- Standardize features (StandardScaler)
- Optional PCA by #components or by explained variance target
- Save outputs to CSV and/or NPZ + metadata.json
- Reproducible and CLI-friendly

Usage examples:
    python wine_preprocess.py --output-dir ./processed_wine
    python wine_preprocess.py --pca-var 0.95 --output-dir ./processed_wine_pca95
    python wine_preprocess.py --pca-k 8 --save-format both
    python wine_preprocess.py --source sklearn --keep-original-classes


"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

def load_wine_ucimlrepo() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Wine from ucimlrepo (id=109). Returns (X_df, y_series)."""
    from ucimlrepo import fetch_ucirepo
    wine = fetch_ucirepo(id=109)
    X = wine.data.features.copy()
    y_df = wine.data.targets.copy()
    # y may be a DataFrame with column "class"
    if "class" in y_df.columns:
        y = y_df["class"].copy()
    else:
        y = y_df.iloc[:, 0].copy()
    return X, y

def load_wine_sklearn() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Wine from sklearn as a fallback. Returns (X_df, y_series)."""
    from sklearn.datasets import load_wine
    obj = load_wine(as_frame=True)
    X = obj.data.copy()
    y = pd.Series(obj.target.copy(), name="class")
    return X, y

def clean_feature_names(cols: List[str]) -> List[str]:
    """Normalize feature names to be file/CLI friendly."""
    cleaned = []
    for c in cols:
        cc = c.strip()
        # Replace spaces and slashes with underscores
        cc = cc.replace(" ", "_").replace("/", "_").replace("-", "_")
        # Some ucimlrepo versions use leading '0D...' (zero-D) instead of 'OD...'
        if cc.startswith("0D"):
            cc = "OD" + cc[2:]
        # Specific long name standardization
        cc = cc.replace("OD280_0D315", "OD280_OD315")
        cleaned.append(cc)
    return cleaned

def maybe_remap_labels(y: pd.Series, keep_original: bool) -> Tuple[pd.Series, Dict[str, Any]]:
    """Optionally map labels {1,2,3} -> {0,1,2}. Returns y and mapping metadata."""
    mapping_meta: Dict[str, Any] = {}
    uniq = sorted(pd.unique(y))
    if keep_original:
        mapping_meta["mapping"] = {str(v): int(v) for v in uniq}
        mapping_meta["note"] = "kept original label values"
        y_mapped = y.copy()
    else:
        # Remap to 0..K-1 in sorted order
        mapping = {orig: i for i, orig in enumerate(uniq)}
        mapping_meta["mapping"] = {str(k): int(v) for k, v in mapping.items()}
        mapping_meta["note"] = "remapped labels to 0..K-1 in sorted order"
        y_mapped = y.map(mapping)
    mapping_meta["classes_sorted"] = [int(v) for v in uniq]
    return y_mapped.astype(int), mapping_meta

def standardize_features(X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Standardize X via StandardScaler. Returns array and scaler metadata."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X.values)
    meta = {
        "scaler": "StandardScaler",
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "n_features_in_": int(getattr(scaler, "n_features_in_", X.shape[1])),
    }
    return X_std, meta

def apply_pca(X_std: np.ndarray, feature_names: List[str], pca_k: int | None, pca_var: float | None):
    """Optionally apply PCA with either fixed components or variance target."""
    if pca_k is None and pca_var is None:
        return X_std, feature_names, {}

    from sklearn.decomposition import PCA
    if pca_k is not None and pca_var is not None:
        raise ValueError("Set only one of --pca-k or --pca-var.")

    if pca_k is not None:
        pca = PCA(n_components=int(pca_k), svd_solver="full", random_state=0)
    else:
        # pca_var is a float in (0,1]; PCA will choose #components to meet variance target
        pca = PCA(n_components=float(pca_var), svd_solver="full", random_state=0)

    X_pca = pca.fit_transform(X_std)
    k = X_pca.shape[1]
    new_names = [f"PC{i+1}" for i in range(k)]
    meta = {
        "pca": "sklearn.decomposition.PCA",
        "n_components_": int(k),
        "explained_variance_ratio_": pca.explained_variance_ratio_.tolist(),
        "singular_values_": pca.singular_values_.tolist(),
    }
    return X_pca, new_names, meta

def save_outputs(
    X_array: np.ndarray,
    y: pd.Series,
    feature_names: List[str],
    outdir: Path,
    save_format: str,
    meta: Dict[str, Any],
):
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV
    if save_format in ("csv", "both"):
        X_df = pd.DataFrame(X_array, columns=feature_names)
        y_df = pd.DataFrame({"label": y.values})
        X_df.to_csv(outdir / "X.csv", index=False)
        y_df.to_csv(outdir / "y.csv", index=False)

    # NPZ
    if save_format in ("npz", "both"):
        np.savez_compressed(
            outdir / "data.npz",
            X=X_array.astype(np.float32),
            y=y.values.astype(np.int64),
            feature_names=np.array(feature_names, dtype=object),
        )

    # Metadata JSON
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Preprocess UCI Wine dataset.")
    parser.add_argument("--source", choices=["ucimlrepo", "sklearn"], default="ucimlrepo",
                        help="Where to load the dataset from.")
    parser.add_argument("--output-dir", default="./processed_wine", help="Directory to save outputs.")
    parser.add_argument("--keep-original-classes", action="store_true",
                        help="Keep labels as in the source (e.g., 1/2/3). By default we remap to 0/1/2.")
    parser.add_argument("--pca-k", type=int, default=None,
                        help="Apply PCA to k components. Set either this or --pca-var.")
    parser.add_argument("--pca-var", type=float, default=None,
                        help="Apply PCA to retain this fraction of explained variance (0-1].")
    parser.add_argument("--save-format", choices=["csv", "npz", "both"], default="both",
                        help="Which file formats to save.")
    args = parser.parse_args()

    # 1) Load
    if args.source == "ucimlrepo":
        try:
            X, y = load_wine_ucimlrepo()
            source_used = "ucimlrepo"
        except Exception as e:
            print(f"[WARN] ucimlrepo failed: {e}\nFalling back to sklearn...")
            X, y = load_wine_sklearn()
            source_used = "sklearn"
    else:
        X, y = load_wine_sklearn()
        source_used = "sklearn"

    # 2) Clean columns
    feat_names = clean_feature_names(list(X.columns))
    X.columns = feat_names

    # 3) Ensure y is a 1D Series named 'label'
    if isinstance(y, pd.DataFrame):
        if "class" in y.columns:
            y = y["class"].copy()
        else:
            y = y.iloc[:, 0].copy()
    y.name = "label"

    # 4) Optional label remapping
    y, label_map_meta = maybe_remap_labels(y, keep_original=args.keep_original_classes)

    # 5) Missing values check (dataset has none, but we assert for safety)
    if X.isna().any().any():
        print("[INFO] Missing values detected; imputing by column mean.")
        X = X.fillna(X.mean(numeric_only=True))

    # 6) Standardize
    X_std, scaler_meta = standardize_features(X)

    # 7) Optional PCA
    X_proc, out_feat_names, pca_meta = apply_pca(X_std, feat_names, args.pca_k, args.pca_var)

    # 8) Save
    outdir = Path(args.output_dir)
    meta_all = {
        "source_used": source_used,
        "n_samples": int(X_proc.shape[0]),
        "n_features_out": int(X_proc.shape[1]),
        "original_feature_names": feat_names,
        "label_mapping": label_map_meta,
        "scaler_meta": scaler_meta,
        "pca_meta": pca_meta,
    }
    save_outputs(X_proc, y, out_feat_names, outdir, args.save_format, meta_all)

    # 9) Print summary
    cls, cnt = np.unique(y.values, return_counts=True)
    print("=== Wine preprocessing done ===")
    print("Source:          ", source_used)
    print("Output dir:      ", outdir.resolve())
    print("Output features: ", len(out_feat_names))
    print("Classes/counts:  ", dict(zip(cls.tolist(), cnt.tolist())))
    print("Saved:           ", args.save_format, "(X.csv/y.csv and/or data.npz + metadata.json)")

if __name__ == "__main__":
    main()
