"""
Preprocess the UCI Census Income (Adult) dataset for two-sample testing and other ML experiments.

Key points (mirrors the style of wine_preprocess.py, but tailored for mixed-type features):
- Load strictly from ucimlrepo (id=20)
- Clean/trim strings and treat "?" as missing
- Target mapping to {0,1}: "<=50K"(0), ">50K"(1)
- ColumnTransformer: Numeric (median-impute + StandardScaler), Categorical (most-frequent impute + OneHotEncoder)
- Save outputs to CSV and/or NPZ + metadata.json
- Reproducible and CLI-friendly (no train/val/test split here to match wine_preprocess.py behavior)

Usage examples:
    python census_preprocess.py --output-dir ./processed_census
    python census_preprocess.py --save-format both
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse

def load_census_ucimlrepo() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Census Income (Adult) from ucimlrepo (id=20). Returns (X_df, y_series)."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=20)
    X = ds.data.features.copy()
    y_df = ds.data.targets.copy()
    # Common target column names: "income" or "class"
    if "income" in y_df.columns:
        y = y_df["income"].copy()
    elif "class" in y_df.columns:
        y = y_df["class"].copy()
    else:
        y = y_df.iloc[:, 0].copy()
    return X, y

def clean_base_feature_names(cols: List[str]) -> List[str]:
    """Normalize original (pre-OHE) feature names to be file/CLI friendly."""
    cleaned = []
    for c in cols:
        cc = c.strip()
        cc = cc.replace(" ", "_").replace("/", "_").replace("-", "_")
        cleaned.append(cc)
    return cleaned

def map_income_labels(y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """Map income labels to {0,1}. Handles variants with trailing periods."""
    mapping = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}
    y_mapped = y.map(mapping)
    if y_mapped.isna().any():
        bad_vals = sorted(pd.unique(y[y_mapped.isna()]).tolist())
        raise ValueError(f"Unknown target values encountered: {bad_vals}")
    meta = {
        "original_unique_values": sorted(pd.unique(y).tolist()),
        "mapping": {"<=50K": 0, ">50K": 1},
        "note": "Trailing periods mapped equivalently."
    }
    return y_mapped.astype(int), meta

def build_preprocessor(X: pd.DataFrame):
    """Build a ColumnTransformer for numeric+categorical preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        sparse_threshold=0.3  # keep sparse when many OHE columns
    )
    return pre, num_cols, cat_cols

def get_ohe_feature_names(pre, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Get output feature names after ColumnTransformer (num + OHE cat)."""
    out_names: List[str] = []

    # For numeric pipeline, keep original names
    out_names.extend(num_cols)

    # For categorical, extract OHE names
    cat_ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    # scikit-learn >= 1.0 supports get_feature_names_out with input_features
    ohe_names = cat_ohe.get_feature_names_out(cat_cols).tolist()
    out_names.extend(ohe_names)
    return out_names

def to_dense_if_needed(X):
    """Convert sparse to dense numpy array when saving CSV; keep sparse otherwise."""
    if sparse.issparse(X):
        return X.toarray()
    return X

def save_outputs(
    X_matrix,
    y: pd.Series,
    feature_names: List[str],
    outdir: Path,
    save_format: str,
    meta: Dict[str, Any],
):
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV
    if save_format in ("csv", "both"):
        X_dense = to_dense_if_needed(X_matrix)
        X_df = pd.DataFrame(X_dense, columns=feature_names)
        y_df = pd.DataFrame({"label": y.values})
        X_df.to_csv(outdir / "X.csv", index=False)
        y_df.to_csv(outdir / "y.csv", index=False)

    # NPZ (compressed). For sparse, prefer scipy.sparse .npz
    if save_format in ("npz", "both"):
        if sparse.issparse(X_matrix):
            sparse.save_npz(outdir / "X_sparse.npz", X_matrix.tocsr())
            np.save(outdir / "y.npy", y.values.astype(np.int64))
            np.save(outdir / "feature_names.npy", np.array(feature_names, dtype=object))
        else:
            np.savez_compressed(
                outdir / "data.npz",
                X=X_matrix.astype(np.float32),
                y=y.values.astype(np.int64),
                feature_names=np.array(feature_names, dtype=object),
            )

    # Metadata JSON
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Preprocess UCI Census Income (Adult) dataset from ucimlrepo.")
    parser.add_argument("--output-dir", default="./processed_census", help="Directory to save outputs.")
    parser.add_argument("--save-format", choices=["csv", "npz", "both"], default="both",
                        help="Which file formats to save.")
    args = parser.parse_args()

    # 1) Load from ucimlrepo only (no CSV fallback)
    X, y_raw = load_census_ucimlrepo()
    base_feat_names = clean_base_feature_names(list(X.columns))
    X.columns = base_feat_names

    # 2) Clean text columns: strip and normalize missing symbols
    X = X.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    X = X.replace("?", np.nan)

    # 3) Map target to {0,1}
    y, label_meta = map_income_labels(y_raw)

    # 4) Build & fit preprocessor on full dataset (to mirror wine_preprocess style)
    pre, num_cols, cat_cols = build_preprocessor(X)
    X_proc = pre.fit_transform(X)

    # 5) Construct output feature names
    out_feature_names = get_ohe_feature_names(pre, num_cols, cat_cols)

    # 6) Save
    outdir = Path(args.output_dir)
    meta_all = {
        "source_used": "ucimlrepo:id=20",
        "n_samples": int(X_proc.shape[0]),
        "n_features_out": int(X_proc.shape[1]),
        "original_feature_names": base_feat_names,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "label_mapping": label_meta,
        "preprocess": {
            "numeric": "SimpleImputer(median) + StandardScaler",
            "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown=ignore)",
        },
        "note": "No PCA or splitting to match wine_preprocess.py behavior.",
    }
    save_outputs(X_proc, y, out_feature_names, outdir, args.save_format, meta_all)

    # 7) Print summary
    cls, cnt = np.unique(y.values, return_counts=True)
    print("=== Census preprocessing done ===")
    print("Source:          ucimlrepo (id=20)")
    print("Output dir:      ", outdir.resolve())
    print("Output features: ", len(out_feature_names))
    print("Classes/counts:  ", dict(zip(cls.tolist(), cnt.tolist())))
    if args.save_format in ("csv", "both") and sparse.issparse(X_proc):
        print("[Note] CSV was saved in dense form; for efficient reloading, prefer NPZ (X_sparse.npz).")
    print("Saved:           ", args.save_format, "(X.csv/y.csv and/or X_sparse.npz + y.npy + feature_names.npy + metadata.json)")

if __name__ == "__main__":
    main()
