import pandas as pd
import numpy as np
import re

LEAKY_PATTERNS = [
    r"^pred\d+_a[124]$",
    r"^true_.*$",
    r"^rapport$",
    r"^r_b\d+$",
    r"^(a1|a2|a4)_(imp|pair|sup9|infegal9)$",
]

def _is_leaky(col: str) -> bool:
    for pat in LEAKY_PATTERNS:
        if re.match(pat, col):
            return True
    return False

def build_feature_frame(df, prono_cols, num_features, cat_features):
    X = pd.DataFrame(index=df.index)
    for c in prono_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in num_features:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in cat_features:
        X[c] = df[c].astype(str).fillna("NA") if c in df.columns else "NA"
    safe = set(prono_cols + num_features + cat_features)
    for c in list(df.columns):
        if c not in safe and _is_leaky(c):
            if c in X.columns:
                X.drop(columns=[c], inplace=True, errors="ignore")
    return X
