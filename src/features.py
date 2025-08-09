import re
import pandas as pd
from typing import Dict

def make_race_id(df: pd.DataFrame, fields):
    return df[fields].astype(str).agg("_".join, axis=1)

def _sanitize_colnames(cols):
    out = []
    for c in cols:
        if not isinstance(c, str):
            c = str(c)
        # remplace [, ], <, >, espaces et tabulations par _
        c = re.sub(r"[\\[\\]<>\\s]", "_", c)
        # compresse les doublons de _ et strip
        c = re.sub(r"_+", "_", c).strip("_")
        out.append(c)
    return out

def build_dataset(courses: pd.DataFrame, cfg: Dict):
    # On part du DataFrame déjà lu (sep défini dans train/predict)
    df = courses.drop_duplicates().copy()

    # --- Features de date ---
    date_col = cfg["columns"]["date"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dow"] = df[date_col].dt.dayofweek

    # --- Identifiant de course ---
    rid_fields = cfg["columns"]["race_id_fields"]
    df["race_id"] = make_race_id(df, rid_fields)

    # --- Suppression colonnes à fuite (rapport, r_b1, a1_*, pred*_a1, etc.) ---
    # Si l'utilisateur ne fournit pas `drop_cols_regex`, on met une valeur sûre.
    drop_regex = cfg["columns"].get("drop_cols_regex", r"^(a1_.*|pred\d+_a1|rapport|r_b1)$")
    pat = re.compile(drop_regex)
    to_drop = [c for c in df.columns if pat.search(str(c))]
    keep = [c for c in df.columns if c not in to_drop]
    df = df[keep]

    # --- Cible : a1 ---
    target = cfg["columns"]["target"]
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable.")

    # Convertit en numérique; non-convertibles -> NaN, puis on écarte
    df[target] = pd.to_numeric(df[target], errors="coerce")
    n_before = len(df)
    df = df[df[target].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[features] {n_dropped} ligne(s) supprimée(s) car {target} manquant/non numérique.")

    y = df[target].astype(int)  # classes = numéros 1..N

    # --- Encodage catégoriel ---
    cat_cols = [c for c in cfg["columns"].get("categorical", []) if c in df.columns]
    X = df.drop(columns=[target])
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    # --- Sélection numérique + nettoyage des noms ---
    race_id = X["race_id"].copy()
    X = X.drop(columns=["race_id"])
    X = X.select_dtypes(include="number").fillna(0)

    # Nettoie les noms de colonnes pour XGBoost (interdit [, ], <, >)
    X.columns = _sanitize_colnames(X.columns)

    meta = {"race_id": race_id}
    return X, y, meta, df
