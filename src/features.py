import re
import pandas as pd
from typing import Dict


def make_race_id(df: pd.DataFrame, fields):
    """Construit un identifiant unique de course à partir de colonnes."""
    return df[fields].astype(str).agg("_".join, axis=1)


def _sanitize_colnames(cols):
    """Nettoie les noms de colonnes pour compatibilité XGBoost."""
    out = []
    for c in cols:
        if not isinstance(c, str):
            c = str(c)
        # remplace [, ], <, >, espaces et tabulations par _
        c = re.sub(r"[\[\]<>\\s]", "_", c)
        # compresse les doublons de _ et strip
        c = re.sub(r"_+", "_", c).strip("_")
        out.append(c)
    return out


def build_dataset(courses: pd.DataFrame, cfg: Dict):
    """
    Construit X, y, meta pour un modèle multiclasses prédisant `columns.target` (ex: a2).

    Anti-fuite par défaut :
      - rapport
      - r_b\d+
      - si_a\d+_dans_pronos
      - pred\d+_a\d+
      - a\d+_.*
      - toutes les colonnes a\d+ ≠ target
    """
    # Copie et dédoublonnage
    df = courses.drop_duplicates().copy()

    # --- Config clés ---
    date_col = cfg["columns"]["date"]
    rid_fields = cfg["columns"]["race_id_fields"]
    target = cfg["columns"]["target"]

    # --- Features de date ---
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dow"] = df[date_col].dt.dayofweek

    # --- Identifiant de course ---
    df["race_id"] = make_race_id(df, rid_fields)

    # --- Suppression colonnes à fuite ---
    default_drop_regex = r"^(?:r_b\d+|si_a\d+_dans_pronos|pred\d+_a\d+|a\d+_.*)$"

    # On retire aussi rapport sauf si keep_rapport est True dans config
    if not cfg["columns"].get("keep_rapport", False):
        default_drop_regex = r"^(?:rapport|" + default_drop_regex[3:]

    drop_regex = cfg["columns"].get("drop_cols_regex", default_drop_regex)
    pat = re.compile(drop_regex)
    to_drop = [c for c in df.columns if pat.search(str(c))]
    keep = [c for c in df.columns if c not in to_drop]
    df = df[keep]

    # --- Cible ---
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' introuvable.")

    # Conversion cible en numérique
    df[target] = pd.to_numeric(df[target], errors="coerce")
    n_before = len(df)
    df = df[df[target].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[features] {n_dropped} ligne(s) supprimée(s) car {target} manquant/non numérique.")

    y = df[target].astype(int)  # classes = numéros 1..N

    # --- Construction X ---
    X = df.drop(columns=[target])

    # Retirer toutes les autres colonnes a\d+ qui ne sont pas la cible
    other_arrival_cols = [c for c in X.columns if re.fullmatch(r"a\d+", str(c)) and c != target]
    if other_arrival_cols:
        X = X.drop(columns=other_arrival_cols)

    # --- Encodage catégoriel ---
    cat_cols = [c for c in cfg["columns"].get("categorical", []) if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    # --- Sélection numérique + nettoyage des noms ---
    if "race_id" not in X.columns:
        raise ValueError("La colonne 'race_id' est manquante avant la phase finale.")
    race_id = X["race_id"].copy()

    X = X.drop(columns=["race_id"], errors="ignore")
    X = X.select_dtypes(include="number").fillna(0)

    # Nettoie les noms de colonnes
    X.columns = _sanitize_colnames(X.columns)

    meta = {"race_id": race_id}
    return X, y, meta, df
