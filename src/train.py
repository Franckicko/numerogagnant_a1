# src/train.py
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from .metrics import (
    multiclass_logloss,
    logloss_multiclass_raceaware,
    logloss_top4_renorm,
    hit_at_k,
)
from .features import build_feature_frame
    # load_raw lit Courses_Completes.csv ; save_metrics_history écrit metrics_history.csv
from .data_io import load_raw, save_metrics_history
from .model_registry import maybe_promote

ROOT = Path(__file__).resolve().parents[1]


class PreprocBooster:
    """Applique le preprocessor sklearn puis appelle un Booster XGBoost.train (multi:softprob)."""
    def __init__(self, preprocessor, booster, num_classes: int):
        self.pre = preprocessor
        self.booster = booster
        self.num_classes = int(num_classes)

    def predict_proba(self, X_df):
        import xgboost as xgb
        Xt = self.pre.transform(X_df)
        d = xgb.DMatrix(Xt)
        # XGBoost 3.x : utiliser iteration_range ; fallback ntree_limit si nécessaire
        try:
            best_it = getattr(self.booster, "best_iteration", None)
            if best_it is not None:
                preds = self.booster.predict(d, iteration_range=(0, int(best_it) + 1))
            else:
                preds = self.booster.predict(d)
        except TypeError:
            # très vieux fallback
            ntree = getattr(self.booster, "best_ntree_limit", 0)
            preds = self.booster.predict(d, ntree_limit=int(ntree) if ntree else 0)
        # reshape (n, K)
        return preds.reshape(-1, self.num_classes)


def _build_xgb_train_params(cfg_model: dict, num_classes: int):
    """Mappe la config sklearn -> params xgb.train"""
    params = cfg_model.get("params", {}).copy()
    # defaults robustes
    n_estimators = int(params.pop("n_estimators", 600))
    learning_rate = float(params.pop("learning_rate", 0.05))
    max_depth = int(params.pop("max_depth", 6))
    subsample = float(params.pop("subsample", 0.8))
    colsample_bytree = float(params.pop("colsample_bytree", 0.8))
    reg_lambda = float(params.pop("reg_lambda", 1.0))
    random_state = int(params.pop("random_state", 42))
    tree_method = params.pop("tree_method", "hist")  # plus rapide

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": int(num_classes),
        "eval_metric": "mlogloss",
        "eta": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": reg_lambda,
        "seed": random_state,
        "tree_method": tree_method,
    }
    # on ignore proprement tout le reste
    return xgb_params, n_estimators


def main():
    # --- Config & données
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}

    df = load_raw().copy().sort_values(cfg["date_col"]).reset_index(drop=True)

    # --- Cible et classes (1..N réelles)
    y_real = df[cfg["target_col"]].astype(int)
    num_classes = int(y_real.max())
    classes_real = np.arange(1, num_classes + 1)

    # --- Split temporel (queue en validation)
    valid_ratio = float(training_cfg.get("valid_ratio", 0.2))
    cut = int(len(df) * (1 - valid_ratio))
    train_df, valid_df = df.iloc[:cut], df.iloc[cut:]

    # --- Garde-fou promotion
    min_valid = int(training_cfg.get("min_valid", 300))
    force_no_promo = len(valid_df) < min_valid
    if force_no_promo:
        print(f"Validation trop petite (N={len(valid_df)} < {min_valid}) → pas de promotion.")

    prono_cols = cfg["prono_cols"]
    num_features = cfg["num_features"]
    cat_features = cfg["cat_features"]

    # --- Features & prépro (anti-leak)
    Xtr_df = build_feature_frame(train_df, prono_cols, num_features, cat_features)
    ytr = train_df[cfg["target_col"]].astype(int) - 1  # 0..K-1
    Xva_df = build_feature_frame(valid_df, prono_cols, num_features, cat_features)
    yva = valid_df[cfg["target_col"]].astype(int) - 1  # 0..K-1

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_features + prono_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    pre.fit(Xtr_df)

    # matrices pour xgboost
    import xgboost as xgb
    Xtr = pre.transform(Xtr_df)
    Xva = pre.transform(Xva_df)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xva, label=yva)

    # --- xgboost.train avec early stopping
    xgb_params, num_boost_round = _build_xgb_train_params(cfg["model"], num_classes)
    es_rounds = int(training_cfg.get("early_stopping_rounds", 80))
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=es_rounds,
        verbose_eval=False,
    )

    # wrapper pour prédire proba derrière le prépro
    wrapped = PreprocBooster(pre, booster, num_classes)

    # --- Validation: VRAI logloss + FAUX logloss Top-4 + Hits
    proba = wrapped.predict_proba(Xva_df)  # (n, K)
    labels_enc = np.arange(0, num_classes)

    ll_global = multiclass_logloss(yva, proba, labels=labels_enc)
    partants_va = valid_df["partants"].values if "partants" in valid_df.columns else np.full(len(yva), num_classes)
    ll_race = logloss_multiclass_raceaware(yva, proba, partants_va)

    y_true_1based = valid_df[cfg["target_col"]].astype(int).values
    ll_top4_items, hit4_list, hit1_list = [], [], []
    for j, row in enumerate(valid_df.itertuples(index=False)):
        pairs = []
        for k in range(1, 9):
            numero = getattr(row, f"prono{k}", None)
            if numero is None or pd.isna(numero):
                continue
            numero = int(numero)
            p = float(proba[j, numero - 1]) if 1 <= numero <= num_classes else 0.0
            pairs.append((numero, p))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:4]
        top_nums = [n for n, _ in top]
        top_probs = [p for _, p in top]
        ll_top4_items.append(logloss_top4_renorm(y_true_1based[j], top_nums, top_probs))
        hit4_list.append(hit_at_k(top_nums, y_true_1based[j], k=4))
        hit1_list.append(hit_at_k(top_nums, y_true_1based[j], k=1))

    ll_top4 = float(np.mean(ll_top4_items)) if ll_top4_items else float("nan")
    hit4_val = float(np.mean(hit4_list)) if hit4_list else float("nan")
    hit1_val = float(np.mean(hit1_list)) if hit1_list else float("nan")

      # --- Sauvegarde challenger (versionné, NE PAS écraser le champion)
    MID = f"a1_{time.strftime('%Y%m%d-%H%M%S')}_xgb"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    challenger_path = models_dir / f"{MID}.joblib"
    joblib.dump({"model": wrapped, "classes": classes_real.tolist()}, challenger_path)

    # --- Décision de promotion (critère: ll_race)
    if force_no_promo:
        promoted, prev = False, None
    else:
        eps = float(training_cfg.get("epsilon_promotion", 1e-4))
        promoted, prev = maybe_promote({"model_id": MID, "file_path": challenger_path}, ll_race, epsilon=eps)

    # --- Canonique du champion + archivage de l'ancien si promotion
    from shutil import copyfile
    canon = models_dir / "xgb_multiclass.joblib"
    if promoted:
        # archiver l'ancien champion s'il existe
        try:
            if prev and prev.get("file_path"):
                old_path = Path(prev["file_path"])
                if old_path.exists():
                    archive_dir = models_dir / "archive"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    copyfile(old_path, archive_dir / f"{ts}_{old_path.name}")
        except Exception as e:
            print(f"[WARN] Archivage de l'ancien champion impossible: {e}")
        # mettre à jour le chemin canonique utilisé par la prod/prédiction
        copyfile(challenger_path, canon)
    else:
        # non promu → on ne touche pas au champion canonique
        pass

    # --- Log métriques
    save_metrics_history(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": MID,
            "val_logloss_global": float(ll_global),
            "val_logloss_raceaware": float(ll_race),
            "val_logloss_top4": float(ll_top4),
            "val_hit1": float(hit1_val),
            "val_hit4": float(hit4_val),
            "promoted": int(promoted),
        }
    )

    print(
        "Validation — "
        f"global: {ll_global:.4f} | "
        f"race-aware: {ll_race:.4f} | "
        f"top4: {ll_top4:.4f} | "
        f"H@1: {hit1_val:.3f} | H@4: {hit4_val:.3f} | "
        f"promoted={promoted}"
    )
