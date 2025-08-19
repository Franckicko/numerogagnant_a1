# src/train.py
from __future__ import annotations

import json, shutil, yaml, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb

from .features import build_dataset

# ---------- chemins ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = MODELS_DIR / "xgb_multiclass.joblib"      # meilleur courant
BEST_META = MODELS_DIR / "best_metrics.json"          # logloss best persistant
BEST_PARAMS_PATH = MODELS_DIR / "xgb_best_params.yaml"
METRICS_HISTORY_CSV = PROC_DIR / "metrics_history.csv"

# ---------- utils ----------
def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z -= np.max(z, axis=1, keepdims=True)
    np.exp(z, out=z)
    z_sum = np.sum(z, axis=1, keepdims=True)
    z /= z_sum
    return z

def _logloss_from_margins(margins, y_enc, T=1.0, n_classes=None):
    proba = _softmax(margins / T)
    labels = np.arange(proba.shape[1] if n_classes is None else n_classes)
    return log_loss(y_enc, proba, labels=labels)

def _fit_temperature(margins, y_enc, n_classes, lo=0.05, hi=5.0, tol=1e-5):
    """Recherche par section dorée pour minimiser le logloss (calibration température)."""
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi
    invphi2 = (3 - 5 ** 0.5) / 2

    a, b = lo, hi
    h = b - a
    if h <= tol:
        return 1.0

    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    fc = _logloss_from_margins(margins, y_enc, T=c, n_classes=n_classes)
    fd = _logloss_from_margins(margins, y_enc, T=d, n_classes=n_classes)

    for _ in range(n):
        if fc < fd:
            b, d, fd = d, c, fc
            h *= invphi
            c = a + invphi2 * h
            fc = _logloss_from_margins(margins, y_enc, T=c, n_classes=n_classes)
        else:
            a, c, fc = c, d, fd
            h *= invphi
            d = a + invphi * h
            fd = _logloss_from_margins(margins, y_enc, T=d, n_classes=n_classes)

    return float((a + b) / 2.0)

def _metrics_hitk(y_true_enc, proba, k=4):
    order = np.argsort(-proba, axis=1)[:, :k]
    hit1 = (order[:, 0] == y_true_enc).mean()
    hitk = np.mean([yy in row for yy, row in zip(y_true_enc, order)])
    return float(hit1), float(hitk)

def _append_history(row: dict):
    df_row = pd.DataFrame([row])
    if METRICS_HISTORY_CSV.exists():
        df_row.to_csv(METRICS_HISTORY_CSV, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df_row.to_csv(METRICS_HISTORY_CSV, index=False, encoding="utf-8")

# ---------- public API ----------
def train_and_save(config_path: str | Path):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    sep = cfg["data"].get("csv_sep", ",")
    target = cfg["columns"]["target"]

    # 1) Données → features
    courses = pd.read_csv(ROOT / cfg["data"]["courses_csv"], sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)
    feat_cols = X.columns.tolist()

    # 2) Encodage labels
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))
    n_classes = len(le.classes_)

    # 3) Split train / valid
    test_size = float(cfg.get("training", {}).get("test_size", 0.2))
    random_state = int(cfg.get("training", {}).get("random_state", 42))
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # 4) DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    dall = xgb.DMatrix(X, label=y_enc)

    # 5) Paramètres XGBoost
    params = cfg["training"]["xgb_params"].copy()
    params["objective"] = "multi:softprob"
    params["num_class"] = n_classes
    params.setdefault("eval_metric", "mlogloss")

    # num_boost_round: compatibilité avec n_estimators
    num_round = params.pop("num_boost_round", None)
    if num_round is None:
        if "n_estimators" in params:
            warnings.warn('xgboost: "n_estimators" n\'est pas utilisé par xgb.train, utilisation comme num_boost_round.', UserWarning)
            num_round = int(params.pop("n_estimators"))
        else:
            num_round = 1000
    early = int(cfg["training"].get("early_stopping_rounds", 100))

    # 6) Entraînement
    evals = [(dtrain, "train"), (dvalid, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=evals,
        early_stopping_rounds=early,
        verbose_eval=False
    )

    # 7) Calibration température (sur validation)
    margins_va = model.predict(dvalid, output_margin=True)
    T_star = _fit_temperature(margins_va, y_va, n_classes)

    # 8) Métriques (post-calibration) ALL + VALID
    margins_all = model.predict(dall, output_margin=True)
    proba_cal_all = _softmax(margins_all / T_star)
    ll_all = log_loss(y_enc, proba_cal_all, labels=np.arange(n_classes))
    hit1_all, hit4_all = _metrics_hitk(y_enc, proba_cal_all, k=4)

    proba_cal_va = _softmax(margins_va / T_star)
    ll_va = log_loss(y_va, proba_cal_va, labels=np.arange(n_classes))
    hit1_va, hit4_va = _metrics_hitk(y_va, proba_cal_va, k=4)

    # 9) Nouveau bundle
    new_bundle = {
        "model": model,  # Booster
        "label_encoder": le,
        "feature_columns": feat_cols,
        "calibrator": {"type": "temperature", "T": float(T_star)}
    }

    # 10) Versionner le run
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cand_path = MODELS_DIR / f"xgb_multiclass_{ts}.joblib"
    joblib.dump(new_bundle, cand_path)

    # 11) Lire BEST logloss persistant (si présent)
    best_ll = float("inf")
    if BEST_META.exists():
        try:
            best_ll = float(json.loads(BEST_META.read_text(encoding="utf-8")).get("logloss_valid", np.inf))
        except Exception:
            pass

    # 12) Comparer au meilleur : ADOPTE si meilleur OU égal (eps)
    eps = 1e-8
    adopted = False
    if ll_va <= best_ll + eps:
        shutil.copy2(cand_path, BEST_PATH)
        BEST_META.write_text(json.dumps({
            "datetime": ts,
            "logloss_valid": float(ll_va),
            "hit1_valid": float(hit1_va),
            "hit4_valid": float(hit4_va),
            "temperature_T": float(T_star),
            "feature_count": int(len(feat_cols)),
            "candidate_path": cand_path.name,
            "params": {**params, "num_boost_round": int(num_round)}
        }, indent=2), encoding="utf-8")
        adopted = True
        print(f"✅ Nouveau modèle adopté (LogLoss VALID {ll_va:.4f} ≤ Best {best_ll:.4f}).")
        print(f"   Version sauvegardée → {cand_path}")
        print(f"   Best → {BEST_PATH}")
    else:
        print(f"↩️ Nouveau modèle ignoré (LogLoss VALID {ll_va:.4f} > Best {best_ll:.4f}).")
        print(f"   Candidate conservé pour historique → {cand_path}")

    # 13) Sauvegarde des meilleurs paramètres (info)
    BEST_PARAMS_PATH.write_text(
        yaml.safe_dump(
            {"xgb_params": {**params, "num_boost_round": int(num_round)}, "temperature_T": float(T_star)},
            sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8"
    )

    # 14) Historique lisible par l’app
    hist_row = {
        "datetime": ts,
        "phase": "VALID",
        "Hit@1": float(hit1_va),
        "Hit@4": float(hit4_va),
        "LogLoss": float(ll_va),
        "adopted": bool(adopted),
        "target": str(target),
        "samples": int(len(X)),
        "features": int(len(feat_cols)),
        "T": float(T_star)
    }
    _append_history(hist_row)

    # 15) Retour pour Streamlit (utilise VALID)
    return {
        "status": "updated_best" if adopted else "kept_old_best",
        "hit1": float(hit1_va),
        "hit4": float(hit4_va),
        "logloss": float(ll_va),
        # extra
        "hit1_all": float(hit1_all),
        "hit4_all": float(hit4_all),
        "logloss_all": float(ll_all),
        "T": float(T_star),
        "versioned_path": str(cand_path),
        "best_path": str(BEST_PATH),
        "adopted": bool(adopted),
    }

if __name__ == "__main__":
    train_and_save(ROOT / "config.yaml")
