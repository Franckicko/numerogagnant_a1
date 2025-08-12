# src/train.py
import yaml, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb
from .features import build_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "xgb_multiclass.joblib"

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
    """Recherche par section dorée pour minimiser le logloss."""
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi
    invphi2 = (3 - 5 ** 0.5) / 2

    a, b = lo, hi
    h = b - a
    if h <= tol:
        return 1.0

    # points internes
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

    # 4) Préparer DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    dall = xgb.DMatrix(X, label=y_enc)

    # 5) Paramètres XGBoost
    params = cfg["training"]["xgb_params"].copy()
    params["objective"] = "multi:softprob"
    params["num_class"] = n_classes
    params.setdefault("eval_metric", "mlogloss")

    early = int(cfg["training"].get("early_stopping_rounds", 100))

    # 6) Entraînement
    evals = [(dtrain, "train"), (dvalid, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 1000),
        evals=evals,
        early_stopping_rounds=early,
        verbose_eval=False
    )

    # 7) Calibration température
    margins_va = model.predict(dvalid, output_margin=True)
    T_star = _fit_temperature(margins_va, y_va, n_classes)

    # 8) Métriques globales (post-calibration)
    margins_all = model.predict(dall, output_margin=True)
    proba_cal = _softmax(margins_all / T_star)

    ll = log_loss(y_enc, proba_cal, labels=np.arange(n_classes))
    hit1, hit4 = _metrics_hitk(y_enc, proba_cal, k=4)

    # 9) Sauvegarde bundle
    bundle = {
        "model": model,  # Booster natif
        "label_encoder": le,
        "feature_columns": feat_cols,
        "calibrator": {"type": "temperature", "T": float(T_star)}
    }
    joblib.dump(bundle, MODEL_PATH)

    # 10) Sauvegarde params
    (MODELS_DIR / "xgb_best_params.yaml").write_text(
        yaml.safe_dump({"xgb_params": params, "temperature_T": float(T_star)}, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # 11) Logs
    print(f"✅ Modèle entraîné et sauvegardé → {MODEL_PATH}")
    print(f"   Temperature T = {T_star:.4f}")
    print(f"📊 Métriques globales (post-calibration)")
    print(f" - Hit@1={hit1:.3f}")
    print(f" - Hit@4={hit4:.3f}")
    print(f" - LogLoss={ll:.4f}")

    return {"hit1": hit1, "hit4": hit4, "logloss": ll, "T": T_star}

if __name__ == "__main__":
    train_and_save(ROOT / "config.yaml")
