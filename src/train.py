# src/train.py
import yaml, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from .features import build_dataset

# dossier racine du projet
ROOT = Path(__file__).resolve().parents[1]


def train_and_save(cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Entraîne le modèle XGBoost multiclass sur l'historique,
    sauvegarde models/xgb_multiclass.joblib et renvoie des métriques.
    Conçu pour être appelé depuis l'UI Streamlit.
    """
    if cfg_path is None:
        cfg_path = ROOT / "config.yaml"

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    sep = cfg["data"].get("csv_sep", ",")

    courses = pd.read_csv(ROOT / cfg["data"]["courses_csv"], sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)

    # encodage des classes (a1 -> 0..K-1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y_enc
    )

    num_class = len(le.classes_)
    params = cfg["training"]["xgb_params"].copy()
    params["num_class"] = num_class  # pour compat xgboost>=2

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # sauvegarde du bundle
    (ROOT / "models").mkdir(exist_ok=True, parents=True)
    out_path = ROOT / "models" / "xgb_multiclass.joblib"
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_columns": X.columns.tolist(),
        },
        out_path,
    )

    # métriques
    proba_test = model.predict_proba(X_test)  # (n, K)
    order = np.argsort(-proba_test, axis=1)[:, :4]

    class_labels = le.classes_
    top_labels = class_labels[order]          # (n, 4)
    true_labels = class_labels[y_test]        # (n,)

    hit1 = float((top_labels[:, 0] == true_labels).mean())
    hit4 = float(np.mean([t in row for t, row in zip(true_labels, top_labels)]))
    ll = float(log_loss(y_test, proba_test, labels=np.arange(num_class)))

    return {
        "path": str(out_path),
        "score_train": float(model.score(X_train, y_train)),
        "score_test": float(model.score(X_test, y_test)),
        "hit1": hit1,
        "hit4": hit4,
        "logloss": ll,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(num_class),
        "classes": class_labels.tolist(),
    }


def main() -> None:
    """Exécution en CLI : python -m src.train"""
    info = train_and_save(ROOT / "config.yaml")

    print(f"✅ Modèle (multiclass) entraîné → {info['path']}")
    print("Classes :", info["classes"])
    print(f"Score train : {info['score_train']:.3f}")
    print(f"Score test  : {info['score_test']:.3f}")
    print(f"Hit@1 (Top-1 exact)     : {info['hit1']:.3f}")
    print(f"Hit@4 (gagnant dans Top-4): {info['hit4']:.3f}")
    print(f"LogLoss (test)          : {info['logloss']:.4f}")


if __name__ == "__main__":
    main()
