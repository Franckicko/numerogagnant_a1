# src/predict.py
import yaml, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
import xgboost as xgb
from .features import build_dataset
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

def _predict_proba_any(model, X, n_classes: int):
    """Retourne proba (n_samples, n_classes) pour Booster ou XGBClassifier."""
    # Cas scikit-learn API
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    # Cas Booster natif
    dmat = xgb.DMatrix(X)
    raw = model.predict(dmat)  # multi:softprob -> déjà proba
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw.reshape(-1, n_classes)
    return raw

def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
    sep = cfg["data"].get("csv_sep", ",")
    target = cfg["columns"]["target"]

    # Charger bundle
    bundle = joblib.load(ROOT / "models" / "xgb_multiclass.joblib")
    model = bundle["model"]
    le = bundle["label_encoder"]
    feat_cols = bundle["feature_columns"]
    calibrator = bundle.get("calibrator", {})

    # Construire features
    courses = pd.read_csv(ROOT / cfg["data"]["courses_csv"], sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)
    X = X.reindex(columns=feat_cols, fill_value=0)

    n_classes = len(le.classes_)

    # Prédiction proba
    proba = _predict_proba_any(model, X, n_classes=n_classes)

    # Calibration température si dispo
    if calibrator.get("type") == "temperature":
        T = float(calibrator.get("T", 1.0))
        # Idéal: recalculer les margins puis softmax(margins / T)
        margins = None
        try:
            # Booster natif
            if not hasattr(model, "predict_proba"):
                dmat = xgb.DMatrix(X)
                margins = model.predict(dmat, output_margin=True)
            else:
                # XGBClassifier (si version supporte output_margin)
                margins = model.predict(X, output_margin=True)
        except Exception:
            margins = None

        if margins is not None:
            # softmax sur margins/T
            z = margins / T
            z = z - z.max(axis=1, keepdims=True)
            proba = np.exp(z)
            proba /= proba.sum(axis=1, keepdims=True)
        else:
            # fallback (approx) si margins indisponible: “tempérer” les proba
            z = np.log(np.clip(proba, 1e-15, 1.0))
            z = z / T
            z = z - z.max(axis=1, keepdims=True)
            proba = np.exp(z)
            proba /= proba.sum(axis=1, keepdims=True)

    # Top-K
    topk = 4
    top_idx = (-proba).argsort(axis=1)[:, :topk]
    top_labels = np.vstack([le.inverse_transform(row) for row in top_idx])
    rows = np.arange(proba.shape[0])[:, None]
    top_proba = proba[rows, top_idx]

    # DataFrame sortie
    out_rows = []
    for rid, labels, probs, true in zip(meta["race_id"], top_labels, top_proba, y):
        row = {"race_id": rid, f"true_{target}": int(true)}
        for i, (lab, p) in enumerate(zip(labels, probs), start=1):
            row[f"pred{i}_{target}"] = int(lab)
            row[f"proba{i}"] = float(p)
        out_rows.append(row)
    out = pd.DataFrame(out_rows)

    # Sauvegarde CSV
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions_top4.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Prédictions sauvegardées → {out_path}")

    # Métriques globales
    hit1 = (out[f"pred1_{target}"] == out[f"true_{target}"]).mean()
    hit4 = out.apply(
        lambda r: r[f"true_{target}"] in [r[f"pred{i}_{target}"] for i in range(1, 5)],
        axis=1
    ).mean()
    y_true_enc = le.transform(out[f"true_{target}"])
    ll = log_loss(y_true_enc, proba, labels=np.arange(n_classes))
    proba1_mean = out["proba1"].mean()

    print("📊 Métriques globales")
    print(f" - n={len(out)}")
    print(f" - Hit@1={hit1:.3f}")
    print(f" - Hit@4={hit4:.3f}")
    print(f" - LogLoss={ll:.4f}")
    print(f" - Proba1_mean={proba1_mean:.3f}")

     # Sauvegarde des métriques dans un fichier texte
    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(
        "\n".join([
            f"n={len(out)}",
            f"Hit@1={hit1:.3f}",
            f"Hit@4={hit4:.3f}",
            f"LogLoss={ll:.4f}",
            f"Proba1_mean={proba1_mean:.3f}"
        ]),
        encoding="utf-8"
    )
    print(f"📝 Métriques écrites → {metrics_path}")

    # --- Historique en CSV ---
    history_path = out_dir / "metrics_history.csv"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "datetime": run_time,
        "n": len(out),
        "Hit@1": round(hit1, 4),
        "Hit@4": round(hit4, 4),
        "LogLoss": round(ll, 4),
        "Proba1_mean": round(proba1_mean, 4)
    }])

    if history_path.exists():
        hist_df = pd.read_csv(history_path)
        hist_df = pd.concat([hist_df, new_row], ignore_index=True)
    else:
        hist_df = new_row

    hist_df.to_csv(history_path, index=False, encoding="utf-8")
    print(f"📈 Historique mis à jour → {history_path}")

if __name__ == "__main__":
    main()
