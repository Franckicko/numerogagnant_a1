import yaml, joblib
import pandas as pd
from pathlib import Path
from .features import build_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
    sep = cfg["data"].get("csv_sep", ",")
    bundle = joblib.load(ROOT/"models"/"xgb_multiclass.joblib")
    model = bundle["model"]
    le = bundle["label_encoder"]
    feat_cols = bundle["feature_columns"]

    courses = pd.read_csv(ROOT / cfg["data"]["courses_csv"], sep=sep, encoding="utf-8")
    X, y, meta, df = build_dataset(courses, cfg)
    # Align columns
    X = X.reindex(columns=feat_cols, fill_value=0)

    proba = model.predict_proba(X)  # shape (n_samples, num_class)

    topk = 4
    top_idx = (-proba).argsort(axis=1)[:, :topk]
    top_labels = le.inverse_transform(top_idx)
    top_proba = proba[[range(proba.shape[0])]*topk, top_idx.T].T

    out_rows = []
    for rid, labels, probs, true in zip(meta["race_id"], top_labels, top_proba, y):
        row = {"race_id": rid, "true_a1": int(true)}
        for i, (lab, p) in enumerate(zip(labels, probs), start=1):
            row[f"pred{i}_a1"] = int(lab)
            row[f"proba{i}"] = float(p)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(ROOT/"data"/"processed"/"predictions_top4.csv", index=False, encoding="utf-8")
    print("✅ Prédictions sauvegardées → data/processed/predictions_top4.csv")

if __name__ == "__main__":
    main()
