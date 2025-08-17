# src/update_truth_a1.py
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml

from .data_io import load_raw
from .features import build_feature_frame

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "data" / "processed" / "predictions_top4.csv"

def get_model():
    path = ROOT / "models" / "xgb_multiclass.joblib"
    if not path.exists():
        raise FileNotFoundError("Champion introuvable. Lance d'abord `python -m src.train`.")
    blob = joblib.load(path)
    return blob["model"], blob.get("classes", None)

def parse_args():
    p = argparse.ArgumentParser(description="Valider la vérité (true_a1) et calculer hits + logloss_item.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--hippodrome", required=True)
    p.add_argument("--numcourse", required=True)
    p.add_argument("--true", required=True, type=int, help="numéro gagnant pour a1")
    return p.parse_args()

def main():
    args = parse_args()
    race_id = f"{args.date}_{args.hippodrome}_{args.numcourse}"
    true_num = int(args.true)

    # 1) récupérer la ligne de la course dans le brut
    raw = load_raw()
    raw["race_id_norm"] = raw["date"].dt.strftime("%Y-%m-%d") + "_" + raw["hippodrome"].astype(str) + "_" + raw["numcourse"].astype(str)
    row = raw.loc[raw["race_id_norm"] == race_id]
    if row.empty:
        raise ValueError(f"Course introuvable dans le brut: {race_id}")
    row = row.iloc[0]

    # 2) reconstruire les features comme en prod
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    feat_row = {
        "date": pd.to_datetime(args.date),
        "hippodrome": args.hippodrome,
        "numcourse": args.numcourse,
        "partants": row.get("partants"),
        "distance": row.get("distance"),
        "discipline": str(row.get("discipline", "galop")).lower(),
    }
    for k in range(1, 9):
        feat_row[f"prono{k}"] = row.get(f"prono{k}")
    X = build_feature_frame(pd.DataFrame([feat_row]), cfg["prono_cols"], cfg["num_features"], cfg["cat_features"])

    # 3) proba de toutes les classes via le champion
    model, classes = get_model()
    proba_vec = model.predict_proba(X)[0]
    num_classes = int(max(classes)) if classes is not None else len(proba_vec)

    # proba de la vraie classe (0-based index)
    if 1 <= true_num <= num_classes:
        p_true = float(proba_vec[true_num - 1])
    else:
        p_true = 0.0
    logloss_item = -np.log(max(p_true, 1e-15))

    # 4) charger/mettre à jour predictions_top4.csv
    if PRED_PATH.exists():
        dfp = pd.read_csv(PRED_PATH)
    else:
        dfp = pd.DataFrame(columns=[
            "race_id","true_a1","pred1_a1","pred2_a1","pred3_a1","pred4_a1",
            "proba1","proba2","proba3","proba4"
        ])

    # upsert (si la course existe déjà on met à jour, sinon on crée)
    if (dfp["race_id"] == race_id).any():
        mask = dfp["race_id"] == race_id
    else:
        dfp = pd.concat([dfp, pd.DataFrame([{"race_id": race_id}])], ignore_index=True)
        mask = dfp["race_id"] == race_id

    # s'assurer des colonnes
    for c in ["true_a1","pred1_a1","pred2_a1","pred3_a1","pred4_a1",
              "proba1","proba2","proba3","proba4","hit1","hit2","hit4",
              "p_true","logloss_item"]:
        if c not in dfp.columns:
            dfp[c] = pd.NA

    # hits
    def _get_int(val):
        try: return int(val)
        except: return None
    preds = [_get_int(dfp.loc[mask, f"pred{i}_a1"].iloc[0]) for i in range(1,5)]
    hit1 = 1 if true_num == (preds[0] or -1) else 0
    hit2 = 1 if true_num in [p for p in preds[:2] if p] else 0
    hit4 = 1 if true_num in [p for p in preds if p] else 0

    # write
    dfp.loc[mask, "true_a1"] = true_num
    dfp.loc[mask, "hit1"] = hit1
    dfp.loc[mask, "hit2"] = hit2
    dfp.loc[mask, "hit4"] = hit4
    dfp.loc[mask, "p_true"] = p_true
    dfp.loc[mask, "logloss_item"] = logloss_item

    dfp.to_csv(PRED_PATH, index=False, encoding="utf-8")
    print(f"[A1] {race_id}  true={true_num}  p_true={p_true:.4f}  logloss_item={logloss_item:.4f}  hits: H@1={hit1}, H@2={hit2}, H@4={hit4}")

if __name__ == "__main__":
    main()
