# src/update_truth_a1.py
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml

from .features import build_feature_frame
from .metrics import logloss_top4_renorm

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "data" / "processed" / "predictions_top4.csv"

def get_model():
    path = ROOT / "models" / "xgb_multiclass.joblib"
    if not path.exists():
        raise FileNotFoundError("Champion introuvable. Lance d'abord `python -m src.train`.")
    blob = joblib.load(path)
    return blob["model"], blob.get("classes", None)

def parse_args():
    p = argparse.ArgumentParser(description="Valider la vérité (true_a1) + hits + 2 logloss (race-aware & top4).")
    p.add_argument("--date", required=True)
    p.add_argument("--hippodrome", required=True)
    p.add_argument("--numcourse", required=True)
    p.add_argument("--true", required=True, type=int)
    p.add_argument("--pronos", required=True, help="8 numéros: 9,13,15,12,16,1,6,2")
    p.add_argument("--partants", required=True, type=int)
    p.add_argument("--distance", required=True, type=float)
    p.add_argument("--discipline", default="galop")
    return p.parse_args()

def main():
    a = parse_args()
    race_id = f"{a.date}_{a.hippodrome}_{a.numcourse}"
    true_num = int(a.true)
    pronos = [int(x.strip()) for x in a.pronos.split(",")]
    if len(pronos) != 8: raise ValueError("Il faut exactement 8 pronos.")

    # features 1 ligne (comme en prod)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    feat = {
        "date": pd.to_datetime(a.date),
        "hippodrome": a.hippodrome,
        "numcourse": a.numcourse,
        "partants": int(a.partants),
        "distance": float(a.distance),
        "discipline": str(a.discipline).lower(),
        **{f"prono{i}": n for i, n in enumerate(pronos, 1)},
    }
    X = build_feature_frame(pd.DataFrame([feat]), cfg["prono_cols"], cfg["num_features"], cfg["cat_features"])

    # proba champion
    model, classes = get_model()
    proba = model.predict_proba(X)[0]
    Kmax = len(proba)
    K = int(a.partants)

    # p_true race-aware (renormalisée sur 1..K)
    denom = float(proba[:K].sum()) if K <= Kmax else float(proba.sum())
    p_true_race = (float(proba[true_num - 1]) / denom) if (1 <= true_num <= Kmax and denom > 0) else (1.0 / max(K, 1))
    ll_race_item = -np.log(max(p_true_race, 1e-15))

    # Top-4 renorm (depuis le fichier s'il existe, sinon on dérive des pronos)
    if PRED_PATH.exists():
        dfp = pd.read_csv(PRED_PATH)
    else:
        dfp = pd.DataFrame(columns=["race_id"])

    mask = (dfp["race_id"] == race_id) if "race_id" in dfp.columns else pd.Series([], dtype=bool)
    if mask.any():
        row = dfp.loc[mask].iloc[0]
        top_nums = [int(row.get(f"pred{i}_a1")) for i in range(1,5)]
        top_probs = [float(row.get(f"proba{i}")) for i in range(1,5)]
    else:
        pairs = []
        for n in pronos:
            p = float(proba[n - 1]) if 1 <= n <= Kmax else 0.0
            pairs.append((n, p))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_nums = [n for n, _ in pairs[:4]]
        top_probs = [p for _, p in pairs[:4]]

    ll_top4_item = logloss_top4_renorm(true_num, top_nums, top_probs)

    # hits
    hit1 = int(true_num == top_nums[0]) if top_nums else 0
    hit2 = int(true_num in top_nums[:2])
    hit4 = int(true_num in top_nums[:4])

    # upsert
    needed_cols = [
        "race_id","true_a1","pred1_a1","pred2_a1","pred3_a1","pred4_a1","proba1","proba2","proba3","proba4",
        "hit1","hit2","hit4","p_true_raceaware","logloss_item_raceaware","logloss_item_top4"
    ]
    for c in needed_cols:
        if c not in dfp.columns: dfp[c] = pd.NA
    if not mask.any():
        dfp = pd.concat([dfp, pd.DataFrame([{"race_id": race_id}])], ignore_index=True)
        mask = dfp["race_id"] == race_id

    dfp.loc[mask, "true_a1"] = true_num
    dfp.loc[mask, "hit1"] = hit1
    dfp.loc[mask, "hit2"] = hit2
    dfp.loc[mask, "hit4"] = hit4
    dfp.loc[mask, "p_true_raceaware"] = p_true_race
    dfp.loc[mask, "logloss_item_raceaware"] = ll_race_item
    dfp.loc[mask, "logloss_item_top4"] = ll_top4_item

    dfp.to_csv(PRED_PATH, index=False, encoding="utf-8")
    print(f"[A1] {race_id}  true={true_num} | H@1={hit1} H@2={hit2} H@4={hit4} | p_true_race={p_true_race:.4f} | ll_race={ll_race_item:.4f} | ll_top4={ll_top4_item:.4f}")

if __name__ == "__main__":
    main()
