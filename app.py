import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib, yaml
from datetime import date

# --- Compatibilité anciens modèles : wrapper PreprocBooster
# Certains modèles ont été picklés avec une classe PreprocBooster déclarée en __main__.
# On redéclare ici un wrapper minimal qui sait prédire dans tous les cas.
class PreprocBooster:
    def __init__(self, pre=None, clf=None, pipe=None, model=None):
        # selon la version sauvegardée, le pickled state peut contenir pipe OU (pre, clf) OU model
        self.pre = pre
        self.clf = clf
        self.pipe = pipe
        self.model = model

    def predict_proba(self, X):
        # 1) Pipeline sklearn complet
        if self.pipe is not None and hasattr(self.pipe, "predict_proba"):
            return self.pipe.predict_proba(X)
        # 2) Objet 'model' stocké directement
        if self.model is not None and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # 3) Préprocesseur + modèle séparés
        if self.pre is not None and self.clf is not None:
            Xt = self.pre.transform(X)
            return self.clf.predict_proba(Xt)
        raise AttributeError("PreprocBooster: composants manquants (pipe/model ou pre+clf).")
# --- chemins
ROOT = Path(__file__).resolve().parent
DATA_PROC = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "xgb_multiclass.joblib"

# --- imports projet
from src.features import build_feature_frame
from src.metrics import logloss_top4_renorm, hit_at_k

# --- helpers persistants
def ensure_files():
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    preds = DATA_PROC / "predictions_top4.csv"
    if not preds.exists():
        pd.DataFrame(
            columns=[
                "race_id", "true_a1",
                "pred1_a1", "pred2_a1", "pred3_a1", "pred4_a1",
                "proba1", "proba2", "proba3", "proba4",
                "hit1", "hit2", "hit4",
                "p_true_raceaware", "logloss_item_raceaware", "logloss_item_top4",
            ]
        ).to_csv(preds, index=False, encoding="utf-8")
    return preds

def load_model_and_cfg():
    blob = joblib.load(MODEL_PATH)
    model = blob["model"]          # wrapper : .predict_proba(DataFrame)
    classes = blob.get("classes", None)
    num_classes = int(max(classes)) if classes is not None else None
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return model, num_classes, cfg

def append_or_update_row(row: dict):
    preds_path = ensure_files()
    df = pd.read_csv(preds_path)
    if (df["race_id"] == row["race_id"]).any():
        mask = (df["race_id"] == row["race_id"])
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(preds_path, index=False, encoding="utf-8")
    return df

def predict_top4(args):
    model, num_classes, cfg = load_model_and_cfg()
    pronos = [int(x.strip()) for x in args["pronos"].split(",")]
    row = {
        "date": pd.to_datetime(args["date"]),
        "hippodrome": args["hippodrome"],
        "numcourse": args["numcourse"],
        "partants": int(args["partants"]),
        "distance": float(args["distance"]),
        "discipline": str(args.get("discipline", "galop")).lower(),
        **{f"prono{i}": n for i, n in enumerate(pronos, 1)},
    }
    X = build_feature_frame(pd.DataFrame([row]), cfg["prono_cols"], cfg["num_features"], cfg["cat_features"])
    proba_vec = model.predict_proba(X)[0]
    if num_classes is None:
        num_classes = len(proba_vec)

    pairs = []
    for numero in pronos:
        p = float(proba_vec[int(numero) - 1]) if 1 <= int(numero) <= num_classes else 0.0
        pairs.append((int(numero), p))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:4]
    top_nums = [n for n, _ in top]
    top_probs = [p for _, p in top]

    out = {
        "race_id": f'{pd.to_datetime(args["date"]).strftime("%Y-%m-%d")}_{args["hippodrome"]}_{args["numcourse"]}',
        "pred1_a1": top_nums[0] if len(top_nums) > 0 else None,
        "pred2_a1": top_nums[1] if len(top_nums) > 1 else None,
        "pred3_a1": top_nums[2] if len(top_nums) > 2 else None,
        "pred4_a1": top_nums[3] if len(top_nums) > 3 else None,
        "proba1": top_probs[0] if len(top_probs) > 0 else None,
        "proba2": top_probs[1] if len(top_probs) > 1 else None,
        "proba3": top_probs[2] if len(top_probs) > 2 else None,
        "proba4": top_probs[3] if len(top_probs) > 3 else None,
    }
    return out, top_nums, top_probs

def update_truth(args):
    """Calcule p_true race-aware + 2 logloss + hits, et upsert."""
    model, num_classes, cfg = load_model_and_cfg()
    pronos = [int(x.strip()) for x in args["pronos"].split(",")]
    K = int(args["partants"])
    row = {
        "date": pd.to_datetime(args["date"]),
        "hippodrome": args["hippodrome"],
        "numcourse": args["numcourse"],
        "partants": K,
        "distance": float(args["distance"]),
        "discipline": str(args.get("discipline", "galop")).lower(),
        **{f"prono{i}": n for i, n in enumerate(pronos, 1)},
    }
    X = build_feature_frame(pd.DataFrame([row]), cfg["prono_cols"], cfg["num_features"], cfg["cat_features"])
    proba = model.predict_proba(X)[0]
    if num_classes is None:
        num_classes = len(proba)

    # Top4 depuis le fichier si dispo, sinon à partir des proba/pronos
    preds_path = ensure_files()
    dfp = pd.read_csv(preds_path)
    race_id = f'{pd.to_datetime(args["date"]).strftime("%Y-%m-%d")}_{args["hippodrome"]}_{args["numcourse"]}'
    mask = (dfp["race_id"] == race_id) if "race_id" in dfp.columns else pd.Series([], dtype=bool)
    if mask.any():
        rowp = dfp.loc[mask].iloc[0]
        top_nums = [int(rowp.get(f"pred{i}_a1")) for i in range(1, 5)]
        top_probs = [float(rowp.get(f"proba{i}")) for i in range(1, 5)]
    else:
        pairs = []
        for n in pronos:
            p = float(proba[n - 1]) if 1 <= n <= num_classes else 0.0
            pairs.append((n, p))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_nums = [n for n, _ in pairs[:4]]
        top_probs = [p for _, p in pairs[:4]]

    # p_true race-aware
    denom = float(proba[:K].sum()) if K <= len(proba) else float(proba.sum())
    true_num = int(args["true"])
    p_true_race = (
        float(proba[true_num - 1]) / denom
        if (1 <= true_num <= len(proba) and denom > 0)
        else (1.0 / max(K, 1))
    )
    ll_race = -np.log(max(p_true_race, 1e-15))
    ll_top4 = logloss_top4_renorm(true_num, top_nums, top_probs)

    hit1 = int(true_num == top_nums[0]) if top_nums else 0
    hit2 = int(true_num in top_nums[:2])
    hit4 = int(true_num in top_nums[:4])

    out = {
        "race_id": race_id,
        "true_a1": true_num,
        "hit1": hit1,
        "hit2": hit2,
        "hit4": hit4,
        "p_true_raceaware": p_true_race,
        "logloss_item_raceaware": ll_race,
        "logloss_item_top4": ll_top4,
    }
    return out

def compute_dashboard_metrics(df):
    # agrège sur tout le fichier (courses validées uniquement)
    d = df[df["true_a1"].notna()].copy()
    if d.empty:
        return None
    hit1 = pd.to_numeric(d["hit1"], errors="coerce")
    hit4 = pd.to_numeric(d["hit4"], errors="coerce")
    ll_race = pd.to_numeric(d["logloss_item_raceaware"], errors="coerce")
    return {"H1": float(hit1.mean()), "H4": float(hit4.mean()), "LL": float(ll_race.mean())}

def block_two_periods(df, cutoff_str):
    if df.empty:
        return None, None
    df = df.copy()

    def extract_date(rid):
        try:
            return rid.split("_", 1)[0]
        except:
            return None

    df["date"] = pd.to_datetime(df["race_id"].astype(str).map(extract_date), errors="coerce")
    df = df[df["date"].notna() & df["true_a1"].notna()]
    cut = pd.to_datetime(cutoff_str)

    def agg(x):
        return {
            "n": len(x),
            "hit4": float(pd.to_numeric(x["hit4"], errors="coerce").mean()) if len(x) else np.nan,
            "ll_race": float(pd.to_numeric(x["logloss_item_raceaware"], errors="coerce").mean()) if len(x) else np.nan,
            "ll_top4": float(pd.to_numeric(x["logloss_item_top4"], errors="coerce").mean()) if len(x) else np.nan,
        }

    return agg(df[df["date"] <= cut]), agg(df[df["date"] > cut])


# ========================= UI ========================= #
st.set_page_config(page_title="Numero Gagnant – A1", layout="wide")

st.title("🎯 Numero Gagnant – A1 (Top-4)")
preds_path = ensure_files()
dfp = pd.read_csv(preds_path)
dash = compute_dashboard_metrics(dfp)
c1, c2, c3 = st.columns(3)
c1.metric("Hit@1 (exact gagnant)", f"{(dash['H1']*100):.1f}%" if dash else "—")
c2.metric("Hit@4 (gagnant ∈ Top-4)", f"{(dash['H4']*100):.1f}%" if dash else "—")
c3.metric("Vrai LogLoss (race-aware)", f"{dash['LL']:.4f}" if dash else "—")

tabs = st.tabs(["📜 Historique", "➕ Nouvelle course", "✅ Validation", "📈 2 périodes"])

# ---- Historique
with tabs[0]:
    if dfp.empty:
        st.info("Aucune course prédite pour le moment.")
    else:
        rid = st.selectbox("Course", options=dfp["race_id"].tolist(), key="hist_course")
        row = dfp[dfp["race_id"] == rid].iloc[0]
        st.write("**Top-4 prédits**")
        st.dataframe(
            pd.DataFrame(
                {
                    "rang": [1, 2, 3, 4],
                    "numero": [row.get("pred1_a1"), row.get("pred2_a1"), row.get("pred3_a1"), row.get("pred4_a1")],
                    "proba": [row.get("proba1"), row.get("proba2"), row.get("proba3"), row.get("proba4")],
                }
            )
        )

# ---- Nouvelle course
with tabs[1]:
    st.subheader("Renseigner la course & lancer la prédiction Top-4")
    col1, col2, col3 = st.columns(3)
    dte = col1.date_input("Date", value=date.today(), key="new_date")
    hip = col2.text_input("Hippodrome", value="Deauville", key="new_hip")
    nc = col3.text_input("Numcourse", value="C3", key="new_numcourse")
    col4, col5, col6 = st.columns(3)
    part = col4.number_input("Partants", min_value=2, max_value=24, value=16, step=1, key="new_partants")
    dist = col5.number_input("Distance (m)", min_value=400, max_value=5000, value=2000, step=50, key="new_dist")
    disc = col6.selectbox("Discipline", options=["galop", "trot"], index=0, key="new_disc")
    pronos = st.text_input(
        "Pronos (8 numéros séparés par des virgules)",
        value="9,13,15,12,16,1,6,2",
        key="new_pronos",
    )

    if st.button("🔮 Prédire Top-4", key="btn_predict"):
        args = {
            "date": dte.isoformat(),
            "hippodrome": hip.strip(),
            "numcourse": nc.strip(),
            "partants": int(part),
            "distance": float(dist),
            "discipline": disc,
            "pronos": pronos,
        }
        out, top_nums, top_probs = predict_top4(args)
        row = {"race_id": out["race_id"], **out, "true_a1": None}
        dfp = append_or_update_row(row)
        st.success(f"Top-4 pour {out['race_id']}")
        st.dataframe(
            pd.DataFrame(
                {
                    "rang": [1, 2, 3, 4],
                    "numero": [out["pred1_a1"], out["pred2_a1"], out["pred3_a1"], out["pred4_a1"]],
                    "proba": [out["proba1"], out["proba2"], out["proba3"], out["proba4"]],
                }
            )
        )

# ---- Validation
with tabs[2]:
    st.subheader("Valider l’arrivée (vérité) et mettre à jour les métriques")
    if dfp.empty:
        st.info("Aucune course prédite à valider.")
    else:
        rid = st.selectbox("Course à valider", options=dfp["race_id"].tolist(), key="val_rid")
        # on laisse saisir/éditer aussi les méta au cas où :
        c1, c2, c3 = st.columns(3)
        dte2 = c1.text_input("Date (YYYY-MM-DD)", value=rid.split("_", 1)[0], key="val_date")
        hip2 = c2.text_input("Hippodrome", value=rid.split("_")[1], key="val_hip")
        nc2 = c3.text_input("Numcourse", value=rid.split("_")[2], key="val_numcourse")
        c4, c5, c6 = st.columns(3)
        part2 = c4.number_input(
            "Partants",
            min_value=2,
            max_value=24,
            value=int(
                dfp[dfp["race_id"] == rid].get("partants", pd.Series([16])).iloc[0]
                if "partants" in dfp.columns
                else 16
            ),
            step=1,
            key="val_partants",
        )
        dist2 = c5.number_input("Distance (m)", min_value=400, max_value=5000, value=2000, step=50, key="val_dist")
        disc2 = c6.selectbox("Discipline", options=["galop", "trot"], index=0, key="val_disc")
        pronos2 = st.text_input(
            "Pronos (rappel)",
            value=",".join(
                [
                    str(int(dfp.loc[dfp["race_id"] == rid, f"pred{i}_a1"].iloc[0]))
                    if pd.notna(dfp.loc[dfp["race_id"] == rid, f"pred{i}_a1"].iloc[0])
                    else ""
                    for i in range(1, 5)
                ]
            ),
            key="val_pronos",
        )
        true_num = st.number_input("Numéro gagnant (a1)", min_value=1, max_value=24, value=6, step=1, key="val_true")
        if st.button("💾 Valider", key="btn_validate"):
            args = {
                "date": dte2,
                "hippodrome": hip2,
                "numcourse": nc2,
                "true": int(true_num),
                "pronos": pronos2 if pronos2.count(",") >= 3 else "9,13,15,12,16,1,6,2",
                "partants": int(part2),
                "distance": float(dist2),
                "discipline": disc2,
            }
            upd = update_truth(args)
            dfp = append_or_update_row(upd)
            st.success(
                f"Mise à jour OK — H@4={upd['hit4']}  |  ll_race={upd['logloss_item_raceaware']:.4f}  |  ll_top4={upd['logloss_item_top4']:.4f}"
            )

# ---- 2 périodes
with tabs[3]:
    st.subheader("Comparer 2 périodes (fixe vs dynamique)")
    cutoff = st.date_input("Date de coupure", value=date.today(), key="cutoff_date")
    b1, b2 = block_two_periods(dfp, cutoff.isoformat())

    def fmt_pct(x):
        return "—" if (x is None or np.isnan(x)) else f"{100*x:0.2f}%"

    def fmt_ll(x):
        return "—" if (x is None or np.isnan(x)) else f"{x:0.4f}"

    colA, colB = st.columns(2)
    if b1 is None:
        st.info("Aucune course validée.")
    else:
        with colA:
            st.markdown(f"### ≤ {cutoff.isoformat()}  (N={b1['n']})")
            st.write(f"Hit@4 : {fmt_pct(b1['hit4'])}")
            st.write(f"Vrai LogLoss : {fmt_ll(b1['ll_race'])}")
            st.write(f"Faux LogLoss Top-4 : {fmt_ll(b1['ll_top4'])}")
        with colB:
            st.markdown(f"### > {cutoff.isoformat()}  (N={b2['n']})")
            st.write(f"Hit@4 : {fmt_pct(b2['hit4'])}")
            st.write(f"Vrai LogLoss : {fmt_ll(b2['ll_race'])}")
            st.write(f"Faux LogLoss Top-4 : {fmt_ll(b2['ll_top4'])}")
