# app.py — Numero Gagnant A1 (Top-4)

from pathlib import Path
from datetime import date
import joblib, yaml
import pandas as pd
import numpy as np
import streamlit as st

# --- Page & debug minimal ----------------------------------------------------
st.set_page_config(page_title="Numero Gagnant – A1", layout="wide")
st.write("🚀 App boot…")

# Affiche les versions (utile pour diagnostiquer les environnements)
try:
    import sklearn, numpy as _np, pandas as _pd
    st.caption(f"sklearn {sklearn.__version__} | numpy {_np.__version__} | pandas {_pd.__version__}")
except Exception:
    pass

# --- Chemins -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PROC = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "xgb_multiclass.joblib"

# --- Imports projet ----------------------------------------------------------
from src.features import build_feature_frame
from src.metrics import logloss_top4_renorm, hit_at_k

# --- Shim pour ceux des modèles picklés avec un wrapper "PreprocBooster"
# Il reconstruit un vrai Pipeline et expose predict_proba(X)
from sklearn.pipeline import Pipeline

class PreprocBooster:
    def __init__(self, pre=None, clf=None, pipe=None, model=None):
        self.pre = pre
        self.clf = clf
        self.pipe = pipe
        self.model = model
        self._cached = None  # pipeline reconstruit / modèle utilisable

    def _as_estimator(self):
        if self._cached is not None:
            return self._cached
        # 1) déjà un pipeline complet
        if self.pipe is not None:
            self._cached = self.pipe
        # 2) un modèle final prêt
        elif self.model is not None:
            self._cached = self.model
        # 3) préprocesseur + classifieur => Pipeline(pre, clf)
        elif (self.pre is not None) and (self.clf is not None):
            self._cached = Pipeline([("pre", self.pre), ("clf", self.clf)])
        else:
            # dernier recours: l’objet lui-même (laisser lever si pas d’API)
            self._cached = self
        return self._cached

    # API attendue par l’app
    def predict_proba(self, X):
        est = self._as_estimator()
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)
        raise AttributeError("PreprocBooster shim: aucun 'predict_proba' disponible")

# =============================================================================
#                               UTILITAIRES
# =============================================================================
def ensure_files() -> Path:
    """S'assure que le dossier processed et le CSV existent, retourne le chemin du CSV."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    preds = DATA_PROC / "predictions_top4.csv"
    if not preds.exists():
        pd.DataFrame(
            columns=[
                "race_id", "date", "hippodrome", "numcourse",
                "partants", "distance", "discipline",
                "true_a1",
                "pred1_a1","pred2_a1","pred3_a1","pred4_a1",
                "proba1","proba2","proba3","proba4",
                "hit1","hit2","hit4",
                "p_true_raceaware","logloss_item_raceaware","logloss_item_top4",
            ]
        ).to_csv(preds, index=False, encoding="utf-8")
    return preds


def load_model_and_cfg():
    blob = joblib.load(MODEL_PATH)
    # Le modèle sauvé peut être {"model": obj, "classes": [...] } ou directement l'objet
    if isinstance(blob, dict):
        model = blob.get("model")
        classes = blob.get("classes")
    else:
        model = blob
        classes = None

    # Si c’est un PreprocBooster (shim), on garde : il expose predict_proba()
    # Sinon, on le laisse tel quel (Pipeline, XGBClassifier, …)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    num_classes = int(max(classes)) if classes is not None else None
    return model, num_classes, cfg


def append_or_update_row(row: dict) -> pd.DataFrame:
    """Upsert d'une ligne dans le CSV des prédictions/validations."""
    preds_path = ensure_files()
    df = pd.read_csv(preds_path)
    if not df.empty and "race_id" in df.columns and (df["race_id"] == row["race_id"]).any():
        mask = df["race_id"] == row["race_id"]
        for k, v in row.items():
            if k in df.columns:
                df.loc[mask, k] = v
            else:
                # si nouvelle colonne, on l'ajoute proprement
                df[k] = np.nan
                df.loc[mask, k] = v
    else:
        # s'assure que toutes colonnes existent
        for k in row.keys():
            if k not in df.columns:
                df[k] = np.nan
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(preds_path, index=False, encoding="utf-8")
    return df


def _race_id(dte: str, hippodrome: str, numcourse: str) -> str:
    return f'{pd.to_datetime(dte).strftime("%Y-%m-%d")}_{hippodrome}_{numcourse}'


# =============================================================================
#                           PRÉDICTION & MISE À JOUR
# =============================================================================
def predict_top4(args: dict):
    """Calcule le Top-4 ordonné à partir des pronos et des proba du modèle."""
    model, classes, cfg = load_model_and_cfg()
    pronos = [int(x.strip()) for x in str(args["pronos"]).split(",") if x.strip()]
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
    num_classes = int(max(classes)) if classes is not None else len(proba_vec)

    # proba par numéro de nos pronos uniquement, triées décroissantes
    pairs = []
    for numero in pronos:
        p = float(proba_vec[int(numero) - 1]) if 1 <= int(numero) <= num_classes else 0.0
        pairs.append((int(numero), p))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:4]
    top_nums = [n for n, _ in top]
    top_probs = [p for _, p in top]

    out = {
        "race_id": _race_id(args["date"], args["hippodrome"], args["numcourse"]),
        "date": pd.to_datetime(args["date"]).strftime("%Y-%m-%d"),
        "hippodrome": args["hippodrome"],
        "numcourse": args["numcourse"],
        "partants": int(args["partants"]),
        "distance": float(args["distance"]),
        "discipline": str(args.get("discipline", "galop")).lower(),
        "pred1_a1": top_nums[0] if len(top_nums) > 0 else np.nan,
        "pred2_a1": top_nums[1] if len(top_nums) > 1 else np.nan,
        "pred3_a1": top_nums[2] if len(top_nums) > 2 else np.nan,
        "pred4_a1": top_nums[3] if len(top_nums) > 3 else np.nan,
        "proba1": top_probs[0] if len(top_probs) > 0 else np.nan,
        "proba2": top_probs[1] if len(top_probs) > 1 else np.nan,
        "proba3": top_probs[2] if len(top_probs) > 2 else np.nan,
        "proba4": top_probs[3] if len(top_probs) > 3 else np.nan,
    }
    return out, top_nums, top_probs


def update_truth(args: dict):
    """Calcule p_true (race-aware) + logloss (race/top4) + hits, et upsert."""
    model, classes, cfg = load_model_and_cfg()
    pronos = [int(x.strip()) for x in str(args["pronos"]).split(",") if x.strip()]
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
    num_classes = int(max(classes)) if classes is not None else len(proba)

    # Récupère un Top-4 déjà stocké si présent (sinon à partir du modèle)
    preds_path = ensure_files()
    dfp = pd.read_csv(preds_path)
    race_id = _race_id(args["date"], args["hippodrome"], args["numcourse"])
    mask = (dfp["race_id"] == race_id) if "race_id" in dfp.columns and not dfp.empty else pd.Series([], dtype=bool)
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

    # p_true race-aware (renormalisation sur les K partants)
    denom = float(proba[:K].sum()) if K <= len(proba) else float(proba.sum())
    true_num = int(args["true"])
    p_true_race = (float(proba[true_num - 1]) / denom) if (1 <= true_num <= len(proba) and denom > 0) else (1.0 / max(K, 1))
    ll_race = -np.log(max(p_true_race, 1e-15))
    ll_top4 = logloss_top4_renorm(true_num, top_nums, top_probs)

    hit1 = int(true_num == top_nums[0]) if top_nums else 0
    hit2 = int(true_num in top_nums[:2])
    hit4 = int(true_num in top_nums[:4])

    out = {
        "race_id": race_id,
        "date": pd.to_datetime(args["date"]).strftime("%Y-%m-%d"),
        "hippodrome": args["hippodrome"],
        "numcourse": args["numcourse"],
        "partants": K,
        "distance": float(args["distance"]),
        "discipline": str(args.get("discipline", "galop")).lower(),
        "true_a1": true_num,
        "hit1": hit1, "hit2": hit2, "hit4": hit4,
        "p_true_raceaware": p_true_race,
        "logloss_item_raceaware": ll_race,
        "logloss_item_top4": ll_top4,
    }
    return out


# =============================================================================
#                               MÉTRIQUES UI
# =============================================================================
def compute_dashboard_metrics(df: pd.DataFrame):
    d = df[df["true_a1"].notna()].copy() if not df.empty else pd.DataFrame()
    if d.empty:
        return None
    hit1 = pd.to_numeric(d["hit1"], errors="coerce")
    hit4 = pd.to_numeric(d["hit4"], errors="coerce")
    ll_race = pd.to_numeric(d["logloss_item_raceaware"], errors="coerce")
    return {"H1": float(hit1.mean()), "H4": float(hit4.mean()), "LL": float(ll_race.mean())}


def block_two_periods(df: pd.DataFrame, cutoff_str: str):
    if df.empty:
        return None, None
    df = df.copy()

    def extract_date(rid):
        try:
            return rid.split("_", 1)[0]
        except Exception:
            return None

    df["date"] = pd.to_datetime(df["race_id"].astype(str).map(extract_date), errors="coerce")
    df = df[df["date"].notna() & df["true_a1"].notna()]
    cut = pd.to_datetime(cutoff_str)

    def agg(x):
        if len(x) == 0:
            return {"n": 0, "hit4": np.nan, "ll_race": np.nan, "ll_top4": np.nan}
        return {
            "n": int(len(x)),
            "hit4": float(pd.to_numeric(x["hit4"], errors="coerce").mean()),
            "ll_race": float(pd.to_numeric(x["logloss_item_raceaware"], errors="coerce").mean()),
            "ll_top4": float(pd.to_numeric(x["logloss_item_top4"], errors="coerce").mean()),
        }

    return agg(df[df["date"] <= cut]), agg(df[df["date"] > cut])


# =============================================================================
#                                   UI
# =============================================================================
try:
    preds_path = ensure_files()
    dfp = pd.read_csv(preds_path)
    dash = compute_dashboard_metrics(dfp)

    st.title("🎯 Numero Gagnant – A1 (Top-4)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hit@1 (exact gagnant)", f"{(dash['H1']*100):.1f}%" if dash else "—")
    c2.metric("Hit@4 (gagnant ∈ Top-4)", f"{(dash['H4']*100):.1f}%" if dash else "—")
    c3.metric("Vrai LogLoss (race-aware)", f"{dash['LL']:.4f}" if dash else "—")

    # ========================= UI ========================= #

tabs = st.tabs([
    "📜 Historique",
    "➕ Nouvelle course",
    "✅ Validation",
    "📈 2 périodes",
])

# ---- Historique
with tabs[0]:
    if dfp.empty:
        st.info("Aucune course prédite pour le moment.")
    else:
        rid = st.selectbox("Course", options=dfp["race_id"].tolist(), key="hist_rid")
        row = dfp[dfp["race_id"] == rid].iloc[0]
        st.write("**Top-4 prédits**")
        st.dataframe(pd.DataFrame({
            "rang": [1, 2, 3, 4],
            "numero": [row.get("pred1_a1"), row.get("pred2_a1"),
                       row.get("pred3_a1"), row.get("pred4_a1")],
            "proba": [row.get("proba1"), row.get("proba2"),
                      row.get("proba3"), row.get("proba4")]
        }))

# ---- Nouvelle course
with tabs[1]:
    try:
        st.subheader("Renseigner la course & lancer la prédiction Top-4")
        col1, col2, col3 = st.columns(3)
        dte = col1.date_input("Date", value=date.today(), key="nc_date")
        hip = col2.text_input("Hippodrome", value="Deauville", key="nc_hip")
        nc  = col3.selectbox("Numcourse", [f"C{i}" for i in range(1, 11)], index=2, key="nc_course")

        col4, col5, col6 = st.columns(3)
        part = col4.number_input("Partants", min_value=2, max_value=24, value=16, step=1, key="nc_partants")
        dist = col5.number_input("Distance (m)", min_value=400, max_value=5000, value=2000, step=50, key="nc_dist")
        disc = col6.selectbox("Discipline", options=["galop", "trot"], index=0, key="nc_disc")

        st.caption("Pronos (8 numéros)")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pd1, pd2, pd3, pd4 = st.columns(4)
        p1 = pc1.number_input("prono1", 1, 24, 9,  key="p1")
        p2 = pc2.number_input("prono2", 1, 24, 13, key="p2")
        p3 = pc3.number_input("prono3", 1, 24, 15, key="p3")
        p4 = pc4.number_input("prono4", 1, 24, 12, key="p4")
        p5 = pd1.number_input("prono5", 1, 24, 16, key="p5")
        p6 = pd2.number_input("prono6", 1, 24, 1,  key="p6")
        p7 = pd3.number_input("prono7", 1, 24, 6,  key="p7")
        p8 = pd4.number_input("prono8", 1, 24, 2,  key="p8")

        if st.button("🔮 Prédire Top-4", key="btn_predire_top4"):
            args = {
                "date": dte.isoformat(),
                "hippodrome": hip.strip(),
                "numcourse": nc.strip(),
                "partants": int(part),
                "distance": float(dist),
                "discipline": disc,
                "pronos": ",".join(map(str, [p1, p2, p3, p4, p5, p6, p7, p8])),
            }
            out, top_nums, top_probs = predict_top4(args)
            row = {"race_id": out["race_id"], **out, "true_a1": None}
            dfp = append_or_update_row(row)
            st.success(f"Top-4 pour {out['race_id']}")
            st.dataframe(pd.DataFrame({
                "rang":   [1, 2, 3, 4],
                "numero": [out['pred1_a1'], out['pred2_a1'], out['pred3_a1'], out['pred4_a1']],
                "proba":  [out['proba1'],   out['proba2'],   out['proba3'],   out['proba4']],
            }))
    except Exception as e:
        st.error("Erreur dans 'Nouvelle course'")
        st.exception(e)

# ---- Validation
with tabs[2]:
    try:
        st.subheader("Valider l’arrivée (vérité) et mettre à jour les métriques")
        if dfp.empty:
            st.info("Aucune course prédite à valider.")
        else:
            rid = st.selectbox("Course à valider", options=dfp["race_id"].tolist(), key="val_rid")
            c1, c2, c3 = st.columns(3)
            dte2 = c1.text_input("Date (YYYY-MM-DD)", value=rid.split("_", 1)[0], key="val_date")
            hip2 = c2.text_input("Hippodrome", value=rid.split("_")[1], key="val_hip")
            nc2  = c3.text_input("Numcourse", value=rid.split("_")[2], key="val_course")
            c4, c5, c6 = st.columns(3)
            part2 = c4.number_input("Partants", 2, 24, 16, 1, key="val_part")
            dist2 = c5.number_input("Distance (m)", 400, 5000, 2000, 50, key="val_dist")
            disc2 = c6.selectbox("Discipline", ["galop", "trot"], index=0, key="val_disc")

            # on re-propose les 8 pronos (par défaut : ceux du Top-4 si dispo)
            pr_def = []
            for i in range(1, 5):
                v = dfp.loc[dfp["race_id"] == rid, f"pred{i}_a1"]
                pr_def.append(int(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else i)
            pr_def += [5, 6, 7, 8]
            c7, c8, c9, c10 = st.columns(4)
            c11, c12, c13, c14 = st.columns(4)
            q1 = c7.number_input("prono1", 1, 24, pr_def[0], key="val_p1")
            q2 = c8.number_input("prono2", 1, 24, pr_def[1], key="val_p2")
            q3 = c9.number_input("prono3", 1, 24, pr_def[2], key="val_p3")
            q4 = c10.number_input("prono4", 1, 24, pr_def[3], key="val_p4")
            q5 = c11.number_input("prono5", 1, 24, pr_def[4], key="val_p5")
            q6 = c12.number_input("prono6", 1, 24, pr_def[5], key="val_p6")
            q7 = c13.number_input("prono7", 1, 24, pr_def[6], key="val_p7")
            q8 = c14.number_input("prono8", 1, 24, pr_def[7], key="val_p8")

            true_num = st.number_input("Numéro gagnant (a1)", 1, 24, 6, 1, key="val_true")
            if st.button("💾 Valider", key="btn_valider"):
                args = {
                    "date": dte2, "hippodrome": hip2, "numcourse": nc2,
                    "true": int(true_num),
                    "pronos": ",".join(map(str, [q1, q2, q3, q4, q5, q6, q7, q8])),
                    "partants": int(part2), "distance": float(dist2), "discipline": disc2,
                }
                upd = update_truth(args)
                dfp = append_or_update_row(upd)
                st.success(
                    f"Mise à jour OK — H@4={upd['hit4']}  |  "
                    f"ll_race={upd['logloss_item_raceaware']:.4f}  |  "
                    f"ll_top4={upd['logloss_item_top4']:.4f}"
                )
    except Exception as e:
        st.error("Erreur dans 'Validation'")
        st.exception(e)

# ---- 2 périodes
with tabs[3]:
    try:
        st.subheader("Comparer 2 périodes (fixe vs dynamique)")
        cutoff = st.date_input("Date de coupure", value=date.today(), key="cutoff")
        b1, b2 = block_two_periods(dfp, cutoff.isoformat())
        def fmt_pct(x): return "—" if (x is None or np.isnan(x)) else f"{100*x:0.2f}%"
        def fmt_ll(x):  return "—" if (x is None or np.isnan(x)) else f"{x:0.4f}"
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
    except Exception as e:
        st.error("Erreur dans '2 périodes'")
        st.exception(e)
    import traceback
    st.error("❌ Erreur au chargement de l’application")
    st.code(traceback.format_exc())
