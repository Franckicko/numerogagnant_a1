# app.py
from __future__ import annotations

import os
import streamlit as st
import numpy as np
import pandas as pd, yaml, joblib, json
from pathlib import Path
from sklearn.metrics import log_loss
from datetime import datetime, date
import xgboost as xgb

from src.features import build_dataset
from src.train import train_and_save  # bouton de ré-entraînement

# ---------- constantes ----------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgb_multiclass.joblib"
COURSES_PATH = ROOT / "data" / "raw" / "Courses_Completes_a1.csv"
PENDING_PATH = ROOT / "data" / "raw" / "pending_courses.csv"
HIPPODROMES_PATH = ROOT / "data" / "raw" / "hippodrome.csv"
DRAFT_PATH = ROOT / "data" / "raw" / "ui_draft.json"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions_top4.csv"
METRICS_HIST_PATH = ROOT / "data" / "processed" / "metrics_history.csv"

# ---------- config ----------
st.set_page_config(page_title="Numero Gagnant - Multiclass", layout="wide")
cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
sep = cfg["data"].get("csv_sep", ",")
rid_fields = cfg["columns"]["race_id_fields"]
target_col = cfg["columns"]["target"]  # 'a1'
cat_cfg = cfg["columns"].get("categorical", [])

st.title(f"🎯 Numero Gagnant – Prédiction de {target_col} (Top 4)")

# ---------- session init ----------
# (pas de valeur par défaut pour new_date -> évite le warning Streamlit)
DEFAULTS = {
    "new_discipline": "", "new_hippodrome": "", "new_numcourse": "",
    "new_partants": 12, "new_distance": 2700,
    **{f"new_prono{i}": 1 for i in range(1, 9)},
    "new_row_cached": None, "new_pred_cached": None,
    "__X__": None, "__y__": None, "__meta__": None,
    "__model__": None, "__le__": None, "__feat_cols__": None, "__calib__": None
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- brouillon (persistant) ----------
def _serialize_for_json(val):
    if isinstance(val, date):
        return val.isoformat()
    return val

def _save_draft():
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    payload = {
        "new_date": st.session_state.get("new_date"),
        "new_discipline": st.session_state.get("new_discipline"),
        "new_hippodrome": st.session_state.get("new_hippodrome"),
        "new_numcourse": st.session_state.get("new_numcourse"),
        "new_partants": st.session_state.get("new_partants"),
        "new_distance": st.session_state.get("new_distance"),
        **{f"new_prono{i}": st.session_state.get(f"new_prono{i}") for i in range(1, 9)},
    }
    payload = {k: _serialize_for_json(v) for k, v in payload.items()}
    with open(DRAFT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_draft_into_session():
    if not DRAFT_PATH.exists():
        return
    try:
        data = json.loads(Path(DRAFT_PATH).read_text(encoding="utf-8"))
    except Exception:
        return
    # Date
    if "new_date" in data and data["new_date"]:
        try:
            y_, m_, d_ = [int(x) for x in str(data["new_date"]).split("-")]
            st.session_state["new_date"] = date(y_, m_, d_)
        except Exception:
            pass
    # Strings
    for k in ("new_discipline", "new_hippodrome", "new_numcourse"):
        if k in data and data[k] is not None:
            st.session_state[k] = data[k]
    # Entiers
    for k in ("new_partants", "new_distance", *[f"new_prono{i}" for i in range(1, 9)]):
        if k in data and data[k] is not None:
            try:
                st.session_state[k] = int(float(data[k]))
            except Exception:
                pass

# Charger brouillon avant l’UI
_load_draft_into_session()

# ---------- helpers modèle/proba ----------
def predict_proba_any(model, X, n_classes: int):
    """Renvoie (n_samples, n_classes) pour Booster ou XGBClassifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = proba.reshape(-1, n_classes)
        return proba
    dmat = xgb.DMatrix(X)
    raw = np.asarray(model.predict(dmat))  # Booster: multi:softprob
    if raw.ndim == 1:
        raw = raw.reshape(-1, n_classes)
    return raw

def apply_temperature(model, X, proba, calibrator):
    """Applique la calibration température si présente. Utilise margins si possible."""
    if not calibrator or calibrator.get("type") != "temperature":
        return proba
    T = float(calibrator.get("T", 1.0))
    margins = None
    try:
        if hasattr(model, "predict_proba"):
            margins = model.predict(X, output_margin=True)
        else:
            dmat = xgb.DMatrix(X)
            margins = model.predict(dmat, output_margin=True)
    except Exception:
        margins = None
    if margins is not None:
        z = margins / T
        z = z - z.max(axis=1, keepdims=True)
        proba = np.exp(z)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba
    # fallback via log(proba)
    z = np.log(np.clip(proba, 1e-15, 1.0)) / T
    z = z - z.max(axis=1, keepdims=True)
    proba = np.exp(z)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba

# ---------- utils data ----------
def make_race_id(df: pd.DataFrame) -> pd.Series:
    return df[rid_fields].astype(str).agg("_".join, axis=1)

def backup(path: Path):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bkp = path.with_name(f"{path.stem}_backup_{ts}.csv")
    if path.exists():
        pd.read_csv(path, sep=sep, encoding="utf-8").to_csv(bkp, index=False, sep=sep, encoding="utf-8")
    return bkp

def upsert_pending(save_row: dict) -> str:
    """Ajoute / met à jour dans pending_courses.csv (clé = race_id)."""
    save_row = save_row.copy()
    save_row["race_id"] = "_".join(str(save_row.get(k, "")) for k in rid_fields)
    try:
        hist = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
    except Exception:
        hist = pd.DataFrame()
    if "race_id" not in hist.columns and all(k in hist.columns for k in rid_fields):
        hist["race_id"] = make_race_id(hist)
    if "race_id" in hist.columns and save_row["race_id"] in set(hist["race_id"].astype(str)):
        return "exists_hist"
    if PENDING_PATH.exists():
        pend = pd.read_csv(PENDING_PATH, sep=sep, encoding="utf-8")
    else:
        pend = pd.DataFrame()
    for k in list(save_row.keys()) + [target_col, "rapport"]:
        if k not in pend.columns:
            pend[k] = pd.NA
    if "race_id" in pend.columns:
        mask = pend["race_id"].astype(str) == save_row["race_id"]
        if mask.any():
            idx = pend.index[mask][0]
            for k, v in save_row.items():
                if k in [target_col, "rapport"] and pd.notna(pend.at[idx, k]):
                    continue
                pend.at[idx, k] = v
            pend.to_csv(PENDING_PATH, index=False, sep=sep, encoding="utf-8")
            return "updated"
    pend = pd.concat([pend, pd.DataFrame([save_row])], ignore_index=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    pend.to_csv(PENDING_PATH, index=False, sep=sep, encoding="utf-8")
    return "inserted"

def _reload_all():
    """Relit CSV + modèle et recalcule features. Stocke en session."""
    courses = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)
    st.session_state["__X__"] = X
    st.session_state["__y__"] = y
    st.session_state["__meta__"] = meta
    bundle = joblib.load(MODEL_PATH)
    st.session_state["__model__"] = bundle["model"]
    st.session_state["__le__"] = bundle["label_encoder"]
    st.session_state["__feat_cols__"] = bundle["feature_columns"]
    st.session_state["__calib__"] = bundle.get("calibrator")

def _ensure_loaded():
    if st.session_state["__X__"] is None:
        _reload_all()

def _to_int_display(values):
    out = []
    for v in values:
        try:
            out.append(int(float(v)))
        except Exception:
            out.append(v)
    return out

# ---------- chargement mémoire ----------
_ensure_loaded()
X = st.session_state["__X__"]; y = st.session_state["__y__"]
meta = st.session_state["__meta__"]; model = st.session_state["__model__"]
le = st.session_state["__le__"]; feat_cols = st.session_state["__feat_cols__"]
calibrator = st.session_state["__calib__"]
X_all = X.reindex(columns=feat_cols, fill_value=0)
class_labels = le.classes_
n_classes = len(class_labels)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## Configuration")
    st.json(cfg)
    st.markdown("---")
    if st.button("🔄 Recharger données & métriques"):
        _reload_all()
        st.toast("Données/métriques rechargées.", icon="✅")
        st.rerun()
    if st.button("🔁 Ré-entraîner le modèle"):
        with st.spinner("Entraînement en cours..."):
            info = train_and_save(ROOT / "config.yaml")
        _reload_all()
        st.success(f"Modèle ré-entraîné • Hit@1={info['hit1']:.3f} • Hit@4={info['hit4']:.3f} • LogLoss={info['logloss']:.4f}")
        st.rerun()
with st.sidebar.expander("🛠️ Diagnostic (chemins)"):
    st.write("App path:", Path(__file__).resolve())
    st.write("Working dir:", Path.cwd())
    st.write("Courses CSV:", COURSES_PATH.resolve())
    st.write("Model path:", MODEL_PATH.resolve())

# ---------- onglets ----------
tab_dash, tab_explore, tab_hist, tab_new, tab_validate = st.tabs(
    ["📊 Dashboard", "🔎 Explorer", "Historique (données existantes)", "➕ Nouvelle course", "✅ Validation des résultats"]
)

# ======= Dashboard =======
with tab_dash:
    st.subheader("Vue d’ensemble")
    proba_all = apply_temperature(model, X_all, predict_proba_any(model, X_all, n_classes), calibrator)
    order_all = np.argsort(-proba_all, axis=1)[:, :4]
    top_labels_all = class_labels[order_all]
    y_enc_all = le.transform(y.values)
    true_labels_all = class_labels[y_enc_all]

    hit1 = float((top_labels_all[:, 0] == true_labels_all).mean())
    hit4 = float(np.mean([t in row for t, row in zip(true_labels_all, top_labels_all)]))
    ll = float(log_loss(y_enc_all, proba_all, labels=np.arange(n_classes)))
    proba1_mean = float(proba_all[np.arange(len(proba_all)), order_all[:, 0]].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Hit@1 (Top-1 exact)", f"{hit1:.3f}")
    c2.metric("Hit@4 (Gagnant dans Top-4)", f"{hit4:.3f}")
    c3.metric("LogLoss", f"{ll:.4f}")
    st.caption(f"Proba1_mean: {proba1_mean:.3f}")

    st.markdown("### Historique des runs")
    if METRICS_HIST_PATH.exists():
        hist = pd.read_csv(METRICS_HIST_PATH, encoding="utf-8")
        if not hist.empty and "datetime" in hist.columns:
            try:
                hist["datetime"] = pd.to_datetime(hist["datetime"], errors="coerce")
                hist = hist.dropna(subset=["datetime"]).sort_values("datetime")
                idx = hist["datetime"].astype(str)
            except Exception:
                idx = hist["datetime"]
            cA, cB = st.columns(2)
            with cA:
                cols = [c for c in ["Hit@1", "Hit@4"] if c in hist.columns]
                if cols and len(hist) > 0:
                    st.line_chart(hist.set_index(idx)[cols])
            with cB:
                if "LogLoss" in hist.columns and len(hist) > 0:
                    st.line_chart(hist.set_index(idx)[["LogLoss"]])
            st.dataframe(hist.tail(20), use_container_width=True)
        else:
            st.info("Historique vide. Lance `python -m src.predict` au moins une fois.")
    else:
        st.info("Pas encore de metrics_history.csv. Lance `python -m src.predict`.")

# ======= Explorer =======
def _load_predictions_df():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8")
    if "race_id" in df.columns:
        parts = df["race_id"].astype(str).str.split("_", n=2, expand=True)
        if parts.shape[1] == 3:
            df["date"] = parts[0]
            df["hippodrome"] = parts[1]
            df["numcourse"] = parts[2]
    return df

def _race_detail_row(df_pred, rid):
    if df_pred.empty or rid not in set(df_pred["race_id"]):
        return None
    r = df_pred[df_pred["race_id"] == rid].iloc[0].to_dict()
    top_nums = [int(r.get(f"pred{i}_{target_col}")) for i in range(1, 5)]
    top_probs = [float(r.get(f"proba{i}")) for i in range(1, 5)]
    true_num = int(r.get(f"true_{target_col}"))
    return {
        "race_id": rid, "top_nums": top_nums, "top_probs": top_probs,
        "true": true_num, "hippodrome": r.get("hippodrome"),
        "date": r.get("date"), "numcourse": r.get("numcourse")
    }

with tab_explore:
    st.subheader("Navigation par course & recherche")
    df_pred = _load_predictions_df()

    if df_pred.empty:
        st.info("Aucune prédiction trouvée. Lance `python -m src.predict` pour créer `predictions_top4.csv`.")
    else:
        left, right = st.columns([1, 3])
        with left:
            hippos = ["(tous)"] + (sorted(df_pred["hippodrome"].dropna().unique().tolist())
                                   if "hippodrome" in df_pred.columns else [])
            hippo = st.selectbox("Hippodrome", hippos, index=0, key="explore_hippo")

            mois = ["(tous)"]
            if "date" in df_pred.columns:
                try:
                    tmp = pd.to_datetime(df_pred["date"], errors="coerce")
                    months = tmp.dt.to_period("M").astype(str).dropna().unique().tolist()
                    mois += sorted(months)
                except Exception:
                    pass
            mois_sel = st.selectbox("Mois", mois, index=0, key="explore_month")

        dff = df_pred.copy()
        if hippo != "(tous)" and "hippodrome" in dff.columns:
            dff = dff[dff["hippodrome"] == hippo]
        if mois_sel != "(tous)" and "date" in dff.columns:
            dt = pd.to_datetime(dff["date"], errors="coerce")
            dff = dff[dt.dt.to_period("M").astype(str) == mois_sel]

        race_ids = dff["race_id"].tolist()
        rid = right.selectbox("Course", race_ids, index=0 if race_ids else None, key="explore_course")

        st.markdown("### Tableau des prédictions filtrées")
        show_cols = [c for c in dff.columns if c in (
            "race_id", "date", "hippodrome", "numcourse", f"true_{target_col}",
            f"pred1_{target_col}", "proba1",
            f"pred2_{target_col}", "proba2",
            f"pred3_{target_col}", "proba3",
            f"pred4_{target_col}", "proba4"
        )]
        if show_cols:
            st.dataframe(dff[show_cols], use_container_width=True, height=420)

        if rid:
            st.markdown("### Détail")
            detail = _race_detail_row(df_pred, rid)
            if detail:
                hit = detail["true"] in detail["top_nums"]
                st.write(f"**{rid}** — vrai: **{detail['true']}** — {'✅ dans le Top-4' if hit else '❌ pas dans le Top-4'}")
                fig_data = pd.DataFrame({"rang": [1, 2, 3, 4], "numero": detail["top_nums"], "proba": detail["top_probs"]})
                st.bar_chart(fig_data.set_index("rang")["proba"])

# ======= Historique (données existantes) =======
with tab_hist:
    race_list = meta["race_id"].tolist()
    rid = st.selectbox("Course", race_list, key="hist_course")
    if rid:
        idx = race_list.index(rid)
        x_row = X_all.iloc[[idx]]
        proba_row = apply_temperature(model, x_row, predict_proba_any(model, x_row, n_classes), calibrator)[0]
        order = proba_row.argsort()[::-1][:4]
        labels = class_labels[order]
        numeros = _to_int_display(labels)
        st.subheader("Top 4 numéros prédits")
        st.dataframe(pd.DataFrame({"rang": [1, 2, 3, 4], "numero": numeros, "proba": proba_row[order]}))
        vrai = class_labels[le.transform([y.iloc[idx]])][0]
        st.info(f"Vrai {target_col} : **{int(vrai)}** — "
                f"{'✅ présent dans le Top-4' if int(vrai) in numeros else '❌ absent du Top-4'}")

# ======= Nouvelle course (ENTIERS + persistance) =======
with tab_new:
    st.subheader(f"Saisir une nouvelle course (sans {target_col})")
    numcourse_opts = [f"C{i}" for i in range(1, 11)]
    colA, colB, colC, colD = st.columns(4)

    # IMPORTANT : pas de "value=" -> la valeur vient du Session State si présent
    colA.date_input("date", key="new_date", on_change=_save_draft)
    colB.selectbox("discipline", ["", "trot", "galop"], key="new_discipline", on_change=_save_draft)

    # --- liste hippodromes (robuste + cache) ---
    @st.cache_data(show_spinner=False)
    def _load_hippo_list(path: Path, sep: str) -> list[str]:
        # 1) depuis hippodrome.csv
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            candidates = [c for c in df.columns if c.lower() in ("hippodrome", "nom", "name")]
            col = candidates[0] if candidates else df.columns[0]
            vals = sorted(pd.Series(df[col].astype(str)).dropna().unique().tolist())
            return vals
        except Exception:
            pass
        # 2) fallback depuis l'historique
        try:
            df = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
            if "hippodrome" in df.columns:
                vals = sorted(pd.Series(df["hippodrome"].astype(str)).dropna().unique().tolist())
                return vals
        except Exception:
            pass
        # 3) sinon, liste vide
        return []

    hippo_list = _load_hippo_list(HIPPODROMES_PATH, sep)
    if hippo_list:
        colC.selectbox("hippodrome", [""] + hippo_list, key="new_hippodrome", on_change=_save_draft)
    else:
        colC.text_input("hippodrome", key="new_hippodrome", on_change=_save_draft)

    colD.selectbox("numcourse", [""] + numcourse_opts, key="new_numcourse", on_change=_save_draft)

    col1, col2 = st.columns(2)
    col1.number_input("partants", min_value=12, max_value=20, step=1, format="%d",
                      key="new_partants", on_change=_save_draft)
    col2.number_input("distance (m)", min_value=800, max_value=6000, step=50, format="%d",
                      key="new_distance", on_change=_save_draft)

    st.markdown("**Pronostics (prono1..prono8)**")
    cols = st.columns(8)
    for i in range(1, 9):
        cols[i - 1].number_input(f"prono{i}", min_value=1, max_value=20, step=1, format="%d",
                                 key=f"new_prono{i}", on_change=_save_draft)

    def _required_filled():
        req = [st.session_state.get("new_date"), st.session_state.get("new_discipline"),
               st.session_state.get("new_hippodrome"), st.session_state.get("new_numcourse"),
               st.session_state.get("new_partants")]
        return all(x not in ("", None) for x in req)

    c_pred, c_save, c_reset = st.columns([1, 1, 1])

    if c_pred.button("Prédire le Top-4", type="primary"):
        if not _required_filled():
            st.warning("Merci de renseigner au minimum date, discipline, hippodrome, numcourse et partants.")
        else:
            new_row = {
                "date": pd.to_datetime(st.session_state["new_date"]).strftime("%Y-%m-%d"),
                "discipline": st.session_state["new_discipline"],
                "hippodrome": st.session_state["new_hippodrome"],
                "numcourse": st.session_state["new_numcourse"],
                "partants": int(st.session_state["new_partants"]),
                "distance": int(st.session_state["new_distance"]),
                **{f"prono{i}": int(st.session_state.get(f"new_prono{i}", 1)) for i in range(1, 9)},
            }
            st.session_state["new_row_cached"] = new_row
            _save_draft()

            df_new = pd.DataFrame([new_row])
            df_new["race_id"] = make_race_id(df_new)
            df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
            df_new["year"] = df_new["date"].dt.year
            df_new["month"] = df_new["date"].dt.month
            df_new["day"] = df_new["date"].dt.day
            df_new["dow"] = df_new["date"].dt.dayofweek

            cat_cols = [c for c in cat_cfg if c in df_new.columns]
            X_new = df_new.drop(columns=["date"])
            if cat_cols:
                X_new = pd.get_dummies(X_new, columns=cat_cols, dummy_na=True)
            X_new = X_new.reindex(columns=feat_cols, fill_value=0)

            proba = apply_temperature(model, X_new, predict_proba_any(model, X_new, n_classes), calibrator)[0]
            order = proba.argsort()[::-1][:4]
            labels = class_labels[order]
            numeros = _to_int_display(labels)
            st.session_state["new_pred_cached"] = pd.DataFrame(
                {"rang": [1, 2, 3, 4], "numero": numeros, "proba": proba[order]}
            )
            st.success("Top-4 prédit pour la nouvelle course.")

    if st.session_state.get("new_pred_cached") is not None:
        st.dataframe(st.session_state["new_pred_cached"], use_container_width=True)

    if c_save.button("💾 Enregistrer cette course dans pending_courses.csv"):
        if not _required_filled():
            st.warning("Merci de compléter les champs requis avant l'enregistrement.")
        else:
            base_row = st.session_state.get("new_row_cached") or {
                "date": pd.to_datetime(st.session_state.get("new_date")).strftime("%Y-%m-%d"),
                "discipline": st.session_state.get("new_discipline"),
                "hippodrome": st.session_state.get("new_hippodrome"),
                "numcourse": st.session_state.get("new_numcourse"),
                "partants": int(st.session_state.get("new_partants")),
                "distance": int(st.session_state.get("new_distance")),
                **{f"prono{i}": int(st.session_state.get(f"new_prono{i}", 1)) for i in range(1, 9)},
            }
            status = upsert_pending(base_row)
            if status == "exists_hist":
                st.error("⛔ Cette course est déjà dans l’historique.")
            elif status == "updated":
                st.info("✏️ Fiche déjà enregistrée : mise à jour effectuée.")
                st.session_state["new_pred_cached"] = None
                st.session_state["new_row_cached"] = None
            elif status == "inserted":
                st.success("✅ Course ajoutée dans pending_courses.csv")
                st.session_state["new_pred_cached"] = None
                st.session_state["new_row_cached"] = None
            else:
                st.warning("Action non reconnue.")

            st.caption(f"Fichier : {PENDING_PATH}")
            try:
                st.dataframe(pd.read_csv(PENDING_PATH, sep=sep, encoding="utf-8"), use_container_width=True)
            except Exception as e:
                st.error(f"Impossible de relire le fichier : {e}")

    if c_reset.button("🔄 Réinitialiser la saisie"):
        for k in list(DEFAULTS.keys()) + ["new_date", "new_row_cached", "new_pred_cached"]:
            if k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
            else:
                st.session_state.pop(k, None)
        if DRAFT_PATH.exists():
            try:
                DRAFT_PATH.unlink()
            except Exception:
                pass
        st.rerun()

# ======= Validation des résultats =======
with tab_validate:
    st.subheader("Valider les résultats et fusionner vers l'historique")
    if not PENDING_PATH.exists():
        st.info("Aucune course en attente. Ajoute d’abord une course depuis l’onglet “➕ Nouvelle course”.")
    else:
        pend = pd.read_csv(PENDING_PATH, sep=sep, encoding="utf-8")
        if "race_id" not in pend.columns and all(k in pend.columns for k in rid_fields):
            pend["race_id"] = make_race_id(pend)

        st.write(f"Complète **{target_col}** (obligatoire) et **rapport** (facultatif, décimal avec point).")
        edited = st.data_editor(pend, num_rows="dynamic", use_container_width=True, key="pending_editor")

        if st.button("Valider et fusionner dans l'historique", type="primary"):
            valid_mask = edited[target_col].notna() & (edited[target_col].astype(str).str.strip() != "")
            done = edited[valid_mask].copy()
            left = edited[~valid_mask].copy()

            try:
                hist = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
            except Exception:
                hist = pd.DataFrame()
            if "race_id" not in hist.columns and all(k in hist.columns for k in rid_fields):
                hist["race_id"] = make_race_id(hist)

            already = set(hist.get("race_id", pd.Series([], dtype=str)).astype(str))
            to_add = done[~done["race_id"].astype(str).isin(already)].copy()

            all_cols = list(dict.fromkeys(list(hist.columns) + list(to_add.columns)))
            hist = hist.reindex(columns=all_cols)
            to_add = to_add.reindex(columns=all_cols)

            b1 = backup(COURSES_PATH); b2 = backup(PENDING_PATH)
            updated_hist = pd.concat([hist, to_add], ignore_index=True)
            updated_hist.to_csv(COURSES_PATH, index=False, sep=sep, encoding="utf-8")
            left.to_csv(PENDING_PATH, index=False, sep=sep, encoding="utf-8")

            st.success(f"✅ {len(to_add)} course(s) ajoutée(s) à l'historique. Backups : {b1.name} / {b2.name}")

            colA, colB = st.columns(2)
            if colA.button("🔄 Recharger données & métriques"):
                _reload_all(); st.toast("Données/métriques rechargées.", icon="✅"); st.rerun()
            if colB.button("🔁 Ré-entraîner le modèle maintenant"):
                with st.spinner("Entraînement en cours..."):
                    info = train_and_save(ROOT / "config.yaml")
                _reload_all()
                st.success(
                    f"Modèle ré-entraîné • Hit@1={info['hit1']:.3f} • Hit@4={info['hit4']:.3f} • LogLoss={info['logloss']:.4f}"
                )
                st.rerun()
