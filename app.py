import streamlit as st
import numpy as np
import pandas as pd, yaml, joblib
from pathlib import Path
from sklearn.metrics import log_loss
from datetime import datetime

from src.features import build_dataset
from src.train import train_and_save  # <-- pour le bouton de ré-entraînement

# ---------- constantes ----------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgb_multiclass.joblib"
COURSES_PATH = ROOT / "data" / "raw" / "Courses_Completes.csv"  # on résout pour être explicite
PENDING_PATH = ROOT / "data" / "raw" / "pending_courses.csv"
HIPPODROMES_PATH = ROOT / "data" / "raw" / "hippodrome.csv"

# ---------- config ----------
st.set_page_config(page_title="Numero Gagnant - Multiclass", layout="wide")
st.title("🎯 Numero Gagnant – Prédiction du numéro gagnant (Top 4)")

cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
sep = cfg["data"].get("csv_sep", ",")
rid_fields = cfg["columns"]["race_id_fields"]

# ---------- session init ----------
for k, v in {
    "new_row_cached": None,
    "__X__": None,
    "__y__": None,
    "__meta__": None,
    "__model__": None,
    "__le__": None,
    "__feat_cols__": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------- utils ----------
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

    # refuse si déjà dans l'historique
    try:
        hist = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
    except Exception:
        hist = pd.DataFrame()
    if "race_id" not in hist.columns and all(k in hist.columns for k in rid_fields):
        hist["race_id"] = make_race_id(hist)
    if "race_id" in hist.columns and save_row["race_id"] in set(hist["race_id"].astype(str)):
        return "exists_hist"

    # charge / crée pending
    if PENDING_PATH.exists():
        pend = pd.read_csv(PENDING_PATH, sep=sep, encoding="utf-8")
    else:
        pend = pd.DataFrame()

    # colonnes minimales
    for k in list(save_row.keys()) + ["a1", "rapport"]:
        if k not in pend.columns:
            pend[k] = pd.NA

    # mise à jour si existant
    if "race_id" in pend.columns:
        mask = pend["race_id"].astype(str) == save_row["race_id"]
        if mask.any():
            idx = pend.index[mask][0]
            for k, v in save_row.items():
                if k in ["a1", "rapport"] and pd.notna(pend.at[idx, k]):
                    continue
                pend.at[idx, k] = v
            pend.to_csv(PENDING_PATH, index=False, sep=sep, encoding="utf-8")
            return "updated"

    # insertion
    pend = pd.concat([pend, pd.DataFrame([save_row])], ignore_index=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    pend.to_csv(PENDING_PATH, index=False, sep=sep, encoding="utf-8")
    return "inserted"


def _reload_all():
    """Relit CSV + modèle et recalcule features. Stocke en session."""
    # données
    courses = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)
    st.session_state["__X__"] = X
    st.session_state["__y__"] = y
    st.session_state["__meta__"] = meta

    # modèle
    bundle = joblib.load(MODEL_PATH)
    st.session_state["__model__"] = bundle["model"]
    st.session_state["__le__"] = bundle["label_encoder"]
    st.session_state["__feat_cols__"] = bundle["feature_columns"]


def _ensure_loaded():
    if st.session_state["__X__"] is None:
        _reload_all()


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## Configuration")
    st.json(cfg)

    st.markdown("---")
    if st.button("🔄 Recharger données & métriques"):
        _reload_all()
        st.toast("Données/métriques rechargées.", icon="✅")
        st.experimental_rerun()

    if st.button("🔁 Ré-entraîner le modèle"):
        with st.spinner("Entraînement en cours..."):
            info = train_and_save(ROOT / "config.yaml")
        _reload_all()
        st.success(
            f"Modèle ré-entraîné • Hit@1={info['hit1']:.3f} • Hit@4={info['hit4']:.3f} • LogLoss={info['logloss']:.4f}"
        )
        st.experimental_rerun()


# ---------- charge mémoire ----------
_ensure_loaded()
X = st.session_state["__X__"]
y = st.session_state["__y__"]
meta = st.session_state["__meta__"]
model = st.session_state["__model__"]
le = st.session_state["__le__"]
feat_cols = st.session_state["__feat_cols__"]

X_all = X.reindex(columns=feat_cols, fill_value=0)

# ---------- métriques globales ----------
proba_all = model.predict_proba(X_all)
order_all = np.argsort(-proba_all, axis=1)[:, :4]
class_labels = le.classes_
top_labels_all = class_labels[order_all]
y_enc_all = le.transform(y.values)
true_labels_all = class_labels[y_enc_all]

c1, c2, c3 = st.columns(3)
c1.metric("Hit@1 (Top-1 exact)", f"{(top_labels_all[:, 0] == true_labels_all).mean():.3f}")
c2.metric("Hit@4 (Gagnant dans Top-4)", f"{np.mean([t in row for t, row in zip(true_labels_all, top_labels_all)]):.3f}")
c3.metric("LogLoss", f"{log_loss(y_enc_all, proba_all, labels=np.arange(len(class_labels))):.4f}")

st.divider()

# ---------- liste hippodromes ----------
hippo_list = []
try:
    hippos_df = pd.read_csv(HIPPODROMES_PATH, sep=sep, encoding="utf-8")
    candidates = [c for c in hippos_df.columns if c.lower() in ("hippodrome", "nom", "name")]
    hippo_col = candidates[0] if candidates else hippos_df.columns[0]
    hippo_list = sorted(pd.Series(hippos_df[hippo_col].astype(str).unique()).dropna())
except Exception:
    pass

# ---------- onglets ----------
tab_hist, tab_new, tab_validate = st.tabs([
    "Historique (données existantes)",
    "➕ Nouvelle course",
    "✅ Validation des résultats"
])

# ======= Historique =======
with tab_hist:
    race_list = meta["race_id"].tolist()
    rid = st.selectbox("Course", race_list)
    if rid:
        idx = race_list.index(rid)
        x_row = X_all.iloc[[idx]]
        proba = model.predict_proba(x_row)[0]
        order = proba.argsort()[::-1][:4]
        labels = class_labels[order]

        st.subheader("Top 4 numéros prédits")
        st.dataframe(pd.DataFrame({"rang": [1, 2, 3, 4], "numero": labels, "proba": proba[order]}))

        vrai = true_labels_all[idx]
        st.info(
            f"Vrai gagnant (a1) : **{int(vrai)}** — "
            f"{'✅ présent dans le Top-4' if int(vrai) in labels else '❌ absent du Top-4'}"
        )

# ======= Nouvelle course =======
with tab_new:
    st.subheader("Saisir une nouvelle course (sans a1)")

    numcourse_opts = [f"C{i}" for i in range(1, 11)]
    partants_opts = list(range(12, 21))
    prono_opts = list(range(1, 21))

    colA, colB, colC, colD = st.columns(4)
    date_val = colA.date_input("date")
    discipline_val = colB.selectbox("discipline", ["trot", "galop"])
    hippodrome_val = colC.selectbox("hippodrome", hippo_list) if hippo_list else colC.text_input("hippodrome")
    numcourse_val = colD.selectbox("numcourse", numcourse_opts, index=2)

    col1, col2 = st.columns(2)
    partants_val = col1.selectbox("partants", partants_opts, index=partants_opts.index(16))
    distance_val = col2.text_input("distance (m)", value="2700")  # libre

    st.markdown("**Pronostics (prono1..prono8)**")
    cols = st.columns(8)
    pr_inputs = {f"prono{i}": cols[i - 1].selectbox(f"prono{i}", prono_opts, index=i - 1) for i in range(1, 9)}

    # prédire
    if st.button("Prédire le Top-4", type="primary"):
        new_row = {
            "date": pd.to_datetime(date_val).strftime("%Y-%m-%d"),
            "discipline": discipline_val,
            "hippodrome": hippodrome_val,
            "numcourse": numcourse_val,
            "partants": int(partants_val),
            "distance": int(pd.to_numeric(distance_val, errors="coerce")) if str(distance_val).strip() else None,
            **{f"prono{i}": pr_inputs[f"prono{i}"] for i in range(1, 9)},
        }
        st.session_state["new_row_cached"] = new_row

        df_new = pd.DataFrame([new_row])
        df_new["race_id"] = make_race_id(df_new)
        df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
        df_new["year"] = df_new["date"].dt.year
        df_new["month"] = df_new["date"].dt.month
        df_new["day"] = df_new["date"].dt.day
        df_new["dow"] = df_new["date"].dt.dayofweek

        cat_cols = [c for c in cfg["columns"].get("categorical", []) if c in df_new.columns]
        X_new = df_new.drop(columns=["date"])
        if cat_cols:
            X_new = pd.get_dummies(X_new, columns=cat_cols, dummy_na=True)
        X_new = X_new.reindex(columns=feat_cols, fill_value=0)

        proba = model.predict_proba(X_new)[0]
        order = proba.argsort()[::-1][:4]
        labels = le.classes_[order]
        st.success("Top-4 prédit pour la nouvelle course :")
        st.dataframe(pd.DataFrame({"rang": [1, 2, 3, 4], "numero": labels, "proba": proba[order]}))

    # enregistrer
    if st.button("💾 Enregistrer cette course dans pending_courses.csv"):
        base_row = st.session_state.get("new_row_cached") or {
            "date": pd.to_datetime(date_val).strftime("%Y-%m-%d"),
            "discipline": discipline_val,
            "hippodrome": hippodrome_val,
            "numcourse": numcourse_val,
            "partants": int(partants_val),
            "distance": int(pd.to_numeric(distance_val, errors="coerce")) if str(distance_val).strip() else None,
            **{f"prono{i}": pr_inputs[f"prono{i}"] for i in range(1, 9)},
        }

        status = upsert_pending(base_row)
        if status == "exists_hist":
            st.error("⛔ Cette course est déjà dans l’historique.")
        elif status == "updated":
            st.info("✏️ Fiche déjà enregistrée : mise à jour effectuée.")
        elif status == "inserted":
            st.success("✅ Course ajoutée dans pending_courses.csv")
        else:
            st.warning("Action non reconnue.")

        st.caption(f"Fichier : {PENDING_PATH}")
        try:
            st.dataframe(pd.read_csv(PENDING_PATH, sep=sep, encoding="utf-8"), use_container_width=True)
        except Exception as e:
            st.error(f"Impossible de relire le fichier : {e}")

# ======= Validation des résultats =======
with tab_validate:
    st.subheader("Valider les résultats et fusionner vers l'historique")

    if not PENDING_PATH.exists():
        st.info("Aucune course en attente. Ajoute d’abord une course depuis l’onglet “➕ Nouvelle course”.")
    else:
        pend = pd.read_csv(PENDING_PATH, sep=sep, encoding="utf-8")
        if "race_id" not in pend.columns and all(k in pend.columns for k in rid_fields):
            pend["race_id"] = make_race_id(pend)

        st.write("Complète **a1** (obligatoire) et **rapport** (facultatif).")
        edited = st.data_editor(
            pend, num_rows="dynamic", use_container_width=True, key="pending_editor"
        )

        if st.button("Valider et fusionner dans l'historique", type="primary"):
            valid_mask = edited["a1"].notna() & (edited["a1"].astype(str).str.strip() != "")
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

            b1 = backup(COURSES_PATH)
            b2 = backup(PENDING_PATH)

            updated_hist = pd.concat([hist, to_add], ignore_index=True)
            updated_hist.to_csv(COURSES_PATH, index=False, sep=sep, encoding="utf-8")
            left.to_csv(PENDING_PATH, index=False, sep=sep, encoding="utf-8")

            st.success(f"✅ {len(to_add)} course(s) ajoutée(s) à l'historique. Backups : {b1.name} / {b2.name}")

            colA, colB = st.columns(2)
            if colA.button("🔄 Recharger données & métriques"):
                _reload_all()
                st.toast("Données/métriques rechargées.", icon="✅")
                st.experimental_rerun()

            if colB.button("🔁 Ré-entraîner le modèle maintenant"):
                with st.spinner("Entraînement en cours..."):
                    info = train_and_save(ROOT / "config.yaml")
                _reload_all()
                st.success(
                    f"Modèle ré-entraîné • Hit@1={info['hit1']:.3f} • Hit@4={info['hit4']:.3f} • LogLoss={info['logloss']:.4f}"
                )
                st.experimental_rerun()
