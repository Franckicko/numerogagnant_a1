import streamlit as st
import numpy as np
import pandas as pd, yaml, joblib, json
from pathlib import Path
from sklearn.metrics import log_loss
from datetime import datetime, date
from functools import partial

from src.features import build_dataset
from src.train import train_and_save  # <-- pour le bouton de ré-entraînement

# ---------- constantes ----------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgb_multiclass.joblib"
COURSES_PATH = ROOT / "data" / "raw" / "Courses_Completes.csv"
PENDING_PATH = ROOT / "data" / "raw" / "pending_courses.csv"
HIPPODROMES_PATH = ROOT / "data" / "raw" / "hippodrome.csv"
DRAFT_PATH = ROOT / "data" / "raw" / "ui_draft.json"

# ---------- config ----------
st.set_page_config(page_title="Numero Gagnant - Multiclass", layout="wide")
st.title("🎯 Numero Gagnant – Prédiction du numéro gagnant (Top 4)")

cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
sep = cfg["data"].get("csv_sep", ",")
rid_fields = cfg["columns"]["race_id_fields"]

# ---------- session init ----------
DEFAULTS = {
    "new_date": None,                  # st.date_input stockera un date
    "new_discipline": "",              # "", "trot", "galop"
    "new_hippodrome": "",              # "" ou un nom de HIPPODROME
    "new_numcourse": "",               # "", "C1".. "C10"
    "new_partants": "",                # texte, normalisé entier 12..20
    "new_distance": "",                # texte, normalisé entier >=0
    **{f"new_prono{i}": "" for i in range(1, 9)},  # texte, normalisé entier 1..20
    "new_row_cached": None,
    "new_pred_cached": None,
    "__X__": None,
    "__y__": None,
    "__meta__": None,
    "__model__": None,
    "__le__": None,
    "__feat_cols__": None,
}
for k, v in DEFAULTS.items():
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
    courses = pd.read_csv(COURSES_PATH, sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)
    st.session_state["__X__"] = X
    st.session_state["__y__"] = y
    st.session_state["__meta__"] = meta

    bundle = joblib.load(MODEL_PATH)
    st.session_state["__model__"] = bundle["model"]
    st.session_state["__le__"] = bundle["label_encoder"]
    st.session_state["__feat_cols__"] = bundle["feature_columns"]

def _ensure_loaded():
    if st.session_state["__X__"] is None:
        _reload_all()

# ---- affichage INT pour labels
def _to_int_display(values):
    out = []
    for v in values:
        try: out.append(int(float(v)))
        except Exception: out.append(v)
    return out

# ---- normalisation & persistance du brouillon
DRAFT_KEYS = [
    "new_date", "new_discipline", "new_hippodrome", "new_numcourse",
    "new_partants", "new_distance", *[f"new_prono{i}" for i in range(1, 9)]
]

def _serialize_for_json(val):
    if isinstance(val, date):
        return val.isoformat()
    return val

def _save_draft():
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    payload = {k: _serialize_for_json(st.session_state.get(k)) for k in DRAFT_KEYS}
    with open(DRAFT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_draft_into_session():
    if not DRAFT_PATH.exists():
        return
    try:
        with open(DRAFT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    # n'écrase que si non présent
    for k, v in data.items():
        if k not in st.session_state or st.session_state[k] in (None, ""):
            if k == "new_date" and isinstance(v, str) and v:
                try:
                    y, m, d = [int(x) for x in v.split("-")]
                    st.session_state[k] = date(y, m, d)
                except Exception:
                    st.session_state[k] = None
            else:
                st.session_state[k] = v

def _norm_int_str(v, minv=None, maxv=None):
    s = "" if v is None else str(v).strip()
    if s == "": return ""
    try: iv = int(float(s))
    except Exception: return ""
    if minv is not None: iv = max(minv, iv)
    if maxv is not None: iv = min(maxv, iv)
    return str(iv)

def _on_change_norm(key, minv=None, maxv=None):
    st.session_state[key] = _norm_int_str(st.session_state.get(key, ""), minv=minv, maxv=maxv)
    _save_draft()

def _on_change_save(key):
    # pour selectbox/date : juste persister
    _save_draft()

def _as_int_or_none(v, minv=None, maxv=None):
    s = "" if v is None else str(v).strip()
    if s == "": return None
    try: iv = int(float(s))
    except Exception: return None
    if minv is not None: iv = max(minv, iv)
    if maxv is not None: iv = min(maxv, iv)
    return iv

# Charger le brouillon éventuel AVANT de créer les widgets
_load_draft_into_session()

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
X = st.session_state["__X__"]; y = st.session_state["__y__"]
meta = st.session_state["__meta__"]; model = st.session_state["__model__"]
le = st.session_state["__le__"]; feat_cols = st.session_state["__feat_cols__"]
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
tab_hist, tab_new, tab_validate = st.tabs(
    ["Historique (données existantes)", "➕ Nouvelle course", "✅ Validation des résultats"]
)

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
        numeros = _to_int_display(labels)

        st.subheader("Top 4 numéros prédits")
        st.dataframe(pd.DataFrame({"rang": [1, 2, 3, 4], "numero": numeros, "proba": proba[order]}))

        vrai = true_labels_all[idx]
        st.info(
            f"Vrai gagnant (a1) : **{int(vrai)}** — "
            f"{'✅ présent dans le Top-4' if int(vrai) in numeros else '❌ absent du Top-4'}"
        )

# ======= Nouvelle course =======
with tab_new:
    st.subheader("Saisir une nouvelle course (sans a1)")

    numcourse_opts = [f"C{i}" for i in range(1, 11)]

    colA, colB, colC, colD = st.columns(4)
    colA.date_input("date", key="new_date", on_change=partial(_on_change_save, "new_date"))
    colB.selectbox("discipline", ["", "trot", "galop"], key="new_discipline",
                   on_change=partial(_on_change_save, "new_discipline"))
    if hippo_list:
        colC.selectbox("hippodrome", [""] + hippo_list, key="new_hippodrome",
                       on_change=partial(_on_change_save, "new_hippodrome"))
    else:
        colC.text_input("hippodrome", key="new_hippodrome", on_change=partial(_on_change_save, "new_hippodrome"))
    colD.selectbox("numcourse", [""] + numcourse_opts, key="new_numcourse",
                   on_change=partial(_on_change_save, "new_numcourse"))

    col1, col2 = st.columns(2)
    col1.text_input("partants", key="new_partants", placeholder="12..20",
                    on_change=partial(_on_change_norm, "new_partants", 12, 20))
    col2.text_input("distance (m)", key="new_distance", placeholder="ex: 2700",
                    on_change=partial(_on_change_norm, "new_distance", 0, None))

    st.markdown("**Pronostics (prono1..prono8)**")
    cols = st.columns(8)
    for i in range(1, 9):
        key = f"new_prono{i}"
        cols[i - 1].text_input(f"prono{i}", key=key, placeholder="1..20",
                               on_change=partial(_on_change_norm, key, 1, 20))

    # --- helpers
    def _required_filled():
        req = [
            st.session_state.get("new_date"),
            st.session_state.get("new_discipline"),
            st.session_state.get("new_hippodrome"),
            st.session_state.get("new_numcourse"),
            st.session_state.get("new_partants"),
        ]
        return all(str(x).strip() not in ("", "None") for x in req)

    # --- Boutons
    c_pred, c_save, c_reset = st.columns([1, 1, 1])

    if c_pred.button("Prédire le Top-4", type="primary"):
        if not _required_filled():
            st.warning("Merci de renseigner au minimum date, discipline, hippodrome, numcourse et partants.")
        else:
            # construire la ligne SANS réécrire de clés de widgets (évite l'APIException)
            new_row = {
                "date": pd.to_datetime(st.session_state["new_date"]).strftime("%Y-%m-%d"),
                "discipline": st.session_state["new_discipline"],
                "hippodrome": st.session_state["new_hippodrome"],
                "numcourse": st.session_state["new_numcourse"],
                "partants": _as_int_or_none(st.session_state["new_partants"], 12, 20),
                "distance": _as_int_or_none(st.session_state["new_distance"], 0, None),
                **{f"prono{i}": _as_int_or_none(st.session_state.get(f"new_prono{i}"), 1, 20) for i in range(1, 9)},
            }
            st.session_state["new_row_cached"] = new_row

            # -- features identiques à l'existant
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
            numeros = _to_int_display(labels)
            st.session_state["new_pred_cached"] = pd.DataFrame(
                {"rang": [1, 2, 3, 4], "numero": numeros, "proba": proba[order]}
            )
            st.success("Top-4 prédit pour la nouvelle course.")

    # -- Affichage persistant du Top-4 si dispo
    if st.session_state.get("new_pred_cached") is not None:
        st.dataframe(st.session_state["new_pred_cached"], use_container_width=True)

    # --- Enregistrer
    if c_save.button("💾 Enregistrer cette course dans pending_courses.csv"):
        if not _required_filled():
            st.warning("Merci de compléter les champs requis avant l'enregistrement.")
        else:
            base_row = st.session_state.get("new_row_cached") or {
                "date": pd.to_datetime(st.session_state.get("new_date")).strftime("%Y-%m-%d"),
                "discipline": st.session_state.get("new_discipline"),
                "hippodrome": st.session_state.get("new_hippodrome"),
                "numcourse": st.session_state.get("new_numcourse"),
                "partants": _as_int_or_none(st.session_state.get("new_partants"), 12, 20),
                "distance": _as_int_or_none(st.session_state.get("new_distance"), 0, None),
                **{f"prono{i}": _as_int_or_none(st.session_state.get(f"new_prono{i}"), 1, 20) for i in range(1, 9)},
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

    # --- Réinitialiser manuellement (si besoin)
    if c_reset.button("🔄 Réinitialiser la saisie"):
        for k in DRAFT_KEYS + ["new_row_cached", "new_pred_cached"]:
            st.session_state[k] = DEFAULTS.get(k, None)
        if DRAFT_PATH.exists():
            try: DRAFT_PATH.unlink()
            except Exception: pass
        st.experimental_rerun()

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
        edited = st.data_editor(pend, num_rows="dynamic", use_container_width=True, key="pending_editor")

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
