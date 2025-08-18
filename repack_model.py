# repack_model.py — reconditionne le modèle en Pipeline propre
from pathlib import Path
import shutil, joblib
from sklearn.pipeline import Pipeline

# Shim pour dépickler d'anciens modèles (picklés avec une classe PreprocBooster en __main__)
class PreprocBooster:
    def __init__(self, pre=None, clf=None, pipe=None, model=None):
        self.pre = pre
        self.clf = clf
        self.pipe = pipe
        self.model = model

ROOT = Path(__file__).resolve().parent
mp = ROOT / "models" / "xgb_multiclass.joblib"

blob = joblib.load(mp)

# Récupère classes et l'objet « modèle »
if isinstance(blob, dict):
    classes = blob.get("classes")
    mdl = blob.get("model")
else:
    classes = None
    mdl = blob

# Essaie d'obtenir un Pipeline utilisable
pipe = None
if getattr(mdl, "pipe", None) is not None:
    pipe = mdl.pipe
elif getattr(mdl, "model", None) is not None:
    pipe = mdl.model
elif getattr(mdl, "pre", None) is not None and getattr(mdl, "clf", None) is not None:
    pipe = Pipeline([("pre", mdl.pre), ("clf", mdl.clf)])
else:
    # au pire, c'est déjà un objet avec predict_proba
    pipe = mdl

# Sauvegarde propre {"model": pipeline, "classes": [...]}
bak = mp.with_suffix(".joblib.bak")
shutil.copy2(mp, bak)
joblib.dump({"model": pipe, "classes": list(classes) if classes is not None else None}, mp)
print(f"OK: modèle reconditionné. Backup -> {bak}")
