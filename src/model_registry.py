# src/model_registry.py
import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "data" / "processed" / "model_registry.json"

def get_champion():
    if REG.exists():
        rec = json.loads(REG.read_text(encoding="utf-8"))
        return {"model_id": rec.get("model_id"), "file_path": rec.get("file_path"), "val_logloss": rec.get("val_logloss")}
    return None

def save_meta(mid, path, val):
    REG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"model_id": mid, "file_path": str(path), "val_logloss": float(val), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    REG.write_text(json.dumps(rec, indent=2), encoding="utf-8")

def maybe_promote(challenger, challenger_val, epsilon=1e-4):
    champ = get_champion()
    if champ is None or (float(challenger_val) < float(champ["val_logloss"]) - float(epsilon)):
        save_meta(challenger["model_id"], challenger["file_path"], challenger_val)
        return True, champ
    return False, champ
