# src/apply_best_params.py
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "config.yaml"
best_path = ROOT / "models" / "xgb_best_params.yaml"

cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
best = yaml.safe_load(open(best_path, "r", encoding="utf-8"))

# merge
cfg.setdefault("training", {}).setdefault("xgb_params", {})
cfg["training"]["xgb_params"].update(best["xgb_params"])

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("✅ config.yaml mis à jour avec xgb_best_params.yaml")
