# src/predict.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data_io import save_predictions_top4
from .features import build_feature_frame

ROOT = Path(__file__).resolve().parents[1]


def get_champion_path() -> Path:
    path = ROOT / "models" / "xgb_multiclass.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Champion model not found. Train first: `python -m src.train`"
        )
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict Top-4 for one race and append to data/processed/predictions_top4.csv"
    )
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--hippodrome", required=True)
    p.add_argument("--numcourse", required=True, help="e.g. C3")
    p.add_argument("--partants", required=True, type=int)
    p.add_argument("--distance", required=True, type=float)
    p.add_argument(
        "--pronos",
        required=True,
        help='8 numbers comma-separated (prono1..8), e.g. "9,13,15,12,16,1,6,2"',
    )
    p.add_argument(
        "--discipline",
        default="galop",
        help='Optional (default="galop").',
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load config & model
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    blob = joblib.load(get_champion_path())
    pipe = blob["model"]
    classes_real = blob.get("classes", None)
    num_classes = int(max(classes_real)) if classes_real is not None else None

    # Parse pronos
    pronos = [int(x.strip()) for x in args.pronos.split(",")]
    if len(pronos) != 8:
        raise ValueError(f"--pronos must contain exactly 8 numbers (got {len(pronos)})")

    # Build single-row features
    row = {
        "date": pd.to_datetime(args.date),
        "hippodrome": args.hippodrome,
        "numcourse": args.numcourse,
        "partants": int(args.partants),
        "distance": float(args.distance),
        "discipline": str(args.discipline).lower(),
    }
    for i, num in enumerate(pronos, start=1):
        row[f"prono{i}"] = num
    X = build_feature_frame(
        pd.DataFrame([row]),
        cfg["prono_cols"],
        cfg["num_features"],
        cfg["cat_features"],
    )

    # Predict proba
    proba_vec = pipe.predict_proba(X)[0]
    if num_classes is None:
        num_classes = len(proba_vec)

    # Map proba -> pronos
    pairs = []
    for numero in pronos:
        p = float(proba_vec[int(numero) - 1]) if 1 <= int(numero) <= num_classes else 0.0
        pairs.append((int(numero), p))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:4]

    out = {
        "race_id": f"{args.date}_{args.hippodrome}_{args.numcourse}",
        "true_a1": None,
        "pred1_a1": top[0][0] if len(top) > 0 else None,
        "pred2_a1": top[1][0] if len(top) > 1 else None,
        "pred3_a1": top[2][0] if len(top) > 2 else None,
        "pred4_a1": top[3][0] if len(top) > 3 else None,
        "proba1": top[0][1] if len(top) > 0 else None,
        "proba2": top[1][1] if len(top) > 1 else None,
        "proba3": top[2][1] if len(top) > 2 else None,
        "proba4": top[3][1] if len(top) > 3 else None,
    }
    save_predictions_top4(pd.DataFrame([out]))

    def fmt(p): return f"{p:.4f}" if p is not None else "-"
    print(
        f"[A1] {out['race_id']}  "
        f"Top4: {out['pred1_a1']}({fmt(out['proba1'])}), "
        f"{out['pred2_a1']}({fmt(out['proba2'])}), "
        f"{out['pred3_a1']}({fmt(out['proba3'])}), "
        f"{out['pred4_a1']}({fmt(out['proba4'])})"
    )


if __name__ == "__main__":
    main()
