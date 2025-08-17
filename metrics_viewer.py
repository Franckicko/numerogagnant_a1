# metrics_viewer.py
import argparse, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "data" / "processed" / "predictions_top4.csv"

def parse_args():
    p = argparse.ArgumentParser(description="Vue métriques A1 sur 2 périodes.")
    p.add_argument("--cutoff", required=True, help="YYYY-MM-DD (inclus dans la 1ère période)")
    return p.parse_args()

def extract_date(race_id: str):
    try:
        return race_id.split("_", 1)[0]
    except Exception:
        return None

def compute_block(df):
    df = df.copy()
    df["hit4"] = pd.to_numeric(df.get("hit4", np.nan), errors="coerce")
    ll_race = pd.to_numeric(df.get("logloss_item_raceaware", np.nan), errors="coerce")
    ll_top4 = pd.to_numeric(df.get("logloss_item_top4", np.nan), errors="coerce")
    return {
        "n_courses": int(len(df)),
        "hit4_rate": float(np.nanmean(df["hit4"])) if len(df) else float("nan"),
        "logloss_raceaware": float(np.nanmean(ll_race)) if len(df) else float("nan"),
        "logloss_top4": float(np.nanmean(ll_top4)) if len(df) else float("nan"),
    }

def main():
    a = parse_args()
    if not P.exists():
        print("Fichier introuvable:", P); return
    df = pd.read_csv(P)
    if "race_id" not in df.columns:
        print("race_id manquant dans predictions_top4.csv"); return

    df["date"] = pd.to_datetime(df["race_id"].astype(str).map(extract_date), errors="coerce")
    df = df[df["date"].notna() & df["true_a1"].notna()]

    cut = pd.to_datetime(a.cutoff)
    bloc1 = compute_block(df[df["date"] <= cut])
    bloc2 = compute_block(df[df["date"] > cut])

    def pct(x): 
        return "—" if np.isnan(x) else f"{100*x:0.2f}%"

    print("\n=== Résumé A1 ===")
    print(f"Période 1 (≤ {a.cutoff}) : N={bloc1['n_courses']}")
    print(f"  Hit@4        : {pct(bloc1['hit4_rate'])}")
    print(f"  LogLoss vrai : {bloc1['logloss_raceaware']:.4f}" if not np.isnan(bloc1['logloss_raceaware']) else "  LogLoss vrai : —")
    print(f"  LogLoss top4 : {bloc1['logloss_top4']:.4f}" if not np.isnan(bloc1['logloss_top4']) else "  LogLoss top4 : —")

    print(f"\nPériode 2 (> {a.cutoff}) : N={bloc2['n_courses']}")
    print(f"  Hit@4        : {pct(bloc2['hit4_rate'])}")
    print(f"  LogLoss vrai : {bloc2['logloss_raceaware']:.4f}" if not np.isnan(bloc2['logloss_raceaware']) else "  LogLoss vrai : —")
    print(f"  LogLoss top4 : {bloc2['logloss_top4']:.4f}" if not np.isnan(bloc2['logloss_top4']) else "  LogLoss top4 : —")
    print()
    print("Astuce UI : affiche ces 2 blocs dans deux cartes « Jusqu’au CUT » / « Depuis CUT+1 (dynamique) ».")

if __name__ == "__main__":
    main()
