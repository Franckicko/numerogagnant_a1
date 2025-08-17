# src/data_io.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

def load_raw():
    path = RAW / "Courses_Completes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "race_id" not in df.columns and set(["date","hippodrome","numcourse"]).issubset(df.columns):
        df["race_id"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + df["hippodrome"].astype(str) + "_" + df["numcourse"].astype(str)
    return df

def save_predictions_top4(df_top4):
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "predictions_top4.csv"
    wanted = [
        "race_id","true_a1","pred1_a1","pred2_a1","pred3_a1","pred4_a1",
        "proba1","proba2","proba3","proba4"
    ]
    for c in wanted:
        if c not in df_top4.columns:
            df_top4[c] = pd.NA
    df_top4 = df_top4[wanted]

    if out.exists():
        old = pd.read_csv(out)
        merged = pd.concat([old, df_top4], ignore_index=True)
        merged = merged.drop_duplicates(subset=["race_id"], keep="last")
        merged[wanted].to_csv(out, index=False, encoding="utf-8")
    else:
        df_top4.to_csv(out, index=False, encoding="utf-8")

def save_metrics_history(metrics_row):
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "metrics_history.csv"
    row_df = pd.DataFrame([metrics_row])
    if out.exists():
        old = pd.read_csv(out)
        merged = pd.concat([old, row_df], ignore_index=True)
        merged.to_csv(out, index=False, encoding="utf-8")
    else:
        row_df.to_csv(out, index=False, encoding="utf-8")
