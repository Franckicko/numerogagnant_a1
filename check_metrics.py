# check_metrics.py
import pandas as pd, numpy as np
from pathlib import Path

PRED_PATH = Path("data/processed/predictions_top4.csv")
OUT_TXT   = Path("data/processed/metrics.txt")

df = pd.read_csv(PRED_PATH)

def hit_at_k(df):
    hit1 = (df['pred1_a2'] == df['true_a2']).mean()
    arr_top4 = df[['pred1_a2','pred2_a2','pred3_a2','pred4_a2']].values
    hit4 = np.mean([t in row for t, row in zip(df['true_a2'], arr_top4)])
    return hit1, hit4

# Sanity checks
dup_in_top4 = (df[['pred1_a2','pred2_a2','pred3_a2','pred4_a2']].nunique(axis=1) < 4).sum()
asc_ok = (df['proba1'] >= df['proba2']).all() and (df['proba2'] >= df['proba3']).all() and (df['proba3'] >= df['proba4']).all()

hit1, hit4 = hit_at_k(df)
proba1_mean = df['proba1'].mean()

lines = []
lines.append(f"n={len(df)}")
lines.append(f"Hit@1={hit1:.3f}")
lines.append(f"Hit@4={hit4:.3f}")
lines.append(f"Proba1_mean={proba1_mean:.3f}")
lines.append(f"Doublons_dans_top4={dup_in_top4}")
lines.append(f"Probas_decroissantes={asc_ok}")

# Segments facultatifs si colonnes dispo (ex: 'discipline', 'hippodrome', 'month', etc.)
for col in ['discipline', 'hippodrome', 'month']:
    if col in df.columns:
        lines.append(f"\n-- Segments par {col} --")
        for key, grp in df.groupby(col):
            h1, h4 = hit_at_k(grp)
            lines.append(f"{col}={key}  n={len(grp)}  Hit@1={h1:.3f}  Hit@4={h4:.3f}")

# Calibration grossière (bins sur proba1)
bins = np.linspace(0, 1, 11)
df['_bin'] = pd.cut(df['proba1'], bins, include_lowest=True)
arr_top1 = df['pred1_a2'].values
calib = df.groupby('_bin').apply(
    lambda g: pd.Series({
        'n': len(g),
        'proba1_mean': g['proba1'].mean(),
        'acc_top1': (g['true_a2'].values == arr_top1[g.index]).mean() if len(g)>0 else np.nan
    })
).reset_index()

lines.append("\n-- Calibration (bin proba1) --")
for _, r in calib.iterrows():
    lines.append(f"{r['_bin']}: n={int(r['n'])}  proba1_mean={r['proba1_mean']:.3f}  acc_top1={r['acc_top1']:.3f}")

OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\n✅ Écrit → {OUT_TXT}")
