# src/metrics.py
import numpy as np
from sklearn.metrics import log_loss

def multiclass_logloss(y_true, proba, labels=None):
    """Logloss global sur toutes les classes (0..K-1)."""
    return log_loss(y_true, proba, labels=labels)

def logloss_multiclass_raceaware(y_true, proba, partants):
    """
    Vrai logloss: pour chaque course i, ne garder que les classes valides 0..K_i-1
    (K_i = partants), renormaliser, puis -log(p_true_renorm).
    y_true est 0-based; proba shape [n, Kmax]; partants = K_i.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    partants = np.asarray(partants, dtype=int)

    n = len(y_true)
    out = 0.0
    for i in range(n):
        k = int(partants[i]) if i < len(partants) else proba.shape[1]
        if k <= 0:
            continue
        p = np.clip(proba[i, :k], 0.0, None)
        s = p.sum()
        if s <= 0:
            p_t = 1.0 / k
        else:
            p /= s
            yi = y_true[i]
            p_t = float(p[yi]) if 0 <= yi < k else (1.0 / k)
        out += -np.log(max(p_t, 1e-15))
    return out / max(n, 1)

def logloss_top4_renorm(true_num, top_nums, top_probs):
    """
    Faux logloss 'Top-4': on ne considère que le Top-4 prédit, renormalisé.
    true_num est 1-based; top_nums = [n1..n4]; top_probs = [p1..p4] (brutes).
    """
    top_nums = list(map(int, top_nums))
    top_probs = np.asarray(top_probs, dtype=float).clip(min=0.0)
    s = top_probs.sum()
    if s <= 0:
        p = 1.0 / 4
    else:
        top_probs = top_probs / s
        p = float(top_probs[top_nums.index(int(true_num))]) if int(true_num) in top_nums else 1.0 / 4
    return -np.log(max(p, 1e-15))

def hit_at_k(preds_sorted, true_label, k=4):
    """Retourne 1 si true_label (1-based) est dans les k premiers de preds_sorted (liste de numéros), sinon 0."""
    return int(int(true_label) in list(map(int, preds_sorted[:k])))
