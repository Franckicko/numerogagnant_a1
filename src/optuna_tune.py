# src/optuna_tune.py
import argparse, yaml, joblib, optuna, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from .features import build_dataset

ROOT = Path(__file__).resolve().parents[1]

def hit_at_k(proba: np.ndarray, y_true: np.ndarray, k: int = 4) -> float:
    order = np.argsort(-proba, axis=1)[:, :k]    # top-k indices (classes encodées)
    return float(np.mean([t in row for t, row in zip(y_true, order)]))

def load_data(cfg_path: Path) -> Tuple[pd.DataFrame, Dict]:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    sep = cfg["data"].get("csv_sep", ",")
    courses = pd.read_csv(ROOT / cfg["data"]["courses_csv"], sep=sep, encoding="utf-8")
    X, y, meta, _ = build_dataset(courses, cfg)
    return (X, y, meta, cfg)

def suggest_params(trial: optuna.Trial, base_params: Dict, num_class: int) -> Dict:
    p = dict(base_params)  # start from config.yaml defaults if you want
    # search space
    p["learning_rate"]   = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    p["max_depth"]       = trial.suggest_int("max_depth", 3, 8)
    p["min_child_weight"]= trial.suggest_int("min_child_weight", 1, 10)
    p["subsample"]       = trial.suggest_float("subsample", 0.6, 1.0)
    p["colsample_bytree"]= trial.suggest_float("colsample_bytree", 0.6, 1.0)
    p["gamma"]           = trial.suggest_float("gamma", 0.0, 0.5)
    p["reg_alpha"]       = trial.suggest_float("reg_alpha", 0.0, 1.0)
    p["reg_lambda"]      = trial.suggest_float("reg_lambda", 0.5, 3.0)
    p["tree_method"]     = "hist"
    p["objective"]       = "multi:softprob"
    p["eval_metric"]     = "mlogloss"
    p["num_class"]       = num_class
    return p

def evaluate_cv(X, y_enc, params: Dict, n_splits: int, random_state: int, n_estimators: int, es_rounds: int,
                optimize_for: str) -> Tuple[float, float]:
    """Retourne (mean_logloss, mean_hit4) en CV stratifiée."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ll_list, h4_list = [], []

    for tr_idx, va_idx in skf.split(X, y_enc):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_va, label=y_va)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=es_rounds,
            verbose_eval=False,
        )
        proba_va = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        ll = log_loss(y_va, proba_va, labels=np.arange(params["num_class"]))
        h4 = hit_at_k(proba_va, y_va, k=4)

        ll_list.append(ll); h4_list.append(h4)

    return float(np.mean(ll_list)), float(np.mean(h4_list))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--optimize_for", type=str, choices=["logloss", "hit4"], default="logloss",
                    help="Choisir la métrique d'optimisation primaire.")
    ap.add_argument("--early_stopping_rounds", type=int, default=100)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--train_best", action="store_true",
                    help="Après recherche, réentraîne un modèle final sur tout le train et le sauvegarde.")
    args = ap.parse_args()

    X, y, meta, cfg = load_data(Path(args.config))
    print(f"[optuna] Data: {X.shape[0]} lignes, {X.shape[1]} features.")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_class = len(le.classes_)

    base_params = cfg.get("training", {}).get("xgb_params", {})
    # objective/metric seront écrasés proprement dans suggest_params
    def objective(trial: optuna.Trial):
        params = suggest_params(trial, base_params, num_class)
        mean_ll, mean_h4 = evaluate_cv(X, y_enc, params, args.cv, args.random_state,
                                       args.n_estimators, args.early_stopping_rounds, args.optimize_for)
        # on stocke l'autre métrique pour suivi
        trial.set_user_attr("mean_hit4", mean_h4)
        trial.set_user_attr("mean_logloss", mean_ll)
        # valeur à optimiser
        return mean_ll if args.optimize_for == "logloss" else -mean_h4

    direction = "minimize" if args.optimize_for == "logloss" else "maximize"
    study = optuna.create_study(direction=direction, study_name=f"xgb_{args.optimize_for}")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n=== Résultats Optuna ===")
    best = study.best_trial
    print(f"Best trial #{best.number}")
    print(f"  value ({args.optimize_for}): {best.value:.4f}")
    print(f"  mean_logloss: {best.user_attrs.get('mean_logloss'):.4f}")
    print(f"  mean_hit4:    {best.user_attrs.get('mean_hit4'):.4f}")
    print("  params:")
    best_params = suggest_params(best, base_params, num_class)  # reconstruit dict final
    # Remplace les valeurs par celles choisies
    for k in list(best_params.keys()):
        if k in best.params:
            best_params[k] = best.params[k]

    # Sauvegarde params + infos
    out_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "xgb_best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"xgb_params": best_params,
                        "optimize_for": args.optimize_for,
                        "best_value": float(best.value),
                        "mean_logloss": float(best.user_attrs.get('mean_logloss')),
                        "mean_hit4": float(best.user_attrs.get('mean_hit4')),
                        "trials": args.trials,
                        "cv": args.cv}, f, sort_keys=False, allow_unicode=True)
    print(f"[optuna] Paramètres sauvegardés → {out_dir/'xgb_best_params.yaml'}")

    if args.train_best:
        # entraînement final (train/val interne + early stopping), sauvegarde bundle
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=cfg.get("training", {}).get("test_size", 0.2),
            random_state=cfg.get("training", {}).get("random_state", 42), stratify=y_enc
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15,
            random_state=cfg.get("training", {}).get("random_state", 42), stratify=y_train
        )
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_val, label=y_val)
        booster = xgb.train(
            params=best_params, dtrain=dtrain, num_boost_round=args.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=args.early_stopping_rounds, verbose_eval=False
        )
        # sauvegarde bundle compatible avec le reste du projet
        joblib.dump(
            {
                "model": booster,
                "label_encoder": le,
                "feature_columns": X.columns.tolist(),
            },
            out_dir / "xgb_multiclass.joblib",
        )
        print(f"[optuna] Modèle final sauvegardé → {out_dir/'xgb_multiclass.joblib'}")

if __name__ == "__main__":
    main()
