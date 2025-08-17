# numerogagnant_a1 (version légère, calquée sur a2/a4)

## Ce qui reste
Place **exactement** ces deux fichiers dans `data/raw/` :
- `Courses_Completes.csv`
- `hippodrome.csv`

*(Noms identiques à ceux que tu as déjà.)*

## Sorties officielles (minimales)
- `data/processed/predictions_top4.csv`
- `data/processed/metrics_history.csv`
- `models/xgb_multiclass.joblib` (champion)

## Entraîner (champion/challenger, anti-régression logloss)
```bash
pip install -r requirements.txt
python src/train.py
```

## Prédire (ajoute/écrase la ligne du jour)
```bash
python src/predict.py --date YYYY-MM-DD --hippodrome "Deauville" --numcourse C3 --partants 16 --distance 2000 --pronos 9,13,15,12,16,1,6,2
```

- Promotion de modèle **uniquement si** `logloss_valid` **baisse** (epsilon configurable).
- Pas de `.venv` ni de backups : projet **léger** et autonome comme a2/a4.