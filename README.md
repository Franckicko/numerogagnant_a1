# Numero Gagnant (multiclass)

Jeu de données: 1 ligne = 1 course. Cible `a1` = numéro gagnant. On prédit un **top 4** de numéros.

## Installation
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Entraînement
```bash
python -m src.train
```

## Prédictions fichier
```bash
python -m src.predict
```

## Interface Streamlit
```bash
streamlit run app.py
```
