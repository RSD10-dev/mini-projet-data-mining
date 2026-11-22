# Mini-projet Data Mining — Classification du score de crédit (Groupe RSD10)

Application Streamlit (10 pages) qui présente l'EDA, les tests statistiques, le benchmark des modèles
et un **prédicteur** de score de crédit (Good / Standard / Poor).

## Structure

```
mini-projet-data-mining/
├── data/processed/     X_train_preprocessed.csv, X_val_preprocessed.csv, y_train.csv, y_val.csv, groups_train.csv
├── notebooks/          01_eda_and_cleaning · 02_preprocessing_encoding · 03_model_training_evaluation
├── results/            resultats_modeles.csv
├── models/             (optionnel) scaler_stats.json
├── src/
│   ├── app.py            application Streamlit
│   ├── models.py         entraînement, évaluation, importance des variables
│   └── preprocessing.py  prétraitement + transformation d'une saisie brute
└── requirements.txt
```

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Au premier lancement, le Random Forest (30 arbres, profondeur 15) est entraîné en quelques secondes.
`python src/models.py` l'entraîne, affiche les métriques et le sauvegarde dans `models/rf_model.joblib`.

## Déploiement (Streamlit Community Cloud)

1. Pousser le projet sur GitHub (avec `data/processed/` et `results/`).
2. Sur share.streamlit.io : *New app* → dépôt → **Main file path : `src/app.py`**.

## Rendre le prédicteur exact (recommandé)

Le `StandardScaler` du notebook 02 n'a pas été sauvegardé. L'application utilise donc une estimation
des moyennes / écarts-types (exacte pour les variables entières, approchée pour `Annual_Income` et
`Total_EMI_per_month`). Pour des valeurs exactes, ajoutez cette cellule **à la fin du notebook 02**
(après `scaler.fit_transform`) puis commitez `models/scaler_stats.json` :

```python
import json, os
os.makedirs("../models", exist_ok=True)
with open("../models/scaler_stats.json", "w") as f:
    json.dump({"features": numerical_features_to_scale,
               "mean": scaler.mean_.tolist(),
               "scale": scaler.scale_.tolist()}, f)
```

L'application détecte le fichier automatiquement.

## Points d'attention

- `Credit_History_Age` est saisi en **mois** (valeurs de 1 à 404 dans les données).
- Le fichier `resultats_modeles.csv` contient 2 lignes « Random Forest » issues de la **validation croisée**
  (Accuracy 0,751) : elles ne sont pas comparables aux scores mesurés sur le jeu de validation.
- Le découpage train / validation des fichiers fournis est fait **par client** (10 000 clients en train, 8 lignes
  chacun) ; le code du notebook 02 présent dans le dépôt utilise pourtant `train_test_split` sans groupes :
  ajoutez-y le code du découpage réellement utilisé.
- `.gitignore` exclut `*.joblib` / `*.pkl` : c'est volontaire (le modèle est réentraîné au démarrage).