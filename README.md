# Classification du score de crédit

Mini-projet de Data Mining, groupe RSD10.

L'objectif est de prédire la catégorie de score de crédit d'un client (`Poor`, `Standard` ou `Good`) à partir de ses informations bancaires et de son comportement de paiement. Le projet couvre toute la chaîne : exploration des données, prétraitement, comparaison de six modèles, optimisation du meilleur d'entre eux et mise à disposition dans une petite application web.

## Sommaire

1. [Jeu de données](#1-jeu-de-données)
2. [Organisation du projet](#2-organisation-du-projet)
3. [Installation et exécution](#3-installation-et-exécution)
4. [Démarche](#4-démarche)
5. [Résultats](#5-résultats)
6. [Application web](#6-application-web)
7. [Conclusion](#7-conclusion)
8. [Limites et perspectives](#8-limites-et-perspectives)
9. [Auteurs](#9-auteurs)

## 1. Jeu de données

Le fichier `train.csv` (dataset *Credit Score Classification*) contient 100 000 observations et 28 attributs. Il décrit 12 500 clients, chacun observé sur 8 mois (de janvier à août). Une même personne apparaît donc 8 fois dans le fichier.

La variable cible `Credit_Score` est déséquilibrée :

| Classe | Part des observations |
|---|---|
| Standard | 53,17 % |
| Poor | 29,00 % |
| Good | 17,83 % |

Les colonnes `ID`, `Customer_ID`, `Name` et `SSN` sont des identifiants ou des données personnelles : elles ne sont pas utilisées comme variables explicatives et ne sont pas affichées dans les analyses.

## 2. Organisation du projet

```
mini-projet-data-mining/
├── data/
│   ├── raw/
│   │   └── train.csv                  # données brutes (jamais modifiées)
│   ├── X_train_preprocessed.csv       # variables explicatives, entraînement
│   ├── X_val_preprocessed.csv         # variables explicatives, validation
│   ├── y_train.csv                    # cible, entraînement
│   └── y_val.csv                      # cible, validation
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_preprocessing_encoding.ipynb
│   └── 03_model_training_evaluation.ipynb
├── results/
│   └── resultats_modeles.csv          # métriques de chaque modèle
├── app/
│   ├── app.py                         # application Streamlit
│   ├── preprocessing.py               # prétraitement partagé
│   ├── train_model.py                 # entraînement du modèle final
│   └── requirements.txt
└── README.md
```

## 3. Installation et exécution

Python 3.9 ou supérieur est recommandé.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter streamlit joblib
```

Les notebooks doivent être exécutés dans l'ordre, car chacun produit les fichiers lus par le suivant :

1. `01_eda_and_cleaning.ipynb` : exploration, contrôle qualité et jeu de données nettoyé (`train_clean.csv`).
2. `02_preprocessing_encoding.ipynb` : lit `train_clean.csv` et produit les quatre fichiers `*_preprocessed.csv` / `y_*.csv`.
3. `03_model_training_evaluation.ipynb` : lit ces fichiers depuis `data/` et écrit les résultats dans `results/resultats_modeles.csv`.

Le notebook 03 prend du temps à cause du SVM (environ 15 minutes d'entraînement) et du GridSearchCV (environ 30 minutes).

L'application se lance séparément, voir la [section 6](#6-application-web).

## 4. Démarche

### 4.1 Exploration et contrôle qualité (notebook 01)

Le fichier brut n'est jamais écrasé. Les corrections de format sont faites en mémoire, dans une copie de travail, uniquement pour rendre les analyses possibles. Les formats récupérables sont corrigés et les valeurs invalides remplacées par `NaN`, ce qui donne le jeu nettoyé utilisé ensuite.

Principaux constats :

- Aucune ligne dupliquée, mais 8 observations par client.
- Sept colonnes contiennent des cellules vides, dont `Monthly_Inhand_Salary` (15 %).
- `Credit_Mix` contient 20 195 valeurs codées `_`, et `Payment_Behaviour` contient 7 600 valeurs codées `!@9#%8`.
- Certaines valeurs numériques se terminent par un `_` parasite, qu'il suffit de retirer pour les récupérer.
- Des valeurs aberrantes sont présentes : 2 770 âges suspects et 4 313 valeurs suspectes pour `Num_of_Loan`.
- Les revenus et les dettes ont une distribution fortement asymétrique.
- Corrélations notables : `Monthly_Inhand_Salary` / `Monthly_Balance` (r = 0,705) et `Delay_from_due_date` / `Outstanding_Debt` (r = 0,572).

Deux tests statistiques ont été menés sur un échantillon d'un enregistrement par client (12 500 clients), afin de limiter la dépendance entre observations d'un même client :

| Test | Variables | Résultat |
|---|---|---|
| χ² d'indépendance | `Credit_Mix` × `Credit_Score` | χ² = 4064,985 ; p < 0,001 ; V de Cramér = 0,450 |
| Kruskal-Wallis | `Outstanding_Debt` selon `Credit_Score` | H = 2411,793 ; p < 0,001 |

Les deux variables testées sont donc liées à la classe de crédit. Ces tests restent exploratoires et ne mesurent pas la performance prédictive.

### 4.2 Prétraitement et encodage (notebook 02)

Toutes les transformations sont ajustées sur l'ensemble d'entraînement uniquement, puis appliquées à la validation, pour éviter toute fuite de données.

1. **Séparation** : les identifiants sont retirés (23 variables explicatives restantes) et `Credit_Score` est isolée comme cible. Le jeu est divisé en 80 % d'entraînement (80 000 lignes) et 20 % de validation (20 000 lignes), avec stratification sur la cible et `random_state=42`.
2. **Imputation** : médiane pour les variables numériques, modalité `Inconnu` pour les variables catégorielles, afin de conserver l'information d'absence.
3. **Encodage** :

   | Variable | Méthode | Colonnes produites |
   |---|---|---|
   | `Month` | Cyclique (sinus / cosinus) | 2 |
   | `Occupation` | One-Hot | 16 |
   | `Type_of_Loan` | Multi-Hot (plusieurs prêts possibles par client) | 10 (9 types + `Inconnu`) |
   | `Credit_Mix` | Ordinal (`Inconnu`=-1, `Bad`=0, `Standard`=1, `Good`=2) | 1 |
   | `Payment_of_Min_Amount` | One-Hot | 3 |
   | `Payment_Behaviour` | One-Hot | 7 |

4. **Normalisation** : `StandardScaler` sur les 17 variables numériques d'origine. Les variables binaires et cycliques ne sont pas modifiées.

Le jeu final comporte **56 variables**, sans aucune valeur manquante, et est exporté en CSV pour la phase de modélisation.

### 4.3 Modélisation, évaluation et optimisation (notebook 03)

Six modèles sont entraînés sur le même jeu prétraité : Zero-R (référence), Naive Bayes, Arbre de décision, KNN, SVM (noyau RBF, `C=1`) et Random Forest (`n_estimators=30`, `max_depth=15`).

Pour chacun, on mesure l'accuracy, la précision, le rappel et le F1-score par classe, les F1 macro et pondéré, la matrice de confusion, ainsi que les temps d'entraînement et de prédiction. Tout est enregistré dans `results/resultats_modeles.csv`.

Le Random Forest, meilleur compromis précision / temps, est ensuite optimisé par `GridSearchCV` : validation croisée stratifiée à 5 plis, 240 ajustements au total, score optimisé = F1 macro.

## 5. Résultats

### 5.1 Comparaison des modèles (ensemble de validation, 20 000 échantillons)

| Modèle | Accuracy | Temps d'entraînement | Temps de prédiction |
|---|---|---|---|
| Zero-R | 53,18 % | — | — |
| Naive Bayes | 57,86 % | — | — |
| Arbre de décision | 59,17 % | 3,07 s | 0,02 s |
| KNN | 62,79 % | — | 4,92 s |
| SVM | 68,65 % | 920 s | environ 150 s |
| **Random Forest** | **68,72 %** | **2,40 s** | **0,07 s** |

Observations :

- **Zero-R** prédit toujours `Standard`. Son accuracy correspond exactement à la proportion de cette classe, et le F1 de `Good` et `Poor` est nul. Tous les autres modèles font mieux, ce qui confirme qu'ils apprennent bien quelque chose.
- **Random Forest et SVM** ont des performances quasi identiques (Weighted F1 d'environ 0,686), mais le SVM est des centaines de fois plus lent. Le Random Forest est donc le choix le plus pratique.
- **KNN** est correct, mais son temps de prédiction est élevé : il doit calculer la distance avec tous les points d'entraînement.
- **Arbre de décision** est limité à environ 59 %, avec un risque de surapprentissage.
- **Naive Bayes** surestime les classes `Good` (rappel de 82,33 %, précision de 40,05 %) et `Poor`, au détriment de `Standard` (rappel de 40,80 %).
- Pour tous les modèles, `Standard` est la mieux reconnue et concentre l'essentiel des confusions. `Good` est la plus difficile à détecter (F1 d'environ 0,58 pour le Random Forest).

### 5.2 Random Forest en validation croisée (5 plis)

| Indicateur | Valeur |
|---|---|
| Accuracy moyenne | 75,12 % (de 74,99 % à 75,35 % selon les plis) |
| F1 macro moyen | 73,22 % |
| Temps d'entraînement cumulé | 8,17 s |

### 5.3 Random Forest optimisé (GridSearchCV)

Meilleurs hyperparamètres retenus : `max_depth=None`, `min_samples_leaf=1`, `min_samples_split=2`.

| Indicateur | Valeur |
|---|---|
| F1 macro (validation croisée) | **78,80 %** |
| Bonnes prédictions sur la validation | Good : 2 165, Standard : 8 042, Poor : 3 633 (soit 13 840 sur 20 000, environ 69,2 %) |
| Temps d'entraînement (recherche sur grille incluse) | 1 752,56 s |
| Temps de prédiction | 0,76 s |

Les erreurs restent concentrées autour de la classe `Standard` : par exemple, 1 280 clients `Standard` prédits `Good`, 1 314 prédits `Poor`, et 1 778 clients `Poor` prédits `Standard`.

## 6. Application web

Une application Streamlit permet de saisir le profil d'un client et d'obtenir le score prédit par le Random Forest, avec la probabilité de chaque classe.

Le dossier `app/` contient :

- `preprocessing.py` : le même prétraitement que le notebook 02 (imputation, encodages, normalisation), appris sur l'entraînement uniquement.
- `train_model.py` : entraîne le Random Forest final et enregistre le modèle et le prétraitement dans `credit_rf.joblib`.
- `app.py` : l'interface web.

Lancement, depuis le dossier `app/` :

```bash
pip install -r requirements.txt
python train_model.py chemin/vers/train_clean.csv
streamlit run app.py
```

`train_model.py` affiche l'accuracy et le F1 macro sur l'ensemble de validation à la fin de l'entraînement. Les hyperparamètres du modèle sont définis en haut du script (`RF_PARAMS`). L'application s'ouvre ensuite dans le navigateur, à l'adresse indiquée par Streamlit (en général `http://localhost:8501`).

Le formulaire reprend les 23 variables explicatives, regroupées en quatre blocs : profil, revenus, crédits et dettes, historique de paiement. Cette application est un outil pédagogique et ne constitue pas une décision de crédit.

## 7. Conclusion

Le modèle retenu est le **Random Forest optimisé par GridSearchCV**. Il obtient le meilleur F1 macro, bat nettement l'arbre de décision seul et le Random Forest de base, et reste très rapide en prédiction (0,76 s pour 20 000 échantillons), ce qui le rend utilisable en situation réelle. Son principal inconvénient est le coût de la recherche d'hyperparamètres.

## 8. Limites et perspectives

**Limites**

- La recherche sur grille est longue (environ 30 minutes).
- Une forêt d'arbres est plus difficile à interpréter qu'un arbre de décision unique.
- La séparation entraînement / validation est aléatoire et stratifiée, mais pas groupée par client. Comme chaque client apparaît 8 fois, des lignes d'un même client peuvent se retrouver des deux côtés, ce qui peut rendre les scores un peu trop optimistes. L'écart entre les scores en validation croisée (75 à 79 %) et l'accuracy sur l'ensemble de validation (environ 69 %) va dans ce sens.
- Le F1 macro de 78,80 % est un score de validation croisée et ne se compare pas directement aux accuracy du tableau 5.1, calculées sur l'ensemble de validation.

**Perspectives**

- Séparer les données par `Customer_ID` (par exemple avec `GroupKFold`) pour une évaluation plus rigoureuse.
- Tester des modèles de boosting (XGBoost, LightGBM, CatBoost).
- Rééquilibrer les classes, par exemple avec SMOTE, pour mieux prédire les classes minoritaires.
- Faire une sélection de variables plus poussée pour réduire la dimension (56 variables actuellement).

## 9. Auteurs

Groupe RSD10 :

- [Prénom Nom] : analyse exploratoire, prétraitement et application
- [Prénom Nom] : modélisation, évaluation et optimisation
