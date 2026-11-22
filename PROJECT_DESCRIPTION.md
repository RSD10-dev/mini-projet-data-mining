# Description du projet

**Titre :** Classification du score de crédit
**Cadre :** Mini-projet de Data Mining
**Groupe :** RSD10

## 1. Contexte

Les établissements financiers doivent évaluer la fiabilité de leurs clients avant d'accorder un crédit. Cette évaluation repose sur un score de crédit, qui résume la situation financière et le comportement de paiement d'une personne. Ce projet propose de reproduire cette démarche avec des méthodes d'apprentissage automatique : à partir des données d'un client, prédire si son score de crédit est `Poor` (faible), `Standard` (moyen) ou `Good` (bon).

## 2. Problématique

Peut-on prédire de manière fiable la catégorie de score de crédit d'un client à partir de ses informations bancaires et de ses habitudes de paiement, et quel algorithme offre le meilleur compromis entre performance et temps de calcul ?

Il s'agit d'un problème de **classification supervisée multiclasse**, avec une cible déséquilibrée (Standard 53,17 %, Poor 29,00 %, Good 17,83 %). Pour cette raison, l'accuracy seule ne suffit pas : les modèles sont aussi jugés sur la précision, le rappel et le F1-score (macro et pondéré).

## 3. Objectifs

1. Explorer le jeu de données et identifier ses problèmes de qualité (valeurs manquantes, codes d'erreur, valeurs aberrantes, déséquilibre des classes).
2. Préparer les données pour la modélisation sans fuite d'information : imputation, encodage des variables catégorielles, normalisation.
3. Construire et comparer plusieurs modèles de classification.
4. Optimiser le meilleur modèle par recherche d'hyperparamètres avec validation croisée.
5. Mettre le modèle final à disposition dans une application web simple.

## 4. Données

- **Source :** dataset *Credit Score Classification*, fichier `train.csv`.
- **Taille :** 100 000 observations, 28 attributs.
- **Structure :** 12 500 clients, chacun observé sur 8 mois (janvier à août).
- **Variable cible :** `Credit_Score` (Poor, Standard, Good).
- **Types de variables :** revenus et salaire, nombre de comptes et de cartes, prêts et types de prêts, dette, retards de paiement, taux d'utilisation du crédit, ancienneté du crédit, comportement de paiement, profession, mois.
- **Données personnelles :** `Name` et `SSN`, ainsi que les identifiants `ID` et `Customer_ID`, sont exclus des variables explicatives.

## 5. Méthodologie

| Étape | Contenu | Notebook |
|---|---|---|
| Exploration et nettoyage | Qualité des données, statistiques descriptives, corrélations, deux tests statistiques (χ² et Kruskal-Wallis) | `01_eda_and_cleaning` |
| Prétraitement | Séparation 80/20 stratifiée, imputation (médiane, « Inconnu »), encodages (cyclique, One-Hot, Multi-Hot, ordinal), StandardScaler. Résultat : 56 variables | `02_preprocessing_encoding` |
| Modélisation | Zero-R, Naive Bayes, Arbre de décision, KNN, SVM, Random Forest | `03_model_training_evaluation` |
| Optimisation | GridSearchCV sur le Random Forest, validation croisée stratifiée à 5 plis, score F1 macro | `03_model_training_evaluation` |
| Déploiement | Application Streamlit de prédiction | `app/` |

Toutes les transformations sont ajustées sur l'ensemble d'entraînement uniquement, afin d'éviter la fuite de données.

## 6. Outils et technologies

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook, Streamlit, joblib.

## 7. Livrables

- Trois notebooks documentés (exploration, prétraitement, modélisation).
- Les jeux de données prétraités et le fichier de résultats `resultats_modeles.csv`.
- Une application web de prédiction (dossier `app/`).
- Un fichier `README.md` décrivant l'installation, la démarche et les résultats.

## 8. Résultats principaux

- Sur l'ensemble de validation, le Random Forest et le SVM sont les meilleurs modèles (environ 68,7 % d'accuracy), mais le Random Forest s'entraîne en 2,4 s contre 920 s pour le SVM.
- Après optimisation par GridSearchCV, le Random Forest atteint un F1 macro de 78,80 % en validation croisée, pour un temps de prédiction de 0,76 s.
- Le **Random Forest optimisé** est retenu comme modèle final.
- La classe `Standard` est la mieux reconnue et concentre la majorité des confusions. `Good` est la plus difficile à détecter.

## 9. Limites et perspectives

**Limites.** La séparation entraînement / validation n'est pas groupée par client, ce qui peut rendre les scores légèrement optimistes. La recherche sur grille est coûteuse en temps. La forêt d'arbres reste moins interprétable qu'un arbre unique.

**Perspectives.** Séparer les données par `Customer_ID`, tester des modèles de boosting (XGBoost, LightGBM, CatBoost), rééquilibrer les classes (SMOTE) et approfondir la sélection de variables.

## 10. Équipe
rsd10
rsd12
siad21
siad23

