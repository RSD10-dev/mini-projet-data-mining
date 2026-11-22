Mini-projet Data Mining — Classification du score de crédit
Groupe RSD10

Ce mini-projet vise à prédire la catégorie de score de crédit d’un client (Poor, Standard ou Good) à partir de ses informations bancaires et de son comportement de paiement.

Le projet couvre toute la chaîne de Data Mining : exploration des données, nettoyage, prétraitement, comparaison de modèles, optimisation du modèle retenu et mise à disposition d’une application web avec Streamlit.

L’application Streamlit présente notamment l’EDA, les tests statistiques, l’exploration des variables, le benchmark des modèles et un prédicteur de score de crédit. Elle comporte 10 pages selon la version de l’application décrite dans le projet.
Sommaire
1. Jeu de données
2. Organisation du projet
3. Installation et exécution
4. Démarche
5. Résultats
6. Application web
7. Déploiement
8. Limites et perspectives
9. Auteurs
1. Jeu de données
Le fichier train.csv (Credit Score Classification) contient 100 000 observations et 28 attributs. Il décrit 12 500 clients, chacun observé sur 8 mois (de janvier à août). Une même personne apparaît donc 8 fois dans le fichier.

La variable cible Credit_Score est déséquilibrée :
Classe	Part des observations
Standard	53,17 %
Poor	29,00 %
Good	17,83 %


Les colonnes ID, Customer_ID, Name et SSN sont des identifiants ou des données personnelles : elles ne sont pas utilisées comme variables explicatives et ne sont pas affichées dans les analyses.
2. Organisation du projet
L’organisation ci-dessous correspond à la structure de l’application actuellement utilisée dans le dossier src/. Les fichiers de données, notebooks et résultats sont conservés dans leurs dossiers respectifs.
mini-projet-data-mining/
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       ├── X_train_preprocessed.csv
│       ├── X_val_preprocessed.csv
│       ├── y_train.csv
│       ├── y_val.csv
│       └── groups_train.csv                 (si disponible)
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_preprocessing_encoding.ipynb
│   └── 03_model_training_evaluation.ipynb
├── results/
│   └── resultats_modeles.csv
├── models/
│   └── scaler_stats.json                    (optionnel)
├── src/
│   ├── app.py                               application Streamlit
│   ├── models.py                            entraînement, évaluation et importance des variables
│   └── preprocessing.py                     prétraitement et transformation d’une saisie brute
├── requirements.txt
└── README.md
3. Installation et exécution
Python 3.9 ou supérieur est recommandé.

Installez les dépendances depuis la racine du projet :
pip install -r requirements.txt
Lancez ensuite l’application Streamlit :
streamlit run src/app.py
Pour entraîner le modèle depuis le module de modélisation, si cette fonctionnalité est prévue dans la version présente du script :
python src/models.py
Au premier lancement, l’application peut entraîner le Random Forest (30 arbres, profondeur maximale de 15) en quelques secondes. Le script src/models.py peut également entraîner le modèle, afficher les métriques et le sauvegarder dans models/rf_model.joblib, selon la configuration du projet.
4. Démarche
Les notebooks doivent être exécutés dans l’ordre, car chacun produit les fichiers utilisés par le suivant.
1. 01_eda_and_cleaning.ipynb : exploration, contrôle qualité et nettoyage du jeu de données.
2. 02_preprocessing_encoding.ipynb : traitement des valeurs manquantes, encodage et normalisation ; production des fichiers prétraités.
3. 03_model_training_evaluation.ipynb : entraînement et évaluation des modèles ; enregistrement des résultats dans results/resultats_modeles.csv.
4.1 Exploration et contrôle qualité (notebook 01)
Le fichier brut n’est jamais écrasé. Les corrections de format sont faites en mémoire, dans une copie de travail, uniquement pour rendre les analyses possibles. Les formats récupérables sont corrigés et les valeurs invalides remplacées par NaN, ce qui donne le jeu nettoyé utilisé ensuite.

Principaux constats :
- Aucune ligne dupliquée, mais 8 observations par client.
- Sept colonnes contiennent des cellules vides, dont Monthly_Inhand_Salary (15 %).
- Credit_Mix contient 20 195 valeurs codées _, et Payment_Behaviour contient 7 600 valeurs codées !@9#%8.
- Certaines valeurs numériques se terminent par un _ parasite, qu’il suffit de retirer pour les récupérer.
- Des valeurs aberrantes sont présentes : 2 770 âges suspects et 4 313 valeurs suspectes pour Num_of_Loan.
- Les revenus et les dettes ont une distribution fortement asymétrique.
- Corrélations notables : Monthly_Inhand_Salary / Monthly_Balance (r = 0,705) et Delay_from_due_date / Outstanding_Debt (r = 0,572).
Deux tests statistiques ont été menés sur un échantillon d’un enregistrement par client (12 500 clients), afin de limiter la dépendance entre observations d’un même client :
Test	Variables	Résultat
χ² d’indépendance	Credit_Mix × Credit_Score	χ² = 4064,985 ; p < 0,001 ; V de Cramér = 0,450
Kruskal-Wallis	Outstanding_Debt selon Credit_Score	H = 2411,793 ; p < 0,001


Les deux variables testées sont donc liées à la classe de crédit. Ces tests restent exploratoires et ne mesurent pas la performance prédictive.
4.2 Prétraitement et encodage (notebook 02)
Les transformations sont ajustées sur l’ensemble d’entraînement uniquement, puis appliquées à la validation, pour éviter toute fuite de données.
1. Séparation : les identifiants sont retirés (23 variables explicatives restantes) et Credit_Score est isolée comme cible. Le jeu est divisé en 80 % d’entraînement (80 000 lignes) et 20 % de validation (20 000 lignes), avec stratification sur la cible et random_state=42.
2. Imputation : médiane pour les variables numériques, modalité Inconnu pour les variables catégorielles, afin de conserver l’information d’absence.
3. Encodage :
Variable	Méthode	Colonnes produites
Month	Cyclique (sinus / cosinus)	2
Occupation	One-Hot	16
Type_of_Loan	Multi-Hot (plusieurs prêts possibles par client)	10 (9 types + Inconnu)
Credit_Mix	Ordinal (Inconnu=-1, Bad=0, Standard=1, Good=2)	1
Payment_of_Min_Amount	One-Hot	3
Payment_Behaviour	One-Hot	7


4. Normalisation : StandardScaler sur les 17 variables numériques d’origine. Les variables binaires et cycliques ne sont pas modifiées.
Le jeu final comporte 56 variables, sans aucune valeur manquante, et est exporté en CSV pour la phase de modélisation.
4.3 Modélisation, évaluation et optimisation (notebook 03)
Six modèles sont entraînés sur le même jeu prétraité : Zero-R (référence), Naive Bayes, arbre de décision, KNN, SVM (noyau RBF, C=1) et Random Forest (n_estimators=30, max_depth=15).

Pour chacun, on mesure l’accuracy, la précision, le rappel et le F1-score par classe, les F1 macro et pondéré, la matrice de confusion, ainsi que les temps d’entraînement et de prédiction. Tout est enregistré dans results/resultats_modeles.csv.

Le Random Forest, meilleur compromis précision / temps dans cette comparaison, est ensuite optimisé par GridSearchCV : validation croisée stratifiée à 5 plis, 240 ajustements au total, score optimisé = F1 macro.
5. Résultats
5.1 Comparaison des modèles (ensemble de validation, 20 000 échantillons)
Modèle	Accuracy	Temps d’entraînement	Temps de prédiction
Zero-R	53,18 %	—	—
Naive Bayes	57,86 %	—	—
Arbre de décision	59,17 %	3,07 s	0,02 s
KNN	62,79 %	—	4,92 s
SVM	68,65 %	920 s	environ 150 s
Random Forest	68,72 %	2,40 s	0,07 s


Observations :
- Zero-R prédit toujours Standard. Son accuracy correspond exactement à la proportion de cette classe, et le F1 de Good et Poor est nul. Tous les autres modèles font mieux, ce qui confirme qu’ils apprennent bien quelque chose.
- Random Forest et SVM ont des performances quasi identiques (Weighted F1 d’environ 0,686), mais le SVM est des centaines de fois plus lent. Le Random Forest est donc le choix le plus pratique.
- KNN est correct, mais son temps de prédiction est élevé : il doit calculer la distance avec tous les points d’entraînement.
- Arbre de décision est limité à environ 59 %, avec un risque de surapprentissage.
- Naive Bayes surestime les classes Good (rappel de 82,33 %, précision de 40,05 %) et Poor, au détriment de Standard (rappel de 40,80 %).
- Pour tous les modèles, Standard est la mieux reconnue et concentre l’essentiel des confusions. Good est la plus difficile à détecter (F1 d’environ 0,58 pour le Random Forest).
5.2 Random Forest en validation croisée (5 plis)
Indicateur	Valeur
Accuracy moyenne	75,12 % (de 74,99 % à 75,35 % selon les plis)
F1 macro moyen	73,22 %
Temps d’entraînement cumulé	8,17 s


5.3 Random Forest optimisé (GridSearchCV)
Meilleurs hyperparamètres retenus : max_depth=None, min_samples_leaf=1, min_samples_split=2.
Indicateur	Valeur
F1 macro (validation croisée)	78,80 %
Bonnes prédictions sur la validation	Good : 2 165, Standard : 8 042, Poor : 3 633 (13 840 sur 20 000, environ 69,2 %)
Temps d’entraînement (recherche sur grille incluse)	1 752,56 s
Temps de prédiction	0,76 s


Les erreurs restent concentrées autour de la classe Standard : par exemple, 1 280 clients Standard prédits Good, 1 314 prédits Poor, et 1 778 clients Poor prédits Standard.
6. Application web
Une application Streamlit permet de saisir le profil d’un client et d’obtenir le score prédit par le Random Forest, avec la probabilité de chaque classe.

L’application présente 10 pages, notamment :
- Home et Executive Summary ;
- Dataset Explorer et EDA Dashboard ;
- Statistical Analytics Suite et Features Explorer ;
- Comparison et Model Benchmark ;
- Credit Score Risk Predictor ;
- Key Discoveries and Findings et informations sur le projet.
Dans la structure actuelle du dépôt, le dossier src/ contient :
- app.py : interface et navigation de l’application Streamlit ;
- models.py : entraînement, évaluation et importance des variables ;
- preprocessing.py : prétraitement et transformation d’une saisie brute.
Le StandardScaler du notebook 02 n’ayant pas été sauvegardé dans la version décrite, l’application peut utiliser une estimation des moyennes et écarts-types. Cette estimation est exacte pour les variables entières, mais approchée pour Annual_Income et Total_EMI_per_month.

Pour obtenir les valeurs exactes, ajoutez à la fin du notebook 02, après scaler.fit_transform, la cellule suivante puis commitez models/scaler_stats.json :
import json, os

os.makedirs("../models", exist_ok=True)
with open("../models/scaler_stats.json", "w") as f:
    json.dump({
        "features": numerical_features_to_scale,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }, f)
L’application détecte le fichier automatiquement, selon l’implémentation de src/preprocessing.py.
7. Déploiement
Pour déployer l’application avec Streamlit Community Cloud :
1. Pousser le projet sur GitHub en incluant les fichiers nécessaires à l’application, notamment data/processed/, results/ et requirements.txt.
2. Sur Streamlit Community Cloud (share.streamlit.io), choisir New app.
3. Sélectionner le dépôt et la branche du projet.
4. Indiquer src/app.py comme Main file path.
8. Limites et perspectives
Points d’attention et limites
- Credit_History_Age est saisi en mois (valeurs de 1 à 404 dans les données).
- Le fichier resultats_modeles.csv contient deux lignes « Random Forest » issues de la validation croisée (Accuracy 0,751) : elles ne sont pas comparables aux scores mesurés sur le jeu de validation.
- Le découpage train / validation des fichiers fournis est décrit comme étant fait par client (10 000 clients en train, 8 lignes chacun), tandis que le code du notebook 02 présent dans le dépôt utilise train_test_split sans groupes. Il faut vérifier et documenter le découpage réellement utilisé.
- .gitignore exclut *.joblib / *.pkl : c’est volontaire si le modèle est réentraîné au démarrage.
- La recherche sur grille est longue (environ 30 minutes).
- Une forêt d’arbres est plus difficile à interpréter qu’un arbre de décision unique.
- Une séparation aléatoire et stratifiée, mais non groupée par client, peut placer des lignes d’un même client dans les deux ensembles et rendre les scores trop optimistes.
- Le F1 macro de 78,80 % est un score de validation croisée et ne se compare pas directement aux accuracy du tableau 5.1, calculées sur l’ensemble de validation.
Perspectives
- Séparer les données par Customer_ID (par exemple avec GroupKFold) pour une évaluation plus rigoureuse.
- Tester des modèles de boosting (XGBoost, LightGBM, CatBoost).
- Rééquilibrer les classes, par exemple avec SMOTE, pour mieux prédire les classes minoritaires.
- Faire une sélection de variables plus poussée pour réduire la dimension (56 variables actuellement).
9. Auteurs
Groupe RSD10 :
- RSD10
- RSD12
- SIAD-21
- SIAD23