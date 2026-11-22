
"""
models.py
=========
Entraînement, évaluation et analyse des modèles de classification du Credit Score
(Good / Standard / Poor), d'après le notebook `03_model_training_evaluation.ipynb`.

Usage en ligne de commande (entraîne et sauvegarde le Random Forest final) :

    python src/models.py
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:  # exécution via « streamlit run src/app.py » ou « python src/models.py »
    import preprocessing as pp
except ImportError:  # exécution en tant que package
    from src import preprocessing as pp

CLASSES = ["Good", "Standard", "Poor"]  # ordre utilisé dans resultats_modeles.csv
RF_PARAMS = dict(n_estimators=30, max_depth=15, random_state=42, n_jobs=-1)  # modèle retenu (notebook 03)
RF_MODEL_PATH = pp.MODELS_DIR / "rf_model.joblib"

# Grille explorée par GridSearchCV dans le notebook 03 (scoring = f1_macro, StratifiedKFold k=5)
GRID_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

QUICK_MODELS = {
    "ZERO_R": lambda: DummyClassifier(strategy="most_frequent"),
    "NAIVE": lambda: GaussianNB(),
    "DECISION_TREE": lambda: DecisionTreeClassifier(random_state=42),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "RANDOM_FOREST": lambda: RandomForestClassifier(**RF_PARAMS),
}


# --------------------------------------------------------------------------- #
# Résultats déjà calculés
# --------------------------------------------------------------------------- #
# Lignes du CSV issues de la validation croisée 5 plis SUR L'ENTRAÎNEMENT (notebook 03, cellule CV) :
# ce ne sont pas des scores sur le jeu de validation (hold-out) et ils ne sont pas comparables aux autres.
CV_ROWS = {"model_RANDOM_FOREST", "model_RANDOM_FOREST-OPTIMISATION"}
PROTOCOL_HOLDOUT = "Validation (20 000 lignes)"
PROTOCOL_CV = "CV 5 plis (sur train)"

# Résultat du GridSearchCV (notebook 03) : meilleurs hyperparamètres et score CV associé
GRIDSEARCH_BEST_PARAMS = {"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 200}
GRIDSEARCH_CV_BEST_SCORE = 0.7880  # F1 macro moyen en CV (StratifiedKFold non groupé)


def load_results() -> pd.DataFrame:
    """Charge results/resultats_modeles.csv ; ajoute « Nom » (lisible) et « Protocole »."""
    df = pd.read_csv(pp.RESULTS_DIR / "resultats_modeles.csv")
    df["Nom"] = df["Modele"].str.replace("^model_", "", regex=True)
    df["Protocole"] = np.where(df["Modele"].isin(CV_ROWS), PROTOCOL_CV, PROTOCOL_HOLDOUT)
    return df


def best_model_row(results: pd.DataFrame) -> pd.Series:
    """Meilleure ligne selon le Macro-F1, parmi les scores sur le jeu de validation uniquement."""
    holdout = results[results["Protocole"] == PROTOCOL_HOLDOUT]
    return holdout.loc[holdout["Macro_F1"].idxmax()]


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """Mêmes indicateurs que ceux de resultats_modeles.csv."""
    out = {"Accuracy": float(accuracy_score(y_true, y_pred))}
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    for cls, p, r, f in zip(CLASSES, prec, rec, f1):
        out[f"Precision_{cls}"] = float(p)
        out[f"Recall_{cls}"] = float(r)
        out[f"F1_{cls}"] = float(f)
    out["Macro_F1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["Weighted_F1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return out


def confusion_matrix_df(y_true, y_pred, labels: Iterable[str] = pp.CLASS_ORDER) -> pd.DataFrame:
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"Réel : {l}" for l in labels], columns=[f"Prédit : {l}" for l in labels])


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def train_random_forest(X_train, y_train, **overrides) -> RandomForestClassifier:
    params = {**RF_PARAMS, **overrides}
    return RandomForestClassifier(**params).fit(X_train, y_train)


def load_or_train_rf(X_train, y_train, path=RF_MODEL_PATH) -> RandomForestClassifier:
    """Charge le Random Forest sauvegardé, sinon l'entraîne (≈ 10 s) et tente de le sauvegarder."""
    if path.exists():
        try:
            model = joblib.load(path)
            if getattr(model, "n_features_in_", None) == X_train.shape[1]:
                return model
        except Exception:
            pass  # fichier illisible → on réentraîne
    model = train_random_forest(X_train, y_train)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
    except OSError:
        pass  # système de fichiers en lecture seule : sans importance
    return model


def run_quick_benchmark(X_train, y_train, X_val, y_val, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Réentraîne les modèles rapides et retourne un tableau au format de resultats_modeles.csv."""
    rows = []
    for name in (names or QUICK_MODELS):
        model = QUICK_MODELS[name]()
        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0
        t0 = time.time()
        y_pred = model.predict(X_val)
        t_pred = time.time() - t0
        rows.append({"Nom": name, **evaluate_predictions(y_val, y_pred),
                     "Temps_Entrainement": t_fit, "Temps_Prediction": t_pred})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Importance des variables
# --------------------------------------------------------------------------- #
def rf_importance(model: RandomForestClassifier, columns: Iterable[str]) -> pd.DataFrame:
    return (pd.DataFrame({"Variable": list(columns), "Importance_RF": model.feature_importances_})
            .sort_values("Importance_RF", ascending=False).reset_index(drop=True))


def permutation_importance_df(model, X_val, y_val, n_samples=3000, n_repeats=3, seed=42) -> pd.DataFrame:
    """Importance par permutation (baisse du Macro-F1) sur un échantillon de la validation."""
    idx = np.random.RandomState(seed).choice(len(X_val), size=min(n_samples, len(X_val)), replace=False)
    result = permutation_importance(model, X_val.iloc[idx], y_val.iloc[idx], scoring="f1_macro",
                                    n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"Variable": X_val.columns, "Importance_Permutation": result.importances_mean})
            .sort_values("Importance_Permutation", ascending=False).reset_index(drop=True))


def mutual_info_df(X_train, y_train, n_samples=10000, seed=42) -> pd.DataFrame:
    """Information mutuelle entre chaque variable et la cible (échantillon de l'entraînement)."""
    idx = np.random.RandomState(seed).choice(len(X_train), size=min(n_samples, len(X_train)), replace=False)
    discrete = np.array([c not in pp.NUMERIC_FEATURES and c not in ("Month_sin", "Month_cos")
                         for c in X_train.columns])
    mi = mutual_info_classif(X_train.iloc[idx], y_train.iloc[idx], discrete_features=discrete, random_state=seed)
    return (pd.DataFrame({"Variable": X_train.columns, "Information_Mutuelle": mi})
            .sort_values("Information_Mutuelle", ascending=False).reset_index(drop=True))


def combine_importances(*tables: pd.DataFrame) -> pd.DataFrame:
    """Fusionne plusieurs tableaux d'importance et calcule le rang moyen."""
    merged = tables[0]
    for t in tables[1:]:
        merged = merged.merge(t, on="Variable", how="outer")
    value_cols = [c for c in merged.columns if c != "Variable"]
    for c in value_cols:
        merged[f"Rang_{c.split('_', 1)[1]}"] = merged[c].rank(ascending=False, method="min").astype(int)
    rank_cols = [c for c in merged.columns if c.startswith("Rang_")]
    merged["Rang_moyen"] = merged[rank_cols].mean(axis=1)
    return merged.sort_values("Rang_moyen").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Déséquilibre des classes
# --------------------------------------------------------------------------- #
def imbalance_experiment(X_train, y_train, X_val, y_val, include_smote: bool = False) -> pd.DataFrame:
    """Compare le Random Forest de base à des variantes qui compensent le déséquilibre."""
    variants = {
        "RF de base": dict(),
        "RF class_weight='balanced'": dict(class_weight="balanced"),
        "RF class_weight='balanced_subsample'": dict(class_weight="balanced_subsample"),
    }
    rows = []
    for name, kwargs in variants.items():
        model = train_random_forest(X_train, y_train, **kwargs)
        rows.append({"Variante": name, **evaluate_predictions(y_val, model.predict(X_val))})
    if include_smote:
        try:
            from imblearn.over_sampling import SMOTE

            X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
            model = train_random_forest(X_res, y_res)
            rows.append({"Variante": "RF + SMOTE", **evaluate_predictions(y_val, model.predict(X_val))})
        except ImportError:
            pass
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Prédiction d'un client
# --------------------------------------------------------------------------- #
def predict_one(model, row: pd.DataFrame):
    """Retourne (classe prédite, Series de probabilités ordonnée Poor / Standard / Good)."""
    proba = pd.Series(model.predict_proba(row)[0], index=model.classes_).reindex(pp.CLASS_ORDER)
    return str(proba.idxmax()), proba


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data = pp.load_processed_data()
    print("Entraînement du Random Forest…", RF_PARAMS)
    t0 = time.time()
    rf = train_random_forest(data["X_train"], data["y_train"])
    print(f"Terminé en {time.time() - t0:.1f} s")
    for key, value in evaluate_predictions(data["y_val"], rf.predict(data["X_val"])).items():
        print(f"{key:>18s} : {value:.4f}")
    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, RF_MODEL_PATH, compress=3)
    print("Modèle sauvegardé :", RF_MODEL_PATH)
