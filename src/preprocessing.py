"""
preprocessing.py
================
Prétraitement du jeu de données « Credit Score » (mini-projet Data Mining, groupe RSD10).

Ce module reproduit, de façon réutilisable, les étapes du notebook
`02_preprocessing_encoding.ipynb` :

1. imputation (médiane pour le numérique, « Inconnu » pour le catégoriel) ;
2. encodage cyclique de `Month` (sin / cos) ;
3. One-Hot de `Occupation`, `Payment_of_Min_Amount`, `Payment_Behaviour` ;
4. Multi-Hot de `Type_of_Loan` ;
5. encodage ordinal de `Credit_Mix` (Inconnu=-1, Bad=0, Standard=1, Good=2) ;
6. normalisation (StandardScaler) des 17 variables numériques.

Il sert aussi à transformer une saisie « brute » (formulaire du prédicteur Streamlit)
en une ligne de 56 variables, dans le même ordre que `X_train_preprocessed.csv`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("DM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

TARGET = "Credit_Score"
CLASS_ORDER = ["Poor", "Standard", "Good"]  # ordre « naturel » (du pire au meilleur)

# --------------------------------------------------------------------------- #
# Définition des variables (identique au notebook 02)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTHS_IN_DATA = MONTHS[:8]  # le jeu d'entraînement contient janvier → août

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Inconnu", "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager",
    "Musician", "Scientist", "Teacher", "Writer",
]
LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
]
LOAN_UNKNOWN_COL = "Type_of_Loan_Inconnu"
PAYMENT_MIN = ["NM", "No", "Yes"]
PAYMENT_BEHAVIOURS = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Inconnu",
    "Low_spent_Large_value_payments", "Low_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
]
CREDIT_MIX_MAPPING = {"Inconnu": -1, "Bad": 0, "Standard": 1, "Good": 2}

# Ordre exact des colonnes de X_train_preprocessed.csv (56 variables)
FEATURE_COLUMNS = (
    NUMERIC_FEATURES[:11]
    + ["Credit_Mix"]
    + NUMERIC_FEATURES[11:]
    + ["Month_sin", "Month_cos"]
    + [f"Occupation_{o}" for o in OCCUPATIONS]
    + LOAN_TYPES
    + [LOAN_UNKNOWN_COL]
    + [f"Payment_of_Min_Amount_{p}" for p in PAYMENT_MIN]
    + [f"Payment_Behaviour_{b}" for b in PAYMENT_BEHAVIOURS]
)

# --------------------------------------------------------------------------- #
# Paramètres du StandardScaler
# --------------------------------------------------------------------------- #
# Le scaler du notebook n'a pas été sauvegardé. Ces valeurs (moyenne, écart-type)
# ont été RECONSTITUÉES à partir de X_train_preprocessed.csv :
#   * variables entières (Age, Num_*, Interest_Rate, Delay…, Credit_History_Age) :
#     valeurs exactes (le pas entre deux valeurs standardisées voisines donne l'écart-type) ;
#   * Credit_Utilization_Ratio, Outstanding_Debt, Changed_Credit_Limit… : précision ≈ 0,1 % ;
#   * Annual_Income et Total_EMI_per_month : approximation (valeurs aberrantes dans les données).
# Pour des valeurs exactes : exporter le scaler depuis le notebook 02 (voir README.md)
# dans models/scaler_stats.json — il sera alors utilisé automatiquement.
SCALER_FALLBACK: Dict[str, Tuple[float, float]] = {
    "Age": (33.3296, 10.6336),
    "Annual_Income": (62776.2954, 434478.9564),
    "Monthly_Inhand_Salary": (4027.1271, 2960.5142),
    "Num_Bank_Accounts": (5.3509, 2.5703),
    "Num_Credit_Card": (5.5194, 2.0494),
    "Interest_Rate": (14.4897, 8.6583),
    "Num_of_Loan": (3.5076, 2.3910),
    "Delay_from_due_date": (21.0700, 14.8657),
    "Num_of_Delayed_Payment": (13.4585, 5.9432),
    "Changed_Credit_Limit": (10.3247, 6.7102),
    "Num_Credit_Inquiries": (5.7643, 3.8049),
    "Outstanding_Debt": (1426.7831, 1156.3815),
    "Credit_Utilization_Ratio": (32.2896, 5.1152),
    "Credit_History_Age": (220.9910, 95.4773),  # en MOIS
    "Total_EMI_per_month": (135.9242, 414.3994),
    "Amount_invested_monthly": (189.9761, 191.7715),
    "Monthly_Balance": (401.5573, 213.1614),
}


def load_scaler_params() -> Tuple[Dict[str, Tuple[float, float]], str]:
    """Retourne ({variable: (moyenne, écart-type)}, source).

    Ordre de priorité : models/scaler_stats.json → models/scaler.joblib → estimation.
    """
    json_path = MODELS_DIR / "scaler_stats.json"
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            params = {f: (float(m), float(s)) for f, m, s in zip(raw["features"], raw["mean"], raw["scale"])}
            if set(NUMERIC_FEATURES) <= set(params):
                return params, "scaler_stats.json (exact)"
        except (OSError, ValueError, KeyError):
            pass
    joblib_path = MODELS_DIR / "scaler.joblib"
    if joblib_path.exists():
        try:
            import joblib

            scaler = joblib.load(joblib_path)
            names = list(getattr(scaler, "feature_names_in_", NUMERIC_FEATURES))
            params = {n: (float(m), float(s)) for n, m, s in zip(names, scaler.mean_, scaler.scale_)}
            if set(NUMERIC_FEATURES) <= set(params):
                return params, "scaler.joblib (exact)"
        except Exception:  # fichier corrompu ou version sklearn différente
            pass
    return dict(SCALER_FALLBACK), "estimation à partir des données prétraitées"


# --------------------------------------------------------------------------- #
# Transformation d'un DataFrame « brut nettoyé » → 56 variables
# --------------------------------------------------------------------------- #
CATEGORICAL_RAW = ["Month", "Occupation", "Type_of_Loan", "Credit_Mix",
                   "Payment_of_Min_Amount", "Payment_Behaviour"]


def transform_dataframe(
    df: pd.DataFrame,
    scaler_params: Optional[Dict[str, Tuple[float, float]]] = None,
    medians: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Applique le prétraitement complet à un DataFrame de données nettoyées.

    `df` doit contenir les 17 variables numériques et les 6 variables catégorielles
    (`CATEGORICAL_RAW`). Les colonnes ID, Customer_ID, Name, SSN et Credit_Score sont ignorées.
    """
    params = scaler_params or load_scaler_params()[0]
    d = df.copy()

    # 1. Imputation
    for col in NUMERIC_FEATURES:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        fill = (medians or {}).get(col, d[col].median())
        d[col] = d[col].fillna(fill)
    for col in CATEGORICAL_RAW:
        d[col] = d[col].astype("string").str.strip().fillna("Inconnu").replace({"": "Inconnu"}).astype(str)

    out = pd.DataFrame(index=d.index)

    # 2. Variables numériques normalisées
    for col in NUMERIC_FEATURES:
        mean, scale = params[col]
        out[col] = (d[col] - mean) / scale

    # 3. Credit_Mix (ordinal)
    out["Credit_Mix"] = d["Credit_Mix"].map(CREDIT_MIX_MAPPING).fillna(-1).astype(int)

    # 4. Month (cyclique)
    month_number = d["Month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    if month_number.isna().any():
        raise ValueError(f"Mois inconnu(s) : {sorted(d.loc[month_number.isna(), 'Month'].unique())}")
    out["Month_sin"] = np.sin(2 * np.pi * (month_number - 1) / 12)
    out["Month_cos"] = np.cos(2 * np.pi * (month_number - 1) / 12)

    # 5. One-Hot
    for occ in OCCUPATIONS:
        out[f"Occupation_{occ}"] = (d["Occupation"] == occ).astype(float)
    for val in PAYMENT_MIN:
        out[f"Payment_of_Min_Amount_{val}"] = (d["Payment_of_Min_Amount"] == val).astype(float)
    for val in PAYMENT_BEHAVIOURS:
        out[f"Payment_Behaviour_{val}"] = (d["Payment_Behaviour"] == val).astype(float)

    # 6. Multi-Hot des types de prêts
    for loan in LOAN_TYPES:
        out[loan] = d["Type_of_Loan"].apply(
            lambda v, loan=loan: int(v != "Inconnu" and loan in [x.strip() for x in v.split(",")])
        )
    out[LOAN_UNKNOWN_COL] = (d["Type_of_Loan"] == "Inconnu").astype(int)

    return out[FEATURE_COLUMNS]


def build_feature_row(raw: dict, scaler_params=None) -> pd.DataFrame:
    """Transforme une saisie brute (dict du formulaire) en DataFrame d'une ligne (56 colonnes).

    `raw["Type_of_Loan"]` peut être une liste de types de prêts ou une chaîne « A, B, C ».
    Une liste vide est encodée « Inconnu » (comme les valeurs manquantes à l'entraînement).
    """
    row = dict(raw)
    loans = row.get("Type_of_Loan", "Inconnu")
    if isinstance(loans, (list, tuple, set)):
        loans = ", ".join(loans) if len(loans) else "Inconnu"
    row["Type_of_Loan"] = loans
    return transform_dataframe(pd.DataFrame([row]), scaler_params=scaler_params)


def inverse_scale_numeric(X_scaled: pd.DataFrame, scaler_params=None) -> pd.DataFrame:
    """Retourne les 17 variables numériques à l'échelle d'origine (utile pour les tests/démos)."""
    params = scaler_params or load_scaler_params()[0]
    return pd.DataFrame({c: X_scaled[c] * params[c][1] + params[c][0] for c in NUMERIC_FEATURES})


# --------------------------------------------------------------------------- #
# Chargement des données prétraitées
# --------------------------------------------------------------------------- #
def load_processed_data() -> dict:
    """Charge X_train / y_train / X_val / y_val / groups_train depuis data/processed/."""
    needed = ["X_train_preprocessed.csv", "y_train.csv", "X_val_preprocessed.csv", "y_val.csv"]
    missing = [f for f in needed if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Fichiers introuvables dans {DATA_DIR} : {', '.join(missing)}")

    data = {
        "X_train": pd.read_csv(DATA_DIR / "X_train_preprocessed.csv"),
        "y_train": pd.read_csv(DATA_DIR / "y_train.csv")[TARGET],
        "X_val": pd.read_csv(DATA_DIR / "X_val_preprocessed.csv"),
        "y_val": pd.read_csv(DATA_DIR / "y_val.csv")[TARGET],
    }
    groups_path = DATA_DIR / "groups_train.csv"
    data["groups_train"] = pd.read_csv(groups_path).iloc[:, 0] if groups_path.exists() else None
    return data


def feature_group(column: str) -> str:
    """Regroupe les colonnes One/Multi-Hot par variable d'origine (pour l'importance agrégée)."""
    if column.startswith("Occupation_"):
        return "Occupation"
    if column.startswith("Payment_of_Min_Amount_"):
        return "Payment_of_Min_Amount"
    if column.startswith("Payment_Behaviour_"):
        return "Payment_Behaviour"
    if column in LOAN_TYPES or column == LOAN_UNKNOWN_COL:
        return "Type_of_Loan"
    if column in ("Month_sin", "Month_cos"):
        return "Month"
    return column


def decode_onehot(X: pd.DataFrame, prefix: str) -> pd.Series:
    """Reconstruit la variable catégorielle d'origine à partir de ses colonnes One-Hot."""
    cols = [c for c in X.columns if c.startswith(prefix + "_")]
    return X[cols].idxmax(axis=1).str[len(prefix) + 1:]


def decode_credit_mix(X: pd.DataFrame) -> pd.Series:
    inverse = {v: k for k, v in CREDIT_MIX_MAPPING.items()}
    return X["Credit_Mix"].map(inverse)