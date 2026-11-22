"""
app.py — Dashboard Streamlit « Credit Score Classification » (mini-projet Data Mining, groupe RSD10)

Lancer en local :   streamlit run src/app.py
Déployer :          Streamlit Community Cloud → fichier principal : src/app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

try:  # « streamlit run src/app.py » ajoute le dossier src/ au chemin d'import
    import preprocessing as pp
    import models as md
except ImportError:  # lancement depuis la racine en tant que package
    from src import models as md
    from src import preprocessing as pp

st.set_page_config(page_title="Credit Score — Data Mining", page_icon="💳", layout="wide")

GROUP = "RSD10"
CLASS_COLORS = {"Poor": "#d62728", "Standard": "#f2a900", "Good": "#2ca02c"}
CLASS_ENCODING = {"Poor": 0, "Standard": 1, "Good": 2}
N_CUSTOMERS = 12_500  # clients distincts (notebook 01) : 8 observations mensuelles chacun

PRETTY = {
    "ZERO_R": "Zero-R", "SVM": "SVM (RBF)", "RANDOM": "Random Forest (30 arbres)",
    "NAIVE": "Naive Bayes", "KNN": "KNN (k=5)", "DECISION_TREE": "Arbre de décision",
    "RANDOM_FOREST": "RF — moyenne CV 5 plis", "RANDOM_FOREST-OPTIMISATION": "RF — moyenne CV 5 plis (bis)",
    "RANDOM_FOREST_GRIDSEARCH": "Random Forest (GridSearch)",
}

CSS = """
<style>
.kpi {border: 1px solid rgba(128,128,128,.3); border-left: 5px solid #1f4fd8; border-radius: 8px;
      padding: 14px 8px; text-align: center; background: rgba(128,128,128,.07); min-height: 96px;}
.kpi .l {font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; opacity: .75;}
.kpi .v {font-size: 1.55rem; font-weight: 700; line-height: 1.25;}
.badge {border-radius: 10px; padding: 18px; text-align: center; font-size: 2rem; font-weight: 700; color: white;}
</style>
"""


# =========================================================================== #
# Utilitaires
# =========================================================================== #
def pretty(name: str) -> str:
    return PRETTY.get(name, name)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>',
                     unsafe_allow_html=True)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(show_spinner="Chargement des données prétraitées…")
def load_data() -> dict:
    return pp.load_processed_data()


@st.cache_resource(show_spinner="Entraînement du Random Forest (≈ 5 s)…")
def get_rf():
    d = load_data()
    return md.load_or_train_rf(d["X_train"], d["y_train"])


@st.cache_data(show_spinner=False)
def rf_val_predictions():
    return get_rf().predict(load_data()["X_val"])


@st.cache_data(show_spinner="Calcul de l'importance par permutation…")
def cached_permutation():
    d = load_data()
    return md.permutation_importance_df(get_rf(), d["X_val"], d["y_val"])


@st.cache_data(show_spinner="Calcul de l'information mutuelle…")
def cached_mutual_info():
    d = load_data()
    return md.mutual_info_df(d["X_train"], d["y_train"])


@st.cache_data(show_spinner=False)
def cached_results() -> pd.DataFrame:
    return md.load_results()


def full_labeled(d, split="train") -> pd.DataFrame:
    X, y = (d["X_train"], d["y_train"]) if split == "train" else (d["X_val"], d["y_val"])
    return X.assign(**{pp.TARGET: y.values})


def one_row_per_customer(d):
    """1 observation par client (les 8 mois d'un client ne sont pas indépendants)."""
    groups = d["groups_train"]
    if groups is None:
        return d["X_train"], d["y_train"]
    mask = ~groups.duplicated().values
    return d["X_train"][mask], d["y_train"][mask]


def hbar(ax, labels, values, color="#1f4fd8", fmt="{:.3f}"):
    pos = np.arange(len(labels))
    ax.barh(pos, values, color=color)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for p, v in zip(pos, values):
        ax.text(v, p, " " + fmt.format(v), va="center", fontsize=8)


# =========================================================================== #
# 1. Accueil
# =========================================================================== #
def page_home():
    d = load_data()
    res = cached_results()
    best = md.best_model_row(res)
    n_obs = len(d["X_train"]) + len(d["X_val"])

    st.title("💳 Credit Score Classification")
    st.caption("Mini-projet Data Mining · Classification multiclasse Good / Standard / Poor · "
               f"Groupe {GROUP}")

    st.subheader("📌 Tableau de bord exécutif")
    kpi_row([
        ("Observations (train + val)", f"{n_obs:,}".replace(",", " ")),
        ("Clients distincts", f"{N_CUSTOMERS:,}".replace(",", " ")),
        ("Variables (après encodage)", f"{d['X_train'].shape[1]}"),
        ("Classes", "3"),
        ("Meilleur modèle (validation)", pretty(best["Nom"])),
        ("Accuracy", f"{best['Accuracy']:.3f}"),
        ("Macro-F1", f"{best['Macro_F1']:.3f}"),
    ])
    st.write("")
    st.warning(
        "Les deux lignes « RF — moyenne CV 5 plis » du fichier de résultats (Accuracy 0,751) sont des "
        "moyennes de validation croisée sur le jeu d'entraînement, pas des scores sur le jeu de validation : "
        "elles ne sont pas comparables aux autres lignes. Détails dans la page **6. Benchmark des modèles**.")

    left, right = st.columns(2)
    with left:
        st.subheader("🎯 Objectifs de Data Mining")
        st.markdown(
            "- **Qualité des données** : auditer 100 000 observations (valeurs manquantes, codes d'erreur, "
            "valeurs aberrantes) et documenter les décisions de nettoyage.\n"
            "- **Prétraitement** : imputation, encodage (cyclique, One-Hot, Multi-Hot, ordinal) et normalisation, "
            "ajustés sur l'entraînement uniquement.\n"
            "- **Modélisation** : comparer Zero-R, Naive Bayes, KNN, Arbre, Random Forest et SVM.\n"
            "- **Optimisation** : validation croisée et GridSearchCV.\n"
            "- **Déploiement** : prédire le score d'un client dans ce tableau de bord.")
    with right:
        st.subheader("📐 Cadre d'évaluation")
        y_all = pd.concat([d["y_train"], d["y_val"]])
        share = y_all.value_counts(normalize=True).reindex(pp.CLASS_ORDER) * 100
        st.markdown(
            f"Les classes sont déséquilibrées : **Standard {share['Standard']:.1f} %**, "
            f"**Poor {share['Poor']:.1f} %**, **Good {share['Good']:.1f} %**. "
            "L'Accuracy seule est donc trompeuse (Zero-R atteint déjà 0,53) ; on suit aussi :\n"
            "- le **Macro-F1** (moyenne des F1 des 3 classes) ;\n"
            "- la **précision / rappel / F1 par classe**, surtout pour *Good* (classe minoritaire) ;\n"
            "- la **matrice de confusion** ;\n"
            "- les **temps** d'entraînement et de prédiction.")
        fig, ax = plt.subplots(figsize=(5, 1.8))
        hbar(ax, pp.CLASS_ORDER, share.values, color=[CLASS_COLORS[c] for c in pp.CLASS_ORDER], fmt="{:.1f} %")
        ax.set_xlim(0, 70)
        ax.set_xlabel("% des observations")
        fig.tight_layout()
        show_fig(fig)


# =========================================================================== #
# 2. Explorateur
# =========================================================================== #
def page_explorer():
    d = load_data()
    st.header("🗂️ Explorateur de données")
    st.caption("Données **prétraitées** (imputées, encodées, normalisées) — fichiers `data/processed/`. "
               "Les 17 variables numériques sont standardisées (moyenne 0, écart-type 1).")
    split = st.radio("Jeu de données", ["Entraînement (80 000 lignes)", "Validation (20 000 lignes)"], horizontal=True)
    df = full_labeled(d, "train" if split.startswith("Entra") else "val")

    tab1, tab2, tab3 = st.tabs(["Aperçu & filtres", "Structure", "Statistiques descriptives"])
    with tab1:
        c1, c2 = st.columns(2)
        cls = c1.selectbox("Classe Credit_Score", ["Toutes"] + pp.CLASS_ORDER)
        n = c2.slider("Nombre de lignes affichées", 5, 500, 25)
        view = df if cls == "Toutes" else df[df[pp.TARGET] == cls]
        st.write(f"{len(view):,} lignes correspondent au filtre.".replace(",", " "))
        st.dataframe(view.head(n))
        st.download_button("⬇️ Télécharger l'aperçu (CSV)", view.head(n).to_csv(index=False).encode("utf-8"),
                           file_name="apercu_credit_score.csv", mime="text/csv")
    with tab2:
        info = pd.DataFrame({"Type": df.dtypes.astype(str), "Valeurs distinctes": df.nunique(),
                             "Manquantes": df.isna().sum()})
        info.index.name = "Variable"
        st.dataframe(info)
        st.caption(f"{df.shape[0]:,} lignes × {df.shape[1]} colonnes (56 variables explicatives + la cible).".replace(",", " "))
    with tab3:
        cols = st.multiselect("Variables", pp.NUMERIC_FEATURES, default=pp.NUMERIC_FEATURES[:6])
        if cols:
            st.dataframe(df[cols].describe().T.round(3))
        else:
            st.info("Choisissez au moins une variable.")


# =========================================================================== #
# 3. EDA
# =========================================================================== #
def page_eda():
    d = load_data()
    X, y = d["X_train"], d["y_train"]
    st.header("📊 Analyse exploratoire (EDA)")
    t1, t2, t3, t4, t5 = st.tabs(["Qualité des données", "Variable cible", "Distributions", "Corrélations",
                                  "Variables catégorielles"])

    with t1:
        st.markdown("Résultats de l'audit du notebook `01_eda_and_cleaning` sur le jeu **brut** "
                    "(100 000 lignes × 28 colonnes, 0 doublon, 12 500 clients × 8 mois).")
        missing = pd.DataFrame({
            "Variable": ["Credit_Mix", "Monthly_Inhand_Salary", "Type_of_Loan", "Name", "Credit_History_Age",
                         "Amount_invested_monthly", "Payment_Behaviour", "Occupation",
                         "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Monthly_Balance"],
            "Manquantes": [20195, 15002, 11408, 9985, 9030, 8784, 7600, 7062, 7002, 2091, 1965, 1209],
        })
        missing["% manquant"] = (missing["Manquantes"] / 1000).round(2)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            hbar(ax, missing["Variable"], missing["% manquant"], fmt="{:.2f} %")
            ax.set_xlabel("% de valeurs manquantes (après harmonisation des formats)")
            ax.set_xlim(0, 25)
            fig.tight_layout()
            show_fig(fig)
        with c2:
            st.markdown("**Anomalies signalées (non supprimées)**")
            st.dataframe(pd.DataFrame({
                "Contrôle": ["Age ≤ 0", "Age > 120", "Num_of_Loan < 0", "Num_of_Loan > 100"],
                "Effectif": [886, 1884, 3876, 437]}), hide_index=True)
            st.caption("Codes d'erreur détectés : suffixes « _ », « _______ » et « !@9#%8 » "
                       "(Credit_Mix, Occupation, Payment_Behaviour…).")

    with t2:
        counts = pd.concat([y, d["y_val"]]).value_counts().reindex(pp.CLASS_ORDER)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.bar(counts.index, counts.values, color=[CLASS_COLORS[c] for c in counts.index])
            ax.set_ylabel("Observations (train + val)")
            for i, v in enumerate(counts.values):
                ax.text(i, v, f"{v:,}".replace(",", " "), ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            show_fig(fig)
        with c2:
            st.dataframe(pd.DataFrame({"Effectif": counts, "Pourcentage": (counts / counts.sum() * 100).round(2)}))
            ratio = counts.max() / counts.min()
            st.markdown(f"Rapport entre la classe majoritaire et la minoritaire : **{ratio:.1f} : 1**. "
                        "Le jeu est modérément déséquilibré (voir la page *Déséquilibre des classes*).")

    with t3:
        col = st.selectbox("Variable numérique (standardisée)", pp.NUMERIC_FEATURES, index=11)
        c1, c2 = st.columns(2)
        lo, hi = X[col].quantile([0.01, 0.99])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for cls in pp.CLASS_ORDER:
                vals = X.loc[y == cls, col]
                if hi > lo:
                    vals = vals[(vals >= lo) & (vals <= hi)]
                ax.hist(vals, bins=40, alpha=0.5, density=True, label=cls, color=CLASS_COLORS[cls])
            ax.set_xlabel(col)
            ax.set_ylabel("Densité")
            ax.legend()
            ax.set_title("Distribution par classe (1ᵉʳ–99ᵉ percentile)", fontsize=10)
            fig.tight_layout()
            show_fig(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            data = [X.loc[y == cls, col] for cls in pp.CLASS_ORDER]
            bp = ax.boxplot(data, showfliers=False, patch_artist=True)
            for patch, cls in zip(bp["boxes"], pp.CLASS_ORDER):
                patch.set_facecolor(CLASS_COLORS[cls])
                patch.set_alpha(0.6)
            ax.set_xticks(range(1, len(pp.CLASS_ORDER) + 1))
            ax.set_xticklabels(pp.CLASS_ORDER)
            ax.set_ylabel(col)
            ax.set_title("Boîtes à moustaches par classe", fontsize=10)
            fig.tight_layout()
            show_fig(fig)

    with t4:
        corr = X[pp.NUMERIC_FEATURES].corr()
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=80, ha="right", fontsize=7)
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.index, fontsize=7)
            fig.colorbar(im, ax=ax, label="Corrélation de Pearson")
            fig.tight_layout()
            show_fig(fig)
        with c2:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            top = upper.reindex(upper.abs().sort_values(ascending=False).index).head(8)
            st.markdown("**Associations linéaires les plus fortes**")
            st.dataframe(top.rename("corrélation").round(3).reset_index().rename(
                columns={"level_0": "Variable A", "level_1": "Variable B"}), hide_index=True)
        st.markdown("**Corrélation avec la cible** (Poor = 0, Standard = 1, Good = 2)")
        cor_t = X.corrwith(y.map(CLASS_ENCODING)).dropna()
        cor_t = cor_t.reindex(cor_t.abs().sort_values(ascending=False).index).head(12)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        hbar(ax, cor_t.index, cor_t.values, fmt="{:+.2f}")
        fig.tight_layout()
        show_fig(fig)

    with t5:
        c1, c2 = st.columns(2)
        with c1:
            occ = pp.decode_onehot(X, "Occupation").value_counts()
            fig, ax = plt.subplots(figsize=(6, 4.2))
            hbar(ax, occ.index, occ.values, fmt="{:,.0f}")
            ax.set_title("Occupation (« Inconnu » = manquant)", fontsize=10)
            fig.tight_layout()
            show_fig(fig)
        with c2:
            mix = pp.decode_credit_mix(X)
            table = pd.crosstab(mix, y, normalize="index")[pp.CLASS_ORDER] * 100
            fig, ax = plt.subplots(figsize=(6, 4.2))
            table.plot(kind="bar", stacked=True, ax=ax, color=[CLASS_COLORS[c] for c in pp.CLASS_ORDER], rot=0)
            ax.set_ylabel("% de la modalité")
            ax.set_xlabel("Credit_Mix")
            ax.set_title("Credit_Mix selon le Credit_Score", fontsize=10)
            fig.tight_layout()
            show_fig(fig)


# =========================================================================== #
# 4. Analyse statistique
# =========================================================================== #
def page_stats():
    d = load_data()
    st.header("🔬 Analyse statistique")
    tab1, tab2 = st.tabs(["Résultats du notebook", "Test interactif"])

    with tab1:
        st.markdown("Les tests ont été réalisés sur **1 ligne par client** (12 500 clients) afin de respecter "
                    "l'hypothèse d'indépendance : un client apparaît 8 fois dans le jeu complet.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Test 1 — χ² (Credit_Mix × Credit_Score)")
            kpi_row([("χ²", "4 064,99"), ("ddl", "4"), ("p-value", "≈ 0"), ("V de Cramér", "0,450")])
            st.markdown("Association **forte** : les clients dont le *Credit_Mix* est « Bad » sont surtout *Poor*, "
                        "ceux dont il est « Good » sont plus souvent *Good* que dans les autres groupes.")
        with c2:
            st.subheader("Test 2 — Kruskal–Wallis (Outstanding_Debt)")
            kpi_row([("H", "2 411,79"), ("p-value", "≈ 0"), ("Médiane Good", "732"), ("Médiane Poor", "1 933")])
            st.markdown("La dette restante diffère nettement selon la classe : médiane 732 (Good), 1 005 (Standard), "
                        "1 933 (Poor).")
        st.info("Kruskal–Wallis est préféré à l'ANOVA car la dette est asymétrique et non gaussienne.")

    with tab2:
        Xc, yc = one_row_per_customer(d)
        st.caption(f"Échantillon : {len(Xc):,} clients de l'entraînement (1 ligne par client).".replace(",", " "))
        kind = st.radio("Type de test", ["χ² d'indépendance (variable catégorielle)",
                                         "Kruskal–Wallis (variable numérique)"], horizontal=True)
        if kind.startswith("χ²"):
            choice = st.selectbox("Variable catégorielle", ["Credit_Mix", "Payment_of_Min_Amount",
                                                           "Payment_Behaviour", "Occupation"])
            series = pp.decode_credit_mix(Xc) if choice == "Credit_Mix" else pp.decode_onehot(Xc, choice)
            table = pd.crosstab(series, yc)[pp.CLASS_ORDER]
            chi2, p, dof, _ = stats.chi2_contingency(table)
            n = table.values.sum()
            cramer = np.sqrt(chi2 / n / min(table.shape[0] - 1, table.shape[1] - 1))
            kpi_row([("χ²", f"{chi2:,.1f}".replace(",", " ")), ("ddl", f"{dof}"),
                     ("p-value", f"{p:.2e}"), ("V de Cramér", f"{cramer:.3f}")])
            st.dataframe(table)
            verdict = "association significative" if p < 0.05 else "pas d'association significative"
            st.markdown(f"→ Au seuil de 5 % : **{verdict}** entre `{choice}` et `Credit_Score`. "
                        "Le V de Cramér mesure la force de l'association (≈ 0,1 faible · 0,3 moyenne · 0,5 forte).")
        else:
            col = st.selectbox("Variable numérique", pp.NUMERIC_FEATURES, index=11)
            groups = [Xc.loc[yc == cls, col] for cls in pp.CLASS_ORDER]
            h, p = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            epsilon2 = h / (n - 1)
            kpi_row([("H", f"{h:,.1f}".replace(",", " ")), ("p-value", f"{p:.2e}"),
                     ("ε² (taille d'effet)", f"{epsilon2:.3f}"), ("n", f"{n:,}".replace(",", " "))])
            med = pd.DataFrame({"Classe": pp.CLASS_ORDER,
                                "Médiane (standardisée)": [g.median() for g in groups],
                                "Moyenne (standardisée)": [g.mean() for g in groups]}).round(3)
            st.dataframe(med, hide_index=True)
            verdict = "des différences significatives" if p < 0.05 else "pas de différence significative"
            st.markdown(f"→ Au seuil de 5 % : **{verdict}** de `{col}` entre les trois classes.")


# =========================================================================== #
# 5. Importance des variables
# =========================================================================== #
def page_importance():
    d = load_data()
    st.header("⚖️ Comparaison de l'importance des variables")
    st.caption("Trois méthodes complémentaires : importance interne du Random Forest (impureté), importance par "
               "permutation (baisse du Macro-F1 sur la validation) et information mutuelle (dépendance avec la cible).")
    model = get_rf()
    imp_rf = md.rf_importance(model, d["X_train"].columns)
    t1, t2, t3, t4 = st.tabs(["Random Forest", "Permutation", "Information mutuelle", "Comparaison"])

    with t1:
        c1, c2 = st.columns(2)
        top_n = c1.slider("Nombre de variables", 5, 30, 15)
        aggregate = c2.checkbox("Agréger les colonnes One-Hot par variable d'origine", value=True)
        table = imp_rf.copy()
        if aggregate:
            table["Variable"] = table["Variable"].map(pp.feature_group)
            table = table.groupby("Variable", as_index=False)["Importance_RF"].sum().sort_values(
                "Importance_RF", ascending=False)
        table = table.head(top_n)
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(table) + 1))
        hbar(ax, table["Variable"], table["Importance_RF"])
        ax.set_xlabel("Importance (somme = 1)")
        fig.tight_layout()
        show_fig(fig)

    def lazy_block(tab, key, label, compute, column):
        with tab:
            if st.button(f"Calculer ({label})", key=f"btn_{key}"):
                st.session_state[key] = compute()
            if key in st.session_state:
                table = st.session_state[key].head(15)
                fig, ax = plt.subplots(figsize=(8, 5))
                hbar(ax, table["Variable"], table[column])
                fig.tight_layout()
                show_fig(fig)
            else:
                st.info("Cliquez sur le bouton pour lancer le calcul (quelques secondes).")

    lazy_block(t2, "imp_perm", "importance par permutation", cached_permutation, "Importance_Permutation")
    lazy_block(t3, "imp_mi", "information mutuelle", cached_mutual_info, "Information_Mutuelle")

    with t4:
        if "imp_perm" in st.session_state and "imp_mi" in st.session_state:
            merged = md.combine_importances(imp_rf, st.session_state["imp_perm"], st.session_state["imp_mi"])
            st.dataframe(merged.head(20).round(4), hide_index=True)
            st.caption("Rang 1 = variable la plus importante ; le rang moyen résume l'accord entre les 3 méthodes.")
        else:
            st.info("Calculez d'abord la permutation et l'information mutuelle dans les onglets précédents.")


# =========================================================================== #
# 6. Benchmark des modèles
# =========================================================================== #
def plot_confusion(cm: pd.DataFrame, normalize: bool):
    values = cm.values.astype(float)
    shown = values / values.sum(axis=1, keepdims=True) if normalize else values
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_xticklabels([c.replace("Prédit : ", "") for c in cm.columns])
    ax.set_yticks(range(3))
    ax.set_yticklabels([c.replace("Réel : ", "") for c in cm.index])
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    for i in range(3):
        for j in range(3):
            txt = f"{shown[i, j]:.0%}" if normalize else f"{int(values[i, j]):,}".replace(",", " ")
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if shown[i, j] > shown.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def page_benchmark():
    d = load_data()
    res = cached_results()
    st.header("🏁 Benchmark des modèles & hyperparamètres")
    t1, t2, t3, t4, t5 = st.tabs(["Tableau comparatif", "Graphiques", "Matrice de confusion",
                                  "Hyperparamètres & validation croisée", "Benchmark rapide (live)"])

    with t1:
        cols = ["Nom", "Protocole", "Accuracy", "Macro_F1", "Weighted_F1", "F1_Good", "F1_Standard", "F1_Poor",
                "Recall_Good", "Recall_Standard", "Recall_Poor", "Temps_Entrainement", "Temps_Prediction"]
        table = res[cols].copy()
        table["Nom"] = table["Nom"].map(pretty)
        st.dataframe(table.round(4), hide_index=True)
        st.caption("Les lignes « CV 5 plis (sur train) » sont des moyennes de validation croisée ; les autres sont "
                   "mesurées sur les 20 000 lignes du jeu de validation (clients jamais vus à l'entraînement). "
                   "Temps en secondes.")

    with t2:
        include_cv = st.checkbox("Inclure les lignes de validation croisée", value=False)
        data = res if include_cv else res[res["Protocole"] == md.PROTOCOL_HOLDOUT]
        metric = st.selectbox("Indicateur", ["Macro_F1", "Accuracy", "Weighted_F1"])
        c1, c2 = st.columns(2)
        with c1:
            ordered = data.sort_values(metric, ascending=False)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            hbar(ax, ordered["Nom"].map(pretty), ordered[metric])
            ax.set_xlim(0, 1)
            ax.set_title(metric, fontsize=10)
            fig.tight_layout()
            show_fig(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            width = 0.27
            x = np.arange(len(data))
            for k, cls in enumerate(["Good", "Standard", "Poor"]):
                ax.bar(x + (k - 1) * width, data[f"F1_{cls}"], width, label=cls, color=CLASS_COLORS[cls])
            ax.set_xticks(x)
            ax.set_xticklabels(data["Nom"].map(pretty), rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("F1 par classe")
            ax.legend()
            fig.tight_layout()
            show_fig(fig)
        fig, ax = plt.subplots(figsize=(8, 3))
        hbar(ax, data["Nom"].map(pretty), data["Temps_Entrainement"].clip(lower=0.01), color="#7a7a7a", fmt="{:.1f} s")
        ax.set_xscale("log")
        ax.set_xlabel("Temps d'entraînement (échelle logarithmique)")
        fig.tight_layout()
        show_fig(fig)

    with t3:
        st.markdown("Random Forest **léger** (30 arbres, profondeur 15) — celui utilisé par le prédicteur — "
                    "évalué sur le jeu de validation.")
        cm = md.confusion_matrix_df(d["y_val"], rf_val_predictions())
        normalize = st.radio("Affichage", ["Effectifs", "Normalisé par ligne (rappel)"], horizontal=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(plot_confusion(cm, normalize.startswith("Normalis")))
        with c2:
            metrics = md.evaluate_predictions(d["y_val"], rf_val_predictions())
            st.dataframe(pd.DataFrame({
                "Classe": md.CLASSES,
                "Précision": [metrics[f"Precision_{c}"] for c in md.CLASSES],
                "Rappel": [metrics[f"Recall_{c}"] for c in md.CLASSES],
                "F1": [metrics[f"F1_{c}"] for c in md.CLASSES]}).round(3), hide_index=True)
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("Macro-F1", f"{metrics['Macro_F1']:.4f}")

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Random Forest de base** : " + ", ".join(f"`{k}={v}`" for k, v in md.RF_PARAMS.items()))
            st.markdown("**Grille du GridSearchCV** (scoring = F1 macro, `StratifiedKFold` k = 5 → 240 ajustements)")
            st.dataframe(pd.DataFrame({"Hyperparamètre": list(md.GRID_PARAM_GRID),
                                       "Valeurs testées": [str(v) for v in md.GRID_PARAM_GRID.values()]}),
                         hide_index=True)
            st.markdown("**Meilleure combinaison** : " + ", ".join(
                f"`{k}={v}`" for k, v in md.GRIDSEARCH_BEST_PARAMS.items()))
        with c2:
            gs = res[res["Nom"] == "RANDOM_FOREST_GRIDSEARCH"].iloc[0]
            kpi_row([("F1 macro en CV (GridSearch)", f"{md.GRIDSEARCH_CV_BEST_SCORE:.3f}"),
                     ("F1 macro sur la validation", f"{gs['Macro_F1']:.3f}")])
            st.write("")
            st.markdown(
                "⚠️ **Pourquoi un tel écart (0,788 → 0,665) ?** Le découpage train/validation est fait **par client** "
                "(10 000 clients en train), mais la validation croisée du notebook utilise `StratifiedKFold`, qui "
                "répartit les 8 mois d'un même client entre plis d'apprentissage et de test. Le modèle « reconnaît » "
                "alors des clients déjà vus et le score CV est optimiste ; le GridSearch a de plus retenu des arbres "
                "de profondeur illimitée, qui mémorisent ces clients.")
        st.code(
            "from sklearn.model_selection import StratifiedGroupKFold\n"
            "groups = pd.read_csv('data/processed/groups_train.csv').iloc[:, 0]\n"
            "cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "GridSearchCV(model, param_grid, scoring='f1_macro', cv=cv).fit(X_train, y_train, groups=groups)",
            language="python")

    with t5:
        names = st.multiselect("Modèles à réentraîner", list(md.QUICK_MODELS),
                               default=["ZERO_R", "NAIVE", "DECISION_TREE", "RANDOM_FOREST"])
        st.caption("Le SVM (≈ 15 min d'entraînement) et le GridSearch (≈ 30 min) ne sont pas rejoués ici.")
        if st.button("▶️ Lancer le benchmark") and names:
            with st.spinner("Entraînement en cours…"):
                live = md.run_quick_benchmark(d["X_train"], d["y_train"], d["X_val"], d["y_val"], names)
            show = live[["Nom", "Accuracy", "Macro_F1", "F1_Good", "F1_Standard", "F1_Poor",
                         "Temps_Entrainement", "Temps_Prediction"]]
            st.dataframe(show.round(4), hide_index=True)


# =========================================================================== #
# 7. Prédicteur
# =========================================================================== #
def page_predictor():
    st.header("🔮 Prédicteur de score de crédit")
    st.caption("Saisissez le profil d'un client : le Random Forest estime s'il sera classé **Poor**, "
               "**Standard** ou **Good**.")
    model = get_rf()
    params, source = pp.load_scaler_params()
    if source.startswith("estimation"):
        st.info("Le scaler exact du notebook n'est pas encore exporté (`models/scaler_stats.json`) : la "
                "normalisation est estimée à partir des données prétraitées (voir README). Les prédictions "
                "restent cohérentes mais peuvent différer très légèrement.")

    with st.form("predict_form"):
        st.subheader("👤 Profil")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Âge", 14, 99, 33)
        occupation = c2.selectbox("Profession", [o for o in pp.OCCUPATIONS if o != "Inconnu"] + ["Inconnu"])
        month = c3.selectbox("Mois d'observation", pp.MONTHS_IN_DATA)

        st.subheader("💰 Revenus et épargne")
        c1, c2, c3, c4 = st.columns(4)
        annual_income = c1.number_input("Revenu annuel", 0.0, 1_000_000.0, 37578.61, step=500.0)
        monthly_salary = c2.number_input("Salaire net mensuel", 0.0, 20_000.0, 3093.75, step=50.0)
        invested = c3.number_input("Montant investi / mois", 0.0, 2_000.0, 128.95, step=10.0)
        balance = c4.number_input("Solde mensuel", 0.0, 2_000.0, 336.73, step=10.0)

        st.subheader("🏦 Comptes, cartes et prêts")
        c1, c2, c3, c4 = st.columns(4)
        n_accounts = c1.number_input("Comptes bancaires", 0, 11, 6)
        n_cards = c2.number_input("Cartes de crédit", 0, 11, 5)
        interest = c3.number_input("Taux d'intérêt (%)", 1, 34, 13)
        n_loans = c4.number_input("Nombre de prêts", 0, 9, 3)
        c1, c2 = st.columns([3, 1])
        loan_types = c1.multiselect("Types de prêts (laisser vide si inconnu)", pp.LOAN_TYPES,
                                    default=["Personal Loan", "Auto Loan", "Student Loan"])
        emi = c2.number_input("Mensualités (EMI) / mois", 0.0, 2_000.0, 69.25, step=5.0)

        st.subheader("📅 Historique de paiement")
        c1, c2, c3, c4 = st.columns(4)
        delay = c1.number_input("Retard moyen de paiement (jours)", -5, 67, 18)
        delayed = c2.number_input("Paiements en retard", 0, 28, 14)
        inquiries = c3.number_input("Demandes de crédit", 0, 17, 6)
        history = c4.number_input("Ancienneté de l'historique (mois)", 1, 404, 221)
        c1, c2, c3, c4 = st.columns(4)
        outstanding = c1.number_input("Dette restante", 0.0, 5_000.0, 1166.16, step=50.0)
        utilization = c2.number_input("Utilisation du crédit (%)", 20.0, 50.0, 32.31, step=0.5)
        limit_change = c3.number_input("Variation de la limite de crédit", -7.0, 37.0, 9.4, step=0.5)
        credit_mix = c4.selectbox("Credit_Mix", ["Standard", "Good", "Bad", "Inconnu"])
        c1, c2 = st.columns(2)
        pay_min = c1.selectbox("Paiement du minimum", ["Yes", "No", "NM"],
                               help="NM = information non disponible (Not Mentioned)")
        behaviour = c2.selectbox("Comportement de paiement", [b for b in pp.PAYMENT_BEHAVIOURS if b != "Inconnu"] + ["Inconnu"],
                                 index=5)
        submitted = st.form_submit_button("Prédire le score de crédit", type="primary")

    if not submitted:
        return

    raw = {
        "Age": age, "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": n_accounts, "Num_Credit_Card": n_cards, "Interest_Rate": interest,
        "Num_of_Loan": n_loans, "Delay_from_due_date": delay, "Num_of_Delayed_Payment": delayed,
        "Changed_Credit_Limit": limit_change, "Num_Credit_Inquiries": inquiries,
        "Outstanding_Debt": outstanding, "Credit_Utilization_Ratio": utilization,
        "Credit_History_Age": history, "Total_EMI_per_month": emi,
        "Amount_invested_monthly": invested, "Monthly_Balance": balance,
        "Month": month, "Occupation": occupation, "Type_of_Loan": loan_types, "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": pay_min, "Payment_Behaviour": behaviour,
    }
    row = pp.build_feature_row(raw, scaler_params=params)
    label, proba = md.predict_one(model, row)

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f'<div class="badge" style="background:{CLASS_COLORS[label]}">{label}</div>',
                    unsafe_allow_html=True)
        st.caption(f"Confiance du modèle : {proba[label]:.0%}")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 2.2))
        hbar(ax, proba.index, proba.values, color=[CLASS_COLORS[c] for c in proba.index], fmt="{:.1%}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilité estimée")
        fig.tight_layout()
        show_fig(fig)
    st.caption("Ce modèle est un exercice académique (≈ 69 % d'Accuracy en validation) : il ne doit pas servir à "
               "décider d'un véritable octroi de crédit.")
    with st.expander("Voir les 56 variables envoyées au modèle"):
        st.dataframe(row.T.rename(columns={0: "Valeur"}))


# =========================================================================== #
# 8. Déséquilibre des classes
# =========================================================================== #
def page_imbalance():
    d = load_data()
    st.header("⚖️ Déséquilibre des classes")
    counts = d["y_train"].value_counts().reindex(pp.CLASS_ORDER)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(pd.DataFrame({"Effectif (train)": counts,
                                   "Part": (counts / counts.sum() * 100).round(1).astype(str) + " %"}))
        st.markdown(f"Rapport Standard : Good = **{counts['Standard'] / counts['Good']:.1f} : 1**. "
                    "Un classifieur peut gagner en Accuracy en privilégiant *Standard* au détriment de *Good*.")
    with c2:
        st.markdown(
            "Pistes pour compenser le déséquilibre :\n"
            "- **`class_weight='balanced'`** : pénalise davantage les erreurs sur les classes rares ;\n"
            "- **`class_weight='balanced_subsample'`** : idem, recalculé sur chaque échantillon bootstrap ;\n"
            "- **SMOTE** (suréchantillonnage synthétique) — nécessite `pip install imbalanced-learn` ;\n"
            "- choisir le **Macro-F1** et le **rappel par classe** comme indicateurs de référence.")

    st.subheader("🧪 Expérience (Random Forest léger)")
    try:
        import imblearn  # noqa: F401
        has_smote = True
    except ImportError:
        has_smote = False
    use_smote = st.checkbox("Inclure SMOTE", value=False, disabled=not has_smote,
                            help=None if has_smote else "Installez imbalanced-learn pour activer cette option.")
    if st.button("▶️ Lancer la comparaison"):
        with st.spinner("Entraînement des variantes (≈ 10–20 s)…"):
            st.session_state["imbalance"] = md.imbalance_experiment(
                d["X_train"], d["y_train"], d["X_val"], d["y_val"], include_smote=use_smote)
    if "imbalance" in st.session_state:
        res = st.session_state["imbalance"]
        st.dataframe(res[["Variante", "Accuracy", "Macro_F1", "Recall_Good", "Recall_Standard", "Recall_Poor"]]
                     .round(4), hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        x = np.arange(len(res))
        width = 0.27
        for k, cls in enumerate(["Good", "Standard", "Poor"]):
            ax.bar(x + (k - 1) * width, res[f"Recall_{cls}"], width, label=f"Rappel {cls}", color=CLASS_COLORS[cls])
        ax.set_xticks(x)
        ax.set_xticklabels(res["Variante"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        show_fig(fig)
        st.caption("Compenser le déséquilibre augmente nettement le rappel de *Good* ; l'Accuracy baisse légèrement "
                   "et le Macro-F1 progresse un peu : c'est un compromis à choisir selon le coût des erreurs.")


# =========================================================================== #
# 9. Conclusions
# =========================================================================== #
def page_conclusions():
    d = load_data()
    res = cached_results()
    best = md.best_model_row(res)
    st.header("💡 Résultats clés & limites")

    holdout = res[res["Protocole"] == md.PROTOCOL_HOLDOUT]
    zero = holdout[holdout["Nom"] == "ZERO_R"].iloc[0]
    kpi_row([("Référence Zero-R", f"{zero['Accuracy']:.3f}"),
             ("Meilleur modèle", pretty(best["Nom"])),
             ("Accuracy", f"{best['Accuracy']:.3f}"),
             ("Macro-F1", f"{best['Macro_F1']:.3f}")])
    st.write("")

    top = md.rf_importance(get_rf(), d["X_train"].columns)
    top_groups = (top.assign(Variable=top["Variable"].map(pp.feature_group))
                  .groupby("Variable")["Importance_RF"].sum().sort_values(ascending=False).head(5))

    st.subheader("Principaux constats")
    st.markdown(
        f"1. **Les modèles à base d'arbres / de noyau dominent** : {pretty(best['Nom'])} (Macro-F1 "
        f"{best['Macro_F1']:.3f}), le Random Forest léger et le SVM sont à moins d'un point l'un de l'autre ; "
        "KNN, Arbre seul et Naive Bayes restent en retrait.\n"
        f"2. **Gain réel mais modeste** : on passe de {zero['Accuracy']:.0%} d'Accuracy (Zero-R) à environ 69 %, "
        "avec un Macro-F1 d'environ 0,66.\n"
        f"3. **Variables les plus déterminantes** (Random Forest, regroupées) : "
        f"{', '.join(f'`{v}`' for v in top_groups.index)}.\n"
        "4. **La classe *Good* est la plus difficile** (F1 ≈ 0,58) : minoritaire et souvent confondue avec *Standard*.\n"
        "5. **Le coût de calcul n'est pas anodin** : le GridSearch ne gagne que ≈ 0,5 point d'Accuracy sur le "
        "Random Forest léger pour un modèle ≈ 30× plus lourd (≈ 500 Mo contre 17 Mo) ; le prédicteur de cette "
        "application utilise donc la version légère.")

    st.subheader("Limites")
    st.markdown(
        "- **Validation croisée non groupée** : `StratifiedKFold` mélange les mois d'un même client entre plis ; "
        "les scores CV (0,751 et 0,788) sont optimistes. Utiliser `StratifiedGroupKFold` avec `groups_train.csv`.\n"
        "- **Information manquante** : 20 % de `Credit_Mix` et 15 % de `Monthly_Inhand_Salary` sont imputés.\n"
        "- **Observations répétées** : 8 lignes par client, non indépendantes ; les modèles n'exploitent pas "
        "l'historique du client dans le temps.\n"
        "- **Données académiques** : ce modèle n'est pas destiné à un usage réel d'octroi de crédit.")

    st.subheader("Perspectives")
    st.markdown(
        "- Corriger la validation croisée (groupée par client) puis relancer le GridSearch ;\n"
        "- tester le boosting (XGBoost, LightGBM) ;\n"
        "- construire des variables **par client** (moyennes, tendances sur les 8 mois) ;\n"
        "- traiter le déséquilibre (`class_weight`, SMOTE) et calibrer les probabilités ;\n"
        "- expliquer les prédictions individuelles (SHAP).")


# =========================================================================== #
# 10. À propos
# =========================================================================== #
def page_about():
    st.header("ℹ️ À propos du projet")
    st.markdown(
        f"**Groupe {GROUP}** — Mini-projet de Data Mining : classification du score de crédit "
        "(Good / Standard / Poor) à partir de 28 variables bancaires.")

    st.subheader("👥 Équipe")
    st.dataframe(pd.DataFrame({
        "Membre": ["rouiched noura ", "tamendjari tilissa ", "ghewalem amani","drissi malek"],
        "Rôle": ["EDA & tests statistiques", "Prétraitement & encodage", "Modélisation & application Streamlit"],
    }), hide_index=True)
    st.caption("À remplacer par les noms réels du groupe dans `src/app.py` (fonction `page_about`).")

    st.subheader("🧭 Pipeline")
    st.markdown(
        "1. `01_eda_and_cleaning` — audit qualité, EDA, tests statistiques ;\n"
        "2. `02_preprocessing_encoding` — découpage 80/20 par client, imputation, encodages, normalisation ;\n"
        "3. `03_model_training_evaluation` — 6 algorithmes, validation croisée, GridSearchCV ;\n"
        "4. `src/` + cette application — code réutilisable et tableau de bord.")
    st.code(
        "mini-projet-data-mining/\n"
        "├── data/processed/   X_*_preprocessed.csv, y_*.csv, groups_train.csv\n"
        "├── notebooks/        01_eda_and_cleaning · 02_preprocessing_encoding · 03_model_training_evaluation\n"
        "├── results/          resultats_modeles.csv\n"
        "├── models/           (optionnel) scaler_stats.json, rf_model.joblib\n"
        "├── src/              app.py · models.py · preprocessing.py\n"
        "└── requirements.txt",
        language="text")
    st.subheader("▶️ Exécution")
    st.code("pip install -r requirements.txt\nstreamlit run src/app.py", language="bash")
    st.markdown("**Technologies** : Python · pandas · scikit-learn · SciPy · Matplotlib · Streamlit.")


# =========================================================================== #
# Navigation
# =========================================================================== #
PAGES = {
    "1. Accueil & Résumé": page_home,
    "2. Explorateur de données": page_explorer,
    "3. Analyse exploratoire (EDA)": page_eda,
    "4. Analyse statistique": page_stats,
    "5. Importance des variables": page_importance,
    "6. Benchmark des modèles": page_benchmark,
    "7. Prédicteur de score": page_predictor,
    "8. Déséquilibre des classes": page_imbalance,
    "9. Résultats clés & limites": page_conclusions,
    "10. À propos": page_about,
}


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 💳 Credit Score DM")
    st.sidebar.markdown("**Dashboard Data Mining**")
    choice = st.sidebar.radio("Menu de navigation", list(PAGES))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Groupe {GROUP}**")
    st.sidebar.caption("Classification du score de crédit · Good / Standard / Poor")
    try:
        PAGES[choice]()
    except FileNotFoundError as err:
        st.error(f"{err}\n\nVérifiez que les dossiers `data/processed/` et `results/` sont présents à la racine du projet.")


main()
