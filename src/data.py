"""Chargement et mise en forme du jeu de données German Credit (UCI).

Source : Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning
Repository. https://doi.org/10.24432/C5NC77 — licence CC BY 4.0.
1 000 demandes de crédit, 20 variables, cible : bon (1) / mauvais (2) payeur.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOCAL_FILE = DATA_DIR / "german.data.csv"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

# Noms lisibles des 20 variables, dans l'ordre du fichier d'origine.
COLUMNS = [
    "statut_compte", "duree_mois", "historique_credit", "objet_credit",
    "montant", "epargne", "anciennete_emploi", "taux_effort",
    "situation_perso", "autres_debiteurs", "anciennete_residence", "patrimoine",
    "age", "autres_credits", "logement", "nb_credits_banque", "emploi",
    "nb_personnes_charge", "telephone", "travailleur_etranger", "cible",
]

# Traduction des codes UCI (A11, A12...) en libellés compréhensibles.
CODES = {
    "statut_compte": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "pas de compte"},
    "historique_credit": {"A30": "aucun crédit / tout remboursé", "A31": "tout remboursé (cette banque)",
                          "A32": "crédits en cours à jour", "A33": "retards passés", "A34": "compte critique"},
    "objet_credit": {"A40": "voiture neuve", "A41": "voiture occasion", "A42": "mobilier",
                     "A43": "radio/TV", "A44": "électroménager", "A45": "réparations",
                     "A46": "études", "A47": "vacances", "A48": "formation", "A49": "professionnel",
                     "A410": "autre"},
    "epargne": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                "A64": ">= 1000 DM", "A65": "inconnue / aucune"},
    "anciennete_emploi": {"A71": "sans emploi", "A72": "< 1 an", "A73": "1-4 ans",
                          "A74": "4-7 ans", "A75": ">= 7 ans"},
    "situation_perso": {"A91": "homme divorcé/séparé", "A92": "femme div./sép./mariée",
                        "A93": "homme célibataire", "A94": "homme marié/veuf", "A95": "femme célibataire"},
    "autres_debiteurs": {"A101": "aucun", "A102": "co-emprunteur", "A103": "garant"},
    "patrimoine": {"A121": "immobilier", "A122": "épargne / assurance vie", "A123": "voiture / autre",
                   "A124": "inconnu / aucun"},
    "autres_credits": {"A141": "banque", "A142": "magasins", "A143": "aucun"},
    "logement": {"A151": "locataire", "A152": "propriétaire", "A153": "logé gratuitement"},
    "emploi": {"A171": "sans emploi / non qualifié non résident", "A172": "non qualifié résident",
               "A173": "employé qualifié", "A174": "cadre / indépendant"},
    "telephone": {"A191": "non", "A192": "oui"},
    "travailleur_etranger": {"A201": "oui", "A202": "non"},
}

# Variables sensibles : on les retire du modèle pour éviter une discrimination
# (sexe / situation familiale, nationalité). Bonne pratique en scoring réel.
SENSITIVE = ["situation_perso", "travailleur_etranger"]


def load_raw() -> pd.DataFrame:
    """Lit le fichier local, sinon le télécharge depuis l'UCI (sans compte)."""
    if LOCAL_FILE.exists():
        return pd.read_csv(LOCAL_FILE, header=None, names=COLUMNS)
    df = pd.read_csv(UCI_URL, sep=" ", header=None, names=COLUMNS)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(LOCAL_FILE, header=False, index=False)
    return df


def load_dataset(drop_sensitive: bool = True):
    """Renvoie (X, y) avec des libellés lisibles.

    y = 1 si le client a fait défaut (« mauvais payeur »), 0 sinon.
    """
    df = load_raw()
    for col, mapping in CODES.items():
        df[col] = df[col].map(mapping).astype("category")
    y = (df.pop("cible") == 2).astype(int).rename("defaut")
    if drop_sensitive:
        df = df.drop(columns=SENSITIVE)
    return df, y


CATEGORICAL = [c for c in CODES if c not in SENSITIVE]
