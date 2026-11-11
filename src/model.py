"""Entraînement, évaluation et choix du seuil de décision."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42

# Matrice de coût fournie avec le jeu de données (UCI) :
# accorder un crédit à un mauvais payeur coûte 5, refuser un bon payeur coûte 1.
COST_FN = 5  # défaut non détecté
COST_FP = 1  # bon client refusé


def make_logreg(X: pd.DataFrame):
    """Modèle de référence : régression logistique (le standard historique en banque)."""
    cat = X.select_dtypes("category").columns.tolist()
    num = X.columns.difference(cat).tolist()
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=2000, C=0.5))


def make_xgb(y: pd.Series, **params):
    """XGBoost avec gestion native des variables catégorielles."""
    default = dict(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, enable_categorical=True, tree_method="hist",
        eval_metric="auc", random_state=SEED,
        # rééquilibre les classes (70 % bons / 30 % mauvais)
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    )
    default.update(params)
    return XGBClassifier(**default)


def gini(y_true, proba) -> float:
    """Indice de Gini = 2*AUC - 1, la métrique préférée des équipes risque."""
    return 2 * roc_auc_score(y_true, proba) - 1


def oof_proba(model, X, y, n_splits=5):
    """Probabilités « hors échantillon » par validation croisée."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def business_cost(y_true, proba, threshold) -> float:
    """Coût moyen par dossier pour un seuil donné."""
    tn, fp, fn, tp = confusion_matrix(y_true, proba >= threshold).ravel()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)


def best_threshold(y_true, proba):
    """Seuil qui minimise le coût métier (au lieu du 0,5 par défaut)."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = [business_cost(y_true, proba, t) for t in grid]
    i = int(np.argmin(costs))
    return grid[i], costs[i], grid, np.array(costs)
