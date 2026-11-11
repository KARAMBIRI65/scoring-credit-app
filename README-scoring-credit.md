# Scoring crédit

Prédire si un client va rembourser son crédit et expliquer la décision.

Les données viennent du jeu German Credit, 1 000 dossiers de crédit allemands.

J'ai comparé une régression logistique et XGBoost, et utilisé SHAP pour voir ce qui compte dans chaque décision.

Sur 200 clients de test, le Gini est de 0,63 pour XGBoost et 0,61 pour la régression logistique.

Ce qui compte le plus, c'est le solde du compte courant, la durée du crédit et le montant.

J'ai ensuite repris ce travail dans une application web (scoring-credit-app).

Pour le lancer, installer les dépendances avec `pip install -r requirements.txt` puis ouvrir `notebooks/scoring_credit.ipynb`.

Issa Aziz Karambiri
