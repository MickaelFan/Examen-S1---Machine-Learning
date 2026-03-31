# Projet Morpion IA — TEAM FIFA
## Examen Final de Machine Learning - Semestre 1
### [Institut Supérieur de Polytechnique de Madagascar (ISPM)](https://ispm-edu.com/)

---

## Informations du Groupe
* **Nom du groupe :** TEAM FIFA
* **Promotion :** ESIIA 4
* **Membres :**
    * RAZAFIMBELO Toky Faniry (N° 34)
    * RAVELONARIVO Fanantenana Mickael (N° 26)
    * RAMEFIARISON Fabio Fandresena (N° 35)
    * FANAMBIHARINDRAINY Schenyolla Anderssen (N° 37)
    * IALISOA Iris Fifaliana (N° 33)
    * RAKOTOARIMANANA Tojo Ny Aina (N° 38)

---

## Description du Projet
Ce projet porte sur le développement d'une Intelligence Artificielle appliquée au jeu du Morpion (Tic-Tac-Toe). L'objectif est d'étudier la transition entre un algorithme de recherche classique (Minimax avec élagage Alpha-Beta) et des modèles d'apprentissage automatique (Machine Learning). Le résultat final est une IA hybride intégrant des modèles prédictifs (XGBoost, Random Forest) pour l'évaluation stratégique des positions au sein d'une interface graphique interactive.

---

## Structure du Répertoire
* `generator/` : Scripts utilisant l'algorithme Minimax-Alphabeta pour la génération de données synthétiques.
* `ressources/dataset.csv` : Base de données finale comprenant 10 000 parties (caractéristiques de jeu et cibles).
* `notebook.ipynb` : Travaux d'analyse exploratoire des données (EDA), définition de la Baseline et entraînement des modèles avancés.
* `game.py` : Application principale avec interface graphique (GUI) proposant trois modes de jeu (Humain, ML Pur, Hybride).
* `README.md` : Documentation technique et rapport de synthèse.

---

## Résultats du Machine Learning
Les modèles ont été évalués et comparés sur les cibles de victoire (`x_wins`) et de match nul (`is_draw`) après un entraînement sur 10 000 échantillons.

| Modèle | Cible | F1-Score | Accuracy | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | x_wins | 0.9124 | 0.8860 | 0.8992 |
| Random Forest | x_wins | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | x_wins | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | is_draw | 0.0000 | 0.8755 | 0.4686 |
| Random Forest | is_draw | 0.0000 | 0.8745 | 0.4981 |
| XGBoost | is_draw | 0.0076 | 0.8695 | 0.4969 |

---

## Réponses aux Questions (Section 5 du sujet)

### Q1 : Analyse des coefficients et importance des caractéristiques
L'analyse des poids du modèle et de l'importance des variables révèle que les caractéristiques liées à la case centrale (index 4) et aux coins présentent les scores les plus élevés.
* **Justification :** La case centrale est statistiquement la plus avantageuse car elle intercepte le plus grand nombre de lignes, colonnes et diagonales (4 combinaisons possibles). Les coins participent à 3 combinaisons. Le modèle identifie que le contrôle de ces zones est le facteur déterminant pour maximiser la probabilité de victoire.

### Q2 : Justification du choix de la métrique (F1-Score vs Accuracy)
Le jeu de données présente un déséquilibre de classes, les matchs nuls étant statistiquement moins fréquents que les victoires dans un jeu semi-aléatoire.
* L'**Accuracy** est trompeuse ici : un modèle pourrait obtenir un score de 87% en prédisant systématiquement "pas de match nul", tout en étant incapable de remplir sa fonction.
* Le **F1-Score** est privilégié car il force le modèle à être performant à la fois sur la Précision et le Rappel, garantissant que l'IA identifie réellement les situations critiques.

### Q3 : Analyse de la difficulté de prédiction (Linéarité)
La cible `is_draw` est plus complexe à modéliser que la cible `x_wins`.
* **Explication :** Une victoire repose sur un motif linéaire simple (trois symboles alignés). À l'inverse, un match nul est une configuration globale de "blocage" où aucune ligne n'est complétée. Cette condition nécessite la capture d'interactions non-linéaires complexes entre l'ensemble des cases, ce que les modèles linéaires comme la Régression Logistique ne parviennent pas à traiter parfaitement.

### Q4 : Avantages de l'approche Hybride
L'architecture hybride (Minimax à profondeur 3 + évaluation par XGBoost) représente la solution optimale pour ce projet :
1. **Tactique à court terme :** L'algorithme Minimax assure qu'aucune erreur tactique immédiate n'est commise (victoire ratée ou défaite directe).
2. **Stratégie à long terme :** Le modèle de Machine Learning agit comme une fonction heuristique de "perception", permettant à l'IA de sentir si une position est favorable même sans voir la fin de la partie.
Cette synergie produit une IA robuste et stratégiquement supérieure à un algorithme classique limité en profondeur.

---

## Vidéo de Présentation
La démonstration de l'interface utilisateur ainsi que l'explication technique de la démarche sont disponibles via le lien suivant :
**(https://www.youtube.com/watch?v=okxXcW1MLZU)**

---
