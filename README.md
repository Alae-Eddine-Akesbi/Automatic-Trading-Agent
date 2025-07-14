# Agent de Trading Automatique Basé sur le Q-Learning

Ce projet présente la conception, l'entraînement et l'évaluation d'un agent de trading automatique utilisant l'algorithme d'apprentissage par renforcement **Q-Learning**. L'agent est entraîné pour prendre des décisions d'achat, de vente ou de conservation sur des actions (par défaut, Apple Inc. - AAPL) en se basant sur des indicateurs techniques dérivés des données historiques du marché.

Le projet est divisé en trois composantes principales :
1.  **Prétraitement des Données** : Un pipeline complet pour collecter, nettoyer et enrichir les données boursières.
2.  **Entraînement de l'Agent** : L'implémentation et l'entraînement de l'agent Q-Learning.
3.  **Interface Utilisateur Interactive** : Une application web développée avec Streamlit pour visualiser et interagir avec l'agent.

## 🚀 Fonctionnalités

- **Collecte de Données Dynamique** : Récupère les données boursières historiques directement depuis Yahoo Finance.
- **Feature Engineering** : Calcule plusieurs indicateurs techniques pour enrichir l'état de l'environnement :
    - Moyennes Mobiles (MA)
    - Relative Strength Index (RSI)
    - Bandes de Bollinger
    - MACD (Moving Average Convergence Divergence)
    - Volume Normalisé
- **Agent Q-Learning Tabulaire** : Un agent simple et efficace qui apprend une politique de trading optimale en explorant un environnement simulé.
- **Backtesting** : Évalue les performances de l'agent sur des données non vues (out-of-sample) pour mesurer sa rentabilité et sa robustesse.
- **Interface Interactive** : Une application Streamlit permet de :
    - Choisir n'importe quel ticker d'action disponible sur Yahoo Finance.
    - Sélectionner une plage de dates pour l'entraînement et le test.
    - Lancer l'entraînement en direct et suivre la progression.
    - Visualiser les performances via des graphiques interactifs (profits, signaux d'achat/vente).
    - Animer les décisions de l'agent jour par jour sur le graphique des prix.

## 🛠️ Structure du Projet

- `Pretraitement_Donnees_AAPL.ipynb` : Notebook Jupyter détaillant les étapes de scraping, nettoyage et analyse exploratoire des données.
- `Q-Learning_Trading_AAPL.ipynb` : Notebook Jupyter pour l'entraînement et l'évaluation de l'agent Q-Learning.
- `data/` : Contient les fichiers CSV générés (données brutes, nettoyées et normalisées).
- `ui/` : Répertoire contenant l'application Streamlit.
    - `app.py` : Le script principal de l'application Streamlit.
    - `agent.py` : La classe `QLearningAgent`.
    - `environment.py` : La classe `TradingEnvironment` qui simule le marché.
    - `data_utils.py` : Fonctions pour le calcul des indicateurs techniques.
- `img/` : Contient des images et des captures d'écran du projet.

## 🔧 Comment l'utiliser

1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/Alae-Eddine-Akesbi/Automatic-Trading-Agent.git
    cd Automatic-Trading-Agent
    ```

2.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : Un fichier `requirements.txt` serait à créer pour lister les dépendances comme pandas, numpy, streamlit, yfinance, plotly, etc.)*

3.  **Lancez l'application Streamlit :**
    ```bash
    streamlit run ui/app.py
    ```

4.  Ouvrez votre navigateur à l'adresse locale fournie (généralement `http://localhost:8501`).

5.  Utilisez la barre latérale pour configurer les paramètres (ticker, dates) et lancez l'entraînement !

## Auteurs

Ce projet a été réalisé par :

- **EL HACHYMI Ahmed Yassine**
- **AKESBI Alae-Eddine**
- **BAGUENA Mohammed Amine**
- **AJI Othman**

## Encadré par :  

**Mr. Mohamed Khalifa BOUTAHIR**

Projet développé dans le cadre d’une démonstration d’agent de trading intelligent basé sur le Q-Learning.
