# Projet de Trading Algorithmique avec Réseaux de Neurones

## Description
Ce projet est un système de trading algorithmique qui utilise des réseaux de neurones profonds (LSTM) pour prédire les prix de clôture des paires de devises Forex, notamment EUR/USD. Il intègre des indicateurs techniques avancés, une sélection de caractéristiques (features) et une communication en temps réel entre MetaTrader 5 (MQL5) et Python via des sockets.

## Fonctionnalités principales
1. **Préparation des données** :
   - Récupération des données de marché (ticks, candles) via MQL5.
   - Calcul d'indicateurs techniques (ADX, RSI, MACD, Ichimoku, etc.).
   - Normalisation et standardisation des données.

2. **Modélisation** :
   - Architecture LSTM avec TensorFlow/Keras.
   - Entraînement itératif sur plusieurs jeux de données.
   - Sauvegarde et chargement des modèles.

3. **Sélection de caractéristiques** :
   - Utilisation de RFECV (Recursive Feature Elimination with Cross-Validation) pour identifier les indicateurs les plus pertinents.

4. **Communication** :
   - Échange de données entre MQL5 et Python via des sockets pour des prédictions en temps réel.

5. **Backtest et visualisation** :
   - Calcul de métriques (RMSE, précision).
   - Visualisation des prédictions vs données réelles.

## Structure des fichiers
- **dnn-MonoOutput-tensorflow.py** : Script Python pour l'entraînement du modèle LSTM.
- **indicators.mqh** : Implémentation des indicateurs techniques en MQL5.
- **market_data.mqh** : Récupération et traitement des données de marché.
- **MetaData.mqh** : Calcul de corrélations entre actifs.
- **RFECV.py** : Sélection de caractéristiques avec Random Forest.
- **socket_server.py** : Serveur socket pour la communication Python-MQL5.
- **structure.mqh** : Définition des structures de données personnalisées.

## Prérequis
- **MetaTrader 5** (avec environnement MQL5).
- **Python 3.x** avec les bibliothèques :
  - TensorFlow/Keras
  - scikit-learn
  - pandas, numpy, matplotlib
  - socket

## Installation
1. Placer les fichiers `.mqh` dans le dossier `Include` de MetaTrader 5.
2. Configurer le script Python (`socket_server.py`) pour écouter sur le port 9090.
3. Exécuter le script MQL5 (`main.mqh`) pour lancer la collecte des données et la communication.

## Utilisation
1. **Entraînement du modèle** :
   - Exécuter `dnn-MonoOutput-tensorflow.py` pour entraîner le modèle sur les données historiques.
   - Les modèles sont sauvegardés dans le dossier spécifié.

2. **Prédiction en temps réel** :
   - Lancer `socket_server.py` pour écouter les requêtes de MQL5.
   - Le script MQL5 envoie les données actuelles et reçoit les prédictions.

3. **Visualisation** :
   - Les résultats sont affichés via matplotlib et peuvent être exportés en CSV.

## Exemple de sortie
- Précision des prédictions (test/train).
- Graphiques comparant les prix réels et prédits.
- Classement des indicateurs par importance (RFECV).

## Auteurs
- **Hedge Ltd.** (équipe de développement trading algorithmique).

## Licence
Propriétaire. Tous droits réservés.

---

Pour toute question ou support, contactez l'équipe technique à l'adresse : support@hedge.com
