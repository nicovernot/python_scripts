# 🎲 Générateur Intelligent de Grilles Keno v2.0

## 📋 Description

Ce générateur avancé utilise l'apprentissage automatique (XGBoost) et l'analyse statistique pour optimiser la génération de grilles Keno. Il analyse les patterns historiques et utilise des stratégies adaptatives pour maximiser les chances de succès.

## 🎯 Caractéristiques Principales

### 🤖 Machine Learning
- **XGBoost** : 10 modèles spécialisés (un par position)
- **Features cycliques** : Encodage temporel et numérique
- **Stratégie adaptative** : Pondération dynamique ML/Fréquence
- **Prédictions intelligentes** : Probabilités calculées pour chaque numéro

### 📊 Analyse Statistique
- **Fréquences** : Analyse des numéros les plus/moins sortis
- **Retards** : Calcul du retard de chaque numéro
- **Paires fréquentes** : Identification des combinaisons récurrentes
- **Zones d'équilibre** : Répartition par zones (1-17, 18-35, 36-52, 53-70)

### 🎲 Spécificités Keno
- **70 numéros** : Pool complet (1 à 70)
- **20 numéros tirés** : Par tirage standard
- **10 numéros sélectionnés** : Grilles optimisées
- **Scoring avancé** : Évaluation multi-critères

## 🚀 Installation et Utilisation

### Prérequis
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn scipy statsmodels pyarrow tqdm
```

### Utilisation Simple
```bash
# Script interactif
./lancer_keno.sh

# Ou directement
python keno_generator_advanced.py --quick
```

### Options Avancées
```bash
# Génération personnalisée
python keno_generator_advanced.py --grids 100 --output mes_grilles.csv

# Réentraînement des modèles
python keno_generator_advanced.py --retrain --grids 50

# Mode silencieux
python keno_generator_advanced.py --silent --grids 20
```

## 📁 Structure des Fichiers

```
keno/
├── keno_generator_advanced.py    # Script principal
├── convert_keno_data.py          # Conversion CSV → Parquet
├── test_keno_quick.py           # Test rapide
├── lancer_keno.sh               # Script de lancement
├── keno_data/                   # Données historiques
│   ├── keno_202010.csv         # Données sources
│   └── keno_202010.parquet     # Données optimisées
├── ../keno_models/              # Modèles ML
│   ├── metadata.json           # Métadonnées
│   └── xgb_keno_*.pkl          # Modèles XGBoost
└── ../keno_output/              # Résultats
    ├── grilles_keno_*.csv      # Grilles générées
    └── rapport_keno.md         # Rapports d'analyse
```

## 🎯 Exemples de Résultats

### Top 5 Grilles Recommandées
```
1. [ 1 -  2 -  3 -  5 -  7 - 13 - 41 - 48 - 50 - 61] | Score: 13.2442
2. [16 - 20 - 23 - 30 - 34 - 35 - 39 - 49 - 61 - 64] | Score: 13.0017
3. [ 1 -  4 -  5 - 13 - 29 - 36 - 48 - 50 - 60 - 64] | Score: 12.8371
4. [ 8 - 11 - 16 - 21 - 22 - 26 - 28 - 32 - 46 - 50] | Score: 12.7392
5. [ 8 - 16 - 21 - 22 - 23 - 25 - 27 - 32 - 34 - 66] | Score: 12.7314
```

### Performance
- **Génération** : ~10 secondes pour 10 grilles (modèles pré-entraînés)
- **Entraînement** : ~7 minutes pour l'ensemble des modèles
- **Données** : 3,520 tirages historiques analysés

## 🔧 Configuration

### Paramètres Keno
```python
KENO_PARAMS = {
    'total_numbers': 70,        # Numéros de 1 à 70
    'numbers_per_draw': 20,     # 20 numéros tirés
    'player_selection': 10,     # 10 numéros sélectionnés
    'min_selection': 2,         # Minimum sélectionnable
    'max_selection': 10,        # Maximum sélectionnable
}
```

### Stratégie Adaptative
```python
adaptive_weights = {
    'ml_weight': 0.6,      # 60% ML
    'freq_weight': 0.4,    # 40% Fréquence
}
```

## 📊 Scoring et Évaluation

Le système de scoring prend en compte :

1. **Fréquences normalisées** : Probabilité basée sur l'historique
2. **Équilibre des zones** : Répartition homogène 1-17, 18-35, 36-52, 53-70
3. **Paires fréquentes** : Bonus pour les combinaisons récurrentes
4. **Dispersion optimale** : Évitement des séquences consécutives excessives

## 🎲 Stratégies Intégrées

### Machine Learning (60%)
- Prédictions XGBoost par position
- Features cycliques temporelles
- Probabilités calculées pour chaque numéro

### Analyse Fréquentielle (40%)
- 50% numéros "chauds" (fréquents)
- 30% numéros "froids" (en retard)
- 20% sélection aléatoire équilibrée

## 🔄 Mise à Jour des Données

Pour actualiser avec de nouvelles données :

1. Remplacer `keno_data/keno_202010.csv`
2. Exécuter `python convert_keno_data.py`
3. Relancer avec `--retrain` pour mettre à jour les modèles

## ⚠️ Avertissements

- **Aléatoire** : Chaque tirage Keno reste totalement aléatoire
- **Probabilités** : Les prédictions sont basées sur l'analyse historique
- **Responsabilité** : Jouer de manière responsable et modérée
- **Garanties** : Aucune garantie de gain fournie

## 🛠️ Dépannage

### Erreurs Communes
```bash
# Modules manquants
pip install -r requirements.txt

# Données non trouvées
python convert_keno_data.py

# Modèles corrompus
python keno_generator_advanced.py --retrain
```

## 📈 Améliorations Futures

- [ ] Interface graphique (GUI)
- [ ] API REST pour intégration
- [ ] Analyse de patterns avancés
- [ ] Optimisation multi-objectifs
- [ ] Base de données temps réel

---

**Version** : 2.0  
**Auteur** : Assistant IA  
**Date** : Août 2025  

🎲 *"L'analyse intelligente au service du hasard maîtrisé"* 🎲
