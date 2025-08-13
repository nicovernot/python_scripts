# 🎯 Guide d'utilisation - Système d'Analyse Loto/Keno

Bienvenue dans le système d'analyse et de génération pour les jeux Loto et Keno !

## 🚀 Démarrage rapide

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Tests du système

Avant de commencer, vérifiez que tout fonctionne :

```bash
# Tests essentiels (rapide)
python test/run_all_tests.py --essential

# Tests complets (recommandé)
python test/run_all_tests.py --fast
```

## 🎲 Utilisation Loto

### Télécharger les données

```bash
python loto/result.py
```

### Générer des grilles optimisées

```bash
# 3 grilles avec visualisations et export des stats
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3 --plots --export-stats --config loto/strategies.yml

# Génération rapide (5 grilles)
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 5 --config loto/strategies.yml

# Avec stratégie spécifique
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3 --strategy agressive --config loto/strategies.yml
```

### Stratégies disponibles

- **equilibre** (défaut) : Approche équilibrée
- **agressive** : Favorise les tendances récentes
- **conservatrice** : Basée sur l'historique long terme
- **ml_focus** : Priorité au machine learning

## 🎰 Utilisation Keno

### Télécharger les données

```bash
python keno/results_clean.py
```

### Analyser et générer des recommandations

```bash
# Analyse complète avec visualisations
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv --plots --export-stats

# Analyse rapide
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv
```

## 📊 Structure des sorties

### Loto
```
loto_analyse_plots/           # Graphiques d'analyse
├── frequences_loto.png
├── heatmap_loto.png
├── retards_loto.png
└── ...

loto_stats_exports/           # Statistiques CSV
├── frequences_loto.csv
├── paires_loto.csv
└── ...

grilles.csv                   # Grilles générées
```

### Keno
```
keno_analyse_plots/           # Graphiques d'analyse
├── frequences_keno.png
├── heatmap_keno.png
└── ...

keno_stats_exports/           # Statistiques CSV
├── frequences_keno.csv
├── zones_keno.csv
└── ...

keno_output/
└── recommandations_keno.txt  # Recommandations
```

## 🧪 Tests et validation

### Tests par catégorie

```bash
# Tests essentiels seulement
python test/run_all_tests.py --essential

# Tests Loto seulement
python test/run_all_tests.py --loto

# Tests Keno seulement
python test/run_all_tests.py --keno

# Tests de performance
python test/run_all_tests.py --performance

# Tous les tests sauf performance (rapide)
python test/run_all_tests.py --fast

# Tous les tests (complet)
python test/run_all_tests.py
```

### Tests individuels

```bash
python test/test_essential.py    # Dépendances et structure
python test/test_loto.py         # Système Loto complet
python test/test_keno.py         # Système Keno complet
python test/test_performance.py # Performance et mémoire
```

## 🔧 Dépannage

### Problèmes courants

**❌ Erreur "Module not found"**
```bash
pip install -r requirements.txt
```

**❌ Fichier CSV manquant**
```bash
# Pour Loto
python loto/result.py

# Pour Keno
python keno/results_clean.py
```

**❌ Erreur de configuration**
```bash
# Vérifier que les fichiers de configuration existent
ls loto/strategies.yml
ls loto/strategies_ml.yml
```

**❌ Tests échoués**
```bash
# Lancer les tests de diagnostic
python test/run_all_tests.py --essential

# Si problème avec les tests, vérifier les logs détaillés
python test/test_loto.py --verbose
```

### Diagnostic système

```bash
# Vérifier l'environnement
python --version
pip list

# Tester les imports critiques
python test/test_essential.py

# Vérifier l'espace disque
du -sh loto_analyse_plots/ keno_analyse_plots/
```

## 📁 Organisation du projet

```
📦 loto_keno/
├── 📁 loto/                    # Système Loto
│   ├── duckdb_loto.py         # Générateur principal
│   ├── result.py              # Téléchargement données
│   ├── strategies.py          # Stratégies avancées
│   ├── strategies.yml         # Configuration
│   └── loto_data/             # Données téléchargées
├── 📁 keno/                    # Système Keno
│   ├── duckdb_keno.py         # Analyseur principal
│   ├── results_clean.py       # Téléchargement données
│   └── keno_data/             # Données téléchargées
├── 📁 test/                    # Tests du système
│   ├── run_all_tests.py       # Lanceur principal
│   ├── test_essential.py      # Tests essentiels
│   ├── test_loto.py           # Tests Loto
│   ├── test_keno.py           # Tests Keno
│   └── test_performance.py    # Tests performance
├── requirements.txt            # Dépendances Python
├── .gitignore                 # Fichiers ignorés
└── README.md                  # Ce guide
```

## 💡 Conseils d'utilisation

### Performance optimale

1. **Lancez les tests** avant la première utilisation
2. **Téléchargez les données** régulièrement (hebdomadaire)
3. **Utilisez `--fast`** pour les tests de développement
4. **Sauvegardez** vos grilles générées favorites

### Workflow recommandé

```bash
# 1. Tests hebdomadaires
python test/run_all_tests.py --fast

# 2. Mise à jour des données
python loto/result.py
python keno/results_clean.py

# 3. Génération Loto
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3 --plots --config loto/strategies.yml

# 4. Analyse Keno
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv --plots
```

## 🎯 Fonctionnalités avancées

### Machine Learning

Le système utilise des modèles ML (XGBoost, GradientBoosting) pour :
- Prédire les probabilités des numéros
- Optimiser les combinaisons
- Scorer les grilles générées

### Stratégies personnalisées

Modifiez `loto/strategies.yml` pour ajuster :
- Poids des différents critères
- Priorités de sélection
- Paramètres ML

### Exports et visualisations

- **CSV** : Toutes les statistiques exportables
- **PNG** : Graphiques haute qualité
- **Markdown** : Rapports formatés

---

✨ **Bon jeu et que la chance soit avec vous !** 🍀
