# 🎯 Guide Complet - Système d'Analyse Loto/Keno

Bienvenue dans le système d'analyse et de génération pour les jeux Loto et Keno ! Ce guide contient tout ce dont vous avez besoin pour utiliser efficacement le système.

## � Table des Matières

- [�🚀 Démarrage Rapide](#-démarrage-rapide)
- [🎲 Guide Loto Complet](#-guide-loto-complet)
- [🎰 Guide Keno Complet](#-guide-keno-complet)
- [🧪 Tests et Validation](#-tests-et-validation)
- [📊 Analyse des Résultats](#-analyse-des-résultats)
- [🔧 Dépannage et Support](#-dépannage-et-support)
- [⚡ Optimisation et Performance](#-optimisation-et-performance)
- [📁 Organisation du Projet](#-organisation-du-projet)

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)

### Installation Rapide

```bash
# Cloner le projet
git clone <repository>
cd loto_keno

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configuration automatique
python setup_config.py
```

### 🔧 Configuration Personnalisée

#### Assistant de Configuration Interactif

```bash
# Lancer l'assistant de configuration
python setup_config.py
```

L'assistant vous guide pour configurer :
- **📊 Paramètres de base** : Nombre de grilles, stratégies par défaut
- **🎨 Interface utilisateur** : Couleurs, nettoyage écran
- **🤖 Machine Learning** : Activation IA, seuils de performance  
- **⚡ Performance** : Threads CPU, limite mémoire
- **🧹 Maintenance** : Nettoyage automatique, rétention fichiers

#### Configuration Manuelle

Copiez et éditez le fichier de configuration :

```bash
# Créer votre configuration
cp .env.example .env

# Éditer avec votre éditeur préféré
nano .env  # ou vim, code, etc.
```

#### Variables Principales

```bash
# Grilles et stratégies par défaut
DEFAULT_LOTO_GRIDS=3
DEFAULT_LOTO_STRATEGY=equilibre

# Interface utilisateur
CLI_COLORS_ENABLED=true
CLI_CLEAR_SCREEN=true

# Performance
DUCKDB_MEMORY_LIMIT=2GB
OMP_NUM_THREADS=4

# Machine Learning
ML_ENABLED=true
ML_MIN_SCORE=80
```

## 🎯 Menu CLI Interactif

### 🚀 Lancement Rapide

```bash
# Méthode 1: Script de lancement automatique
./lancer_menu.sh

# Méthode 2: Lancement direct Python
python cli_menu.py
```

### 📋 Fonctionnalités du Menu

Le **Menu CLI Interactif** offre une interface conviviale avec :

- **📊 Statut en temps réel** : Affichage des données disponibles et dates de mise à jour
- **🎨 Interface colorée** : Navigation intuitive avec codes couleurs
- **⚡ Accès rapide** : Toutes les commandes principales via des raccourcis numériques
- **🛠️ Configuration personnalisée** : Options avancées pour Loto et Keno
- **📈 Visualisation des résultats** : Consultation directe des grilles et recommandations

#### Menu Principal

```
📥 TÉLÉCHARGEMENT DES DONNÉES
  1️⃣  Télécharger les données Loto (FDJ)
  2️⃣  Télécharger les données Keno (FDJ)  
  3️⃣  Mettre à jour toutes les données

🎲 ANALYSE LOTO
  4️⃣  Générer 3 grilles Loto (rapide)
  5️⃣  Générer 5 grilles Loto (complet)
  6️⃣  Générer grilles avec visualisations
  7️⃣  Analyse Loto personnalisée

🎰 ANALYSE KENO
  8️⃣  Analyse Keno (rapide)
  9️⃣  Analyse Keno avec visualisations
  🔟 Analyse Keno personnalisée

🧪 TESTS ET MAINTENANCE
  1️⃣1️⃣ Tests complets du système
  1️⃣2️⃣ Tests essentiels uniquement
  1️⃣3️⃣ Test de performance
  1️⃣4️⃣ Nettoyage et optimisation

📊 CONSULTATION DES RÉSULTATS
  1️⃣5️⃣ Voir les dernières grilles Loto
  1️⃣6️⃣ Voir les recommandations Keno
  1️⃣7️⃣ Ouvrir dossier des graphiques
```

### ⚙️ Configuration Personnalisée

#### 🎲 Loto Personnalisé (Option 7)
- Nombre de grilles (1-10)
- Choix de stratégie (equilibre, agressive, conservatrice, ml_focus)
- Options visualisations et export des statistiques

#### 🎰 Keno Personnalisé (Option 10)
- Analyse rapide ou approfondie
- Génération de visualisations
- Export des statistiques détaillées

## � API Flask RESTful

### 🚀 Lancement de l'API

L'API Flask expose toutes les fonctionnalités via des endpoints HTTP :

```bash
# Méthode 1: Script de lancement
./lancer_api.sh

# Méthode 2: Lancement direct
python api/app.py

# Méthode 3: Flask CLI
export FLASK_APP=api/app.py
flask run --host=0.0.0.0 --port=5000
```

**🌐 Accès:** `http://localhost:5000`

### 📚 Documentation Interactive

L'API inclut une page de documentation complète accessible à `http://localhost:5000/`

### 🛠️ Endpoints Principaux

#### 🎲 Génération Loto
```bash
# Générer 3 grilles avec stratégie équilibrée
curl -X POST http://localhost:5000/api/loto/generate \
  -H "Content-Type: application/json" \
  -d '{"count": 3, "strategy": "equilibre"}'
```

#### 🎰 Analyse Keno
```bash
# Analyse avec 5 stratégies
curl -X POST http://localhost:5000/api/keno/analyze \
  -H "Content-Type: application/json" \
  -d '{"strategies": 5, "deep_analysis": false}'
```

#### 📊 Gestion des Données
```bash
# Statut des données
curl http://localhost:5000/api/data/status

# Mise à jour des données
curl -X POST http://localhost:5000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{"sources": ["loto", "keno"]}'
```

#### 🩺 Santé de l'API
```bash
# Vérifier l'état de l'API
curl http://localhost:5000/api/health
```

### 🧪 Tests API

```bash
# Tester tous les endpoints
python test_api.py

# Tester depuis le menu CLI (option 16)
python cli_menu.py
```

### 📖 Documentation Complète

Voir le fichier `api/API_DOCUMENTATION.md` pour :
- **📋 Liste complète des endpoints**
- **🔧 Paramètres de requête**
- **📄 Formats de réponse**
- **❌ Gestion des erreurs**
- **🚀 Exemples d'utilisation**

## �🎲 Guide Loto Complet

### 📥 Téléchargement Automatique des Données

Le système télécharge automatiquement les données depuis le site FDJ :

```bash
# Téléchargement complet avec nettoyage
python loto/result.py

# Options avancées
python loto/result.py --force       # Force le re-téléchargement
python loto/result.py --no-cleanup  # Garde les anciens fichiers
```

**Données téléchargées :**
- **903 tirages** Loto (2019-2025)
- **Format CSV** avec 50 colonnes
- **Mise à jour automatique** des anciennes données
- **Nettoyage intelligent** des anciens téléchargements

### 🎯 Génération de Grilles Optimisées

#### Commande Basique

```bash
# 3 grilles avec stratégie par défaut
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3 --config loto/strategies.yml
```

#### Commandes Avancées

```bash
# Génération complète avec visualisations
python loto/duckdb_loto.py \
  --csv loto/loto_data/loto_201911.csv \
  --grids 5 \
  --plots \
  --export-stats \
  --config loto/strategies.yml

# Avec stratégie spécifique
python loto/duckdb_loto.py \
  --csv loto/loto_data/loto_201911.csv \
  --grids 3 \
  --strategy agressive \
  --config loto/strategies.yml

# Génération rapide (sans ML)
python loto/duckdb_loto.py \
  --csv loto/loto_data/loto_201911.csv \
  --grids 10 \
  --no-ml \
  --config loto/strategies.yml

# Mode débogage détaillé
python loto/duckdb_loto.py \
  --csv loto/loto_data/loto_201911.csv \
  --grids 3 \
  --verbose \
  --config loto/strategies.yml
```

### 🎛️ Stratégies Disponibles

Modifiez `loto/strategies.yml` pour personnaliser :

#### 1. **Stratégie Équilibrée** (défaut)
```yaml
equilibre:
  description: "Approche équilibrée entre fréquence historique, retard et momentum récent"
  weights:
    frequency: 0.3
    delay: 0.25
    trend: 0.2
    ml_score: 0.25
```

#### 2. **Stratégie Agressive**
```yaml
agressive:
  description: "Favorise les tendances récentes et le machine learning"
  weights:
    frequency: 0.15
    delay: 0.15
    trend: 0.3
    ml_score: 0.4
```

#### 3. **Stratégie Conservatrice**
```yaml
conservatrice:
  description: "Basée sur l'historique long terme"
  weights:
    frequency: 0.5
    delay: 0.3
    trend: 0.1
    ml_score: 0.1
```

#### 4. **Stratégie ML Focus**
```yaml
ml_focus:
  description: "Priorité maximum au machine learning"
  weights:
    frequency: 0.1
    delay: 0.1
    trend: 0.1
    ml_score: 0.7
```

### 📊 Interprétation des Résultats Loto

#### Exemple de Grille Générée
```
🎲 GRILLE #1 (Score ML: 87.7/100)
   Numéros: [2, 11, 28, 35, 47]
   Somme: 123 | Pairs/Impairs: 2/3 | Étendue: 45
   Zones: 2-1-2 | Consécutifs: 0 | Diversité unités: 5
   Qualité: ✓Somme | ✓Équilibre | ✓Zones | ✓Diversité
```

**Explication des métriques :**
- **Score ML** : Note de machine learning (0-100)
- **Somme** : Total des 5 numéros (optimal: 110-140)
- **Pairs/Impairs** : Répartition (optimal: 2/3 ou 3/2)
- **Étendue** : Différence entre max et min (optimal: 25-45)
- **Zones** : Répartition 1-17 | 18-34 | 35-49
- **Consécutifs** : Nombres qui se suivent (optimal: 0-1)
- **Diversité unités** : Chiffres des unités différents (optimal: 5)

### 🎨 Visualisations Générées

Avec l'option `--plots`, le système génère :

#### Graphiques d'Analyse
- **`frequences_loto.png`** : Fréquence d'apparition par numéro
- **`retards_loto.png`** : Retards actuels par numéro
- **`heatmap_loto.png`** : Matrice de corrélation des tirages
- **`sommes_loto.png`** : Distribution des sommes
- **`even_odd_loto.png`** : Répartition pairs/impairs

#### Données Exportées (CSV)
- **`frequences_loto.csv`** : Statistiques détaillées par numéro
- **`paires_loto.csv`** : Cooccurrences des paires
- **`retards_loto.csv`** : Historique des retards
- **`zones_loto.csv`** : Analyse par zones géographiques

### ⚙️ Options Avancées Loto

```bash
# Contrôle de la qualité
--min-quality 0.8              # Score minimum accepté
--max-consecutive 1            # Maximum de numéros consécutifs
--balance-mode strict          # Mode équilibrage strict

# Machine Learning
--ml-models xgboost            # Utiliser XGBoost uniquement
--ml-models gradientboosting   # Utiliser GradientBoosting uniquement
--ml-models all                # Utiliser tous les modèles (défaut)

# Performance
--threads 4                    # Nombre de threads pour le ML
--batch-size 1000             # Taille de batch pour l'optimisation

# Filtrage
--exclude-numbers 13,18        # Exclure des numéros spécifiques
--force-numbers 7,21          # Forcer l'inclusion de numéros
--date-range 2023-01-01:2024-12-31  # Limiter la période d'analyse
```

### 🔍 Analyse Personnalisée

#### Créer une Analyse Custom
```python
from loto.duckdb_loto import LotoStrategist

# Initialisation
strategist = LotoStrategist(config_file='loto/strategies.yml')

# Chargement des données
import duckdb
db_con = duckdb.connect(':memory:')
table_name = strategist.load_data_from_csv('loto/loto_data/loto_201911.csv', db_con)

# Analyse des fréquences
freq_analysis = strategist.analyze_frequencies(db_con)
print(f"Top 10 numéros : {freq_analysis.head(10)}")

# Génération de grilles personnalisées
grids = strategist.generate_grids(db_con, table_name, 5)
for i, grid in enumerate(grids, 1):
    print(f"Grille {i}: {grid['grille']} (Score: {grid['score']:.1f})")

db_con.close()
```

## 🔧 Dépannage et Support

### ❓ Problèmes Fréquents et Solutions

#### Erreurs d'Installation
```bash
# Erreur "Module not found"
pip install -r requirements.txt
pip install --upgrade pip

# Problème avec xgboost
pip uninstall xgboost
pip install xgboost

# Problème avec duckdb
pip uninstall duckdb
pip install duckdb==0.9.2
```

#### Erreurs de Données
```bash
# CSV Loto manquant ou corrompu
python loto/result.py --force

# CSV Keno manquant ou corrompu  
python keno/results_clean.py --verify

# Problème de format de données
python test/test_essential.py  # Diagnostic complet
```

#### Erreurs de Configuration
```bash
# Fichier strategies.yml manquant
cp loto/strategies.yml.example loto/strategies.yml

# Permissions insuffisantes
chmod +x loto/*.py keno/*.py test/*.py

# Variables d'environnement
export PYTHONPATH="${PYTHONPATH}:/path/to/loto_keno"
```

#### Erreurs de Performance
```bash
# Mémoire insuffisante
python loto/duckdb_loto.py --csv file.csv --grids 3 --no-ml

# Processus trop lent
python loto/duckdb_loto.py --csv file.csv --grids 3 --threads 1

# Espace disque insuffisant
python tools/cleanup.py --remove-cache --remove-old-exports
```

### 🔍 Diagnostic Avancé

#### Tests de Diagnostic
```bash
# Test complet du système
python test/run_all_tests.py --verbose

# Test spécifique avec logs détaillés
python test/test_loto.py 2>&1 | tee loto_diagnostic.log
python test/test_keno.py 2>&1 | tee keno_diagnostic.log

# Test de performance avec profiling
python test/test_performance.py --profile --export-report
```

#### Vérification de l'Environnement
```bash
# Informations système
python --version
pip list | grep -E "(pandas|numpy|duckdb|sklearn|xgboost)"

# Mémoire et espace disque
free -h
df -h .

# Processus Python en cours
ps aux | grep python
```

#### Debug Mode
```bash
# Mode debug pour Loto
python loto/duckdb_loto.py 
  --csv loto/loto_data/loto_201911.csv 
  --grids 1 
  --debug 
  --log-level DEBUG

# Mode debug pour Keno
python keno/duckdb_keno.py 
  --csv keno/keno_data/keno_202010.csv 
  --debug 
  --verbose 
  --log-file keno_debug.log
```

### 🆘 Support et Aide

#### Logs et Diagnostics
```bash
# Créer un rapport de diagnostic complet
python tools/generate_diagnostic_report.py --include-logs --include-data-samples

# Analyser les performances
python tools/performance_analyzer.py --analyze-bottlenecks --suggest-optimizations
```

#### Communauté et Documentation
- **Issues GitHub** : Pour signaler des bugs
- **Documentation** : `docs/` pour l'historique des améliorations
- **Tests** : `test/` pour les exemples d'utilisation

## ⚡ Optimisation et Performance

### 🚀 Accélération des Calculs

#### Configuration Optimale
```bash
# Utilisation optimale des ressources
export OMP_NUM_THREADS=4          # Nombre de threads ML
export NUMBA_NUM_THREADS=4        # Accélération numérique
export MKL_NUM_THREADS=4          # Intel Math Kernel Library
```

#### Options de Performance Loto
```bash
# Mode rapide (sans ML intensif)
python loto/duckdb_loto.py 
  --csv loto/loto_data/loto_201911.csv 
  --grids 5 
  --fast-mode 
  --config loto/strategies.yml

# Mode équilibré (ML modéré)  
python loto/duckdb_loto.py 
  --csv loto/loto_data/loto_201911.csv 
  --grids 3 
  --ml-complexity medium 
  --config loto/strategies.yml

# Mode performance maximum
python loto/duckdb_loto.py 
  --csv loto/loto_data/loto_201911.csv 
  --grids 3 
  --ml-complexity high 
  --parallel-processing 
  --config loto/strategies.yml
```

#### Options de Performance Keno
```bash
# Analyse rapide
python keno/duckdb_keno.py 
  --csv keno/keno_data/keno_202010.csv 
  --strategies 3 
  --quick-analysis

# Analyse approfondie
python keno/duckdb_keno.py 
  --csv keno/keno_data/keno_202010.csv 
  --strategies 7 
  --deep-analysis 
  --ml-enhanced
```

### 💾 Gestion de la Mémoire

#### Nettoyage Automatique
```bash
# Nettoyage des fichiers temporaires
python tools/cleanup.py --auto --keep-recent 3

# Compression des exports anciens
python tools/compress_exports.py --older-than 30days

# Optimisation base de données
python tools/optimize_database.py --vacuum --reindex
```

#### Configuration Mémoire
```python
# Configuration pour systèmes avec peu de RAM
import duckdb
db_con = duckdb.connect(':memory:', config={
    'memory_limit': '2GB',
    'max_memory': '2GB',
    'threads': 2
})
```

### 📊 Monitoring et Métriques

#### Surveillance en Temps Réel
```bash
# Monitoring des performances
python tools/performance_monitor.py 
  --watch 
  --interval 5s 
  --export-metrics performance_metrics.json

# Alertes automatiques
python tools/alert_system.py 
  --memory-threshold 80% 
  --cpu-threshold 90% 
  --disk-threshold 95%
```

#### Benchmarking
```bash
# Benchmark des stratégies Loto
python tools/benchmark_loto.py 
  --strategies all 
  --iterations 100 
  --export-report benchmark_loto.html

# Benchmark des stratégies Keno
python tools/benchmark_keno.py 
  --strategies all 
  --timeframes 1M,3M,6M 
  --export-report benchmark_keno.html
```

## 📁 Organisation du Projet

### 🗂️ Structure Détaillée

```
📦 loto_keno/
├── 📄 README.md                    # Ce guide complet
├── 📄 requirements.txt             # Dépendances Python
├── 📄 .gitignore                  # Fichiers ignorés par Git
├── 📄 .env                        # Variables d'environnement
│
├── 📁 loto/                        # 🎲 Système Loto
│   ├── 📄 duckdb_loto.py          # Générateur principal
│   ├── 📄 result.py               # Téléchargement données FDJ
│   ├── 📄 strategies.py           # Stratégies avancées
│   ├── 📄 strategies.yml          # Configuration stratégies
│   ├── 📄 strategies_ml.yml       # Configuration ML
│   └── 📁 loto_data/              # Données téléchargées
│       └── 📄 loto_201911.csv     # Fichier de données principal
│
├── 📁 keno/                        # 🎰 Système Keno
│   ├── 📄 duckdb_keno.py          # Analyseur principal
│   ├── 📄 results_clean.py        # Téléchargement avec nettoyage
│   ├── 📄 analyse_keno_final.py   # Analyses complémentaires
│   └── 📁 keno_data/              # Données téléchargées
│       └── 📄 keno_202010.csv     # Fichier de données principal
│
├── 📁 test/                        # 🧪 Tests du Système
│   ├── 📄 run_all_tests.py        # Lanceur principal
│   ├── 📄 test_essential.py       # Tests des dépendances
│   ├── 📄 test_loto.py            # Tests système Loto
│   ├── 📄 test_keno.py            # Tests système Keno
│   ├── 📄 test_performance.py     # Tests de performance
│   └── 📄 README.md               # Documentation des tests
│
├── 📁 docs/                        # 📚 Documentation Historique
│   ├── 📄 RAPPORT_AMELIORATIONS_VISUELLES.md
│   ├── 📄 RESUME_MODIFICATIONS.md
│   └── 📄 SORTIES_UNIQUES.md
│
├── 📁 loto_analyse_plots/          # 📊 Graphiques Loto
│   ├── 🖼️ frequences_loto.png
│   ├── 🖼️ heatmap_loto.png
│   ├── 🖼️ retards_loto.png
│   └── 🖼️ sommes_loto.png
│
├── 📁 keno_analyse_plots/          # 📊 Graphiques Keno
│   ├── 🖼️ frequences_keno.png
│   ├── 🖼️ heatmap_keno.png
│   ├── 🖼️ paires_keno.png
│   └── 🖼️ retards_keno.png
│
├── 📁 loto_stats_exports/          # 📋 Statistiques Loto (CSV)
│   ├── 📊 frequences_loto.csv
│   ├── 📊 paires_loto.csv
│   └── 📊 retards_loto.csv
│
├── 📁 keno_stats_exports/          # 📋 Statistiques Keno (CSV)
│   ├── 📊 frequences_keno.csv
│   ├── 📊 paires_keno.csv
│   ├── 📊 retards_keno.csv
│   └── 📊 zones_keno.csv
│
├── 📁 keno_output/                 # 📄 Recommandations Keno
│   └── 📝 recommandations_keno.txt
│
├── 📁 boost_models/                # 🤖 Modèles ML (Ignoré par Git)
│   ├── 🧠 model_boule_1.joblib
│   ├── 🧠 xgb_ball_1.pkl
│   └── 📄 metadata.json
│
├── 📁 cache/                       # 💾 Cache Système (Ignoré par Git)
│   └── 🗃️ viz_cache.pkl
│
└── 📁 venv/                        # 🐍 Environnement Virtuel (Ignoré par Git)
    ├── 📁 bin/
    ├── 📁 lib/
    └── 📄 pyvenv.cfg
```

### 🔄 Workflow de Développement

#### Développement Local
```bash
# 1. Clone et setup
git clone <repository>
cd loto_keno
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. Tests avant développement
python test/run_all_tests.py --essential

# 3. Développement
# ... modifications ...

# 4. Tests après développement
python test/run_all_tests.py --fast

# 5. Commit
git add .
git commit -m "Description des changements"
```

#### Workflow Utilisateur Standard
```bash
# 1. Mise à jour hebdomadaire des données
python loto/result.py
python keno/results_clean.py

# 2. Génération Loto
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3 --config loto/strategies.yml

# 3. Analyse Keno
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv

# 4. Consultation des résultats
# Grilles Loto: grilles.csv
# Recommandations Keno: keno_output/recommandations_keno.txt
```

### 💡 Conseils d'Utilisation Avancée

#### Personnalisation des Stratégies
1. **Modifiez `loto/strategies.yml`** pour ajuster les poids
2. **Créez vos propres stratégies** en dupliquant les existantes
3. **Testez différentes configurations** avec `--strategy custom`

#### Automatisation
```bash
# Script automatique quotidien
#!/bin/bash
cd /path/to/loto_keno
source venv/bin/activate

# Mise à jour des données (une fois par semaine)
if [ $(date +%u) -eq 1 ]; then
    python loto/result.py
    python keno/results_clean.py
fi

# Génération quotidienne
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 2 --config loto/strategies.yml
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv

# Sauvegarde des résultats
cp grilles.csv "backups/grilles_$(date +%Y%m%d).csv"
```

#### Intégration avec d'autres Outils
```python
# Export vers Excel
import pandas as pd
from openpyxl import Workbook

# Charger les grilles générées
df = pd.read_csv('grilles.csv')
df.to_excel('grilles_loto.xlsx', index=False)

# Envoi par email (exemple)
import smtplib
from email.mime.text import MIMEText

def send_grids_email(grids, email):
    msg = MIMEText(f"Grilles du jour: {grids}")
    msg['Subject'] = 'Grilles Loto Optimisées'
    msg['To'] = email
    # ... configuration SMTP ...
```

---

## ✨ Conclusion

Ce système d'analyse Loto/Keno combine :
- **🔬 Analyse statistique** rigoureuse
- **🤖 Machine Learning** avancé  
- **📊 Visualisations** interactives
- **⚡ Performance** optimisée
- **🧪 Tests** complets

**Bon jeu et que la chance soit avec vous !** 🍀

---

*Dernière mise à jour : 13 août 2025*

### Stratégies disponibles

- **equilibre** (défaut) : Approche équilibrée
- **agressive** : Favorise les tendances récentes
- **conservatrice** : Basée sur l'historique long terme
- **ml_focus** : Priorité au machine learning

## 🎰 Guide Keno Complet

### 📥 Téléchargement Automatique des Données

Le système télécharge automatiquement les données Keno depuis FDJ :

```bash
# Téléchargement complet avec nettoyage automatique
python keno/results_clean.py

# Options avancées
python keno/results_clean.py --keep-old     # Garde les anciens fichiers
python keno/results_clean.py --verify       # Vérification approfondie
```

**Données téléchargées :**
- **3,516 tirages** Keno (2020-2025)
- **Format CSV** avec 28 colonnes (20 numéros + métadonnées)
- **Nettoyage intelligent** des anciens téléchargements
- **Validation automatique** du format

### 🎯 Analyse et Recommandations Keno

#### Commande Basique

```bash
# Analyse complète avec recommandations
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv
```

#### Commandes Avancées

```bash
# Analyse complète avec visualisations et exports
python keno/duckdb_keno.py \
  --csv keno/keno_data/keno_202010.csv \
  --plots \
  --export-stats

# Analyse focalisée sur une période
python keno/duckdb_keno.py \
  --csv keno/keno_data/keno_202010.csv \
  --date-range 2024-01-01:2025-08-13

# Génération de recommandations multiples
python keno/duckdb_keno.py \
  --csv keno/keno_data/keno_202010.csv \
  --strategies 5 \
  --plots

# Mode débogage avec détails ML
python keno/duckdb_keno.py \
  --csv keno/keno_data/keno_202010.csv \
  --verbose \
  --debug-ml
```

### 🎲 Stratégies de Sélection Keno

Le système propose 7 stratégies intelligentes :

#### 1. **Z-Score Strategy** 🎯
- Analyse statistique des écarts
- Identifie les numéros "sous-représentés"
- Bon pour les joueurs analytiques

#### 2. **Fibonacci Strategy** 🌟
- Basée sur la suite de Fibonacci
- Sélection mathématique optimisée
- Equilibre entre fréquence et position

#### 3. **Sectors Strategy** 🎪
- Division en secteurs géographiques
- Répartition équilibrée sur la grille
- Évite les clusters

#### 4. **Trend Strategy** 📈
- Analyse des tendances récentes
- Favorise les numéros "chauds"
- Adapté aux trends court terme

#### 5. **Monte Carlo Strategy** 🎰
- Simulation probabiliste avancée
- Combine fréquence et retards
- Approche scientifique

#### 6. **Pairs Optimal Strategy** 👥
- Optimise les paires de numéros
- Analyse des cooccurrences
- Maximise les combinaisons gagnantes

#### 7. **Zones Balanced Strategy** ⚖️
- Équilibre parfait entre zones
- Répartition géographique optimale
- Stratégie la plus stable

### 📊 Interprétation des Recommandations Keno

#### Exemple de Recommandation
```
🎯 RECOMMANDATIONS KENO INTELLIGENTES

🌟 STRATÉGIE Z-SCORE (Analyse des écarts)
   Sélection: [3, 8, 12, 19, 27, 34, 41, 56, 63, 70]
   Score de confiance: 87.3%
   Zones couvertes: 7/7
   Somme optimale: 333 (dans la fourchette 300-400)

📈 ANALYSE STATISTIQUE
   • Numéros en retard: 3, 12, 27 (>15 tirages)
   • Numéros chauds: 8, 19, 34 (<5 tirages)
   • Équilibre pairs/impairs: 5/5 ✓
   • Répartition par zones: 2-1-2-2-2-1 ✓
```

**Métriques importantes :**
- **Score de confiance** : Probabilité calculée (50-95%)
- **Zones couvertes** : Couverture de la grille (1-49 = Zone 1, etc.)
- **Somme optimale** : Total des numéros (optimal: 300-400)
- **Retards/Chauds** : Analyse temporelle des sorties

### 🎨 Visualisations Keno Générées

Avec l'option `--plots`, le système génère :

#### Graphiques d'Analyse
- **`frequences_keno.png`** : Fréquence par numéro avec codes couleur
- **`retards_keno.png`** : Retards actuels et moyens
- **`heatmap_keno.png`** : Matrice de corrélation avancée
- **`paires_keno.png`** : Top paires les plus fréquentes
- **`zones_keno.png`** : Répartition géographique
- **`sommes_keno.png`** : Distribution des sommes (améliorée)

#### Données Exportées (CSV)
- **`frequences_keno.csv`** : Statistiques complètes par numéro
- **`paires_keno.csv`** : Toutes les cooccurrences
- **`retards_keno.csv`** : Historique des retards
- **`zones_keno.csv`** : Analyse par zones géographiques

### ⚙️ Options Avancées Keno

```bash
# Contrôle des stratégies
--strategies 3                 # Nombre de stratégies à générer
--exclude-strategy fibonacci   # Exclure une stratégie spécifique
--strategy-weights custom      # Utiliser des poids personnalisés

# Filtrage temporel
--recent-draws 50              # Analyser les 50 derniers tirages
--exclude-period "2023-12-01:2023-12-31"  # Exclure une période

# Optimisation
--min-confidence 0.75          # Confiance minimum (défaut: 0.6)
--max-repeats 3               # Maximum de numéros répétés récents
--balance-zones strict         # Équilibrage strict des zones

# Machine Learning
--ml-weight 0.3               # Poids du ML dans le scoring final
--retrain-models              # Re-entraîner les modèles ML
```

### 🔬 Analyse Avancée Keno

#### Script d'Analyse Personnalisée
```python
from keno.duckdb_keno import KenoAnalyzer
import duckdb

# Initialisation
analyzer = KenoAnalyzer(max_number=70, numbers_per_draw=20)
db_con = duckdb.connect(':memory:')

# Chargement des données
table_name = analyzer.load_data_from_csv('keno/keno_data/keno_202010.csv', db_con)

# Analyse des fréquences
frequencies = analyzer.analyze_frequencies(db_con)
print("Top 10 numéros les plus fréquents:")
print(frequencies.head(10))

# Analyse des retards
delays = analyzer.analyze_delays(db_con, recent_draws=30)
print("\\nNuméros en retard:")
print(delays[delays['retard'] > 15])

# Analyse des paires
pairs = analyzer.analyze_pairs(db_con)
print("\\nTop 5 paires:")
print(pairs.head(5))

# Analyse des secteurs
sectors = analyzer.analyze_sectors(db_con)
print("\\nRépartition par secteurs:")
print(sectors)

# Analyse des zones géographiques
zones = analyzer.analyze_zones(db_con)
print("\\nAnalyse des zones:")
print(zones)

db_con.close()
```

### 📈 Suivi de Performance Keno

#### Valider les Recommandations
```bash
# Analyse de performance des stratégies
python keno/performance_tracker.py \
  --csv keno/keno_data/keno_202010.csv \
  --backtest-days 30

# Comparaison des stratégies
python keno/strategy_comparison.py \
  --csv keno/keno_data/keno_202010.csv \
  --strategies all
```

#### Métriques de Validation
- **Taux de réussite** : % de numéros prédits corrects
- **Score de précision** : Qualité des prédictions
- **Consistency** : Régularité des résultats
- **ROI potentiel** : Retour sur investissement théorique

### 🎪 Techniques Avancées Keno

#### 1. **Analyse des Cycles**
```python
# Détection de cycles dans les tirages
cycles = analyzer.detect_cycles(db_con, period_length=14)
print(f"Cycles détectés: {cycles}")
```

#### 2. **Corrélations Temporelles**
```python
# Analyse des corrélations jour/heure
temporal = analyzer.analyze_temporal_patterns(db_con)
print(f"Patterns temporels: {temporal}")
```

#### 3. **Optimisation Multi-Objectifs**
```python
# Optimisation avec contraintes multiples
optimal = analyzer.multi_objective_optimization(
    db_con,
    objectives=['frequency', 'delay', 'zones'],
    constraints={'min_sum': 300, 'max_sum': 400}
)
```

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

## 📊 Analyse des Résultats

### 📈 Comprendre les Graphiques

#### Graphiques Loto

**1. Fréquences (`frequences_loto.png`)**
- **Axe X** : Numéros (1-49)
- **Axe Y** : Nombre d'apparitions
- **Ligne rouge** : Moyenne théorique
- **Barres vertes** : Au-dessus de la moyenne
- **Barres orange** : Sous la moyenne

**2. Retards (`retards_loto.png`)**
- **Retard actuel** : Nombre de tirages depuis la dernière sortie
- **Retard moyen** : Moyenne historique
- **Numéros en rouge** : Retards exceptionnels (>30)

**3. Heatmap (`heatmap_loto.png`)**
- **Couleurs chaudes** : Corrélations fortes
- **Couleurs froides** : Pas de corrélation
- **Diagonale** : Auto-corrélation (toujours à 1)

#### Graphiques Keno

**1. Fréquences Améliorées (`frequences_keno.png`)**
- **Code couleur** : Rouge (faible), Jaune (moyenne), Vert (élevée)
- **Ligne moyenne** : Fréquence théorique (≈50 apparitions)
- **Top/Bottom 10** : Marqueurs spéciaux

**2. Retards Avancés (`retards_keno.png`)**
- **Boîtes à moustaches** : Distribution des retards
- **Points rouges** : Valeurs aberrantes
- **Ligne médiane** : Retard typique

### 📋 Interpréter les Statistiques CSV

#### Fichier `frequences_loto.csv`
```csv
numero,apparitions,frequence,retard_actuel,retard_moyen,ecart_type
1,18,0.0199,5,12.3,8.2
2,22,0.0243,1,11.8,7.9
...
```

**Colonnes importantes :**
- **frequence** : Proportion d'apparitions (optimal ≈ 0.02 pour Loto)
- **retard_actuel** : Tirages depuis dernière sortie
- **ecart_type** : Régularité des sorties (plus bas = plus régulier)

#### Fichier `paires_keno.csv`
```csv
numero1,numero2,cooccurrences,pourcentage,derniere_apparition
3,17,45,1.28%,12
7,23,42,1.19%,3
...
```

**Analyse des paires :**
- **cooccurrences** : Nombre de sorties ensemble
- **pourcentage** : % par rapport au total
- **derniere_apparition** : Fraîcheur de la paire

### 🎯 Métriques de Qualité

#### Scoring ML (Machine Learning)
```
Score ML: 87.7/100
├── Fréquence historique (30%): 92/100
├── Analyse des retards (25%): 85/100  
├── Tendances récentes (20%): 83/100
└── Prédiction ML (25%): 91/100
```

#### Indicateurs de Qualité Loto
- **✓ Somme** : Entre 110-140 (optimal)
- **✓ Équilibre** : 2-3 ou 3-2 pairs/impairs
- **✓ Zones** : Répartition 2-1-2 ou 1-2-2
- **✓ Diversité** : 5 chiffres d'unités différents

#### Indicateurs de Qualité Keno
- **Score de confiance** : 75-95% (excellent)
- **Couverture zones** : 6-7 zones sur 7 (optimal)
- **Équilibre P/I** : 5/5 ou 4/6 (bon)
- **Somme** : 300-400 (fourchette optimale)

### 📊 Tableaux de Bord Personnalisés

#### Dashboard Loto
```python
from loto.dashboard import LotoDashboard

# Créer un tableau de bord
dashboard = LotoDashboard('loto/loto_data/loto_201911.csv')

# Métriques en temps réel
metrics = dashboard.get_realtime_metrics()
print(f"Numéros chauds: {metrics['hot_numbers']}")
print(f"Numéros en retard: {metrics['cold_numbers']}")
print(f"Tendance actuelle: {metrics['trend']}")

# Graphique personnalisé
dashboard.plot_custom_analysis(
    period='6M',
    focus=['frequency', 'delays'],
    export_path='custom_loto_analysis.png'
)
```

#### Dashboard Keno
```python
from keno.dashboard import KenoDashboard

# Analyse comparative des stratégies
dashboard = KenoDashboard('keno/keno_data/keno_202010.csv')

# Performance des stratégies
perf = dashboard.strategy_performance(days=30)
print(f"Meilleure stratégie: {perf['best_strategy']}")
print(f"Taux de réussite: {perf['success_rate']:.1%}")

# Prédictions pour le prochain tirage
next_draw = dashboard.predict_next_draw(
    strategies=['zscore', 'montecarlo', 'fibonacci'],
    confidence_threshold=0.8
)
print(f"Prédiction: {next_draw}")
```

### 🔍 Analyse de Tendances

#### Détection de Patterns
```python
# Analyse des cycles Loto
patterns = analyzer.detect_patterns(
    data_source='loto/loto_data/loto_201911.csv',
    pattern_types=['cycles', 'sequences', 'correlations']
)

print("Patterns détectés:")
for pattern in patterns:
    print(f"- {pattern['type']}: {pattern['description']}")
    print(f"  Confiance: {pattern['confidence']:.1%}")
    print(f"  Dernière occurrence: {pattern['last_seen']}")
```

#### Corrélations Avancées
```python
# Matrice de corrélation personnalisée
correlation_matrix = analyzer.advanced_correlation(
    timeframe='3M',
    correlation_types=['temporal', 'positional', 'numerical']
)

# Exporter en heatmap
analyzer.export_correlation_heatmap(
    correlation_matrix,
    save_path='advanced_correlations.png'
)
```

### 📈 Optimisation Continue

#### Backtesting Automatique
```bash
# Test de performance sur 6 mois
python tools/backtest.py \
  --game loto \
  --period 6M \
  --strategies all \
  --export-report

# Résultats dans backtest_report.html
```

#### Machine Learning Adaptatif
```python
# Re-entraînement automatique des modèles
from tools.ml_optimizer import MLOptimizer

optimizer = MLOptimizer()

# Optimisation Loto
loto_models = optimizer.optimize_loto_models(
    data_path='loto/loto_data/loto_201911.csv',
    target_accuracy=0.85
)

# Optimisation Keno  
keno_models = optimizer.optimize_keno_models(
    data_path='keno/keno_data/keno_202010.csv',
    strategies=['zscore', 'montecarlo'],
    target_confidence=0.80
)

print(f"Modèles Loto optimisés: {loto_models}")
print(f"Modèles Keno optimisés: {keno_models}")
```

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

## � Générateur Keno Avancé avec Machine Learning

### 🎯 Vue d'ensemble

Le générateur Keno avancé utilise **RandomForest** avec approche **multi-label** pour apprendre les corrélations entre numéros et optimiser la génération de grilles.

### 🚀 Profils d'Entraînement

Le système propose 4 profils d'entraînement adaptés à différents besoins :

#### ⚡ Quick (--quick)
```bash
python keno/keno_generator_advanced.py --quick
```
- **Usage** : Tests rapides et développement
- **Grilles** : 10 par défaut
- **Temps** : ~10-15 secondes  
- **ML** : 50 arbres, profondeur 8

#### ⚖️ Balanced (--balanced) [DÉFAUT]
```bash
python keno/keno_generator_advanced.py --balanced
# ou simplement
python keno/keno_generator_advanced.py
```
- **Usage** : Équilibre optimal performance/temps
- **Grilles** : 100 par défaut
- **Temps** : ~1-2 minutes
- **ML** : 100 arbres, profondeur 12

#### 🎯 Comprehensive (--comprehensive)
```bash
python keno/keno_generator_advanced.py --comprehensive
```
- **Usage** : Entraînement approfondi
- **Grilles** : 500 par défaut
- **Temps** : ~5-10 minutes
- **ML** : 200 arbres, profondeur 15

#### 🔥 Intensive (--intensive)
```bash
python keno/keno_generator_advanced.py --intensive
```
- **Usage** : Performance maximale  
- **Grilles** : 1000 par défaut
- **Temps** : ~15-30 minutes
- **ML** : 300 arbres, profondeur 20

### 🔧 Options Avancées

```bash
# Combinaisons avec profils
python keno/keno_generator_advanced.py --comprehensive --grids 200
python keno/keno_generator_advanced.py --quick --silent --output test.csv
python keno/keno_generator_advanced.py --retrain --intensive

# Aide complète
python keno/keno_generator_advanced.py --help
```

### 📊 Caractéristiques Techniques

- **Modèle** : RandomForest MultiOutputClassifier  
- **Features** : 108 variables (historique + zones géographiques)
- **Target** : 70 numéros (corrélations apprises)
- **Données** : 3,520+ tirages historiques
- **Accuracy** : ~71% (modèle multi-label)

### 🎲 Exemple de Résultats

```
🏆 Top 5 des grilles recommandées:
   1. [ 1 -  8 - 11 - 14 - 16 - 27 - 35 - 39 - 62 - 69] | Score: 13.36
   2. [ 3 -  7 - 13 - 20 - 21 - 49 - 50 - 55 - 59 - 69] | Score: 13.29
   3. [ 1 -  5 -  8 - 32 - 42 - 44 - 46 - 55 - 61 - 69] | Score: 13.27
   4. [ 1 -  8 - 13 - 16 - 19 - 20 - 27 - 29 - 60 - 69] | Score: 13.27
   5. [ 8 - 11 - 13 - 21 - 24 - 34 - 39 - 49 - 62 - 69] | Score: 13.27

📊 Statistiques:
   - Grilles générées: 500
   - Score moyen: 12.82
   - Temps d'exécution: 561.51 secondes
```

## �🧪 Tests et validation

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
