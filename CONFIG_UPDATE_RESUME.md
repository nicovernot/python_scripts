# 🔧 Configuration .env - Résumé de la Mise à Jour

## ✨ Vue d'Ensemble

La configuration du système Loto/Keno a été entièrement modernisée pour offrir une gestion centralisée et flexible de tous les paramètres.

## 🚀 Nouveaux Fichiers Ajoutés

### 📄 Fichiers Principaux

1. **`.env`** - Configuration principale (mise à jour complète)
2. **`.env.example`** - Modèle de configuration avec commentaires détaillés
3. **`config_env.py`** - Gestionnaire de configuration Python
4. **`setup_config.py`** - Assistant de configuration interactif

### 🎯 Nouvelles Fonctionnalités

#### 1. **Gestionnaire de Configuration Centralisé**
```python
from config_env import load_config, get_config, get_config_path, get_config_bool

# Chargement automatique
config = load_config()

# Accès simple aux variables
loto_csv = get_config_path('LOTO_CSV_PATH')
ml_enabled = get_config_bool('ML_ENABLED')
default_grids = get_config('DEFAULT_LOTO_GRIDS')
```

#### 2. **Assistant de Configuration Interactif**
```bash
python setup_config.py
```
- Configuration guidée étape par étape
- Validation automatique des paramètres
- Application immédiate des changements
- Test de fonctionnement intégré

#### 3. **Menu CLI Configuré**
Le menu CLI utilise maintenant automatiquement la configuration :
- Nombre de grilles par défaut depuis `.env`
- Stratégie par défaut personnalisable
- Interface adaptée aux préférences utilisateur
- Performance optimisée selon la machine

## 📊 Sections de Configuration

### 📁 **Chemins de Données**
```bash
LOTO_CSV_PATH=loto/loto_data/loto_201911.csv
KENO_CSV_PATH=keno/keno_data/keno_202010.csv
LOTO_PLOTS_PATH=loto_analyse_plots
KENO_PLOTS_PATH=keno_analyse_plots
```

### ⚙️ **Paramètres Génération**
```bash
DEFAULT_LOTO_GRIDS=3
DEFAULT_LOTO_STRATEGY=equilibre
DEFAULT_KENO_STRATEGIES=7
LOTO_CONFIG_FILE=loto/strategies.yml
```

### 🤖 **Machine Learning**
```bash
ML_ENABLED=true
ML_MIN_SCORE=80
ML_TIMEOUT_SECONDS=300
XGB_N_ESTIMATORS=100
```

### 🎨 **Interface Utilisateur**
```bash
CLI_COLORS_ENABLED=true
CLI_CLEAR_SCREEN=true
CLI_SHOW_STATUS=true
CLI_SHOW_PROGRESS=true
```

### ⚡ **Performance**
```bash
MAX_CPU_CORES=
OMP_NUM_THREADS=4
DUCKDB_MEMORY_LIMIT=2GB
DUCKDB_THREADS=4
```

### 🎨 **Visualisations**
```bash
PLOT_STYLE=seaborn-v0_8
FIGURE_DPI=300
COLOR_PALETTE=husl
HEATMAP_COLORMAP=YlOrRd
```

### 🧹 **Maintenance**
```bash
AUTO_CLEANUP=true
CLEANUP_DAYS_OLD=30
MAX_EXPORTS_KEEP=10
```

### 🔧 **Logging et Debug**
```bash
LOG_LEVEL=INFO
LOG_FILE_ENABLED=false
TEST_TIMEOUT=300
```

## 🔄 Migration depuis l'Ancienne Configuration

### Avant
```bash
# Configuration dispersée dans le code
CSV_PATH=~/Téléchargements/loto_201911.csv
DEFAULT_NUM_GRIDS=20
PLOT_STYLE=seaborn
```

### Maintenant
```bash
# Configuration centralisée et organisée
LOTO_CSV_PATH=loto/loto_data/loto_201911.csv
DEFAULT_LOTO_GRIDS=3
PLOT_STYLE=seaborn-v0_8
CLI_COLORS_ENABLED=true
ML_ENABLED=true
AUTO_CLEANUP=true
```

## 🚀 Avantages de la Nouvelle Configuration

### ✅ **Facilité d'Utilisation**
- **Assistant interactif** pour la configuration initiale
- **Valeurs par défaut** intelligentes
- **Validation automatique** des paramètres

### ✅ **Flexibilité**
- **74 variables configurables** pour personnalisation totale
- **Types automatiques** (bool, int, float, path)
- **Priorité environnement** > fichier .env > défaut

### ✅ **Robustesse**
- **Gestion d'erreurs** complète
- **Fallback intelligent** si configuration manquante
- **Validation des chemins** et création automatique

### ✅ **Performance**
- **Optimisation CPU/mémoire** configurable
- **Timeout intelligents** pour éviter les blocages
- **Cache et nettoyage** automatiques

## 📋 Instructions d'Utilisation

### 🚀 **Configuration Rapide**
```bash
# 1. Copier l'exemple
cp .env.example .env

# 2. Configuration interactive
python setup_config.py

# 3. Validation
python config_env.py
```

### 🔧 **Configuration Manuelle**
```bash
# Éditer directement
nano .env

# Tester la configuration
python config_env.py
```

### 🎯 **Intégration dans le Code**
```python
# Dans vos scripts Python
from config_env import get_config, get_config_path, get_config_bool

# Utilisation
csv_path = get_config_path('LOTO_CSV_PATH')
grids_count = get_config('DEFAULT_LOTO_GRIDS', 3)
ml_enabled = get_config_bool('ML_ENABLED', True)
```

## 🧪 **Test de la Configuration**

### Validation Complète
```bash
python config_env.py
```

### Tests du Système
```bash
python test/run_all_tests.py --essential
```

### Test du Menu CLI
```bash
python cli_menu.py
```

## 💡 **Conseils d'Optimisation**

### 🔋 **Systèmes avec Ressources Limitées**
```bash
DEFAULT_LOTO_GRIDS=2
DUCKDB_MEMORY_LIMIT=1GB
OMP_NUM_THREADS=2
ML_ENABLED=false
```

### 🚀 **Systèmes Performants**
```bash
DEFAULT_LOTO_GRIDS=5
DUCKDB_MEMORY_LIMIT=4GB
OMP_NUM_THREADS=8
ML_ENABLED=true
PERFORMANCE_BENCHMARK=true
```

### 🎨 **Mode Terminal Basique**
```bash
CLI_COLORS_ENABLED=false
CLI_CLEAR_SCREEN=false
PLOT_STYLE=classic
```

---

## 🎉 **Résultat Final**

La configuration du système Loto/Keno est maintenant :
- **✅ Centralisée** dans .env
- **✅ Documentée** avec .env.example
- **✅ Interactive** avec setup_config.py
- **✅ Intégrée** dans tous les scripts
- **✅ Flexible** et personnalisable
- **✅ Robuste** avec validation automatique

**Le système est prêt pour une utilisation professionnelle !** 🚀

---

*Mise à jour effectuée le 13 août 2025*
