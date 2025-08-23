# 🎯 SYSTÈME KENO COMPLET - GUIDE FINAL

## 📋 Vue d'ensemble

Le système Keno complet permet d'extraire, analyser et générer des grilles de jeu intelligentes basées sur l'analyse statistique des données historiques de la FDJ. Le système dispose de **deux interfaces principales** :

1. **CLI Simple** (`keno_cli.py`) - Interface en ligne de commande
2. **Menu Interactif** (`cli_menu.py`) - Interface menu colorée avec plus d'options

## 🚀 Démarrage Rapide

### Interface CLI Simple
```bash
python keno_cli.py all --grids 5    # Pipeline complet
python keno_cli.py status           # Statut du système
python keno_cli.py --help           # Aide complète
```

### Interface Menu Interactif (Recommandée)
```bash
python cli_menu.py                  # Menu principal coloré
```

## 🎮 Interface CLI Simple (keno_cli.py)

### Commandes disponibles

#### 📊 Statut du système
```bash
python keno_cli.py status
```

#### 🔄 Extraction de données
```bash
python keno_cli.py extract
```

#### 📈 Analyse statistique
```bash
python keno_cli.py analyze
```

#### 🎲 Génération de grilles
```bash
python keno_cli.py generate --grids 5
```

#### 🧹 Nettoyage du système
```bash
python keno_cli.py clean           # Nettoyage standard
python keno_cli.py clean --deep    # Nettoyage approfondi
```

#### � Pipeline complet
```bash
python keno_cli.py all --grids 3   # Extraction → Analyse → Génération
```

## 🎨 Interface Menu Interactif (cli_menu.py)

Le menu interactif propose une interface colorée avec de nombreuses options :

### � Fonctionnalités Keno disponibles :
- **Option 8** : Analyse Keno complète (nouveaux algorithmes)
- **Option 9** : Pipeline Keno complet avec visualisations  
- **Option 10** : Analyse Keno personnalisée (5 sous-options)
- **Option 14** : Nettoyage et optimisation (3 modes)
- **Option 23** : Statut détaillé du système

### 🔧 Options de nettoyage (Menu Option 14)
- **Standard** : Fichiers temporaires (.pyc, logs, etc.)
- **Approfondi** : Inclut les anciens backups (>30 jours)
- **Statut** : Affichage du statut seulement

### 🎯 Analyse personnalisée Keno (Menu Option 10)
1. **Extraction seule** - Télécharge les dernières données
2. **Analyse statistique seule** - Lance l'analyse complète
3. **Génération de grilles** - Génère 1-10 grilles
4. **Pipeline complet personnalisé** - Tout en une fois
5. **Analyse DuckDB avancée** - 11 stratégies avancées

## 📁 Structure des données

### Format unifié des CSV
```
date,numero_tirage,b1,b2,b3,...,b20
2025-08-17,25457,7,8,9,11,12,15,21,25,28,33,41,42,43,46,50,51,55,56,59,66
```

### Répertoires
- `keno/keno_data/` : Données brutes (60 fichiers, 7,038 tirages)
- `keno_stats_exports/` : Statistiques exportées
- `keno_output/` : Grilles et rapports générés
- `keno_analyse_plots/` : Visualisations

## 🧠 Scripts individuels

### 1. Extracteur automatique
```bash
python keno/extracteur_keno_zip.py
```

### 2. Analyse complète avec DuckDB
```bash
python keno/duckdb_keno.py --auto-consolidated --plots --export-stats
```

### 3. Générateur intelligent v2
```bash
python keno/generateur_keno_intelligent_v2.py --grids 5
```

### 4. Analyse DuckDB avancée (11 stratégies)
```bash
python keno/duckdb_keno.py --csv keno/keno_data/keno_202508.csv --plots --export-stats
```

### 5. Scripts legacy (compatibles)
```bash
python keno/analyse_keno_final.py
python keno/duckdb_keno.py --csv [fichier]
```

## 📊 Stratégies de jeu disponibles

### Via générateur intelligent (3 stratégies)
1. **RETARD** : Numéros en retard d'apparition
2. **ÉQUILIBRÉ** : Mix fréquence/retard optimal  
3. **FRÉQUENTS** : Numéros les plus sortis

### Via DuckDB avancé (11 stratégies)
1. **MIX_INTELLIGENT** : Pondération probabiliste multi-stratégies (Score: 0.95)
2. **MONTECARLO** : Simulation Monte Carlo 10k itérations (Score: 0.90)
3. **HOT** : Numéros les plus fréquents (Score: 0.85)
4. **ZSCORE** : Écarts statistiques significatifs (Score: 0.80)
5. **PAIRS_OPTIMAL** : Optimisation basée sur les paires (Score: 0.80)
6. **COLD** : Numéros en retard (Score: 0.75)
7. **TREND** : Tendance récente (Score: 0.75)
8. **ZONES_BALANCED** : Équilibrage par zones (Score: 0.75)
9. **BALANCED** : Fréquences équilibrées (Score: 0.70)
10. **SECTORS** : Répartition géographique (Score: 0.70)
11. **FIBONACCI** : Suite de Fibonacci (Score: 0.65)

## 📈 Statistiques actuelles

- **Total tirages** : 7,038 (2020-2025)
- **Total boules** : 140,760
- **Numéro le + fréquent** : N°1 (2,122 fois - 1.51%)
- **Numéro le - fréquent** : N°28 (1,902 fois - 1.35%)
- **Paire la + fréquente** : 1-13 (660 fois - 9.4%)

## 🎯 Utilisation recommandée

### Workflow quotidien simple
```bash
python keno_cli.py all --grids 5     # Tout en une commande
```

### Workflow interactif (recommandé)
```bash
python cli_menu.py                   # Menu coloré avec toutes les options
# Choisir option 9 pour pipeline complet
```

### Workflow avancé
```bash
# 1. Extraction
python keno_cli.py extract

# 2. Analyse DuckDB avancée
python keno/duckdb_keno.py --csv keno/keno_data/keno_202508.csv --plots --export-stats

# 3. Génération personnalisée
python keno_cli.py generate --grids 3
```

## ⚡ Performances

- **Extraction** : ~30 secondes (5 ans de données)
- **Analyse complète** : ~10 secondes (7,038 tirages)
- **Génération 5 grilles** : ~5 secondes
- **Pipeline complet** : ~45 secondes
- **Analyse DuckDB** : ~15 secondes (11 stratégies)

## 📝 Nouveautés v2.0

### 🆕 CLI amélioré
- ✅ Commande `clean` avec nettoyage approfondi
- ✅ Meilleure gestion des erreurs
- ✅ Statut détaillé du système

### 🆕 Menu interactif mis à jour
- ✅ Options Keno modernisées (8, 9, 10)
- ✅ Nettoyage intelligent (option 14)
- ✅ Statut détaillé (option 23)
- ✅ Analyse personnalisée avancée

### 🆕 Compatibilité formats
- ✅ Auto-détection ancien/nouveau format
- ✅ Conversion automatique avec backup
- ✅ 60 fichiers convertis au format unifié

### 🆕 Nettoyage automatique
- ✅ Suppression fichiers temporaires
- ✅ Gestion des anciens exports (garde les 5 derniers)
- ✅ Nettoyage des backups anciens (>30 jours)

## 🚨 Points importants

1. **Le Keno reste un jeu de hasard** - Utilisez les analyses de façon responsable
2. **Deux interfaces disponibles** - CLI simple ou menu interactif
3. **Système auto-nettoyant** - Gestion automatique des anciens fichiers
4. **11 stratégies avancées** - Via l'analyse DuckDB
5. **Format unifié** - Compatible avec tous les scripts

## 🔧 Dépannage

### Interface recommandée
```bash
python cli_menu.py     # Menu interactif complet
```

### Erreur de données
```bash
python keno_cli.py extract && python keno_cli.py analyze
```

### Système encombré
```bash
python keno_cli.py clean --deep
```

### Test rapide
```bash
python keno_cli.py status
```

---

**🎯 Système Keno v2.0 - Interface double, 11 stratégies, nettoyage automatique !**

**Recommandation : Utilisez `python cli_menu.py` pour l'interface complète**
