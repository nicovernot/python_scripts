# 🎯 ANALYSEUR KENO - SORTIES UNIQUES

## 📋 Changements Apportés

Le script `duckdb_keno.py` a été modifié pour générer des **sorties uniques** sans horodatage, évitant ainsi l'accumulation de fichiers doublons.

### ✅ Avant vs Après

**AVANT** (avec horodatage) :
```
keno_stats_exports/
├── frequences_keno_20250812_074607.csv
├── frequences_keno_20250812_075123.csv
├── frequences_keno_20250812_075456.csv
├── paires_keno_20250812_074607.csv
├── paires_keno_20250812_075123.csv
└── ...
```

**APRÈS** (fichiers uniques) :
```
keno_stats_exports/
├── frequences_keno.csv         ← TOUJOURS LE MÊME NOM
├── paires_keno.csv            ← REMPLACÉ À CHAQUE EXÉCUTION
├── retards_keno.csv           ← PAS DE DOUBLONS
└── zones_keno.csv             ← FICHIERS UNIQUES
```

## 🚀 Utilisation

### Option 1 : Script Original (Modifié)
```bash
python keno/duckdb_keno.py --csv keno/keno_data/extracted/keno_202010.csv --plots --export-stats
```

### Option 2 : Script Wrapper (Recommandé)
```bash
# Analyse complète avec auto-détection du CSV
python analyse_keno_unique.py --plots --export-stats

# Avec fichier spécifique
python analyse_keno_unique.py --csv data/keno.csv --plots --export-stats

# Nettoyage seulement
python analyse_keno_unique.py --clean
```

## 📁 Fichiers Générés

### 📊 Statistiques CSV (keno_stats_exports/)
- `frequences_keno.csv` - Fréquences des numéros
- `retards_keno.csv` - Retards par numéro
- `paires_keno.csv` - Top paires fréquentes
- `zones_keno.csv` - Répartition par zones

### 📈 Visualisations (keno_analyse_plots/)
- `frequences_keno.png` - Graphique des fréquences
- `retards_keno.png` - Graphique des retards
- `heatmap_keno.png` - Heatmap des fréquences
- `paires_keno.png` - Top 20 des paires

### 📋 Recommandations (keno_output/)
- `recommandations_keno.md` - Stratégies et grilles recommandées (format Markdown)

## 🔧 Fonctionnalités Ajoutées

### 1. Nettoyage Automatique
- ✅ Suppression automatique des anciens fichiers avec horodatage
- ✅ Évite l'accumulation de doublons
- ✅ Garde seulement la version la plus récente

### 2. Script Wrapper Intelligent
- ✅ Auto-détection du fichier CSV le plus récent
- ✅ Nettoyage préalable des anciens fichiers
- ✅ Résumé des fichiers générés avec tailles
- ✅ Gestion d'erreurs améliorée

### 3. Fichiers Uniques
- ✅ Noms de fichiers sans horodatage
- ✅ Remplacement des versions précédentes
- ✅ Structure de dossiers propre

## 💡 Avantages

1. **Espace Disque** : Plus d'accumulation de fichiers doublons
2. **Simplicité** : Toujours les mêmes noms de fichiers
3. **Automatisation** : Plus besoin de nettoyer manuellement
4. **Compatibilité** : Scripts utilisant ces fichiers fonctionnent toujours
5. **Performance** : Moins de fichiers = navigation plus rapide

## 🛠️ Modifications Techniques

### Dans `duckdb_keno.py` :
1. **Suppression des horodatages** dans `export_statistics()`
2. **Ajout de `_clean_old_timestamped_files()`** pour le nettoyage
3. **Message de confirmation** pour les visualisations
4. **Noms de fichiers fixes** au lieu d'horodatés

### Nouveau fichier `analyse_keno_unique.py` :
1. **Wrapper intelligent** avec auto-détection
2. **Nettoyage préalable** automatique
3. **Rapport des fichiers générés** avec tailles
4. **Options flexibles** (clean, plots, export-stats)

## 📝 Exemples d'Utilisation

### Analyse Rapide
```bash
python analyse_keno_unique.py
```

### Analyse Complète
```bash
python analyse_keno_unique.py --plots --export-stats
```

### Nettoyage Manuel
```bash
python analyse_keno_unique.py --clean
```

### Avec Fichier Spécifique
```bash
python analyse_keno_unique.py --csv mes_donnees.csv --plots
```

---

✅ **Résultat** : Fini les doublons ! Vous avez maintenant des sorties CSV et images uniques qui se remplacent automatiquement à chaque analyse.
