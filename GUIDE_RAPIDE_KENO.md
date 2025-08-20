# 🎯 GUIDE RAPIDE KENO

## 🚀 Démarrage Ultra-Rapide

### Option 1 : Menu Interactif (Recommandé)
```bash
python cli_menu.py
# Choisir option 8 ou 9 pour Keno
```

### Option 2 : CLI Direct
```bash
python keno_cli.py all --grids 5
```

## 🎮 Commandes Essentielles

```bash
# Statut complet
python keno_cli.py status

# Pipeline complet (extract + analyze + generate)
python keno_cli.py all --grids 3

# Nettoyage
python keno_cli.py clean

# Analyse avancée
python keno/duckdb_keno.py --csv keno/keno_data/keno_202508.csv --plots
```

## 📊 Menu Interactif - Options Keno

- **Option 8** : Analyse Keno complète
- **Option 9** : Pipeline complet avec visualisations
- **Option 10** : Analyse personnalisée (5 sous-options)
- **Option 14** : Nettoyage système
- **Option 23** : Statut détaillé

## 🎯 Recommandations

1. **Débutant** : `python cli_menu.py` → Option 8
2. **Avancé** : `python keno_cli.py all --grids 5`
3. **Expert** : DuckDB avec 11 stratégies avancées
4. **Maintenance** : Option 14 pour nettoyage

## 📈 Données Actuelles

- **7,038 tirages** de 2020 à 2025
- **60 fichiers de données** au format unifié
- **11 stratégies d'analyse** disponibles
- **3 algorithmes de génération** de grilles

## ⚡ Performance

- **Pipeline complet** : ~45 secondes
- **Analyse seule** : ~10 secondes
- **Génération 5 grilles** : ~5 secondes

---
**Interface recommandée : `python cli_menu.py` pour l'expérience complète !**
