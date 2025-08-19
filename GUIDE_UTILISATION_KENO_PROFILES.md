# 🎯 Guide d'Utilisation Rapide - Profils d'Entraînement

## 🎲 Générateur Keno Avancé

### 🚀 Utilisation Simple

```bash
# Profil par défaut (équilibré)
python keno/keno_generator_advanced.py

# Tests rapides (10 grilles en 15 secondes)
python keno/keno_generator_advanced.py --quick

# Entraînement complet (500 grilles en 10 minutes)
python keno/keno_generator_advanced.py --comprehensive

# Performance maximale (1000 grilles en 30 minutes)
python keno/keno_generator_advanced.py --intensive
```

### 📊 Comparaison des Profils

| Profil | Grilles | Temps | Arbres ML | Profondeur | Usage |
|--------|---------|-------|-----------|------------|-------|
| **quick** | 10 | ~15s | 50 | 8 | Tests/développement |
| **balanced** | 100 | ~2min | 100 | 12 | Usage standard |
| **comprehensive** | 500 | ~10min | 200 | 15 | Analyse approfondie |
| **intensive** | 1000 | ~30min | 300 | 20 | Performance maximale |

### 🔧 Options Combinées

```bash
# Profil quick avec nombre personnalisé de grilles
python keno/keno_generator_advanced.py --quick --grids 25

# Profil comprehensive avec réentraînement
python keno/keno_generator_advanced.py --comprehensive --retrain

# Mode silencieux pour automation
python keno/keno_generator_advanced.py --intensive --silent

# Sortie personnalisée
python keno/keno_generator_advanced.py --balanced --output mes_grilles.csv
```

### 📈 Recommandations d'Usage

#### 🧪 Développement et Tests
```bash
python keno/keno_generator_advanced.py --quick
```
- Idéal pour valider le code
- Retour rapide
- Ressources minimales

#### 📱 Usage Quotidien
```bash
python keno/keno_generator_advanced.py --balanced
```
- Équilibre optimal
- Bon compromis temps/qualité
- Recommandé par défaut

#### 🎯 Analyse Poussée
```bash
python keno/keno_generator_advanced.py --comprehensive
```
- Pour les analyses statistiques
- Meilleure précision
- Plus de grilles générées

#### 🏆 Performance Ultime
```bash
python keno/keno_generator_advanced.py --intensive
```
- Maximum de précision ML
- Pour les gros volumes
- Calcul intensif

## 🔍 Vérification du Statut

### Voir les Paramètres Actuels
```bash
python keno/keno_generator_advanced.py --help
```

### Test Rapide de Fonctionnement
```bash
python test_keno_profiles.py
```

### Validation Complète
```bash
# Test de tous les profils
python keno/keno_generator_advanced.py --quick --grids 5
python keno/keno_generator_advanced.py --balanced --grids 10
python keno/keno_generator_advanced.py --comprehensive --grids 20
python keno/keno_generator_advanced.py --intensive --grids 30
```

## 🎯 Résultats Attendus

### Métriques de Performance
- **Accuracy ML** : ~70-71%
- **Score moyen** : 12.5-13.5
- **Grilles uniques** : 100% (pas de doublons)

### Structure des Sorties
```
keno_output/
├── grilles_keno.csv         # Grilles avec scores
├── rapport_keno.md          # Rapport détaillé
└── logs/                    # Journaux d'exécution
```

### Format des Grilles
```csv
1,8,11,14,16,27,35,39,62,69,13.3554
3,7,13,20,21,49,50,55,59,69,13.2903
1,5,8,32,42,44,46,55,61,69,13.2748
```

## ⚠️ Notes Importantes

1. **Profils exclusifs** : Un seul profil à la fois
2. **Mémoire** : Profils intensive nécessitent ~2GB RAM
3. **CPU** : Temps proportionnel au nombre d'arbres
4. **Stockage** : Modèles sauvegardés dans `keno_models/`
5. **Réentraînement** : Utilisez `--retrain` si données mises à jour

## 🚨 Dépannage

### Problèmes Courants

#### Timeout ou Arrêt
```bash
# Réduire le nombre de grilles pour tests
python keno/keno_generator_advanced.py --intensive --grids 100
```

#### Erreur de Mémoire
```bash
# Utiliser un profil moins intensif
python keno/keno_generator_advanced.py --balanced --grids 50
```

#### Modèles Corrompus
```bash
# Forcer le réentraînement
python keno/keno_generator_advanced.py --retrain --quick
```

### Support
- Documentation complète : `keno/README_KENO_ADVANCED.md`
- Tests : `test_keno_profiles.py`
- Profils détaillés : `KENO_PROFILES_DOCUMENTATION.md`
