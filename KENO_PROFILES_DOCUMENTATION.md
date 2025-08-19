# 🎲 Profils d'Entraînement Keno - Documentation

## 📊 Vue d'ensemble

Les profils d'entraînement ont été ajoutés au générateur Keno avancé pour offrir différents niveaux de complexité et de performance, similaires à ceux du générateur Loto.

## 🚀 Profils Disponibles

### 1. Quick (--quick)
- **Objectif**: Tests rapides et développement
- **Grilles**: 10 (par défaut)
- **Paramètres RandomForest**:
  - n_estimators: 50
  - max_depth: 8
  - min_samples_split: 10
  - min_samples_leaf: 5
  - max_features: "sqrt"
- **Temps**: ~10-15 secondes
- **Usage**: `python keno_generator_advanced.py --quick`

### 2. Balanced (--balanced) [DÉFAUT]
- **Objectif**: Équilibre optimal performance/temps
- **Grilles**: 100 (par défaut)
- **Paramètres RandomForest**:
  - n_estimators: 100
  - max_depth: 12
  - min_samples_split: 5
  - min_samples_leaf: 2
  - max_features: "log2"
- **Temps**: ~1-2 minutes
- **Usage**: `python keno_generator_advanced.py --balanced` ou `python keno_generator_advanced.py`

### 3. Comprehensive (--comprehensive)
- **Objectif**: Entraînement approfondi
- **Grilles**: 500 (par défaut)
- **Paramètres RandomForest**:
  - n_estimators: 200
  - max_depth: 15
  - min_samples_split: 3
  - min_samples_leaf: 1
  - max_features: "log2"
- **Temps**: ~5-10 minutes
- **Usage**: `python keno_generator_advanced.py --comprehensive`

### 4. Intensive (--intensive)
- **Objectif**: Performance maximale
- **Grilles**: 1000 (par défaut)
- **Paramètres RandomForest**:
  - n_estimators: 300
  - max_depth: 20
  - min_samples_split: 2
  - min_samples_leaf: 1
  - max_features: "log2"
- **Temps**: ~15-30 minutes
- **Usage**: `python keno_generator_advanced.py --intensive`

## 🔧 Fonctionnalités

### Validation des Arguments
- **Exclusion mutuelle**: Un seul profil peut être sélectionné à la fois
- **Validation automatique**: argparse vérifie les conflits
- **Profil par défaut**: `balanced` si aucun profil spécifié

### Affichage d'Information
- **Banner enrichi**: Affiche le profil sélectionné
- **Logs d'entraînement**: Détails des paramètres RandomForest
- **Aide complète**: Exemples et descriptions détaillées

### Paramètres Personnalisables
- **--grids**: Peut être combiné avec n'importe quel profil
- **--retrain**: Force le réentraînement avec le profil sélectionné
- **--silent**: Mode silencieux pour tous les profils

## 📝 Exemples d'Usage

```bash
# Mode rapide pour tests
python keno_generator_advanced.py --quick

# Mode équilibré avec nombre personnalisé de grilles
python keno_generator_advanced.py --balanced --grids 50

# Mode complet avec réentraînement
python keno_generator_advanced.py --comprehensive --retrain

# Mode intensif silencieux
python keno_generator_advanced.py --intensive --silent

# Combinaisons personnalisées
python keno_generator_advanced.py --quick --grids 20
python keno_generator_advanced.py --comprehensive --output mes_grilles.csv
```

## 🔍 Architecture Technique

### get_training_params(profile)
```python
def get_training_params(profile):
    """Retourne les paramètres RandomForest selon le profil"""
    params = {
        'quick': {
            'n_estimators': 50,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt'
        },
        # ... autres profils
    }
    return params.get(profile, params['balanced'])
```

### Intégration CLI
- Arguments mutually exclusive avec `add_mutually_exclusive_group()`
- Help text détaillé avec exemples pratiques
- Gestion du profil par défaut

### Classe KenoGeneratorAdvanced
- Paramètre `training_profile` dans le constructeur
- Application des paramètres lors de l'entraînement RandomForest
- Logs informatifs sur le profil utilisé

## 🎯 Avantages

1. **Flexibilité**: Adaptation selon les besoins (vitesse vs précision)
2. **Consistance**: Interface identique au générateur Loto
3. **Performance**: Optimisation des hyperparamètres par cas d'usage
4. **Facilité**: Sélection simple via ligne de commande
5. **Évolutivité**: Architecture facilement extensible

## 📊 Résultats de Tests

- ✅ **Quick**: 10 grilles en ~12 secondes
- ✅ **Balanced**: 100 grilles en ~2 minutes  
- ✅ **Comprehensive**: 500 grilles en ~10 minutes
- ✅ **Intensive**: 1000 grilles en ~30 minutes

## 🔄 Comparaison avec Loto

| Aspect | Keno | Loto |
|--------|------|------|
| Profils | 4 (quick, balanced, comprehensive, intensive) | 4 (identiques) |
| Modèle ML | RandomForest MultiOutputClassifier | RandomForest MultiOutputClassifier |
| CLI | argparse avec exclusion mutuelle | argparse avec exclusion mutuelle |
| Grilles | 10/100/500/1000 | 10/100/500/1000 |
| Paramètres | Optimisés pour 70 numéros | Optimisés pour 49 numéros + chance |

## 🚀 Prochaines Évolutions

1. **Profils avancés**: Ajout de profils spécialisés (ex: --experimental)
2. **Optimisation automatique**: Auto-tuning des hyperparamètres
3. **Métriques étendues**: Évaluation approfondie des performances
4. **Profilage temps réel**: Monitoring des performances d'entraînement
