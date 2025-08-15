# Guide d'Utilisation - Paramètres Configurables du Générateur Loto Avancé

## 🎯 Vue d'ensemble

Le générateur Loto avancé intègre maintenant des paramètres configurables `n_simulations` et `n_cores` pour optimiser les performances selon vos besoins.

## 📋 Paramètres Disponibles

### Paramètres Principaux
- `--simulations, -s` : Nombre de simulations (100-100,000)
- `--cores, -c` : Nombre de cœurs CPU à utiliser (1-CPU_COUNT)

### Modes Prédéfinis
- `--quick` : Mode rapide (1,000 simulations, tous les cœurs)
- `--intensive` : Mode intensif (10,000 simulations, tous les cœurs)

### Options Supplémentaires
- `--silent` : Mode silencieux (moins de logs)
- `--seed` : Graine aléatoire pour reproductibilité
- `--help, -h` : Affichage de l'aide

## 🚀 Exemples d'Utilisation

### 1. Mode Rapide (Recommandé pour débutants)
```bash
python loto/loto_generator_advanced_Version2.py --quick
# Génère 1,000 grilles en ~1-2 minutes
```

### 2. Mode Intensif (Pour analyses approfondies)
```bash
python loto/loto_generator_advanced_Version2.py --intensive
# Génère 10,000 grilles en ~10-15 minutes
```

### 3. Configuration Personnalisée
```bash
# 500 simulations avec 2 cœurs
python loto/loto_generator_advanced_Version2.py -s 500 -c 2

# 2000 simulations silencieuses
python loto/loto_generator_advanced_Version2.py -s 2000 --silent

# Configuration avec graine fixe (reproductible)
python loto/loto_generator_advanced_Version2.py -s 1000 --seed 42
```

## 🎮 Interface CLI (Menu Principal)

Accédez aux paramètres via l'option **22** du menu principal :

```bash
python cli_menu.py
# Choisir l'option 22 : "Générateur Loto avancé (ML + IA)"
```

### Modes de Configuration CLI

1. **Mode Rapide** : 1,000 simulations optimisées
2. **Mode Standard** : 5,000 simulations équilibrées  
3. **Mode Intensif** : 10,000 simulations approfondies
4. **Mode Personnalisé** : Configuration manuelle complète

## ⚡ Recommandations de Performance

### Pour un Usage Quotidien
- **Mode rapide** : Idéal pour générer quelques grilles rapidement
- **Paramètres** : 1,000-2,000 simulations, tous les cœurs

### Pour une Analyse Approfondie
- **Mode intensif** : Meilleure qualité statistique
- **Paramètres** : 5,000-10,000 simulations, tous les cœurs

### Pour des Tests/Développement
- **Configuration légère** : 100-500 simulations, 1-2 cœurs
- **Mode silencieux** recommandé

## 📊 Comparaison des Temps d'Exécution

| Simulations | Cœurs | Temps Estimé | Usage Recommandé |
|-------------|-------|--------------|------------------|
| 100         | 1     | 30s          | Tests rapides    |
| 500         | 2     | 1m30s        | Usage léger      |
| 1,000       | 4     | 1m30s        | Mode rapide      |
| 2,000       | 4     | 3m           | Usage standard   |
| 5,000       | 4     | 7m           | Analyse poussée  |
| 10,000      | 4     | 15m          | Mode intensif    |

## 🔧 Paramètres Avancés

### Validation Automatique
- Les valeurs sont automatiquement validées
- Plages acceptées : 100-100,000 simulations, 1-CPU_COUNT cœurs
- Messages d'erreur explicites en cas de valeur invalide

### Gestion de l'Environnement
- Détection automatique du chemin Python virtuel
- Fallback vers Python système si nécessaire
- Compatible avec conda, venv, et installations standard

## 🎲 Qualité des Résultats

### Relation Simulations/Qualité
- **100-500** : Résultats basiques, variance élevée
- **1,000-2,000** : Bon équilibre qualité/temps
- **5,000+** : Haute qualité, patterns statistiques stables
- **10,000+** : Qualité optimale, convergence statistique

### Influence du Nombre de Cœurs
- **1 cœur** : Lent mais stable
- **2-4 cœurs** : Bon équilibre
- **8+ cœurs** : Rendements décroissants (I/O bound)

## 🔍 Dépannage

### Problèmes Courants

1. **Erreur "Valeur invalide"**
   - Vérifiez les plages acceptées
   - Utilisez des entiers positifs

2. **Exécution lente**
   - Réduisez le nombre de simulations
   - Augmentez le nombre de cœurs
   - Utilisez `--silent`

3. **Erreur d'environnement**
   - Vérifiez l'activation du venv
   - Utilisez le chemin complet vers Python

### Support et Logs
- Utilisez `--help` pour l'aide complète
- Mode `--silent` pour réduire la verbosité
- Logs détaillés disponibles en mode normal

## 🎯 Bonnes Pratiques

1. **Démarrez avec le mode rapide** pour vous familiariser
2. **Utilisez le mode intensif** pour les analyses importantes
3. **Personnalisez** selon vos contraintes de temps/qualité
4. **Mode silencieux** pour l'intégration dans des scripts
5. **Sauvegardez** les configurations qui fonctionnent bien

---

*Ce guide vous permet d'exploiter pleinement les nouvelles capacités de configuration du générateur Loto avancé. N'hésitez pas à expérimenter avec différents paramètres pour trouver le réglage optimal pour vos besoins !*
