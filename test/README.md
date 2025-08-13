# 🧪 Tests du Projet Loto/Keno

Ce dossier contient tous les tests de validation pour le système d'analyse Loto/Keno.

## 📋 Tests Disponibles

### 🔧 Tests Essentiels
- **`test_essential.py`** - Vérification des dépendances et de la structure du projet
  - Tests d'import des modules critiques
  - Validation de la structure des dossiers
  - Compilation des scripts principaux

### 🎯 Tests Spécifiques

#### Loto
- **`test_loto.py`** - Tests complets du système Loto
  - Vérification des données CSV téléchargées
  - Configuration des scripts
  - Fonctionnement du générateur de grilles
  - Script de téléchargement

#### Keno  
- **`test_keno.py`** - Tests complets du système Keno
  - Vérification des données CSV téléchargées
  - Compilation des scripts d'analyse
  - Fonctionnement de l'analyseur
  - Détection de doublons

### ⚡ Tests de Performance
- **`test_performance.py`** - Vérification des performances du système
  - Temps d'import des modules
  - Performance de chargement des CSV
  - Temps d'exécution des analyses
  - Utilisation mémoire

## 🚀 Utilisation

### Lancer tous les tests
```bash
python test/run_all_tests.py
```

### Tests spécifiques
```bash
# Tests essentiels seulement
python test/run_all_tests.py --essential

# Tests Loto seulement
python test/run_all_tests.py --loto

# Tests Keno seulement  
python test/run_all_tests.py --keno

# Tests de performance seulement
python test/run_all_tests.py --performance

# Tous les tests sauf performance (plus rapide)
python test/run_all_tests.py --fast
```

### Tests individuels
```bash
# Test essentiel
python test/test_essential.py

# Test Loto
python test/test_loto.py

# Test Keno
python test/test_keno.py

# Test performance
python test/test_performance.py
```

## 📊 Interprétation des Résultats

### ✅ Tests Réussis
- Tous les composants fonctionnent correctement
- Le système est opérationnel

### ❌ Tests Échoués
- Vérifiez les messages d'erreur détaillés
- Corrigez les problèmes signalés
- Relancez les tests

### ⚠️  Avertissements
- Le système fonctionne mais avec des limitations
- Certaines fonctionnalités peuvent être dégradées

## 🔍 Dépannage

### Erreurs d'Import
```bash
# Installer les dépendances manquantes
pip install -r requirements.txt

# Vérifier l'environnement Python
python --version
pip list
```

### Erreurs de Données
```bash
# Télécharger les données Loto
python loto/result.py

# Télécharger les données Keno
python keno/results_clean.py
```

### Problèmes de Performance
- Vérifiez la mémoire disponible
- Fermez les applications non nécessaires
- Utilisez `--fast` pour éviter les tests lourds

## 📁 Structure

```
test/
├── run_all_tests.py     # 🎯 Lanceur principal
├── test_essential.py    # 🔧 Tests essentiels
├── test_loto.py         # 🎲 Tests Loto
├── test_keno.py         # 🎰 Tests Keno
├── test_performance.py  # ⚡ Tests performance
└── README.md           # 📖 Cette documentation
```

## ✨ Bonnes Pratiques

1. **Lancez les tests essentiels** avant de commencer à travailler
2. **Testez après chaque modification** importante
3. **Utilisez `--fast`** pour les tests de développement
4. **Lancez tous les tests** avant de commit des changements
5. **Vérifiez les performances** régulièrement

## 🎯 Tests Recommandés par Situation

### Premier lancement
```bash
python test/run_all_tests.py --essential
```

### Développement quotidien
```bash
python test/run_all_tests.py --fast
```

### Avant un commit
```bash
python test/run_all_tests.py
```

### Débogage spécifique
```bash
python test/test_loto.py --verbose
python test/test_keno.py --verbose
```
