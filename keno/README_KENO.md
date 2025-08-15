# 🎯 Module d'Analyse et Génération Keno FDJ

Ce module fournit des outils complets pour analyser les tirages Keno de la FDJ et générer des grilles intelligentes basées sur des analyses statistiques.

## 📋 Vue d'ensemble

Le système comprend plusieurs scripts spécialisés :

- **Analyse statistique** des tirages passés
- **Génération intelligente** de grilles
- **Visualisations** graphiques
- **Rapports détaillés** des tendances

## 🚀 Scripts disponibles

### 1. `demo_analyse_simple.py`
**Démonstration avec données simulées**

```bash
python demo_analyse_simple.py
```

- Génère 15 tirages de démonstration
- Analyse les fréquences et retards
- Propose une grille basée sur l'analyse
- Parfait pour comprendre le fonctionnement

### 2. `generateur_keno_intelligent.py`
**Générateur de grilles intelligent**

```bash
# Génération standard (5 grilles de 10 numéros)
python generateur_keno_intelligent.py

# Personnalisé (10 grilles de 8 numéros)
python generateur_keno_intelligent.py --grilles 10 --taille 8

# Mode purement aléatoire
python generateur_keno_intelligent.py --sans-analyse
```

**Fonctionnalités :**
- Grilles analysées basées sur les tendances statistiques
- Grilles aléatoires pour la diversité
- Statistiques de répartition par zones
- Conseils de jeu intégrés

### 3. `analyse_tirages_fdj.py`
**Analyseur complet des données FDJ** *(en développement)*

```bash
python analyse_tirages_fdj.py
```

- Récupération des vrais tirages FDJ
- Analyses statistiques complètes
- Création de visualisations
- Sauvegarde des données

### 4. `test_analyse.py`
**Tests du système complet**

```bash
python test_analyse.py
```

- Tests avec 30 tirages simulés
- Validation de toutes les analyses
- Tests de sauvegarde et visualisations

## 📊 Types d'analyses

### Analyse des fréquences
- Numéros les plus/moins sortis
- Pourcentages d'apparition
- Écarts par rapport à la moyenne théorique

### Analyse des retards
- Numéros en retard de sortie
- Nombre de tirages depuis la dernière apparition
- Identification des numéros "dus"

### Analyse des paires
- Combinaisons de numéros sortant ensemble
- Fréquence des associations
- Paires gagnantes récurrentes

### Recommandations intelligentes
- Numéros chauds (fréquents récemment)
- Numéros froids (en retard)
- Grilles équilibrées mixant les deux approches

## 🎲 Stratégies de jeu

### Grilles analysées (60% des grilles)
- Basées sur les tendances statistiques récentes
- Utilisent les numéros chauds et en retard
- Adaptatives selon les données disponibles

### Grilles aléatoires (40% des grilles)
- Diversification du portefeuille de jeu
- Couverture de numéros non identifiés par l'analyse
- Équilibrage des risques

### Répartition par zones
Le système surveille la répartition des numéros :
- **Zone 1-23** : Numéros bas
- **Zone 24-46** : Numéros moyens  
- **Zone 47-70** : Numéros hauts

## 📈 Visualisations créées

Les graphiques générés incluent :
- Histogrammes des fréquences
- Graphiques des retards
- Heatmaps des paires gagnantes
- Statistiques par zones

Fichiers sauvegardés dans `../keno_analyse_plots/`

## 💾 Données sauvegardées

### Tirages récupérés
- Format JSON avec date, heure, numéros
- Sauvegarde dans `tirages_fdj/`

### Analyses statistiques
- Données complètes d'analyse
- Format JSON avec métadonnées
- Historique des analyses

### Exports CSV
- Fréquences, retards, paires
- Compatible Excel/LibreOffice
- Sauvegarde dans `../keno_stats_exports/`

## ⚙️ Options de personnalisation

### Taille des grilles
```bash
python generateur_keno_intelligent.py --taille 7    # 7 numéros par grille
python generateur_keno_intelligent.py --taille 12   # 12 numéros par grille
```
- Minimum : 3 numéros
- Maximum : 20 numéros

### Nombre de grilles
```bash
python generateur_keno_intelligent.py --grilles 15  # 15 grilles
```

### Mode d'analyse
```bash
python generateur_keno_intelligent.py --sans-analyse  # Purement aléatoire
```

## 🧪 Tests et validation

Le module inclut des tests complets :

```bash
# Test rapide
python demo_analyse_simple.py

# Test complet
python test_analyse.py

# Vérification du système
python recap_keno.py
```

## 📋 Exemple de sortie

```
🎯 GÉNÉRATEUR DE GRILLES KENO INTELLIGENT
============================================================
📅 15/08/2025 12:22:38
🎲 Génération de 5 grilles de 10 numéros
🧠 Analyse intelligente: Activée

📊 Analyse des tirages précédents...
✅ 20 numéros recommandés identifiés

🎲 Génération de 5 grilles...
   Grille  1:  9 - 14 - 21 - 39 - 41 - 47 - 50 - 53 - 64 - 69 🧠 Analysée
   Grille  2:  5 - 19 - 27 - 40 - 42 - 45 - 50 - 58 - 67 - 70 🧠 Analysée
   Grille  3:  5 - 13 - 14 - 38 - 39 - 47 - 51 - 58 - 64 - 65 🧠 Analysée
   Grille  4: 10 - 19 - 23 - 25 - 26 - 32 - 39 - 42 - 51 - 61 🎲 Aléatoire
   Grille  5: 10 - 12 - 20 - 22 - 23 - 24 - 33 - 35 - 64 - 65 🎲 Aléatoire

📈 Statistiques des grilles générées:
   Zone 1-23: 16 numéros (32.0%)
   Zone 24-46: 16 numéros (32.0%)  
   Zone 47-70: 18 numéros (36.0%)

✅ 5 grilles générées avec succès!
```

## ⚠️ Avertissements importants

1. **Jeu responsable** : Ces outils sont à des fins d'analyse uniquement
2. **Aucune garantie** : Les statistiques n'assurent aucun gain
3. **Budget** : Ne jouez que ce que vous pouvez vous permettre de perdre
4. **Hasard** : Le Keno reste un jeu de hasard malgré les analyses

## 🔧 Installation des dépendances

```bash
pip install beautifulsoup4 requests matplotlib seaborn pandas numpy
```

## 🤝 Intégration avec le système principal

Ce module s'intègre avec le générateur principal :

```bash
# Depuis le répertoire principal
python exemple_lancement.py --jeu keno --nombres-par-grille 8
```

## 📞 Utilisation

Pour commencer :

1. **Démonstration** : `python demo_analyse_simple.py`
2. **Génération** : `python generateur_keno_intelligent.py`  
3. **Récapitulatif** : `python recap_keno.py`

---

**🎯 Bonne chance avec vos analyses et générations de grilles Keno !**
