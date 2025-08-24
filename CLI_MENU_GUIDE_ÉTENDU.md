# 🎮 GUIDE D'UTILISATION - MENU CLI ÉTENDU

## 🌟 NOUVELLES FONCTIONNALITÉS TOP NUMÉROS

Le menu CLI a été enrichi avec 4 nouvelles options dédiées aux analyses TOP numéros équilibrés :

### 🆕 NOUVELLE SECTION : "🎯 TOP NUMÉROS ÉQUILIBRÉS"

```
🎯 TOP NUMÉROS ÉQUILIBRÉS
  2️⃣8️⃣ 🏆 TOP 25 Loto équilibrés (stratégie optimisée)
  2️⃣9️⃣ 🏆 TOP 30 Keno équilibrés (stratégie optimisée)
  3️⃣0️⃣ 📊 Voir TOP 25 Loto (dernière génération)
  3️⃣1️⃣ 📊 Voir TOP 30 Keno (dernière génération)
```

## 🎯 GUIDE DÉTAILLÉ DES NOUVELLES OPTIONS

### Option 28 : 🏆 TOP 25 Loto Équilibrés

**Fonction :** Génère les 25 numéros Loto avec le plus de chances selon la stratégie équilibrée

**Processus interactif :**
1. **Détection automatique** du fichier de données Loto le plus récent
2. **Affichage des informations** : nom du fichier et période couverte
3. **Choix de stratégie :**
   - `1` - Équilibrée (recommandé) : Fréquence 35% + Retard 30% + Paires 20% + Zones 15%
   - `2` - Focus retard : Priorité aux numéros en retard
   - `3` - Focus fréquence : Priorité aux numéros les plus fréquents
4. **Options d'export :**
   - Export statistiques détaillées (O/n)
   - Génération de visualisations (o/N)
5. **Confirmation** et lancement

**Résultat :**
- Fichier CSV : `loto_stats_exports/top_25_numeros_equilibres_loto.csv`
- Rapport Markdown avec analyses détaillées
- Affichage console des TOP 10 avec scores

**Exemple d'utilisation :**
```
🎯 Votre choix: 28
📁 Fichier détecté: loto_201911.csv
📅 Période: 11/2019 → 11/2019

⚙️ Stratégies disponibles:
1️⃣  Équilibrée (recommandé)
2️⃣  Focus retard
3️⃣  Focus fréquence

Stratégie (1-3) [1]: 1
Exporter les statistiques détaillées ? (O/n): O
Générer les visualisations ? (o/N): n
Lancer la génération ? (O/n): O
```

---

### Option 29 : 🏆 TOP 30 Keno Équilibrés

**Fonction :** Génère les 30 numéros Keno avec le plus de chances selon la stratégie équilibrée

**Processus interactif :**
1. **Détection intelligente** : Privilégie le fichier consolidé si disponible
2. **Affichage des informations** du fichier sélectionné
3. **Options d'export :**
   - Export statistiques détaillées (O/n)
   - Génération de visualisations (o/N)
4. **Confirmation** et lancement

**Résultat :**
- Fichier CSV : `keno_stats_exports/top_30_numeros_equilibres_keno.csv`
- Analyse de 11 stratégies probabilistes
- Affichage console des TOP 10 avec détails

**Avantages Keno :**
- Utilise le fichier consolidé pour performance optimale
- Analyse 11 stratégies : HOT, COLD, BALANCED, Z-SCORE, FIBONACCI, etc.
- Scoring composite : Fréquence 30% + Retard 25% + Paires 25% + Zones 20%

---

### Option 30 : 📊 Voir TOP 25 Loto

**Fonction :** Affiche et analyse le dernier fichier TOP 25 Loto généré

**Affichage inclut :**
- **Informations du fichier** : localisation et dernière modification
- **TOP 10 coloré** avec scores, zones, retards et fréquences
- **Répartition par zones** avec pourcentages
- **Suggestions pratiques** :
  - Grille 5 numéros (TOP 5 direct)
  - Système 7 numéros
  - Système 10 numéros

**Exemple de sortie :**
```
🏆 TOP 25 NUMÉROS LOTO ÉQUILIBRÉS
================================================================================
Rang Numéro Score    Zone            Retard Fréq
--------------------------------------------------------------------------------
1    35     0.8750   3               118    96  
2    7      0.7912   1               74     101 
3    11     0.7414   1               85     84  

📍 RÉPARTITION PAR ZONES:
Zone 2 (18-34): 10 numéros (40.0%)
Zone 1 (1-17): 8 numéros (32.0%)
Zone 3 (35-49): 7 numéros (28.0%)

💡 SUGGESTIONS PRATIQUES:
Grille 5 numéros: [35, 7, 11, 12, 28]
Système 7 numéros: [35, 7, 11, 12, 28, 46, 17]
```

---

### Option 31 : 📊 Voir TOP 30 Keno

**Fonction :** Affiche et analyse le dernier fichier TOP 30 Keno généré

**Spécificités Keno :**
- **TOP 10 coloré** des 30 numéros générés
- **Répartition 5 zones** Keno avec pourcentages
- **Suggestions adaptées** :
  - Grille 10 numéros (TOP 10 direct)
  - Système 15 numéros
  - Système 20 numéros

**Avantage :** Consultation rapide sans régénération

## 🚀 WORKFLOW RECOMMANDÉ

### Pour Loto :
1. **Option 28** : Générer TOP 25 avec données récentes
2. **Option 30** : Consulter et choisir les numéros
3. Utiliser les suggestions pratiques pour les grilles

### Pour Keno :
1. **Option 29** : Générer TOP 30 avec fichier consolidé
2. **Option 31** : Consulter les 11 stratégies
3. Choisir selon le style de jeu préféré

## 🔧 PARAMÈTRES ET CONFIGURATIONS

### Stratégies Loto Disponibles

| Stratégie | Description | Pondération |
|-----------|-------------|-------------|
| **Équilibrée** | Approche balancée (recommandé) | Fréq 35% + Retard 30% + Paires 20% + Zones 15% |
| **Focus retard** | Priorité aux numéros en retard | Retard majoré |
| **Focus fréquence** | Priorité aux plus fréquents | Fréquence majorée |

### Options d'Export

| Option | Description | Recommandation |
|--------|-------------|----------------|
| **Statistiques détaillées** | Export CSV complet + rapports | Toujours activé (O) |
| **Visualisations** | Graphiques et heatmaps | Selon besoin (o/N) |

## 🎯 AVANTAGES DU MENU CLI ÉTENDU

### ✅ **Simplicité d'Usage**
- Interface guidée pas à pas
- Détection automatique des fichiers
- Valeurs par défaut intelligentes

### ✅ **Flexibilité**
- Choix de stratégies multiples
- Options d'export modulaires
- Consultation sans régénération

### ✅ **Performance**
- Utilisation des fichiers optimaux
- Cache des résultats précédents
- Génération sur demande uniquement

### ✅ **Intégration Complète**
- Compatible avec tous les autres outils
- Fichiers CSV standardisés
- Workflow cohérent

## 📊 FICHIERS GÉNÉRÉS

### Fichiers CSV Fixes (remplacés à chaque génération)
- `loto_stats_exports/top_25_numeros_equilibres_loto.csv`
- `keno_stats_exports/top_30_numeros_equilibres_keno.csv`

### Structure CSV Commune (10 colonnes)
- `rang` : Position dans le classement
- `numero` : Numéro recommandé  
- `score_composite` : Score final optimisé
- `zone` : Zone géographique d'appartenance
- `frequence`, `score_retard`, `score_paires`, `score_zones` : Scores détaillés
- `retard_actuel` : Nombre de jours depuis dernière sortie
- `freq_absolue` : Fréquence brute d'apparition

## 🏆 CAS D'USAGE OPTIMAUX

### 🎲 **Joueur Loto Régulier**
1. Option 28 chaque semaine avec nouvelles données
2. Option 30 pour consultation rapide avant jeu
3. Utilisation suggestions pratiques selon budget

### 🎰 **Joueur Keno Intensif**
1. Option 29 avec fichier consolidé (base historique large)
2. Option 31 pour analyse rapide entre sessions
3. Alternance entre différentes stratégies selon résultats

### 📈 **Analyste/Développeur**
1. Options de génération avec export complet
2. Analyse des CSV pour études statistiques
3. Comparaison des stratégies dans le temps

---

💡 **Le menu CLI étendu offre maintenant un accès unifié et intuitif à toutes les fonctionnalités TOP numéros équilibrés, avec une interface guidée et des résultats immédiatement exploitables !**
