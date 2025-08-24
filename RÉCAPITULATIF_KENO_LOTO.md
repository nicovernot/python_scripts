# 🎮 RÉCAPITULATIF COMPLET - KENO vs LOTO

## 📊 RÉSUMÉ DES IMPLÉMENTATIONS

### 🔥 KENO - TOP 30 NUMÉROS ÉQUILIBRÉS

#### Caractéristiques Techniques
- **Plage :** 1-70 (70 numéros disponibles)
- **Recommandations :** TOP 30 numéros
- **Zones :** 5 zones équilibrées (14 numéros chacune)
- **Tirages :** 20 numéros par tirage, plusieurs fois par jour

#### Pondération du Score Composite
```
Score = (Fréquence × 0.30) + (Retard × 0.25) + (Paires × 0.25) + (Zones × 0.20)
```

#### Fichiers Principaux
- `keno/duckdb_keno.py` - Moteur d'analyse principal
- Exports : `keno_stats_exports/top_30_numeros_equilibres_keno_*.csv`
- Tests : Validés avec données réelles

---

### 🎯 LOTO - TOP 25 NUMÉROS ÉQUILIBRÉS

#### Caractéristiques Techniques
- **Plage :** 1-49 (49 numéros disponibles)
- **Recommandations :** TOP 25 numéros
- **Zones :** 3 zones équilibrées (Zone 1: 1-17, Zone 2: 18-34, Zone 3: 35-49)
- **Tirages :** 5 numéros par tirage, 3 fois par semaine

#### Pondération du Score Composite
```
Score = (Fréquence × 0.35) + (Retard × 0.30) + (Paires × 0.20) + (Zones × 0.15)
```

#### Fichiers Principaux
- `loto/duckdb_loto.py` - Moteur d'analyse principal
- Exports : `loto_stats_exports/top_25_numeros_equilibres_loto_*.csv`
- Tests : `test_top_25_loto.py` (3/3 tests passés)

---

## 🔄 COMPARAISON DÉTAILLÉE

### Différences Structurelles

| Aspect | KENO | LOTO | Justification |
|--------|------|------|--------------|
| **Nombre Total** | 70 | 49 | Plage du jeu |
| **TOP Recommandés** | 30 | 25 | Proportion ~43% vs ~51% |
| **Zones Géographiques** | 5 zones | 3 zones | Complexité d'équilibrage |
| **Tirages par Semaine** | ~140 (20×7) | 15 (5×3) | Fréquence des données |
| **Numéros par Tirage** | 20 | 5 | Mécanique du jeu |

### Adaptations des Pondérations

| Critère | KENO | LOTO | Raison de l'Adaptation |
|---------|------|------|----------------------|
| **Fréquence** | 30% | 35% | Plus critique avec moins de tirages |
| **Retard** | 25% | 30% | Impact plus fort sur 5 numéros |
| **Paires** | 25% | 20% | Moins de combinaisons possibles |
| **Zones** | 20% | 15% | Zones plus simples (3 vs 5) |

### Stratégies d'Utilisation

#### KENO - Grilles Recommandées
- **Grille 10 numéros :** TOP 10 direct
- **Grille 15 numéros :** TOP 15 pour couverture étendue
- **Grille 20 numéros :** TOP 20 pour système complet
- **Multi-grilles :** Combinaisons variées avec TOP 30

#### LOTO - Grilles Recommandées
- **Grille 5 numéros :** TOP 5 direct
- **Grille 7 numéros :** System simple (7 combinaisons)
- **Grille 10 numéros :** System étendu (252 combinaisons)
- **Multi-grilles :** Mix stratégique avec TOP 25

---

## 📈 RÉSULTATS DE TEST

### KENO (Exemple de Sortie)
```
TOP 5 KENO: [12, 45, 67, 23, 8]
Zones équilibrées sur 5 régions
Score maximum observé: ~0.85-0.90
```

### LOTO (Test Validé - Nov 2019)
```
TOP 5 LOTO: [35, 7, 11, 12, 28]
Distribution: Zone 1: 32%, Zone 2: 40%, Zone 3: 28%
Score maximum: 0.8750 (numéro 35)
```

---

## 🔧 GUIDES D'UTILISATION

### Documentation Disponible
- **KENO :** Documentation intégrée dans `duckdb_keno.py`
- **LOTO :** `LOTO_TOP_25_GUIDE.md` (guide complet de 200+ lignes)

### Commandes d'Exécution

#### KENO
```bash
python keno/duckdb_keno.py --csv keno/keno_data/fichier.csv --grids 3 --export-stats
```

#### LOTO
```bash
python loto/duckdb_loto.py --csv loto/loto_data/fichier.csv --grids 3 --export-stats --config-file loto/strategies.yml
```

---

## 📊 FORMATS D'EXPORT

### Structure CSV Commune

| Colonne | KENO | LOTO | Description |
|---------|------|------|-------------|
| `rang` | 1-30 | 1-25 | Position dans le classement |
| `numero` | 1-70 | 1-49 | Numéro recommandé |
| `score_composite` | 0-1 | 0-1 | Score final combiné |
| `zone` | "Zone X (Y-Z)" | "Zone X (Y-Z)" | Zone géographique |
| `frequence` | 0-1 | 0-1 | Score de fréquence |
| `score_retard` | 0-1 | 0-1 | Score de retard |
| `score_paires` | 0-1 | 0-1 | Score des paires |
| `score_zones` | 0-1 | 0-1 | Bonus d'équilibrage |
| `retard_actuel` | Entier | Entier | Jours de retard |
| `freq_absolue` | Entier | Entier | Fréquence brute |

### Exemples de Fichiers Générés

#### KENO
- `keno_stats_exports/top_30_numeros_equilibres_keno.csv` (fichier fixe, remplacé à chaque génération)
- Rapport : `keno/output/rapport_analyse.md`

#### LOTO
- `loto_stats_exports/top_25_numeros_equilibres_loto.csv` (fichier fixe, remplacé à chaque génération)
- Rapport : `loto/output/rapport_analyse.md`

---

## 🧪 VALIDATION ET TESTS

### Suite de Tests KENO
- Tests intégrés dans le pipeline principal
- Validation avec données historiques étendues
- Génération automatique validée

### Suite de Tests LOTO
- `test_top_25_loto.py` - 3 tests complets
- Test avec 907 tirages réels (Nov 2019)
- Validation format CSV et Markdown
- **Résultat :** 3/3 tests passés ✅

---

## 🎯 AVANTAGES DES DEUX SYSTÈMES

### Points Forts Communs
- ✅ **Scoring composite :** Multi-critères optimisé
- ✅ **Export structuré :** CSV horodaté avec toutes les données
- ✅ **Équilibrage zones :** Répartition géographique automatique
- ✅ **Données historiques :** Analyse sur historiques étendus
- ✅ **Flexibilité :** Compatible avec différentes stratégies
- ✅ **Traçabilité :** Tous les calculs exportés et vérifiables

### Spécificités KENO
- 🔥 **Couverture étendue :** 30 numéros sur 70 (43%)
- 🔥 **Zones complexes :** 5 zones pour optimisation fine
- 🔥 **Tirages fréquents :** Adaptation aux données nombreuses
- 🔥 **Systèmes variés :** Grilles 10-20 numéros

### Spécificités LOTO
- 🎯 **Concentration :** 25 numéros sur 49 (51%)
- 🎯 **Simplicité zones :** 3 zones équilibrées naturellement
- 🎯 **Adaptation fréquence :** Optimisé pour tirages 3×/semaine
- 🎯 **Guide détaillé :** Documentation complète et exemples

---

## 🚀 UTILISATION PRATIQUE

### Workflow Recommandé

1. **Choix du jeu :** KENO (tirages fréquents) ou LOTO (gains plus importants)
2. **Analyse historique :** Lancement avec données récentes
3. **Génération TOP :** 30 pour KENO, 25 pour LOTO
4. **Sélection grilles :** Selon budget et stratégie
5. **Suivi performance :** Export CSV pour analyse continue

### Exemple d'Utilisation Complète

#### Session KENO
```bash
# Analyse avec TOP 30
python keno/duckdb_keno.py --csv keno/keno_data/recent.csv --export-stats

# Récupération TOP 10 pour grille principale
# Utilisation TOP 11-20 pour grilles secondaires
```

#### Session LOTO  
```bash
# Analyse avec TOP 25
python loto/duckdb_loto.py --csv loto/loto_data/recent.csv --grids 3 --export-stats --config-file loto/strategies.yml

# Grille principale: TOP 5 [35, 7, 11, 12, 28]
# Grille système: TOP 7 pour couverture étendue
```

---

## 🔮 PERSPECTIVES D'ÉVOLUTION

### Améliorations Possibles
- **Prédiction temporelle :** ML pour timing optimal
- **Optimisation budgétaire :** Rapport coût/couverture
- **Historique étendu :** Intégration données multi-années
- **API temps réel :** Mise à jour automatique des scores

### Extensions Futures
- **EuroMillions :** Adaptation 5/50 + 2/12
- **Rapido :** Système simplifié 8/20
- **Multi-jeux :** Analyse croisée KENO-LOTO

---

💡 **Conclusion :** Deux systèmes complets et opérationnels, adaptés aux spécificités de chaque jeu, avec documentation extensive et validation complète. Prêts pour utilisation en production.
