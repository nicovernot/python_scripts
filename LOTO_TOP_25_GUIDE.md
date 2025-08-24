# 🎯 GUIDE - TOP 25 NUMÉROS ÉQUILIBRÉS LOTO

## 🌟 PRÉSENTATION

Cette nouvelle fonctionnalité génère automatiquement les **25 numéros avec le plus de chances de sortir** selon une stratégie équilibrée adaptée au Loto Français (5 numéros sur 49). L'analyse combine plusieurs facteurs statistiques optimisés pour maximiser les probabilités de gains.

## 📊 MÉTHODOLOGIE SPÉCIFIQUE LOTO

### Algorithme de Scoring Composite

Le score de chaque numéro est calculé selon la formule adaptée au Loto :
```
Score = (Fréquence × 0.35) + (Retard × 0.30) + (Paires × 0.20) + (Zones × 0.15)
```

### Détail des Critères (adaptés au Loto)

| Critère | Poids | Description | Spécificité Loto |
|---------|-------|-------------|-----------------|
| **Fréquence** | 35% | Apparition historique du numéro | Plus important car moins de tirages |
| **Retard** | 30% | Nombre de jours depuis dernière sortie | Crucial pour 5 numéros sur 49 |
| **Paires** | 20% | Performance moyenne dans les meilleures paires | Associations fréquentes |
| **Zones** | 15% | Bonus d'équilibrage géographique | 3 zones équilibrées |

### Zones Géographiques Loto

- **Zone 1 (1-17) :** 17 numéros - Région basse
- **Zone 2 (18-34) :** 17 numéros - Région moyenne  
- **Zone 3 (35-49) :** 15 numéros - Région haute

## 🚀 UTILISATION

### Lancement de l'Analyse

```bash
# Avec un fichier spécifique et stratégie équilibrée
python loto/duckdb_loto.py --csv loto/loto_data/fichier.csv --grids 3 --export-stats --config-file loto/strategies.yml

# Avec stratégie personnalisée
python loto/duckdb_loto.py --csv loto/loto_data/fichier.csv --strategy focus_retard --export-stats --config-file loto/strategies.yml
```

### Fichiers Générés

1. **CSV détaillé :** `loto_stats_exports/top_25_numeros_equilibres_loto.csv` (fichier fixe, remplacé à chaque génération)
2. **Rapport Markdown :** `loto/output/rapport_analyse.md`
3. **Grilles générées :** `loto_stats_exports/grilles.csv`

## 📋 FORMAT DE SORTIE CSV

### Structure des Colonnes

| Colonne | Description | Type | Spécificité Loto |
|---------|-------------|------|-----------------|
| `rang` | Position dans le classement (1-25) | Integer | Top 25 vs 30 pour Keno |
| `numero` | Numéro Loto (1-49) | Integer | Plage réduite vs 70 pour Keno |
| `score_composite` | Score final combiné | Float (0-1) | Pondération adaptée |
| `zone` | Zone géographique | String | 3 zones équilibrées |
| `frequence` | Score de fréquence normalisé | Float (0-1) | Poids 35% |
| `score_retard` | Score de retard normalisé | Float (0-1) | Poids 30% |
| `score_paires` | Score moyen des paires | Float (0-1) | Poids 20% |
| `score_zones` | Bonus d'équilibrage | Float (0-1) | Poids 15% |
| `retard_actuel` | Nombre de jours de retard | Integer | Spécifique temporel |
| `freq_absolue` | Fréquence absolue d'apparition | Integer | Données brutes |

### Exemple de Données

```csv
rang;numero;score_composite;zone;frequence;score_retard;score_paires;score_zones;retard_actuel;freq_absolue
1;35;0.8750;Zone 3 (35-49);0.8807;1.0000;0.8116;0.6964;118;96
2;7;0.7912;Zone 1 (1-17);0.9266;0.6271;0.9058;0.6507;74;101
3;11;0.7414;Zone 1 (1-17);0.7706;0.7203;0.7899;0.6507;85;84
```

## 🎲 STRATÉGIES D'UTILISATION LOTO

### Grilles de 5 Numéros (Standard)

**Option 1 - TOP 5 Direct :**
- Sélectionnez les 5 premiers numéros du classement
- Couverture optimale avec score maximal
- Exemple : `[35, 7, 11, 12, 28]`

**Option 2 - Mix Stratégique :**
- 2-3 numéros du TOP 5 (scores > 0.70)
- 2-3 numéros du TOP 6-15 (diversification)
- Équilibrage des zones recommandé

### Grilles de 7 Numéros (Système Simple)

- Utilisez les 7 premiers numéros du classement
- Génère 7 combinaisons de 5 numéros
- Exemple : `[35, 7, 11, 12, 28, 46, 17]`
- Couverture équilibrée entre toutes les zones

### Grilles de 10 Numéros (Système Étendu)

- Prenez les 10 premiers numéros
- Génère 252 combinaisons de 5 numéros
- Couverture maximale pour investissement contrôlé
- Exemple : `[35, 7, 11, 12, 28, 46, 17, 15, 6, 24]`

### Grilles Multiples (Système Avancé)

- Utilisez les 15-20 premiers numéros
- Créez plusieurs grilles en alternant les sélections
- Équilibrez entre les différentes zones
- Optimisez selon votre budget

## 📍 ÉQUILIBRAGE DES ZONES LOTO

### Répartition Optimale Observée

Dans l'exemple testé (TOP 25) :
- **Zone 1 (1-17) :** 8 numéros (32%)
- **Zone 2 (18-34) :** 10 numéros (40%)  
- **Zone 3 (35-49) :** 7 numéros (28%)

### Recommandations par Grille

**Grille 5 numéros :**
- Zone 1 : 1-2 numéros
- Zone 2 : 2-3 numéros
- Zone 3 : 1-2 numéros

**Grille 7 numéros :**
- Zone 1 : 2-3 numéros
- Zone 2 : 3-4 numéros
- Zone 3 : 2-3 numéros

## 🔍 INTERPRÉTATION DES SCORES

### Scores Composites Loto

| Score | Interprétation | Action Recommandée |
|-------|----------------|-------------------|
| > 0.80 | Excellent potentiel | Priorité absolue |
| 0.70-0.80 | Très bon potentiel | Fortement recommandé |
| 0.60-0.70 | Bon potentiel | Recommandé |
| < 0.60 | Potentiel moyen | À considérer selon stratégie |

### Scores Individuels

- **Fréquence = 1.0 :** Numéro le plus fréquent historiquement
- **Retard = 1.0 :** Retard maximum (forte probabilité de sortie)
- **Paires = 1.0 :** Excellente performance en associations
- **Zones = 1.0 :** Zone prioritaire pour l'équilibrage

## 📈 AVANTAGES DE LA MÉTHODE LOTO

### ✅ Points Forts Spécifiques

1. **Adaptation Loto :** Pondération optimisée pour 5/49
2. **3 Zones équilibrées :** Répartition géographique naturelle
3. **Historique important :** Analyse sur données étendues
4. **Flexibilité systèmes :** Compatible avec tous types de grilles
5. **Export structuré :** Données traçables et analysables

### 🎯 Cas d'Usage Optimaux

- **Joueurs réguliers :** Stratégie cohérente et documentée
- **Systèmes complexes :** Base solide pour grilles multiples
- **Analyse budgétaire :** Optimisation coût/couverture
- **Suivi temporel :** Évolution des scores dans le temps

## 🔄 COMPARAISON KENO vs LOTO

### Différences Techniques

| Aspect | Keno | Loto | Justification |
|--------|------|------|--------------|
| **Nombres** | 1-70 (TOP 30) | 1-49 (TOP 25) | Plage plus restreinte |
| **Fréquence** | 30% | 35% | Plus critique au Loto |
| **Retard** | 25% | 30% | Impact plus fort |
| **Zones** | 5 zones | 3 zones | Équilibrage simplifié |
| **Tirages** | 20 par jour | 3 par semaine | Fréquence différente |

### Adaptations Spécifiques

- **Pondération :** Fréquence et retard plus importants
- **Zones :** Découpage en 3 zones équilibrées vs 5 pour Keno
- **Export :** TOP 25 vs TOP 30 pour s'adapter à la plage
- **Suggestions :** Grilles 5, 7, 10 vs 10, 15, 20 pour Keno

## 🎮 EXEMPLES PRATIQUES

### Résultat Test (Données Nov 2019)

**TOP 5 Recommandés :** 35, 7, 11, 12, 28

**Analyse :**
- Score maximum : 0.8750 (numéro 35)
- Répartition : 2 en Zone 1, 2 en Zone 2, 1 en Zone 3
- Équilibrage pairs/impairs : 3 impairs, 2 pairs

### Utilisation Pratique

1. **Grille Optimale :** Utilisez directement [35, 7, 11, 12, 28]
2. **Grille Alternative :** Remplacez 1-2 numéros par le TOP 6-10
3. **Système 7 :** Ajoutez [46, 17] pour couverture étendue

---

💡 **Conseil :** Combinez cette analyse avec les stratégies configurables (equilibre, focus_retard, etc.) pour une approche multi-angles adaptée à votre style de jeu.
