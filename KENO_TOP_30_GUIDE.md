# 🎯 GUIDE - TOP 30 NUMÉROS ÉQUILIBRÉS KENO

## 🌟 PRÉSENTATION

Cette nouvelle fonctionnalité génère automatiquement les **30 numéros avec le plus de chances de sortir** selon une stratégie équilibrée multi-critères. L'analyse combine plusieurs facteurs statistiques pour optimiser les probabilités de gains.

## 📊 MÉTHODOLOGIE

### Algorithme de Scoring Composite

Le score de chaque numéro est calculé selon la formule :
```
Score = (Fréquence × 0.30) + (Retard × 0.25) + (Paires × 0.25) + (Zones × 0.20)
```

### Détail des Critères

| Critère | Poids | Description |
|---------|-------|-------------|
| **Fréquence** | 30% | Apparition historique du numéro (normalisée) |
| **Retard** | 25% | Nombre de tirages depuis dernière sortie |
| **Paires** | 25% | Performance moyenne dans les meilleures paires |
| **Zones** | 20% | Bonus d'équilibrage géographique (1-23, 24-46, 47-70) |

## 🚀 UTILISATION

### Lancement de l'Analyse

```bash
# Avec un fichier spécifique
python keno/duckdb_keno.py --csv keno/keno_data/keno_202501.csv --export-stats

# Avec fichier consolidé (recommandé)
python keno/duckdb_keno.py --auto-consolidated --export-stats
```

### Fichiers Générés

1. **CSV détaillé :** `keno_stats_exports/top_30_numeros_equilibres_YYYYMMDD_HHMMSS.csv`
2. **Rapport Markdown :** Section dédiée dans `keno_output/recommandations_keno.md`

## 📋 FORMAT DE SORTIE CSV

### Structure des Colonnes

| Colonne | Description | Type |
|---------|-------------|------|
| `rang` | Position dans le classement (1-30) | Integer |
| `numero` | Numéro Keno (1-70) | Integer |
| `score_composite` | Score final combiné | Float (0-1) |
| `zone` | Zone géographique | String |
| `frequence` | Score de fréquence normalisé | Float (0-1) |
| `score_retard` | Score de retard normalisé | Float (0-1) |
| `score_paires` | Score moyen des paires | Float (0-1) |
| `score_zones` | Bonus d'équilibrage | Float (0-1) |
| `retard_actuel` | Nombre de tirages de retard | Integer |
| `freq_absolue` | Fréquence absolue d'apparition | Integer |

### Exemple de Données

```csv
rang;numero;score_composite;zone;frequence;score_retard;score_paires;score_zones;retard_actuel;freq_absolue
1;64;0.7427;Zone 3 (47-70);0.6296;0.9286;0.75;0.671;13;17
2;38;0.7286;Zone 2 (24-46);1.0;0.2143;0.975;0.6565;3;27
3;11;0.6955;Zone 1 (1-23);0.8889;0.2857;0.8917;0.6726;4;24
```

## 🎲 STRATÉGIES D'UTILISATION

### Grilles de 10 Numéros (Standard)

**Option 1 - TOP 10 Direct :**
- Sélectionnez les 10 premiers numéros du classement
- Couverture optimale avec score maximal

**Option 2 - Mix Stratégique :**
- 3-4 numéros du TOP 5 (scores > 0.65)
- 6-7 numéros du TOP 11-20 (diversification)

### Grilles de 15 Numéros (Recommandé)

- Utilisez les 15 premiers numéros du classement
- Garantit une répartition équilibrée entre toutes les stratégies
- Meilleur rapport probabilité/investissement

### Grilles de 20 Numéros (Couverture Maximum)

- Prenez les 20 premiers numéros
- Couverture optimale pour maximiser les chances
- Idéal pour un investissement mesuré

## 📍 ÉQUILIBRAGE DES ZONES

### Répartition Géographique

- **Zone 1 (1-23) :** Numéros bas
- **Zone 2 (24-46) :** Numéros moyens  
- **Zone 3 (47-70) :** Numéros hauts

### Optimisation Automatique

L'algorithme applique un bonus aux zones sous-représentées pour maintenir un équilibre géographique naturel dans les sélections.

## 🔍 INTERPRÉTATION DES SCORES

### Scores Composites

| Score | Interprétation | Action Recommandée |
|-------|----------------|-------------------|
| > 0.70 | Excellent potentiel | Priorité absolue |
| 0.60-0.70 | Très bon potentiel | Fortement recommandé |
| 0.50-0.60 | Bon potentiel | Recommandé |
| < 0.50 | Potentiel moyen | À considérer selon stratégie |

### Scores Individuels

- **Fréquence = 1.0 :** Numéro le plus fréquent historiquement
- **Retard = 1.0 :** Retard maximum (forte probabilité de sortie)
- **Paires = 1.0 :** Excellente performance en associations
- **Zones = 1.0 :** Zone prioritaire pour l'équilibrage

## 📈 AVANTAGES DE LA MÉTHODE

### ✅ Points Forts

1. **Multi-critères :** Combine 4 analyses différentes
2. **Équilibré :** Évite la sur-pondération d'un seul facteur
3. **Adaptatif :** S'ajuste automatiquement aux données
4. **Documenté :** Export détaillé de tous les calculs
5. **Flexible :** Utilisable pour différentes tailles de grilles

### 🎯 Cas d'Usage Optimaux

- **Joueurs réguliers :** Stratégie cohérente et documentée
- **Analyses statistiques :** Données exportables et traçables
- **Optimisation budgétaire :** Sélection ciblée selon l'investissement
- **Études probabilistes :** Base solide pour recherches avancées

## 🔄 MISE À JOUR DES DONNÉES

### Fréquence Recommandée

- **Mensuelle :** Pour données consolidées complètes
- **Hebdomadaire :** Pour ajustements fins
- **Avant gros jeux :** Pour optimisation maximale

### Automatisation

Le système génère automatiquement les fichiers avec horodatage, permettant de suivre l'évolution des scores dans le temps.

---

💡 **Conseil :** Combinez cette analyse avec les autres stratégies du rapport (MIX INTELLIGENT, MONTE CARLO) pour une approche multi-angles optimale.
