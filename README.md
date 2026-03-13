# Système Loto — Générateur IA

Générateur de grilles Loto basé sur des statistiques Bayésiennes, un modèle XGBoost et une optimisation PuLP.

## Démarrage rapide

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Générateur Bayésien (recommandé)
make bayesian

# Menu interactif complet
make menu
# ou
python cli_menu.py
```

## Architecture du pipeline

```
loto/loto_data/loto_201911.csv
        │
        ▼
   [DuckDB]  ──────────────────────────────────────────────┐
  Statistiques : fréquence, retard, momentum,              │
  co-occurrences (prior Bayésien)                          │
        │                                                   │
        ▼                                                   │
  [XGBoost]                                                 │
  Features : freq×4 fenêtres, retard, zone,                │
  parité, momentum, co-occurrence hot numbers              │
  → vraisemblance P(n | features)                          │
        │                                                   │
        ▼                                                   │
  [Posterior Bayésien]                                      │
  P(n | history) ∝ Prior × Likelihood × CoocBonus          │
        │                                                   │
        ▼                                                   │
  TOP 10 numéros ──────────────────────────────────────────┘
        │
        ▼
   [PuLP — MILP]
  6 grilles optimisées
  Contraintes : zone (BAS/MID/HAUT), parité, diversité
        │
        ▼
  loto_output/bayesian_grilles.json
```

## Structure du projet

```
loto_keno/
├── Makefile                        # Toutes les commandes (make help)
├── cli_menu.py                     # Menu interactif
├── lancer_menu.sh                  # Raccourci bash
│
├── loto/
│   ├── loto_data/
│   │   └── loto_201911.csv         # Données historiques (994 tirages, 2019→2026)
│   ├── generateur_bayesien_loto.py # Générateur principal (Bayésien + PuLP)
│   ├── duckdb_loto.py              # Générateur classique (stratégies YAML)
│   ├── loto_generator_advanced_Version2.py  # Générateur ML avancé (simulations)
│   ├── strategies.yml              # Stratégies pondérées
│   └── strategies_ml.yml           # Stratégies ML
│
├── boost_models/
│   ├── xgb_bayesian_loto.json      # Modèle XGBoost Bayésien (entraîné)
│   ├── xgb_ball_1-5.pkl            # Modèles XGBoost par boule
│   └── model_balls_multilabel.joblib
│
├── grilles/
│   ├── generateur_grilles.py       # Générateur système réduit
│   ├── mes_nombres_favoris.txt     # Numéros personnalisés
│   └── sorties/                    # Grilles exportées
│
├── loto_output/                    # Résultats JSON des générations
├── loto_stats_exports/             # Exports CSV des statistiques
├── loto_analyse_plots/             # Graphiques matplotlib
│
├── api/                            # API Flask
├── test/                           # Tests pytest
└── docs/                           # Documentation technique
```

## Commandes Makefile

```bash
make help            # Liste toutes les commandes

# Générateur Bayésien
make bayesian        # 6 grilles (modèle en cache)
make retrain         # 6 grilles + ré-entraînement XGBoost

# Générateur classique
make loto            # 3 grilles (stratégie équilibrée)
make loto-full       # 5 grilles + export stats
make loto-plots      # 3 grilles + visualisations

# Générateur avancé ML
make advanced-quick  # 1 000 simulations
make advanced        # 10 000 simulations
make advanced-heavy  # 50 000 simulations

# Données & maintenance
make data            # Télécharge les données FDJ
make top25           # Génère le TOP 25 numéros équilibrés
make status          # Vérifie l'environnement
make test            # Lance les tests
make clean           # Supprime fichiers temporaires
make clean-all       # Reset complet (modèles + exports)
```

## Menu interactif (options)

| # | Action |
|---|--------|
| 1 | Télécharger données FDJ |
| 2–5 | Génération rapide / classique / personnalisée |
| **6** | **Générateur Bayésien — 6 grilles optimisées (recommandé)** |
| **7** | **Générateur Bayésien avec ré-entraînement** |
| 8–11 | Générateur avancé ML (simulations) |
| 12–15 | TOP 25 numéros + grilles CSV/Markdown |
| 16–17 | Système réduit (numéros favoris) |
| 18–20 | Résultats, graphiques, statut |
| 21–22 | API Flask |

## Générateur Bayésien — détail

**Fichier :** `loto/generateur_bayesien_loto.py`

```bash
python loto/generateur_bayesien_loto.py           # modèle en cache
python loto/generateur_bayesien_loto.py --retrain # ré-entraînement forcé
```

Le pipeline en 7 étapes :

1. **DuckDB** — lecture CSV + statistiques SQL (fréquence, retard, momentum, co-occurrences)
2. **Prior Bayésien** — fréquence empirique lissée (lissage de Laplace)
3. **Features XGBoost** — 18 features : fréquences multi-fenêtres (10/30/60/100 tirages), retard normalisé, zone, parité, momentum, co-occurrence avec numéros chauds
4. **XGBoost** — classification binaire, validation TimeSeriesSplit 3-fold, sauvegardé dans `boost_models/xgb_bayesian_loto.json`
5. **Posterior** — `P(n|history) ∝ Prior × Vraisemblance × (1 + 0.15 × CoocScore)`
6. **TOP 10** — 10 numéros avec posterior le plus élevé + meilleur numéro chance
7. **PuLP (MILP)** — 6 grilles optimisées, contraintes : ≥1 numéro par zone, 2–4 impairs, ≤3 numéros communs entre grilles

Sortie : `loto_output/bayesian_grilles.json`

## Dépendances principales

| Package | Rôle |
|---------|------|
| `duckdb` | Statistiques SQL sur le CSV |
| `xgboost` | Modèle de vraisemblance |
| `pulp` | Optimisation MILP des grilles |
| `scipy` | Distributions statistiques |
| `scikit-learn` | Validation croisée temporelle |
| `pandas / numpy` | Manipulation des données |
| `matplotlib` | Visualisations |
| `flask` | API REST |

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
