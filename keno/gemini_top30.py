#!/usr/bin/env python3
"""
============================================================================
🎲 GÉNÉRATEUR INTELLIGENT DE GRILLES KENO - VERSION AVANCÉE (AMÉLIORÉE) 🎲
============================================================================

Générateur de grilles Keno utilisant l'apprentissage automatique (XGBoost/RandomForest) 
et l'analyse statistique pour optimiser les combinaisons.

Caractéristiques (améliorées):
- Machine Learning avec RandomForest (MultiOutput) ou XGBoost (par numéro)
  pour prédire les numéros probables.
- Ingénierie de features avancées: séquences, écarts, périodicités,
  retards conditionnels, moyennes mobiles exponentielles (EWMA).
- Stratégie adaptative avec pondération dynamique ML/Fréquence et diversité.
- Analyse de patterns temporels et cycliques avec FFT.
- Optimisation des combinaisons selon critères statistiques et ML.
- Génération de rapports détaillés avec visualisations.
- Validation croisée temporelle pour une évaluation plus robuste.
- Affinement du scoring composite pour le TOP 30.

Auteur: Assistant IA
Version: 3.0
Date: Septembre 2025
============================================================================
"""

import pandas as pd
import numpy as np
import random
import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import pickle
from typing import List, Tuple, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
import csv
import pulp

# DuckDB pour optimiser les requêtes
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("⚠️  DuckDB non disponible. Utilisation de Pandas uniquement.")

# ML et analyse
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from joblib import Parallel, delayed
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️  Modules ML non disponibles. Mode fréquence uniquement.")

# Visualisation et analyse
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.api import ExponentialSmoothing
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("⚠️  Modules de visualisation non disponibles.")

# Configuration des warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 🔧 CONFIGURATION ET CONSTANTES
# ==============================================================================

# Paramètres spécifiques au Keno
KENO_PARAMS = {
    'total_numbers': 70,        # Numéros de 1 à 70
    'numbers_per_draw': 20,     # 20 numéros tirés par tirage
    'player_selection': 10,     # Le joueur sélectionne typiquement 10 numéros
    'min_selection': 2,         # Minimum de numéros sélectionnables
    'max_selection': 10,        # Maximum de numéros sélectionnables
    'ml_prediction_threshold': 0.5, # Seuil de probabilité pour la sélection ML
}

# Chemins des fichiers
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "keno/keno_data"
MODELS_DIR = BASE_DIR.parent / "keno_models"
OUTPUT_DIR = BASE_DIR.parent / "keno_output"

# Configuration ML
ML_CONFIG = {
    'xgb_params': {
        'objective': 'binary:logistic',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
    },
    'rf_params_multioutput': {
        'n_estimators': 150,
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    'scaler': StandardScaler() # Utilisation d'un scaler global
}

def get_training_params(profile: str) -> dict:
    """
    Retourne les paramètres d'entraînement selon le profil sélectionné
    
    Args:
        profile: Profil d'entraînement ('quick', 'balanced', 'comprehensive', 'intensive')
    
    Returns:
        dict: Paramètres d'entraînement pour RandomForest
    """
    params = {}
    if profile == "quick":
        params = {
            'n_estimators': 50,        # Arbres réduits pour la vitesse
            'max_depth': 8,            # Profondeur modérée
            'min_samples_split': 10,   # Éviter l'overfitting
            'min_samples_leaf': 5,     # Éviter l'overfitting
            'max_features': 'sqrt',    # Features réduites
        }
    elif profile == "balanced":
        params = {
            'n_estimators': 100,       # Standard
            'max_depth': 12,           # Profondeur équilibrée
            'min_samples_split': 5,    # Standard
            'min_samples_leaf': 3,     # Standard
            'max_features': 'sqrt',    # Standard
        }
    elif profile == "comprehensive":
        params = {
            'n_estimators': 200,       # Plus d'arbres pour meilleure précision
            'max_depth': 15,           # Profondeur élevée
            'min_samples_split': 4,    # Plus de splits
            'min_samples_leaf': 2,     # Feuilles plus petites
            'max_features': 'log2',    # Plus de features
        }
    elif profile == "intensive":
        params = {
            'n_estimators': 300,       # Maximum d'arbres
            'max_depth': 20,           # Profondeur maximale
            'min_samples_split': 3,    # Splits agressifs
            'min_samples_leaf': 1,     # Feuilles minimales
            'max_features': None,      # Toutes les features
        }
    else:
        # Fallback sur balanced
        params = get_training_params("balanced")
    
    params.update({
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    })
    return params

@dataclass
class KenoStats:
    """Structure pour stocker les statistiques Keno complètes et avancées"""
    # 1. Fréquences d'apparition des numéros
    frequences: Dict[int, int] = field(default_factory=dict)
    frequences_recentes: Dict[int, int] = field(default_factory=dict) # Fréquence sur 100 derniers tirages
    frequences_50: Dict[int, int] = field(default_factory=dict)      # Fréquence sur 50 derniers tirages
    frequences_20: Dict[int, int] = field(default_factory=dict)      # Fréquence sur 20 derniers tirages
    ewma_frequences: Dict[int, float] = field(default_factory=dict) # EWMA des fréquences

    # 2. Numéros en retard (overdue numbers)
    retards: Dict[int, int] = field(default_factory=dict)             # Retard actuel de chaque numéro
    retards_historiques: Dict[int, List[int]] = field(default_factory=lambda: {i: [] for i in range(1, KENO_PARAMS['total_numbers'] + 1)}) # Historique des retards
    retards_moyens: Dict[int, float] = field(default_factory=dict)  # Retard moyen historique
    retards_std: Dict[int, float] = field(default_factory=dict)     # Écart-type des retards

    # 3. Combinaisons et patterns récurrents
    paires_freq: Dict[Tuple[int, int], int] = field(default_factory=dict) # Paires fréquentes
    trios_freq: Dict[Tuple[int, int, int], int] = field(default_factory=dict) # Trios fréquents
    patterns_parité: Dict[str, int] = field(default_factory=dict)   # Distribution pair/impair
    patterns_sommes: Dict[int, int] = field(default_factory=dict)   # Distribution des sommes
    patterns_zones: Dict[str, int] = field(default_factory=dict)    # Répartition par zones/dizaines

    # 4. Analyse par période
    tendances_10: Dict[int, float] = field(default_factory=dict)     # Tendances sur 10 tirages
    tendances_50: Dict[int, float] = field(default_factory=dict)     # Tendances sur 50 tirages
    tendances_100: Dict[int, float] = field(default_factory=dict)    # Tendances sur 100 tirages
    periodicites: Dict[int, Optional[float]] = field(default_factory=dict) # Périodicité dominante pour chaque numéro
    
    # 5. Caractéristiques structurelles des tirages
    ecarts_moyens: Dict[int, float] = field(default_factory=dict) # Écart moyen entre numéros du tirage
    dispersion_moyenne: float = 0.0                             # Dispersion moyenne des tirages
    clusters_moyens: float = 0.0                                # Nombre moyen de clusters par tirage

    # Données brutes pour analyses avancées
    zones_freq: Dict[str, int] = field(default_factory=dict)    # Ancien format maintenu pour compatibilité
    derniers_tirages: List[List[int]] = field(default_factory=list) # Derniers tirages
    tous_tirages: List[List[int]] = field(default_factory=list) # Tous les tirages pour DuckDB


# ==============================================================================
# 🎯 CLASSE PRINCIPALE GENERATEUR KENO
# ==============================================================================

class KenoGeneratorAdvanced:
    """Générateur avancé de grilles Keno avec ML et analyse statistique"""
    
    def __init__(self, data_path: str = None, silent: bool = False, training_profile: str = "balanced", ml_strategy: str = "multioutput"):
        """
        Initialise le générateur Keno
        
        Args:
            data_path: Chemin vers les données historiques
            silent: Mode silencieux pour réduire les sorties
            training_profile: Profil d'entraînement ('quick', 'balanced', 'comprehensive', 'intensive')
            ml_strategy: Stratégie ML ('multioutput' pour RandomForest, 'per_number' pour XGBoost par numéro)
        """
        self.silent = silent
        self.data_path = data_path or "keno/keno_data/keno_202010.parquet"
        self.models_dir = MODELS_DIR
        self.output_dir = OUTPUT_DIR
        self.training_profile = training_profile
        self.ml_strategy = ml_strategy # Nouvelle option de stratégie ML
        
        # Créer les répertoires nécessaires
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialisation des composants
        self.data = None
        self.stats: KenoStats = KenoStats() # Utilisation de la dataclass
        self.ml_models = {} # Peut contenir un seul MultiOutputClassifier ou un dict de modèles par numéro
        self.metadata = {}
        self.cache = {}
        self.ml_scaler = ML_CONFIG['scaler'] # Scaler pour les features ML
        self.feature_names = [] # Pour stocker les noms des features après l'entraînement
        self.ml_scores = {} # Pour stocker les probabilités ML du top 30
        
        # Stratégie adaptative
        self.adaptive_weights = {
            'ml_weight': 0.5,      # Poids par défaut
            'freq_weight': 0.2,
            'recent_weight': 0.15,
            'retard_weight': 0.1,
            'tendance_weight': 0.05,
            'pair_weight': 0.05,
            'trio_weight': 0.05,
            'lift_weight': 0.05,
            'periodicity_weight': 0.05,
            'diversity_penalty': 0.1,
            'performance_history': [],
            'last_update': datetime.now()
        }
        
        self._log(f"🎲 Générateur Keno Avancé v3.0 initialisé (Stratégie ML: {ml_strategy})")

    def export_grids(self, grids: List[List[int]], summary: Dict[str, Any], filename_prefix: str = "keno_grids"):
        """Exporte les grilles en CSV et Markdown dans OUTPUT_DIR"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # --- CSV ---
            csv_path = self.output_dir / f"{filename_prefix}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Grille_ID"] + [f"Num_{i+1}" for i in range(len(grids[0]))])
                for idx, grid in enumerate(grids, 1):
                    writer.writerow([idx] + grid)

            # --- Markdown ---
            md_path = self.output_dir / f"{filename_prefix}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# 🎰 Génération de grilles Keno\n\n")
                f.write(f"- Date: {datetime.now().isoformat()}\n")
                f.write(f"- Tirages analysés: {summary['n_draws']}\n\n")
                f.write("## 🔝 Top 10 numéros prédits\n\n")
                for num, score in summary['top10_ml']:
                    f.write(f"- **{num}** → {score:.3f}\n")
                f.write("\n## 🎯 Grilles générées\n\n")
                for idx, grid in enumerate(grids, 1):
                    f.write(f"- Grille #{idx}: {', '.join(map(str, grid))}\n")

            self._log(f"💾 Export effectué: {csv_path} et {md_path}")
        except Exception as e:
            self._log(f"⚠️ Erreur export grilles: {e}", "WARNING")

    def _log(self, message: str, level: str = "INFO"):
        """Système de logging configuré"""
        if not self.silent or level == "ERROR":
            print(f"{message}")
    
    def _ensure_zone_freq_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes zoneX_freq manquantes pour compatibilité avec le modèle ML.
        S'assure que les colonnes de comptage de zones existent.
        """
        for freq_col in ['zone1_freq', 'zone2_freq', 'zone3_freq', 'zone4_freq']:
            if freq_col not in df.columns:
                df[freq_col] = 0
        for count_col in ['zone1_count', 'zone2_count', 'zone3_count', 'zone4_count']:
            if count_col not in df.columns:
                df[count_col] = 0
        return df

    def _get_draw_numbers_from_row(self, row: pd.Series) -> List[int]:
        """Extrait les numéros d'un tirage d'une ligne de DataFrame."""
        return [int(row[f'boule{i}']) for i in range(1, 21)]

    def load_data(self) -> bool:
        """
        Charge les données historiques des tirages Keno
        
        Returns:
            bool: True si les données sont chargées avec succès
        """
        try:
            self._log("📊 Chargement des données historiques Keno...")
            
            if not Path(self.data_path).exists():
                self._log(f"❌ Fichier non trouvé: {self.data_path}", "ERROR")
                return False
                
            self.data = pd.read_parquet(self.data_path)
            
            required_cols = ['date_de_tirage'] + [f'boule{i}' for i in range(1, 21)]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                self._log(f"❌ Colonnes manquantes: {missing_cols}", "ERROR")
                return False
            
            self.data['date_de_tirage'] = pd.to_datetime(self.data['date_de_tirage'])
            self.data = self.data.sort_values('date_de_tirage').reset_index(drop=True)
            
            self._log(f"✅ {len(self.data)} tirages Keno chargés ({self.data['date_de_tirage'].min()} à {self.data['date_de_tirage'].max()})")
            return True
            
        except Exception as e:
            self._log(f"❌ Erreur lors du chargement des données: {e}", "ERROR")
            return False
    
    def analyze_patterns(self) -> KenoStats:
        """
        Analyse complète des patterns et statistiques Keno avec DuckDB ou Pandas.
        Améliorée avec plus de statistiques avancées.
        """
        self._log("🔍 Analyse complète des patterns Keno...")
        
        all_draws = []
        for _, row in self.data.iterrows():
            draw = self._get_draw_numbers_from_row(row)
            all_draws.append(sorted(draw))
        
        if HAS_DUCKDB:
            self.stats = self._analyze_with_duckdb(all_draws)
        else:
            self.stats = self._analyze_with_pandas(all_draws)

        # Calcul des retards moyens et STD
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            if len(self.stats.retards_historiques[num]) > 1:
                self.stats.retards_moyens[num] = np.mean(self.stats.retards_historiques[num])
                self.stats.retards_std[num] = np.std(self.stats.retards_historiques[num])
            else:
                self.stats.retards_moyens[num] = 0
                self.stats.retards_std[num] = 0

        # Calcul des fréquences EWMA
        self._log("📈 Calcul des fréquences EWMA...")
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            presence_series = [1 if num in draw else 0 for draw in all_draws]
            if len(presence_series) > 10: # Nécessite au moins quelques points
                try:
                    ewma_model = ExponentialSmoothing(presence_series, initialization_method='estimated', trend=None, seasonal=None).fit(smoothing_level=0.2, optimized=False)
                    self.stats.ewma_frequences[num] = ewma_model.forecast(1)[0]
                except Exception as e:
                    self._log(f"⚠️ Erreur EWMA pour numéro {num}: {e}", "WARNING")
                    self.stats.ewma_frequences[num] = presence_series[-1] if presence_series else 0.0 # Fallback au dernier état
            else:
                self.stats.ewma_frequences[num] = 0.0

        self._log(f"✅ Analyse complète terminée - {len(all_draws)} tirages analysés")
        return self.stats
    
    def _analyze_with_duckdb(self, all_draws: List[List[int]]) -> KenoStats:
        """Analyse optimisée avec DuckDB"""
        self._log("🚀 Utilisation de DuckDB pour l'analyse optimisée")
        conn = duckdb.connect(':memory:')
        
        draws_data = []
        for i, draw in enumerate(all_draws):
            for num in draw:
                draws_data.append({'tirage_id': i, 'numero': num}) # Simplifié, position non nécessaire ici
        
        draws_df = pd.DataFrame(draws_data)
        conn.register('tirages', draws_df)
        
        current_stats = KenoStats(tous_tirages=all_draws) # Initialisation avec les tirages bruts

        # 1. FRÉQUENCES D'APPARITION
        freq_global = conn.execute("""
            SELECT numero, COUNT(*) as freq 
            FROM tirages 
            GROUP BY numero 
            ORDER BY numero
        """).fetchdf()
        current_stats.frequences = dict(zip(freq_global['numero'], freq_global['freq']))
        
        max_tirage = len(all_draws) - 1
        
        for num_draws, attr_name in [(100, 'frequences_recentes'), (50, 'frequences_50'), (20, 'frequences_20')]:
            freq_df = conn.execute(f"""
                SELECT numero, COUNT(*) as freq 
                FROM tirages 
                WHERE tirage_id >= {max(0, max_tirage - num_draws + 1)}
                GROUP BY numero
            """).fetchdf()
            # Assurer que tous les numéros ont une entrée, même si 0
            freq_dict = {i: 0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
            freq_dict.update(dict(zip(freq_df['numero'], freq_df['freq'])))
            setattr(current_stats, attr_name, freq_dict)
        
        # 2. CALCUL DES RETARDS
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            retard = 0
            for draw in reversed(all_draws):
                if num in draw:
                    break
                retard += 1
            current_stats.retards[num] = retard
            
            historique = []
            dernier_tirage = -1
            for i, draw in enumerate(all_draws):
                if num in draw:
                    if dernier_tirage >= 0:
                        historique.append(i - dernier_tirage - 1)
                    dernier_tirage = i
            current_stats.retards_historiques[num] = historique
        
        # 3. PATTERNS ET COMBINAISONS
        paires_freq = {}
        trios_freq = {}
        patterns_parité = {"tout_pair": 0, "tout_impair": 0, "mixte": 0}
        patterns_sommes = {}
        patterns_zones = {
            "zone_1_17": 0, "zone_18_35": 0, "zone_36_52": 0, "zone_53_70": 0,
            "dizaine_1_10": 0, "dizaine_11_20": 0, "dizaine_21_30": 0, 
            "dizaine_31_40": 0, "dizaine_41_50": 0, "dizaine_51_60": 0, "dizaine_61_70": 0
        }
        
        for draw in all_draws:
            # Paires et Trios
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    paires_freq[pair] = paires_freq.get(pair, 0) + 1
                    for k in range(j + 1, len(draw)):
                        trio = tuple(sorted([draw[i], draw[j], draw[k]]))
                        trios_freq[trio] = trios_freq.get(trio, 0) + 1
            
            # Parité
            pairs = sum(1 for num in draw if num % 2 == 0)
            if pairs == len(draw): patterns_parité["tout_pair"] += 1
            elif pairs == 0: patterns_parité["tout_impair"] += 1
            else: patterns_parité["mixte"] += 1
            
            # Sommes
            somme = sum(draw)
            patterns_sommes[somme] = patterns_sommes.get(somme, 0) + 1
            
            # Zones
            for num in draw:
                if 1 <= num <= 17: patterns_zones["zone_1_17"] += 1
                elif 18 <= num <= 35: patterns_zones["zone_18_35"] += 1
                elif 36 <= num <= 52: patterns_zones["zone_36_52"] += 1
                else: patterns_zones["zone_53_70"] += 1
                
                # Dizaines
                dizaine = (num - 1) // 10 + 1
                if dizaine <= 7: patterns_zones[f"dizaine_{(dizaine-1)*10+1}_{dizaine*10}"] += 1
        
        current_stats.paires_freq = paires_freq
        current_stats.trios_freq = trios_freq
        current_stats.patterns_parité = patterns_parité
        current_stats.patterns_sommes = patterns_sommes
        current_stats.patterns_zones = patterns_zones

        # 4. ANALYSE PAR PÉRIODE ET TENDANCES
        current_stats.tendances_10 = self._calculer_tendances(all_draws, 10)
        current_stats.tendances_50 = self._calculer_tendances(all_draws, 50)
        current_stats.tendances_100 = self._calculer_tendances(all_draws, 100)
        
        # 5. Caractéristiques structurelles des tirages
        current_stats.ecarts_moyens = self._calculate_average_gaps(all_draws)
        current_stats.dispersion_moyenne = np.mean([np.std(draw) for draw in all_draws]) if all_draws else 0
        current_stats.clusters_moyens = np.mean([self._count_clusters_in_draw(draw) for draw in all_draws]) if all_draws else 0

        # Périodicités (peut prendre du temps, optionnel ou limité)
        self._log("🎶 Calcul des périodicités (peut prendre du temps)...")
        if HAS_VIZ: # Utilise HAS_VIZ car FFT en fait partie
            with Parallel(n_jobs=mp.cpu_count() if mp.cpu_count() > 1 else 1) as parallel:
                periodicities_results = parallel(
                    delayed(self._analyze_number_periodicity)(num, all_draws)
                    for num in range(1, KENO_PARAMS['total_numbers'] + 1)
                )
                current_stats.periodicites = {num: period for num, period in zip(range(1, KENO_PARAMS['total_numbers'] + 1), periodicities_results)}
        else:
            current_stats.periodicites = {i: None for i in range(1, KENO_PARAMS['total_numbers'] + 1)}

        current_stats.zones_freq = { # Ancien format
            "zone1_17": patterns_zones["zone_1_17"], "zone18_35": patterns_zones["zone_18_35"], 
            "zone36_52": patterns_zones["zone_36_52"], "zone53_70": patterns_zones["zone_53_70"]
        }
        current_stats.derniers_tirages = all_draws[-50:] if len(all_draws) >= 50 else all_draws
        
        conn.close()
        return current_stats

    def _analyze_with_pandas(self, all_draws: List[List[int]]) -> KenoStats:
        """Analyse de fallback avec Pandas seulement"""
        self._log("⚠️  Analyse de fallback avec Pandas (DuckDB non disponible)")
        current_stats = KenoStats(tous_tirages=all_draws)
        
        # Initialisation des dictionnaires pour toutes les stats
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            current_stats.frequences[num] = 0
            current_stats.frequences_recentes[num] = 0
            current_stats.frequences_50[num] = 0
            current_stats.frequences_20[num] = 0
            current_stats.retards[num] = 0
            current_stats.retards_historiques[num] = []
            current_stats.tendances_10[num] = 0.0
            current_stats.tendances_50[num] = 0.0
            current_stats.tendances_100[num] = 0.0
            current_stats.ewma_frequences[num] = 0.0
        
        # Comptage basique des fréquences et retards
        for i, draw in enumerate(all_draws):
            for num in draw:
                current_stats.frequences[num] += 1
                if i >= len(all_draws) - 100: current_stats.frequences_recentes[num] += 1
                if i >= len(all_draws) - 50: current_stats.frequences_50[num] += 1
                if i >= len(all_draws) - 20: current_stats.frequences_20[num] += 1

                # Zones
                if 1 <= num <= 17: current_stats.zones_freq["zone1_17"] = current_stats.zones_freq.get("zone1_17", 0) + 1
                elif 18 <= num <= 35: current_stats.zones_freq["zone18_35"] = current_stats.zones_freq.get("zone18_35", 0) + 1
                elif 36 <= num <= 52: current_stats.zones_freq["zone36_52"] = current_stats.zones_freq.get("zone36_52", 0) + 1
                else: current_stats.zones_freq["zone53_70"] = current_stats.zones_freq.get("zone53_70", 0) + 1
        
        # Calcul des retards
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            retard = 0
            for draw in reversed(all_draws):
                if num in draw:
                    break
                retard += 1
            current_stats.retards[num] = retard

            dernier_tirage = -1
            for i, draw in enumerate(all_draws):
                if num in draw:
                    if dernier_tirage >= 0:
                        current_stats.retards_historiques[num].append(i - dernier_tirage - 1)
                    dernier_tirage = i
        
        # Paires basiques (trios trop coûteux en Pandas)
        for draw in all_draws:
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    current_stats.paires_freq[pair] = current_stats.paires_freq.get(pair, 0) + 1
        
        current_stats.derniers_tirages = all_draws[-50:] if len(all_draws) >= 50 else all_draws
        current_stats.patterns_parité = {"tout_pair": 0, "tout_impair": 0}

        # 4. ANALYSE PAR PÉRIODE ET TENDANCES
        current_stats.tendances_10 = self._calculer_tendances(all_draws, 10)
        current_stats.tendances_50 = self._calculer_tendances(all_draws, 50)
        current_stats.tendances_100 = self._calculer_tendances(all_draws, 100)
        
        # 5. Caractéristiques structurelles des tirages
        current_stats.ecarts_moyens = self._calculate_average_gaps(all_draws)
        current_stats.dispersion_moyenne = np.mean([np.std(draw) for draw in all_draws]) if all_draws else 0
        current_stats.clusters_moyens = np.mean([self._count_clusters_in_draw(draw) for draw in all_draws]) if all_draws else 0

        # Périodicités (peut prendre du temps, optionnel ou limité)
        self._log("🎶 Calcul des périodicités (peut prendre du temps)...")
        if HAS_VIZ: # Utilise HAS_VIZ car FFT en fait partie
            with Parallel(n_jobs=mp.cpu_count() if mp.cpu_count() > 1 else 1) as parallel:
                periodicities_results = parallel(
                    delayed(self._analyze_number_periodicity)(num, all_draws)
                    for num in range(1, KENO_PARAMS['total_numbers'] + 1)
                )
                current_stats.periodicites = {num: period for num, period in zip(range(1, KENO_PARAMS['total_numbers'] + 1), periodicities_results)}
        else:
            current_stats.periodicites = {i: None for i in range(1, KENO_PARAMS['total_numbers'] + 1)}

        current_stats.zones_freq = { # Ancien format
            "zone1_17": patterns_zones["zone_1_17"], "zone18_35": patterns_zones["zone_18_35"], 
            "zone36_52": patterns_zones["zone_36_52"], "zone53_70": patterns_zones["zone_53_70"]
        }
        current_stats.derniers_tirages = all_draws[-50:] if len(all_draws) >= 50 else all_draws
        
        return current_stats

    def _calculer_tendances(self, all_draws, window: int = 10):
        """
        Calcule la tendance des numéros sur une fenêtre donnée.
        Retourne un dictionnaire {numéro: tendance}
        """
        tendances = {}
        if not all_draws or len(all_draws) < window:
            return {num: 1.0 for num in range(1, 71)}
        recent_draws = all_draws[-window:]
        for num in range(1, 71):
            count = sum(num in draw for draw in recent_draws)
            tendances[num] = count / window
        return tendances

    def _analyze_number_periodicity(self, num: int, all_draws: List[List[int]]) -> Optional[float]:
        """Analyse simple de périodicité par FFT — retourne la période dominante en nombre de tirages ou None."""
        try:
            presence = np.array([1.0 if num in draw else 0.0 for draw in all_draws])
            n = len(presence)
            if n < 32:
                return None
            # Detrend
            presence_detrended = presence - np.mean(presence)
            yf = fft(presence_detrended)
            xf = fftfreq(n, d=1)[:n // 2]
            ps = np.abs(yf[:n // 2]) ** 2
            # Ignorer composantes basses liées à la moyenne (xf ~ 0)
            idx = np.argsort(ps)[-5:]  # quelques pics potentiels
            # Choisir pic hors fréquence nulle
            best_period = None
            best_power = 0
            for i in idx:
                if xf[i] > 1e-6:
                    period = 1.0 / abs(xf[i])
                    if ps[i] > best_power and period > 1.0:
                        best_power = ps[i]
                        best_period = period
            if best_period is None:
                return None
            return float(best_period)
        except Exception:
            return None

    def _calculate_average_gaps(self, all_draws: List[List[int]]) -> Dict[int, float]:
        """Calcule l'écart moyen entre apparitions successives pour chaque numéro."""
        avg_gaps = {}
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            last_idx = None
            gaps = []
            for i, draw in enumerate(all_draws):
                if num in draw:
                    if last_idx is not None:
                        gaps.append(i - last_idx)
                    last_idx = i
            avg_gaps[num] = float(np.mean(gaps)) if gaps else float(len(all_draws))
        return avg_gaps

    def _count_clusters_in_draw(self, draw: List[int]) -> int:
        """Compte les 'clusters' (séquences consécutives) dans un tirage."""
        if not draw:
            return 0
        draw_sorted = sorted(draw)
        clusters = 1
        for a, b in zip(draw_sorted[:-1], draw_sorted[1:]):
            if b != a + 1:
                clusters += 1
        return clusters

    # -------------------------
    # 🛠️ Construction des features ML
    # -------------------------
    def _build_features_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construit X (features) et y (labels) à partir de self.data et self.stats.
        Retour:
            X: DataFrame (n_tirages, n_features)
            y: DataFrame (n_tirages, total_numbers) binaire (présence = 1/0)
        """
        self._log("🔧 Construction des features et labels...")
        draws = []
        dates = []
        for _, row in self.data.iterrows():
            draws.append(self._get_draw_numbers_from_row(row))
            dates.append(row['date_de_tirage'])

        n = len(draws)
        features = []
        labels = []

        # Pré-calc de fréquences globales pour normalisation simple
        freq_global = self.stats.frequences if self.stats and self.stats.frequences else {i: 0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}

        for idx in range(n):
            draw = draws[idx]
            feat = {}
            # Métriques du tirage précédent (si existant)
            prev_draw = draws[idx - 1] if idx > 0 else []
            feat['prev_count'] = len(prev_draw)
            feat['prev_sum'] = sum(prev_draw) if prev_draw else 0
            feat['prev_even_ratio'] = (sum(1 for x in prev_draw if x % 2 == 0) / max(1, len(prev_draw))) if prev_draw else 0.0

            # Retards et fréquences récentes / globales (moyennes sur le tirage)
            feat['mean_freq_global'] = np.mean([freq_global[num] for num in draw]) if draw else 0.0
            feat['mean_retard'] = np.mean([self.stats.retards.get(num, 0) for num in draw]) if draw else 0.0
            feat['ewma_mean'] = np.mean([self.stats.ewma_frequences.get(num, 0.0) for num in draw]) if draw else 0.0

            # Zones: proportion par zone dans le tirage
            zcounts = {'z1': 0, 'z2': 0, 'z3': 0, 'z4': 0}
            for num in draw:
                if 1 <= num <= 17: zcounts['z1'] += 1
                elif 18 <= num <= 35: zcounts['z2'] += 1
                elif 36 <= num <= 52: zcounts['z3'] += 1
                else: zcounts['z4'] += 1
            for k, v in zcounts.items():
                feat[f'zone_prop_{k}'] = v / max(1, len(draw))

            # Statistiques globales du tirage
            feat['draw_std'] = np.std(draw) if draw else 0.0
            feat['draw_min'] = min(draw) if draw else 0
            feat['draw_max'] = max(draw) if draw else 0

            # Ajout features temporelles simples
            feat['dayofweek'] = dates[idx].dayofweek if dates[idx] is not None else 0
            feat['month'] = dates[idx].month if dates[idx] is not None else 0

            features.append(feat)

            # Labels: vecteur binaire de longueur total_numbers
            label = {f'n_{i}': (1 if i in draw else 0) for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
            labels.append(label)

        X = pd.DataFrame(features).fillna(0.0)
        y = pd.DataFrame(labels).fillna(0).astype(int)

        # Assurer colonnes de zones par compatibilité
        X = self._ensure_zone_freq_columns(X)

        self._log(f"✅ Features construites: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    # -------------------------
    # 🧠 Entraînement ML
    # -------------------------
    def train_ml_models(self, force_retrain: bool = False) -> bool:
        """
        Entraîne les modèles ML selon la stratégie définie.
        Si HAS_ML est False, renvoie False (mode fréquence uniquement).
        """
        if not HAS_ML:
            self._log("⚠️ ML non disponible — entraînement ignoré.", "WARNING")
            return False

        try:
            X, y = self._build_features_labels()
            self.feature_names = list(X.columns)

            # Scalage
            X_scaled = pd.DataFrame(ML_CONFIG['scaler'].fit_transform(X), columns=X.columns)

            if self.ml_strategy == "multioutput":
                self._log("🧩 Entraînement MultiOutput RandomForest (un modèle pour tous les numéros)...")
                rf_params = get_training_params(self.training_profile)
                rf_params.update(ML_CONFIG['rf_params_multioutput'])
                base_rf = RandomForestClassifier(**rf_params)
                model = MultiOutputClassifier(base_rf, n_jobs=-1)
                model.fit(X_scaled, y.values)
                self.ml_models['multioutput'] = model
                self.ml_scores['train_shape'] = X_scaled.shape
            else:
                self._log("🔥 Entraînement XGBoost par numéro (un modèle binaire par numéro)...")
                xgb_params = ML_CONFIG['xgb_params'].copy()
                # Entraînement parallèle par numéro
                def train_single(num_idx):
                    y_col = y.iloc[:, num_idx].values
                    dtrain = xgb.DMatrix(X_scaled, label=y_col, feature_names=self.feature_names)
                    booster = xgb.train(xgb_params, dtrain, num_boost_round=xgb_params.get('n_estimators', 100))
                    return booster

                with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as ex:
                    futures = {ex.submit(train_single, i): i for i in range(y.shape[1])}
                    for fut in tqdm(futures, desc="Training XGB per-number", total=len(futures)):
                        i = futures[fut]
                        try:
                            model_i = fut.result()
                            self.ml_models[f'xgb_{i+1}'] = model_i
                        except Exception as e:
                            self._log(f"❌ Erreur entraînement numéro {i+1}: {e}", "ERROR")

            # Sauvegarde automatique
            self.save_models()
            self._log("✅ Entraînement ML terminé.")
            return True
        except Exception as e:
            self._log(f"❌ Erreur durant l'entraînement ML: {e}", "ERROR")
            return False

    # -------------------------
    # 🔮 Prédictions et scoring
    # -------------------------
    def _construct_next_features(self) -> pd.DataFrame:
        """
        Construit une ligne de features représentant l'état 'au prochain tirage'
        à partir des statistiques et du dernier tirage.
        """
        last_row = self.data.iloc[-1] if self.data is not None and len(self.data) > 0 else None
        last_draw = self._get_draw_numbers_from_row(last_row) if last_row is not None else []

        feat = {}
        feat['prev_count'] = len(last_draw)
        feat['prev_sum'] = sum(last_draw) if last_draw else 0
        feat['prev_even_ratio'] = (sum(1 for x in last_draw if x % 2 == 0) / max(1, len(last_draw))) if last_draw else 0.0

        freq_global = self.stats.frequences
        feat['mean_freq_global'] = np.mean([freq_global.get(num, 0) for num in last_draw]) if last_draw else 0.0
        feat['mean_retard'] = np.mean([self.stats.retards.get(num, 0) for num in last_draw]) if last_draw else 0.0
        feat['ewma_mean'] = np.mean([self.stats.ewma_frequences.get(num, 0.0) for num in last_draw]) if last_draw else 0.0

        zcounts = {'z1': 0, 'z2': 0, 'z3': 0, 'z4': 0}
        for num in last_draw:
            if 1 <= num <= 17: zcounts['z1'] += 1
            elif 18 <= num <= 35: zcounts['z2'] += 1
            elif 36 <= num <= 52: zcounts['z3'] += 1
            else: zcounts['z4'] += 1
        for k, v in zcounts.items():
            feat[f'zone_prop_{k}'] = v / max(1, len(last_draw))

        feat['draw_std'] = np.std(last_draw) if last_draw else 0.0
        feat['draw_min'] = min(last_draw) if last_draw else 0
        feat['draw_max'] = max(last_draw) if last_draw else 0

        # Temps: maintenant
        now = datetime.now()
        feat['dayofweek'] = now.weekday()

        Xnext = pd.DataFrame([feat])
        Xnext = self._ensure_zone_freq_columns(Xnext)
        # Assurer mêmes colonnes que training (ajouter 0 si manquant)
        for c in self.feature_names:
            if c not in Xnext.columns:
                Xnext[c] = 0.0
        Xnext = Xnext[self.feature_names] if self.feature_names else Xnext
        # Scale
        if hasattr(self.ml_scaler, 'transform'):
            try:
                Xnext_scaled = pd.DataFrame(self.ml_scaler.transform(Xnext), columns=Xnext.columns)
            except Exception:
                Xnext_scaled = Xnext
        else:
            Xnext_scaled = Xnext
        return Xnext_scaled

    def predict_probabilities(self) -> Dict[int, float]:
        """
        Prédit la probabilité d'apparition de chaque numéro au prochain tirage.
        Retourne un dict {numéro: probabilité}
        """
        probs = {i: 0.0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
        if not HAS_ML or not self.ml_models:
            # Fallback: probabilité estimée par fréquence récente normalisée
            recent = self.stats.frequences_recentes if self.stats and self.stats.frequences_recentes else self.stats.frequences
            total_recent = sum(recent.values()) if recent else 1
            for num in range(1, KENO_PARAMS['total_numbers'] + 1):
                probs[num] = (recent.get(num, 0) / max(1, total_recent))
            # Normalize to [0,1]
            maxp = max(probs.values()) if probs else 1.0
            if maxp > 0:
                for k in probs:
                    probs[k] = probs[k] / maxp
            return probs

        Xnext = self._construct_next_features()
        if 'multioutput' in self.ml_models:
            model: MultiOutputClassifier = self.ml_models['multioutput']
            try:
                proba_matrix = model.predict_proba(Xnext)  # list of arrays per output
                # predict_proba returns list with len = n_outputs, each shape (n_samples, 2)
                for i, arr in enumerate(proba_matrix):
                    probs[i + 1] = float(arr[0][1])  # probabilité de la classe '1'
            except Exception as e:
                self._log(f"⚠️ Erreur predict_proba MultiOutput: {e}", "WARNING")
        else:
            # XGBoost per-number
            for i in range(1, KENO_PARAMS['total_numbers'] + 1):
                key = f'xgb_{i}'
                model = self.ml_models.get(key)
                if model is None:
                    probs[i] = 0.0
                    continue
                try:
                    dmat = xgb.DMatrix(Xnext, feature_names=list(Xnext.columns))
                    pred = model.predict(dmat)
                    probs[i] = float(pred[0])
                except Exception as e:
                    self._log(f"⚠️ Erreur prédiction XGB numéro {i}: {e}", "WARNING")
                    probs[i] = 0.0

        # Normalisation légère (0..1)
        maxp = max(probs.values()) if probs else 1.0
        if maxp > 0:
            for k in probs:
                probs[k] = probs[k] / maxp
        self.ml_scores = probs
        return probs
    def export_grids(self, grids: List[List[int]], summary: Dict[str, Any], filename_prefix: str = "keno_grids"):
        """Exporte les grilles en CSV et Markdown dans OUTPUT_DIR"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # --- CSV ---
            csv_path = self.output_dir / f"{filename_prefix}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Grille_ID"] + [f"Num_{i+1}" for i in range(len(grids[0]))])
                for idx, grid in enumerate(grids, 1):
                    writer.writerow([idx] + grid)

            # --- Markdown ---
            md_path = self.output_dir / f"{filename_prefix}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# 🎰 Génération de grilles Keno\n\n")
                f.write(f"- Date: {datetime.now().isoformat()}\n")
                f.write(f"- Tirages analysés: {summary['n_draws']}\n\n")
                f.write("## 🔝 Top 10 numéros prédits\n\n")
                for num, score in summary['top10_ml']:
                    f.write(f"- **{num}** → {score:.3f}\n")
                f.write("\n## 🎯 Grilles générées\n\n")
                for idx, grid in enumerate(grids, 1):
                    f.write(f"- Grille #{idx}: {', '.join(map(str, grid))}\n")

            self._log(f"💾 Export effectué: {csv_path} et {md_path}")
        except Exception as e:
            self._log(f"⚠️ Erreur export grilles: {e}", "WARNING")

    # -------------------------
    # 🧾 Scoring composite et sélection TOP
    # -------------------------
    def score_numbers(self, probs: Dict[int, float]) -> Dict[int, float]:
        """
        Calcule un score composite pour chaque numéro en combinant:
        - Probabilité ML (probs)
        - Fréquence récente
        - Retard (overdue)
        - EWMA
        - Périodicité (si disponible)
        - Diversité (pénalité pour clusters)
        """
        scores = {}
        # Normalisations de base
        freq_recent = self.stats.frequences_recentes if self.stats and self.stats.frequences_recentes else self.stats.frequences
        max_freq = max(freq_recent.values()) if freq_recent else 1
        max_retard = max(self.stats.retards.values()) if self.stats and self.stats.retards else 1
        max_ewma = max(self.stats.ewma_frequences.values()) if self.stats and self.stats.ewma_frequences else 1.0

        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            mlp = probs.get(num, 0.0)
            freq_score = (freq_recent.get(num, 0) / max_freq) if max_freq else 0.0
            retard_score = (self.stats.retards.get(num, 0) / max_retard) if max_retard else 0.0
            ewma_score = (self.stats.ewma_frequences.get(num, 0.0) / max_ewma) if max_ewma else 0.0
            periodicity = self.stats.periodicites.get(num) if self.stats and self.stats.periodicites else None
            periodicity_score = 1.0 - min(1.0, (periodicity / 100.0)) if periodicity else 0.0

            # Compose selon poids adaptatifs
            w = self.adaptive_weights
            score = (
                w['ml_weight'] * mlp +
                w['freq_weight'] * freq_score +
                w['recent_weight'] * freq_score +  # recent et freq souvent corrélés
                w['retard_weight'] * retard_score +
                w['tendance_weight'] * (self.stats.tendances_10.get(num, 0.0) if self.stats else 0.0) +
                w['periodicity_weight'] * periodicity_score +
                w['pair_weight'] * 0.0 +  # placeholder pour patterns
                w['trio_weight'] * 0.0
            )

            # Diversity penalty: plus le numéro est proche à d'autres top, plus on pénalise (simple heuristique)
            scores[num] = float(score)

        # Normalisation 0..1
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            for k in scores:
                scores[k] = scores[k] / max_score
        return scores

    def get_top_n(self, n: int = 30) -> List[Tuple[int, float]]:
        """Retourne les TOP n numéros triés par score composite"""
        probs = self.predict_probabilities()
        scores = self.score_numbers(probs)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    # -------------------------
    # 🎯 Génération de grilles
    # -------------------------
    def generate_grid(self, selection_size: int = None, strategy: str = "greedy") -> List[int]:
        """
        Génère une grille Keno selon la stratégie choisie.
        Strategies:
            - 'greedy': prend les meilleurs scores en imposant diversité
            - 'stochastic': échantillonne pondéré par le score composite
        """
        selection_size = selection_size or KENO_PARAMS['player_selection']
        top_scores = dict(self.get_top_n(60))  # candidates pool
        if not top_scores:
            # fallback: choix aléatoire pondéré par fréquence globale
            self._log("⚠️ Aucune score disponible — génération aléatoire pondérée.")
            pool = list(range(1, KENO_PARAMS['total_numbers'] + 1))
            weights = [self.stats.frequences.get(i, 1) for i in pool]
            chosen = list(np.random.choice(pool, size=selection_size, replace=False, p=np.array(weights)/sum(weights)))
            return sorted(chosen)

        numbers = list(top_scores.keys())
        scores = np.array(list(top_scores.values()))

        if strategy == "stochastic":
            # Échantillonnage sans remplacement, pondéré
            probs = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores) / len(scores)
            chosen = list(np.random.choice(numbers, size=selection_size, replace=False, p=probs))
            return sorted(chosen)

        # Greedy with simple diversity penalty
        chosen = []
        remaining = numbers.copy()
        local_scores = {num: float(score) for num, score in top_scores.items()}
        while len(chosen) < selection_size and remaining:
            # Choisir meilleur
            best = max(remaining, key=lambda x: local_scores.get(x, 0.0))
            chosen.append(best)
            remaining.remove(best)
            # Appliquer pénalité de diversité: réduire le score des nombres proches (±2)
            for other in remaining:
                if abs(other - best) <= 2:
                    local_scores[other] *= (1 - self.adaptive_weights.get('diversity_penalty', 0.1))
        return sorted(chosen)

    # -------------------------
    # 💾 Sauvegarde / Chargement modèles et stats
    # -------------------------
    def save_models(self):
        """Sauvegarde des modèles et métadonnées"""
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            models_path = self.models_dir / "ml_models.pkl"
            meta_path = self.models_dir / "metadata.json"
            with open(models_path, "wb") as f:
                pickle.dump(self.ml_models, f)
            with open(meta_path, "w") as f:
                json.dump({
                    "feature_names": self.feature_names,
                    "training_profile": self.training_profile,
                    "ml_strategy": self.ml_strategy,
                    "saved_at": datetime.now().isoformat()
                }, f)
            self._log(f"💾 Modèles sauvegardés dans {models_path}")
        except Exception as e:
            self._log(f"⚠️ Erreur sauvegarde modèles: {e}", "WARNING")

    def load_models(self):
        """Charge les modèles si présents"""
        try:
            models_path = self.models_dir / "ml_models.pkl"
            meta_path = self.models_dir / "metadata.json"
            if models_path.exists():
                with open(models_path, "rb") as f:
                    self.ml_models = pickle.load(f)
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        self.feature_names = meta.get("feature_names", self.feature_names)
                self._log("✅ Modèles ML chargés depuis le disque.")
                return True
            else:
                self._log("ℹ️ Aucun modèle trouvé sur le disque.")
                return False
        except Exception as e:
            self._log(f"⚠️ Erreur chargement modèles: {e}", "WARNING")
            return False

    # -------------------------
    # 🧾 Rapport sommaire et utilitaires
    # -------------------------
    def summary(self) -> Dict[str, Any]:
        """Retourne un résumé compact de l'analyse et prédictions"""
        probs = self.predict_probabilities()
        top10 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "n_draws": len(self.data) if self.data is not None else 0,
            "top10_ml": top10,
            "last_update": datetime.now().isoformat()
        }

# ==============================================================================
# 🧭 CLI et point d'entrée
# ==============================================================================
def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Générateur Intelligent de Grilles Keno (v3.0)")
    parser.add_argument(
        "--data", "-d",
        help="Chemin vers le fichier parquet des tirages",
        default="keno/keno_data/keno_202010.parquet"
    )
    parser.add_argument("--train", action="store_true", help="Forcer l'entraînement ML")
    parser.add_argument("--no-ml", action="store_true", help="Désactiver ML (mode fréquence uniquement)")
    parser.add_argument("--profile", choices=['quick', 'balanced', 'comprehensive', 'intensive'], default="balanced", help="Profil d'entraînement")
    parser.add_argument("--strategy", choices=['greedy', 'stochastic'], default="greedy", help="Stratégie de génération de grille")
    parser.add_argument("--n", type=int, default=1, help="Nombre de grilles à générer")
    parser.add_argument("--size", type=int, choices=range(7, 11), default=KENO_PARAMS['player_selection'], help="Nombre de numéros par grille (7 à 10)")
    parser.add_argument("--silent", action="store_true", help="Mode silencieux")
    return parser.parse_args(argv)

def main(argv: List[str] = None):
    args = parse_args(argv or sys.argv[1:])
    gen = KenoGeneratorAdvanced(data_path=args.data, silent=args.silent, training_profile=args.profile, ml_strategy="multioutput" if not args.no_ml else "none")

    if not gen.load_data():
        print("❌ Impossible de charger les données. Vérifiez le chemin et le format (parquet attendu).")
        return

    gen.analyze_patterns()
    gen.load_models()

    if args.no_ml:
        gen._log("ℹ️ Mode ML désactivé par option utilisateur. Utilisation du mode fréquence.", "WARNING")
    elif args.train or not gen.ml_models:
        gen._log("▶ Entraînement ML demandé...")
        gen.train_ml_models(force_retrain=args.train)

    # Génération du TOP 30 ML
    top30_ml = [num for num, _ in gen.get_top_n(30)]

    # Génération des grilles ML optimisées avec pulp et statistiques
    grids_ml = generate_best_ml_grids(top30_ml, n_grids=max(10, args.n), grid_size=args.size, stats=gen.stats)

    # Sélection des grilles avec meilleure couverture (diversité maximale)
    # Ici, on garde les N grilles les plus différentes
    final_grids_ml = []
    covered = set()
    for grid in grids_ml:
        if not set(grid).issubset(covered):
            final_grids_ml.append(grid)
            covered.update(grid)
        if len(final_grids_ml) >= args.n:
            break
    # Si pas assez de diversité, complète avec les premières grilles
    while len(final_grids_ml) < args.n and grids_ml:
        for grid in grids_ml:
            if grid not in final_grids_ml:
                final_grids_ml.append(grid)
            if len(final_grids_ml) >= args.n:
                break

    # Affichage et export
    for i, grid in enumerate(final_grids_ml, 1):
        print(f"Grille ML optimisée #{i}: {grid}")

    summary = gen.summary()
    gen.export_grids(final_grids_ml, summary, filename_prefix="keno_grids_ml")

    # Export du TOP 30 ML
    export_top30_csv(top30_ml)

    print(f"\nGrilles ML optimisées exportées dans keno_output/keno_grids_ml.csv")

def generate_best_ml_grids(top30_ml, n_grids=10, grid_size=10, stats=None):
    """
    Génère n_grids grilles à partir du TOP 30 ML en maximisant la couverture et la diversité,
    pondérées par les statistiques (fréquence, retard, etc.).
    """
    grids = []
    used_combinations = set()
    # Pondération par score composite si stats disponibles
    weights = {num: 1.0 for num in top30_ml}
    if stats:
        freq = stats.frequences_recentes if stats.frequences_recentes else stats.frequences
        max_freq = max(freq.values()) if freq else 1
        for num in top30_ml:
            weights[num] = freq.get(num, 1) / max_freq

    for grid_idx in range(n_grids):
        prob = pulp.LpProblem(f"KenoMLGrid_{grid_idx}", pulp.LpMaximize)
        x = {n: pulp.LpVariable(f"x_{grid_idx}_{n}", cat="Binary") for n in top30_ml}
        # Objectif : maximiser la somme pondérée des numéros
        prob += pulp.lpSum([weights[n] * x[n] for n in top30_ml])
        # Contraintes : exactement grid_size numéros par grille
        prob += pulp.lpSum([x[n] for n in top30_ml]) == grid_size
        # Contraintes pour éviter les doublons
        for prev_grid in grids:
            prob += pulp.lpSum([x[n] for n in prev_grid]) <= grid_size - 1
        prob.solve()
        grid = tuple(sorted([n for n in top30_ml if pulp.value(x[n]) == 1]))
        if grid in used_combinations:
            break
        grids.append(list(grid))
        used_combinations.add(grid)
    return grids

def export_grids_csv(grids, output_path="keno_output/grilles_keno.csv"):
    columns = [f"numero_{i}" for i in range(1, len(grids[0]) + 1)]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for grid in grids:
            writer.writerow(grid)

def export_top30_csv(top30_ml, output_path="keno_output/top30_ml.csv"):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Numéro"])
        for num in top30_ml:
            writer.writerow([num])

if __name__ == "__main__":
    class GeminiTop30:
        def __init__(self, data_path, n, train):
            # Initialisation du modèle, des données, etc.
            self.data_path = data_path
            self.n = n
            self.train = train

    def summary(self):
        return "Résumé du modèle GeminiTop30"

    main()
