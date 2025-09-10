#!/usr/bin/env python3
"""
============================================================================
🎲 GÉNÉRATEUR INTELLIGENT DE GRILLES KENO - VERSION AVANCÉE 🎲
============================================================================

Générateur de grilles Keno utilisant l'apprentissage automatique (XGBoost) 
et l'analyse statistique pour optimiser les combinaisons.

Caractéristiques:
- Machine Learning avec XGBoost pour prédire les numéros probables
- Stratégie adaptative avec pondération dynamique ML/Fréquence  
- Analyse de patterns temporels et cycliques
- Optimisation des combinaisons selon critères statistiques
- Génération de rapports détaillés avec visualisations

Auteur: Assistant IA
Version: 2.0
Date: Août 2025
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
from dataclasses import dataclass
from tqdm import tqdm
import logging

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
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
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
    'min_selection': 7,         # Minimum de numéros sélectionnables (modifié)
    'max_selection': 10,        # Maximum de numéros sélectionnables
}

# Chemins des fichiers
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "keno_data"
MODELS_DIR = BASE_DIR.parent / "keno_models"
OUTPUT_DIR = BASE_DIR.parent / "keno_output"

# Configuration ML
ML_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'binary:logistic'
}

def get_training_params(profile: str) -> dict:
    """
    Retourne les paramètres d'entraînement selon le profil sélectionné
    
    Args:
        profile: Profil d'entraînement ('quick', 'balanced', 'comprehensive', 'intensive')
    
    Returns:
        dict: Paramètres d'entraînement pour RandomForest
    """
    if profile == "quick":
        return {
            'n_estimators': 50,        # Arbres réduits pour la vitesse
            'max_depth': 8,            # Profondeur modérée
            'min_samples_split': 10,   # Éviter l'overfitting
            'min_samples_leaf': 5,     # Éviter l'overfitting
            'max_features': 'sqrt',    # Features réduites
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    elif profile == "balanced":
        return {
            'n_estimators': 100,       # Standard
            'max_depth': 12,           # Profondeur équilibrée
            'min_samples_split': 5,    # Standard
            'min_samples_leaf': 3,     # Standard
            'max_features': 'sqrt',    # Standard
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    elif profile == "comprehensive":
        return {
            'n_estimators': 200,       # Plus d'arbres pour meilleure précision
            'max_depth': 15,           # Profondeur élevée
            'min_samples_split': 4,    # Plus de splits
            'min_samples_leaf': 2,     # Feuilles plus petites
            'max_features': 'log2',    # Plus de features
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    elif profile == "intensive":
        return {
            'n_estimators': 300,       # Maximum d'arbres
            'max_depth': 20,           # Profondeur maximale
            'min_samples_split': 3,    # Splits agressifs
            'min_samples_leaf': 1,     # Feuilles minimales
            'max_features': None,      # Toutes les features
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    else:
        # Fallback sur balanced
        return get_training_params("balanced")

@dataclass
class KenoStats:
    """Structure pour stocker les statistiques Keno complètes"""
    # 1. Fréquences d'apparition des numéros
    frequences: Dict[int, int]                           # Fréquence globale
    frequences_recentes: Dict[int, int]                  # Fréquence sur 100 derniers tirages
    frequences_50: Dict[int, int]                        # Fréquence sur 50 derniers tirages
    frequences_20: Dict[int, int]                        # Fréquence sur 20 derniers tirages
    
    # 2. Numéros en retard (overdue numbers)
    retards: Dict[int, int]                              # Retard actuel de chaque numéro
    retards_historiques: Dict[int, List[int]]            # Historique des retards
    
    # 3. Combinaisons et patterns récurrents
    paires_freq: Dict[Tuple[int, int], int]              # Paires fréquentes
    trios_freq: Dict[Tuple[int, int, int], int]          # Trios fréquents
    patterns_parité: Dict[str, int]                      # Distribution pair/impair
    patterns_sommes: Dict[int, int]                      # Distribution des sommes
    patterns_zones: Dict[str, int]                       # Répartition par zones/dizaines
    
    # 4. Analyse par période
    tendances_10: Dict[int, float]                       # Tendances sur 10 tirages
    tendances_50: Dict[int, float]                       # Tendances sur 50 tirages  
    tendances_100: Dict[int, float]                      # Tendances sur 100 tirages
    
    # Données brutes pour analyses avancées
    zones_freq: Dict[str, int]                           # Ancien format maintenu pour compatibilité
    derniers_tirages: List[List[int]]                    # Derniers tirages
    tous_tirages: List[List[int]]                        # Tous les tirages pour DuckDB

# ==============================================================================
# 🎯 CLASSE PRINCIPALE GENERATEUR KENO
# ==============================================================================

class KenoGeneratorAdvanced:
    """Générateur avancé de grilles Keno avec ML et analyse statistique"""
    
    def __init__(self, data_path: str = None, silent: bool = False, training_profile: str = "balanced", grid_size: int = 10):
        """
        Initialise le générateur Keno
        
        Args:
            data_path: Chemin vers les données historiques
            silent: Mode silencieux pour réduire les sorties
            training_profile: Profil d'entraînement ('quick', 'balanced', 'comprehensive', 'intensive')
            grid_size: Taille des grilles (7 à 10 numéros)
        """
        self.silent = silent
        self.data_path = data_path or str(DATA_DIR / "keno_202010.parquet")
        self.models_dir = MODELS_DIR
        self.output_dir = OUTPUT_DIR
        self.training_profile = training_profile
        
        # Validation de la taille de grille
        if not (7 <= grid_size <= 10):
            self._log(f"⚠️  Taille de grille invalide ({grid_size}). Utilisation de 10 par défaut.", "ERROR")
            grid_size = 10
        self.grid_size = grid_size
        
        # Créer les répertoires nécessaires
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialisation des composants
        self.data = None
        self.stats = None
        self.ml_models = {}
        self.metadata = {}
        self.cache = {}
        
        # Stratégie adaptative
        self.adaptive_weights = {
            'ml_weight': 0.6,      # 60% ML par défaut
            'freq_weight': 0.4,    # 40% fréquence par défaut  
            'performance_history': [],
            'last_update': datetime.now()
        }
        
        self._log(f"🎲 Générateur Keno Avancé v2.0 initialisé (grilles de {self.grid_size} numéros)")

    def _log(self, message: str, level: str = "INFO"):
        """Système de logging configuré"""
        if not self.silent or level == "ERROR":
            print(f"{message}")
    
    def _ensure_zone_freq_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes zoneX_freq manquantes pour compatibilité avec le modèle ML.
        """
        for freq_col in ['zone1_freq', 'zone2_freq', 'zone3_freq', 'zone4_freq']:
            if freq_col not in df.columns:
                df[freq_col] = 0
        return df

    
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
                
            # Chargement depuis Parquet
            self.data = pd.read_parquet(self.data_path)
            
            # Validation des colonnes requises
            required_cols = ['date_de_tirage'] + [f'boule{i}' for i in range(1, 21)]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                self._log(f"❌ Colonnes manquantes: {missing_cols}", "ERROR")
                return False
            
            # Tri par date
            self.data = self.data.sort_values('date_de_tirage').reset_index(drop=True)
            
            self._log(f"✅ {len(self.data)} tirages Keno chargés ({self.data['date_de_tirage'].min()} à {self.data['date_de_tirage'].max()})")
            return True
            
        except Exception as e:
            self._log(f"❌ Erreur lors du chargement des données: {e}", "ERROR")
            return False
    
    def analyze_patterns(self) -> KenoStats:
        """
        Analyse complète des patterns et statistiques Keno avec DuckDB
        
        Returns:
            KenoStats: Statistiques complètes calculées
        """
        self._log("🔍 Analyse complète des patterns Keno avec DuckDB...")
        
        # Extraction des numéros de tous les tirages
        all_draws = []
        for _, row in self.data.iterrows():
            # Support des différents formats de colonnes
            if 'b1' in self.data.columns:
                draw = [int(row[f'boule{i}']) for i in range(1, 21)]
            else:
                draw = [int(row[f'boule{i}']) for i in range(1, 21)]
            all_draws.append(sorted(draw))
        
        self._log(f"📊 Analyse de {len(all_draws)} tirages")
        
        if HAS_DUCKDB:
            return self._analyze_with_duckdb(all_draws)
        else:
            return self._analyze_with_pandas(all_draws)
    
    def _analyze_with_duckdb(self, all_draws: List[List[int]]) -> KenoStats:
        """Analyse optimisée avec DuckDB"""
        self._log("🚀 Utilisation de DuckDB pour l'analyse optimisée")
        
        # Créer une connexion DuckDB
        conn = duckdb.connect(':memory:')
        
        # Préparer les données pour DuckDB
        draws_data = []
        for i, draw in enumerate(all_draws):
            for num in draw:
                draws_data.append({'tirage_id': i, 'numero': num, 'position': draw.index(num)})
        
        draws_df = pd.DataFrame(draws_data)
        
        # Créer la table dans DuckDB
        conn.register('tirages', draws_df)
        
        # 1. FRÉQUENCES D'APPARITION
        self._log("📊 Calcul des fréquences d'apparition...")
        
        # Fréquence globale
        freq_global = conn.execute("""
            SELECT numero, COUNT(*) as freq 
            FROM tirages 
            GROUP BY numero 
            ORDER BY numero
        """).fetchdf()
        frequences = dict(zip(freq_global['numero'], freq_global['freq']))
        
        # Fréquences récentes (100, 50, 20 derniers tirages)
        max_tirage = len(all_draws) - 1
        
        freq_100 = conn.execute(f"""
            SELECT numero, COUNT(*) as freq 
            FROM tirages 
            WHERE tirage_id >= {max(0, max_tirage - 99)}
            GROUP BY numero
        """).fetchdf()
        frequences_100 = dict(zip(freq_100['numero'], freq_100['freq']))
        
        freq_50 = conn.execute(f"""
            SELECT numero, COUNT(*) as freq 
            FROM tirages 
            WHERE tirage_id >= {max(0, max_tirage - 49)}
            GROUP BY numero
        """).fetchdf()
        frequences_50 = dict(zip(freq_50['numero'], freq_50['freq']))
        
        freq_20 = conn.execute(f"""
            SELECT numero, COUNT(*) as freq 
            FROM tirages 
            WHERE tirage_id >= {max(0, max_tirage - 19)}
            GROUP BY numero
        """).fetchdf()
        frequences_20 = dict(zip(freq_20['numero'], freq_20['freq']))
        
        # 2. CALCUL DES RETARDS
        self._log("⏰ Calcul des retards des numéros...")
        retards = {}
        retards_historiques = {}
        
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            # Retard actuel
            retard = 0
            for draw in reversed(all_draws):
                if num in draw:
                    break
                retard += 1
            retards[num] = retard
            
            # Historique des retards
            historique = []
            dernier_tirage = -1
            for i, draw in enumerate(all_draws):
                if num in draw:
                    if dernier_tirage >= 0:
                        historique.append(i - dernier_tirage - 1)
                    dernier_tirage = i
            retards_historiques[num] = historique
        
        # 3. PATTERNS ET COMBINAISONS
        self._log("🔗 Analyse des patterns et combinaisons...")
        
        # Paires fréquentes
        paires_freq = {}
        for draw in all_draws:
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    paires_freq[pair] = paires_freq.get(pair, 0) + 1
        
        # Trios fréquents (limité aux plus fréquents pour performance)
        trios_freq = {}
        for draw in all_draws[-1000:]:  # Seulement les 1000 derniers pour performance
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    for k in range(j + 1, len(draw)):
                        trio = tuple(sorted([draw[i], draw[j], draw[k]]))
                        trios_freq[trio] = trios_freq.get(trio, 0) + 1
        
        # Patterns de parité
        patterns_parité = {"tout_pair": 0, "tout_impair": 0, "mixte": 0}
        for draw in all_draws:
            pairs = sum(1 for num in draw if num % 2 == 0)
            if pairs == len(draw):
                patterns_parité["tout_pair"] += 1
            elif pairs == 0:
                patterns_parité["tout_impair"] += 1
            else:
                patterns_parité["mixte"] += 1
        
        # Patterns de sommes
        patterns_sommes = {}
        for draw in all_draws:
            somme = sum(draw)
            patterns_sommes[somme] = patterns_sommes.get(somme, 0) + 1
        
        # Patterns de zones/dizaines
        patterns_zones = {
            "zone_1_17": 0, "zone_18_35": 0, "zone_36_52": 0, "zone_53_70": 0,
            "dizaine_1_10": 0, "dizaine_11_20": 0, "dizaine_21_30": 0, 
            "dizaine_31_40": 0, "dizaine_41_50": 0, "dizaine_51_60": 0, "dizaine_61_70": 0
        }
        
        for draw in all_draws:
            # Comptage par zones
            for num in draw:
                if 1 <= num <= 17:
                    patterns_zones["zone_1_17"] += 1
                elif 18 <= num <= 35:
                    patterns_zones["zone_18_35"] += 1
                elif 36 <= num <= 52:
                    patterns_zones["zone_36_52"] += 1
                else:  # 53-70
                    patterns_zones["zone_53_70"] += 1
                
                # Comptage par dizaines
                dizaine = (num - 1) // 10 + 1
                if dizaine <= 7:
                    patterns_zones[f"dizaine_{(dizaine-1)*10+1}_{dizaine*10}"] += 1
        
        # 4. ANALYSE PAR PÉRIODE ET TENDANCES
        self._log("📈 Calcul des tendances par période...")
        
        tendances_10 = self._calculer_tendances(all_draws, 10)
        tendances_50 = self._calculer_tendances(all_draws, 50)
        tendances_100 = self._calculer_tendances(all_draws, 100)
        
        # Zones compatibilité (ancien format)
        zones_freq = {
            "zone1_17": patterns_zones["zone_1_17"],
            "zone18_35": patterns_zones["zone_18_35"], 
            "zone36_52": patterns_zones["zone_36_52"],
            "zone53_70": patterns_zones["zone_53_70"]
        }
        
        # Garder les derniers tirages
        derniers_tirages = all_draws[-50:] if len(all_draws) >= 50 else all_draws
        
        conn.close()
        
        # Initialiser avec des valeurs par défaut pour les champs manquants
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            if num not in frequences:
                frequences[num] = 0
            if num not in frequences_100:
                frequences_100[num] = 0
            if num not in frequences_50:
                frequences_50[num] = 0
            if num not in frequences_20:
                frequences_20[num] = 0
        
        self.stats = KenoStats(
            frequences=frequences,
            frequences_recentes=frequences_100,
            frequences_50=frequences_50,
            frequences_20=frequences_20,
            retards=retards,
            retards_historiques=retards_historiques,
            paires_freq=paires_freq,
            trios_freq=trios_freq,
            patterns_parité=patterns_parité,
            patterns_sommes=patterns_sommes,
            patterns_zones=patterns_zones,
            tendances_10=tendances_10,
            tendances_50=tendances_50,
            tendances_100=tendances_100,
            zones_freq=zones_freq,
            derniers_tirages=derniers_tirages,
            tous_tirages=all_draws
        )
        
        self._log(f"✅ Analyse DuckDB terminée - {len(all_draws)} tirages analysés")
        return self.stats
    
    def _analyze_with_pandas(self, all_draws: List[List[int]]) -> KenoStats:
        """Analyse de fallback avec Pandas seulement"""
        self._log("⚠️  Analyse de fallback avec Pandas (DuckDB non disponible)")
        
        # Implémentation simplifiée pour compatibilité
        frequences = {i: 0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
        retards = {i: 0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
        paires_freq = {}
        zones_freq = {"zone1_17": 0, "zone18_35": 0, "zone36_52": 0, "zone53_70": 0}
        
        # Comptage basique des fréquences
        for draw in all_draws:
            for num in draw:
                frequences[num] += 1
                
                # Zones
                if 1 <= num <= 17:
                    zones_freq["zone1_17"] += 1
                elif 18 <= num <= 35:
                    zones_freq["zone18_35"] += 1
                elif 36 <= num <= 52:
                    zones_freq["zone36_52"] += 1
                else:
                    zones_freq["zone53_70"] += 1
        
        # Calcul des retards
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            retard = 0
            for draw in reversed(all_draws):
                if num in draw:
                    break
                retard += 1
            retards[num] = retard
        
        # Paires basiques
        for draw in all_draws:
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    paires_freq[pair] = paires_freq.get(pair, 0) + 1
        
        derniers_tirages = all_draws[-50:] if len(all_draws) >= 50 else all_draws
        
        # Initialiser avec des valeurs par défaut
        self.stats = KenoStats(
            frequences=frequences,
            frequences_recentes=frequences.copy(),
            frequences_50=frequences.copy(),
            frequences_20=frequences.copy(),
            retards=retards,
            retards_historiques={i: [] for i in range(1, 71)},
            paires_freq=paires_freq,
            trios_freq={},
            patterns_parité={"tout_pair": 0, "tout_impair": 0, "mixte": len(all_draws)},
            patterns_sommes={},
            patterns_zones={},
            tendances_10={i: 0.0 for i in range(1, 71)},
            tendances_50={i: 0.0 for i in range(1, 71)},
            tendances_100={i: 0.0 for i in range(1, 71)},
            zones_freq=zones_freq,
            derniers_tirages=derniers_tirages,
            tous_tirages=all_draws
        )
        
        self._log(f"✅ Analyse Pandas terminée - {len(all_draws)} tirages analysés")
        return self.stats
    
    def _calculer_tendances(self, all_draws: List[List[int]], periode: int) -> Dict[int, float]:
        """Calcule les tendances d'apparition sur une période donnée"""
        if len(all_draws) < periode:
            return {i: 0.0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
        
        tendances = {}
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            # Fréquence récente vs fréquence historique
            recent_draws = all_draws[-periode:]
            freq_recent = sum(1 for draw in recent_draws if num in draw)
            freq_moyenne = freq_recent / periode
            
            # Fréquence historique
            freq_historique = sum(1 for draw in all_draws[:-periode] if num in draw)
            if len(all_draws) > periode:
                freq_historique_moyenne = freq_historique / (len(all_draws) - periode)
            else:
                freq_historique_moyenne = freq_moyenne
            
            # Tendance = ratio récent/historique
            if freq_historique_moyenne > 0:
                tendances[num] = freq_moyenne / freq_historique_moyenne
            else:
                tendances[num] = freq_moyenne * 2  # Bonus si nouveau numéro
        
        return tendances
    
    def add_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features cycliques pour l'apprentissage automatique
        
        Args:
            df: DataFrame avec les données
            
        Returns:
            DataFrame avec les features ajoutées
        """
        df = df.copy()
        
        # Features temporelles cycliques
        df['day_sin'] = np.sin(2 * np.pi * df['date_de_tirage'].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['date_de_tirage'].dt.day / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['date_de_tirage'].dt.month / 12)  
        df['month_cos'] = np.cos(2 * np.pi * df['date_de_tirage'].dt.month / 12)
        
        # Features cycliques pour chaque numéro (1-70 pour Keno)
        for i in range(1, 11):  # Pour les 10 premières boules principales
            if f'boule{i}' in df.columns:
                df[f'boule{i}_sin'] = np.sin(2 * np.pi * df[f'boule{i}'] / KENO_PARAMS['total_numbers'])
                df[f'boule{i}_cos'] = np.cos(2 * np.pi * df[f'boule{i}'] / KENO_PARAMS['total_numbers'])
        
        return df
    
    def enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features avancées pour le ML : séquences, clusters, dispersion, stats paires/trios, features cycliques, etc.
        """
        df = df.copy()
        # Features cycliques déjà présentes
        # Ajout de la dispersion (écart-type des numéros tirés)
        df['dispersion'] = df[[f'boule{i}' for i in range(1, 21)]].std(axis=1)
        # Ajout du nombre de paires et trios fréquents dans le tirage
        if hasattr(self, 'stats') and self.stats:
            def count_pairs(row):
                nums = [int(row[f'boule{i}']) for i in range(1, 21)]
                return sum(1 for i in range(len(nums)) for j in range(i+1, len(nums)) if tuple(sorted([nums[i], nums[j]])) in self.stats.paires_freq)
            def count_trios(row):
                nums = [int(row[f'boule{i}']) for i in range(1, 21)]
                return sum(1 for i in range(len(nums)) for j in range(i+1, len(nums)) for k in range(j+1, len(nums)) if tuple(sorted([nums[i], nums[j], nums[k]])) in self.stats.trios_freq)
            df['nb_paires_freq'] = df.apply(count_pairs, axis=1)
            df['nb_trios_freq'] = df.apply(count_trios, axis=1)
        else:
            df['nb_paires_freq'] = 0
            df['nb_trios_freq'] = 0
        # Ajout de la somme des numéros tirés
        df['somme_tirage'] = df[[f'boule{i}' for i in range(1, 21)]].sum(axis=1)
        # Ajout de la parité (nombre de pairs)
        df['nb_pairs'] = df[[f'boule{i}' for i in range(1, 21)]].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
        # Ajout de la feature cluster (nombre de groupes de numéros consécutifs)
        def count_clusters(row):
            nums = sorted([int(row[f'boule{i}']) for i in range(1, 21)])
            clusters = 1
            for i in range(1, len(nums)):
                if nums[i] - nums[i-1] > 1:
                    clusters += 1
            return clusters
        df['nb_clusters'] = df.apply(count_clusters, axis=1)
        return df
    
    def train_xgboost_models(self, retrain: bool = False) -> bool:
        """
        Entraîne un modèle XGBoost multi-label pour prédire les numéros Keno
        
        Args:
            retrain: Force le réentraînement même si le modèle existe
            
        Returns:
            bool: True si l'entraînement est réussi
        """
        if not HAS_ML:
            self._log("❌ Modules ML non disponibles", "ERROR")
            return False
        
        # Vérification du modèle existant (un seul modèle multi-label)
        model_file = self.models_dir / "xgb_keno_multilabel.pkl"
        if not retrain and model_file.exists():
            self._log("✅ Modèle XGBoost Keno multi-label existant trouvé")
            return self.load_ml_models()
        
        self._log("🤖 Entraînement du modèle XGBoost Keno multi-label...")
        self._log("   📊 Stratégie: 1 modèle multi-label pour apprendre les corrélations")
        
        try:
            # Préparation des données avec features enrichies (optimisé pour éviter la fragmentation)
            df_features = self.add_cyclic_features(self.data)
            df_features = self.enrich_features(df_features)
            
            # Features temporelles
            feature_cols = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
            
            # Préparation des features d'historique avec pd.concat pour éviter la fragmentation
            self._log("   📝 Création des features d'historique...")
            lag_features = {}
            
            for lag in range(1, 6):
                for ball_num in range(1, 21):
                    col_name = f'lag{lag}_boule{ball_num}'
                    if lag < len(df_features):
                        lag_features[col_name] = df_features[f'boule{ball_num}'].shift(lag).fillna(0)
                    else:
                        lag_features[col_name] = pd.Series([0] * len(df_features))
                    feature_cols.append(col_name)
            
            # Ajout des features d'historique en une seule fois
            lag_df = pd.DataFrame(lag_features, index=df_features.index)
            df_features = pd.concat([df_features, lag_df], axis=1)
            
            # Features de fréquence par zone (calculées efficacement)
            self._log("   📝 Calcul des features de zones...")
            zone_features = {
                'zone1_count': [],
                'zone2_count': [],
                'zone3_count': [],
                'zone4_count': [],
                'zone1_freq': [],
                'zone2_freq': [],
                'zone3_freq': [],
                'zone4_freq': []
            }
            
            for idx, row in df_features.iterrows():
                draw_numbers = [int(row[f'boule{i}']) for i in range(1, 21)]
                zone_features['zone1_count'].append(sum(1 for n in draw_numbers if 1 <= n <= 17))
                zone_features['zone2_count'].append(sum(1 for n in draw_numbers if 18 <= n <= 35))
                zone_features['zone3_count'].append(sum(1 for n in draw_numbers if 36 <= n <= 52))
                zone_features['zone4_count'].append(sum(1 for n in draw_numbers if 53 <= n <= 70))
            
            # Ajout des features de zones
            for zone_name, zone_values in zone_features.items():
                # Correction : si la liste est vide ou de mauvaise taille, remplis avec des zéros
                if len(zone_values) != len(df_features):
                    zone_values = [0] * len(df_features)
                df_features[zone_name] = zone_values
                feature_cols.append(zone_name)

            # AJOUT CRUCIAL: Features statistiques étendues (manquantes à l'entraînement)
            if self.stats:
                # Features fréquentielles pour chaque numéro (1-70)
                stats_data = {
                    f'freq_recent_{num}': [self.stats.frequences_recentes.get(num, 0)] * len(df_features)
                    for num in range(1, 71)
                }
                stats_data.update({
                    f'retard_{num}': [self.stats.retards.get(num, 0)] * len(df_features)
                    for num in range(1, 71)
                })
                stats_data.update({
                    f'tendance_{num}': [self.stats.tendances_50.get(num, 1.0)] * len(df_features)
                    for num in range(1, 71)
                })
                
                # Ajouter toutes ces features au DataFrame
                for feature_name, feature_values in stats_data.items():
                    df_features[feature_name] = feature_values
                    feature_cols.append(feature_name)
            else:
                # Créer des valeurs par défaut pour toutes les features statistiques
                for num in range(1, 71):
                    df_features[f'freq_recent_{num}'] = [0] * len(df_features)
                    df_features[f'retard_{num}'] = [0] * len(df_features)
                    df_features[f'tendance_{num}'] = [1.0] * len(df_features)
                    feature_cols.extend([f'freq_recent_{num}', f'retard_{num}', f'tendance_{num}'])
            
            X = df_features[feature_cols].fillna(0)
            
            # Création du target multi-label (matrice 70 colonnes, une par numéro)
            y = np.zeros((len(df_features), KENO_PARAMS['total_numbers']))
            
            for idx, row in df_features.iterrows():
                draw_numbers = [int(row[f'boule{i}']) for i in range(1, 21)]
                for num in draw_numbers:
                    if 1 <= num <= KENO_PARAMS['total_numbers']:
                        y[idx, num - 1] = 1  # Index 0-69 pour numéros 1-70
            
            self._log(f"   📝 Données préparées: {X.shape[0]} tirages, {X.shape[1]} features")
            self._log(f"   📝 Target multi-label: {y.shape[1]} numéros à prédire")
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Configuration spéciale pour multi-label
            ml_config = ML_CONFIG.copy()
            ml_config['objective'] = 'multi:logistic'
            ml_config['num_class'] = 2  # Binaire pour chaque label
            
            # Entraînement du modèle multi-label avec RandomForest (meilleur pour les corrélations)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.multioutput import MultiOutputClassifier
            
            # Obtenir les paramètres selon le profil d'entraînement
            rf_params = get_training_params(self.training_profile)
            
            # Afficher le profil utilisé
            profile_names = {
                'quick': 'Ultra-rapide',
                'balanced': 'Équilibré', 
                'comprehensive': 'Complet',
                'intensive': 'Intensif'
            }
            profile_name = profile_names.get(self.training_profile, 'Équilibré')
            self._log(f"📊 Profil d'entraînement: {profile_name} ({self.training_profile})")
            self._log(f"   • Arbres: {rf_params['n_estimators']}")
            self._log(f"   • Profondeur max: {rf_params['max_depth']}")
            self._log(f"   • Features: {rf_params['max_features']}")
            
            # RandomForest configuré selon le profil
            base_model = RandomForestClassifier(**rf_params)
            
            # Alternative: Essayer aussi XGBoost avec MultiOutputClassifier
            # base_model = xgb.XGBClassifier(
            #     objective='binary:logistic',
            #     n_estimators=150,
            #     max_depth=8,
            #     learning_rate=0.05,
            #     random_state=42,
            #     n_jobs=-1
            # )
            
            # Modèle multi-output
            model = MultiOutputClassifier(base_model, n_jobs=-1)
            
            self._log("   🔄 Entraînement du modèle RandomForest multi-label...")
            model.fit(X_train, y_train)
            
            # Évaluation rapide
            y_pred = model.predict(X_test)
            
            # Calcul de l'accuracy moyenne sur tous les labels
            accuracies = []
            for i in range(y.shape[1]):
                acc = accuracy_score(y_test[:, i], y_pred[:, i])
                accuracies.append(acc)
            
            mean_accuracy = np.mean(accuracies)
            self._log(f"   📊 Accuracy moyenne: {mean_accuracy:.4f}")
            
            # Validation croisée
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(base_model, X_train, y_train, cv=5, n_jobs=-1)
            self._log(f"   📊 Validation croisée (CV=5): Score moyen = {np.mean(cv_scores):.4f}")
            
            # Sauvegarde du modèle
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            self.ml_models['multilabel'] = model
            
            # Sauvegarde des métadonnées
            self.metadata = {
                'features_count': len(feature_cols),
                'model_type': 'xgboost_keno_multilabel',
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'version': '3.0',  # Version avec multi-label
                'keno_params': KENO_PARAMS,
                'mean_accuracy': mean_accuracy,
                'feature_names': feature_cols
            }
            
            metadata_path = self.models_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            self._log("✅ Entraînement du modèle multi-label Keno terminé")
            return True
            
        except Exception as e:
            self._log(f"❌ Erreur lors de l'entraînement: {e}", "ERROR")
            import traceback
            self._log(f"Détails: {traceback.format_exc()}")
            return False
    
    def load_ml_models(self) -> bool:
        """Charge le modèle ML multi-label pré-entraîné"""
        try:
            self._log("📥 Chargement du modèle ML Keno multi-label...")
            
            # Chargement des métadonnées
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Chargement du modèle multi-label
            model_path = self.models_dir / "xgb_keno_multilabel.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.ml_models['multilabel'] = pickle.load(f)
                
                self._log("✅ Modèle multi-label Keno chargé avec succès")
                if 'mean_accuracy' in self.metadata:
                    self._log(f"   📊 Accuracy du modèle: {self.metadata['mean_accuracy']:.4f}")
                return True
            else:
                self._log("❌ Modèle multi-label non trouvé", "ERROR")
                return False
                
        except Exception as e:
            self._log(f"❌ Erreur lors du chargement du modèle: {e}", "ERROR")
            return False
    
    def predict_numbers_ml(self, num_grids: int = 10) -> List[List[int]]:
        """
        Prédit les numéros avec le modèle ML multi-label
        
        Args:
            num_grids: Nombre de grilles à générer
            
        Returns:
            List[List[int]]: Liste des grilles prédites
        """
        if 'multilabel' not in self.ml_models:
            self._log("❌ Modèle ML multi-label non disponible", "ERROR")
            return []
        
        try:
            model = self.ml_models['multilabel']
            
            # Préparation des features pour la prédiction
            current_date = pd.Timestamp.now()
            df_predict = pd.DataFrame({
                'date_de_tirage': [current_date],
                **{f'boule{i}': [0] for i in range(1, 21)}  # Valeurs dummy
            })
            
            # Ajout des features temporelles ET enrichies (comme à l'entraînement)
            df_features = self.add_cyclic_features(df_predict)
            df_features = self.enrich_features(df_features)
            
            # Features d'historique
            lag_features = {}
            if self.stats and self.stats.derniers_tirages:
                for lag in range(1, 6):
                    for ball_num in range(1, 21):
                        col_name = f'lag{lag}_boule{ball_num}'
                        if lag <= len(self.stats.derniers_tirages):
                            last_draw = self.stats.derniers_tirages[-(lag)]
                            lag_features[col_name] = last_draw[ball_num - 1] if ball_num <= len(last_draw) else 0
                        else:
                            lag_features[col_name] = 0
                lag_df = pd.DataFrame([lag_features], index=df_features.index)
            else:
                # Valeurs par défaut si pas d'historique
                lag_features = {f'lag{lag}_boule{ball_num}': 0 for lag in range(1, 6) for ball_num in range(1, 21)}
                lag_df = pd.DataFrame([lag_features], index=df_features.index)

            df_features = pd.concat([df_features, lag_df], axis=1)

            # Features de fréquence par zone
            zone_features = {
                'zone1_count': [0],
                'zone2_count': [0],
                'zone3_count': [0],
                'zone4_count': [0],
                'zone1_freq': [0],
                'zone2_freq': [0],
                'zone3_freq': [0],
                'zone4_freq': [0]
            }
            
            for zone_name, zone_values in zone_features.items():
                df_features[zone_name] = zone_values

            # AJOUT CRUCIAL: Features statistiques étendues pour TOUS les numéros (1-70)
            if self.stats:
                stats_data = {
                    f'freq_recent_{num}': self.stats.frequences_recentes.get(num, 0)
                    for num in range(1, 71)
                }
                stats_data.update({
                    f'retard_{num}': self.stats.retards.get(num, 0)
                    for num in range(1, 71)
                })
                stats_data.update({
                    f'tendance_{num}': self.stats.tendances_50.get(num, 1.0)
                    for num in range(1, 71)
                })
                stats_df = pd.DataFrame([stats_data], index=df_features.index)
                df_features = pd.concat([df_features, stats_df], axis=1)
            else:
                # Créer des valeurs par défaut pour toutes les features statistiques
                stats_data = {}
                for num in range(1, 71):
                    stats_data[f'freq_recent_{num}'] = 0
                    stats_data[f'retard_{num}'] = 0
                    stats_data[f'tendance_{num}'] = 1.0
                stats_df = pd.DataFrame([stats_data], index=df_features.index)
                df_features = pd.concat([df_features, stats_df], axis=1)

            # Utiliser la liste de features de l'entraînement
            feature_cols = self.metadata.get('feature_names', [])

            # CORRECTION DÉFINITIVE: Garantir exactement les mêmes features
            for c in feature_cols:
                if c not in df_features.columns:
                    df_features[c] = 0.0

            # Supprimer les colonnes en trop et réordonner exactement comme à l'entraînement
            df_features = df_features.loc[:, feature_cols]

            X_pred = df_features.fillna(0)
            X_pred = X_pred.select_dtypes(include=[np.number])

            # Vérification du nombre de features
            if X_pred.shape[1] != len(feature_cols):
                self._log(f"⚠️  Mismatch: {X_pred.shape[1]} features vs {len(feature_cols)} attendues", "ERROR")
                return []

            # Normalisation des features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_pred_scaled = scaler.fit_transform(X_pred)

            # Prédiction des probabilités pour tous les numéros
            probabilities = model.predict_proba(X_pred_scaled)

            number_probs = []
            for i in range(min(KENO_PARAMS['total_numbers'], len(probabilities))):
                num = i + 1
                if len(probabilities[i][0]) > 1:
                    prob = probabilities[i][0][1]
                else:
                    prob = 0.5
                number_probs.append((num, prob))

            number_probs.sort(key=lambda x: x[1], reverse=True)
            top30 = number_probs[:30]

            self._log(f"✅ TOP 30 ML calculé - Probabilité moyenne: {np.mean([prob for _, prob in top30]):.4f}")
            return top30

        except Exception as e:
            self._log(f"❌ Erreur lors du calcul TOP 30 ML: {e}", "ERROR")
            import traceback
            self._log(f"Détails: {traceback.format_exc()}")
            return []

    def get_top30_numbers_advanced(self) -> list:
        """
        Sélectionne les 30 meilleurs numéros selon un score composite avancé
        (fréquence globale, fréquence récente, retard, tendance, score ML, paires/trios)
        """
        if not self.stats:
            self._log("Stats non disponibles, impossible de calculer le TOP 30.", "ERROR")
            return []
        scores = {}
        max_freq = max(self.stats.frequences.values()) if self.stats.frequences else 1
        max_freq_recent = max(self.stats.frequences_recentes.values()) if self.stats.frequences_recentes else 1
        max_retard = max(self.stats.retards.values()) if self.stats.retards else 1
        max_tendance = max(self.stats.tendances_50.values()) if self.stats.tendances_50 else 1.0

        # Pondérations dynamiques (exemple)
        ml_weight = 0.15
        freq_weight = 0.25
        recent_weight = 0.20
        retard_weight = 0.15
        tendance_weight = 0.15
        pair_weight = 0.05
        trio_weight = 0.05
        lift_weight = 0.10
        diversity_penalty = 0.10

        # Calcul du lift des paires
        def lift(i, j):
            cofreq = self.stats.paires_freq.get(tuple(sorted([i, j])), 0)
            freq_i = self.stats.frequences.get(i, 1)
            freq_j = self.stats.frequences.get(j, 1)
            return cofreq / (freq_i * freq_j) if freq_i and freq_j else 0

        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            score = 0.0
            score += (self.stats.frequences.get(num, 0) / max_freq) * freq_weight
            score += (self.stats.frequences_recentes.get(num, 0) / max_freq_recent) * recent_weight
            score += (1 - self.stats.retards.get(num, 0) / max_retard) * retard_weight
            score += (self.stats.tendances_50.get(num, 1.0) / max_tendance) * tendance_weight
            if hasattr(self, 'ml_scores') and self.ml_scores and num in self.ml_scores:
                score += self.ml_scores[num] * ml_weight

            # Bonus lift sur les paires
            lift_score = 0
            for other in range(1, 71):
                if other != num:
                    lift_score += lift(num, other)
            score += (lift_score / 69) * lift_weight

            # Bonus pour les paires/trios fréquents
            pair_bonus = sum([self.stats.paires_freq.get(tuple(sorted([num, other])), 0) for other in range(1, 71) if other != num])
            score += (pair_bonus / 1000) * pair_weight
            trio_bonus = sum([self.stats.trios_freq.get(tuple(sorted([num, other1, other2]),), 0)
                              for other1 in range(1, 71) for other2 in range(other1+1, 71)
                              if other1 != num and other2 != num])
            score += (trio_bonus / 5000) * trio_weight

            # Diversité : pénalité si trop de numéros dans la même dizaine ou même parité
            dizaine = (num - 1) // 10
            parite = num % 2
            same_dizaine = sum(1 for other in range(1, 71) if other != num and (other - 1) // 10 == dizaine)
            same_parite = sum(1 for other in range(1, 71) if other != num and other % 2 == parite)
            score -= ((same_dizaine / 69) + (same_parite / 69)) * diversity_penalty

            scores[num] = score

        top30 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:30]
        return [num for num, score in top30]
        
    def run_full_pipeline(self, num_grids: int = 40, profile: str = "balanced"):
        """
        Pipeline complet : chargement des données, analyse, entraînement ML, génération des grilles.
        """
        self._log("🚀 Démarrage du pipeline complet Keno...")
        if not self.load_data():
            self._log("❌ Chargement des données impossible.", "ERROR")
            return
        self.stats = self.analyze_patterns()
        self.train_xgboost_models(retrain=False)
        self.load_ml_models()
        self._log("✅ Pipeline complet terminé.")

    def generate_optimized_grids(self, num_grids: int = 40) -> list:
        """
        Génère des grilles optimisées en privilégiant le TOP 30 ML.
        """
        self._log(f"🎯 Génération de {num_grids} grilles Keno optimisées (TOP 30 ML privilégié, {self.grid_size} numéros/grille)...")
        top30_ml = [num for num, _ in self.predict_numbers_ml()]
        if not top30_ml or len(top30_ml) < self.grid_size:
            self._log("❌ TOP 30 ML indisponible, génération aléatoire.", "ERROR")
            # Fallback : génération aléatoire
            return [sorted(random.sample(range(1, 71), self.grid_size)) for _ in range(num_grids)]
        grids = []
        for _ in range(num_grids):
            grid = sorted(random.sample(top30_ml, self.grid_size))
            grids.append(grid)
        self._log(f"✅ {len(grids)} grilles de {self.grid_size} numéros générées à partir du TOP 30 ML")
        return grids

    def save_results(self, grids: list):
        """Sauvegarde les grilles générées dans un fichier CSV avec colonnes adaptées à la taille"""
        output_path = self.output_dir / "grilles_keno.csv"
        columns = [f"numero_{i}" for i in range(1, self.grid_size + 1)]
        df = pd.DataFrame(grids, columns=columns)
        df.to_csv(output_path, index=False)
        self._log(f"💾 Grilles de {self.grid_size} numéros sauvegardées dans {output_path}")

    def save_top30_ml_csv(self):
        """Sauvegarde le TOP 30 ML dans un fichier CSV"""
        top30_ml = [num for num, _ in self.predict_numbers_ml()]
        output_path = self.output_dir / "top30_ml.csv"
        df = pd.DataFrame(top30_ml, columns=["Numéro"])
        df.to_csv(output_path, index=False)
        self._log(f"💾 TOP 30 ML sauvegardé dans {output_path}")

    def generate_report(self, grids: list) -> str:
        """
        Génère un rapport détaillé sur les grilles produites.
        """
        report = "# Rapport détaillé des grilles Keno\n\n"
        report += f"Nombre de grilles générées : {len(grids)}\n\n"
        report += "## Grilles\n"
        for idx, grid in enumerate(grids, 1):
            report += f"- Grille {idx}: {grid}\n"
        report += "\n"
        # Statistiques sur la diversité des numéros
        all_numbers = [num for grid in grids for num in grid]
        unique_numbers = set(all_numbers)
        report += f"Nombre total de numéros uniques utilisés : {len(unique_numbers)}\n"
        report += f"Liste des numéros uniques : {sorted(unique_numbers)}\n"
        return report

    def update_and_retrain(self):
        """
        Recharge les données, analyse, et réentraîne le modèle ML sur tous les tirages.
        """
        self._log("🔄 Mise à jour des données et réentraînement du modèle ML...")
        self.load_data()
        self.stats = self.analyze_patterns()
        self.train_xgboost_models(retrain=True)
        self.load_ml_models()
        self._log("✅ Modèle ML réentraîné avec les nouveaux tirages.")

    def evaluate_grids_with_model(self, grids: list) -> list:
        """
        Évalue chaque grille générée avec le modèle ML et retourne les scores/probabilités.
        """
        if 'multilabel' not in self.ml_models:
            self._log("❌ Modèle ML non disponible pour l'évaluation.", "ERROR")
            return []
        model = self.ml_models['multilabel']
        scores = []
        for grid in grids:
            current_date = pd.Timestamp.now()
            # Adapter le DataFrame aux grilles de taille variable
            df_data = {'date_de_tirage': [current_date]}
            for i in range(1, 21):
                if i <= len(grid):
                    df_data[f'boule{i}'] = [grid[i-1]]
                else:
                    df_data[f'boule{i}'] = [0]  # Padding avec des zéros
            
            df = pd.DataFrame(df_data)

            # Ajout des features temporelles ET enrichies (comme à l'entraînement)
            df_features = self.add_cyclic_features(df)
            df_features = self.enrich_features(df_features)

            # Reconstruction des features d'historique - EXACTEMENT comme à l'entraînement
            lag_features = {}
            if self.stats and self.stats.derniers_tirages:
                for lag in range(1, 6):
                    for ball_num in range(1, 21):
                        col_name = f'lag{lag}_boule{ball_num}'
                        if lag <= len(self.stats.derniers_tirages):
                            last_draw = self.stats.derniers_tirages[-(lag)]
                            lag_features[col_name] = last_draw[ball_num - 1] if ball_num <= len(last_draw) else 0
                        else:
                            lag_features[col_name] = 0
                lag_df = pd.DataFrame([lag_features], index=df_features.index)
            else:
                # Valeurs par défaut si pas d'historique
                lag_features = {f'lag{lag}_boule{ball_num}': 0 for lag in range(1, 6) for ball_num in range(1, 21)}
                lag_df = pd.DataFrame([lag_features], index=df_features.index)

            df_features = pd.concat([df_features, lag_df], axis=1)

            # Features de fréquence par zone - EXACTEMENT comme à l'entraînement
            zone_features = {
                'zone1_count': [],
                'zone2_count': [],
                'zone3_count': [],
                'zone4_count': [],
                'zone1_freq': [],
                'zone2_freq': [],
                'zone3_freq': [],
                'zone4_freq': []
            }
            
            for idx, row in df_features.iterrows():
                draw_numbers = [int(row[f'boule{i}']) for i in range(1, 21) if row[f'boule{i}'] > 0]
                zone_features['zone1_count'].append(sum(1 for n in draw_numbers if 1 <= n <= 17))
                zone_features['zone2_count'].append(sum(1 for n in draw_numbers if 18 <= n <= 35))
                zone_features['zone3_count'].append(sum(1 for n in draw_numbers if 36 <= n <= 52))
                zone_features['zone4_count'].append(sum(1 for n in draw_numbers if 53 <= n <= 70))
                zone_features['zone1_freq'].append(0)
                zone_features['zone2_freq'].append(0)
                zone_features['zone3_freq'].append(0)
                zone_features['zone4_freq'].append(0)

            # Ajout des features de zones
            for zone_name, zone_values in zone_features.items():
                if len(zone_values) != len(df_features):
                    zone_values = [0] * len(df_features)
                df_features[zone_name] = zone_values

            # AJOUT CRUCIAL: Features statistiques étendues comme à l'entraînement
            if self.stats:
                stats_data = {
                    f'freq_recent_{num}': self.stats.frequences_recentes.get(num, 0)
                    for num in range(1, 71)
                }
                stats_data.update({
                    f'retard_{num}': self.stats.retards.get(num, 0)
                    for num in range(1, 71)
                })
                stats_data.update({
                    f'tendance_{num}': self.stats.tendances_50.get(num, 1.0)
                    for num in range(1, 71)
                })
                stats_df = pd.DataFrame([stats_data], index=df_features.index)
                df_features = pd.concat([df_features, stats_df], axis=1)
            else:
                # Créer des valeurs par défaut pour toutes les features statistiques
                stats_data = {}
                for num in range(1, 71):
                    stats_data[f'freq_recent_{num}'] = 0
                    stats_data[f'retard_{num}'] = 0
                    stats_data[f'tendance_{num}'] = 1.0
                stats_df = pd.DataFrame([stats_data], index=df_features.index)
                df_features = pd.concat([df_features, stats_df], axis=1)

            # Utiliser la liste de features de l'entraînement
            feature_cols = self.metadata.get('feature_names', [])

            # CORRECTION DÉFINITIVE: Ajouter toutes les colonnes manquantes avec 0.0
            for c in feature_cols:
                if c not in df_features.columns:
                    df_features[c] = 0.0

            # Supprimer les colonnes en trop et réordonner exactement comme à l'entraînement
            df_features = df_features.loc[:, feature_cols]

            # Vérification finale du nombre de features
            if df_features.shape[1] != len(feature_cols):
                self._log(f"⚠️  Mismatch: {df_features.shape[1]} features vs {len(feature_cols)} attendues", "ERROR")
                scores.append((grid, 0.0))
                continue

            X_eval = df_features.fillna(0)
            X_eval = X_eval.select_dtypes(include=[np.number])

            # Normalisation des features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_eval_scaled = scaler.fit_transform(X_eval)

            # Prédiction des probabilités pour tous les numéros
            probabilities = model.predict_proba(X_eval_scaled)

            # Calcul du score moyen de la grille
            grid_score = 0.0
            count = 0
            for num in grid:
                if 1 <= num <= KENO_PARAMS['total_numbers']:
                    idx = num - 1  # Index 0-69 pour numéros 1-70
                    if idx < len(probabilities):
                        if len(probabilities[idx]) > 0 and len(probabilities[idx][0]) > 1:
                            prob = probabilities[idx][0][1]
                            grid_score += prob
                            count += 1
            
            if count > 0:
                grid_score /= count
            
            scores.append((grid, grid_score))

        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur avancé de grilles Keno")
    parser.add_argument("--n", type=int, default=10, help="Nombre de grilles à générer")
    parser.add_argument("--grids", type=int, help="Alias pour --n (nombre de grilles à générer)")
    parser.add_argument("--size", type=int, default=10, choices=[7, 8, 9, 10], help="Taille des grilles (7 à 10 numéros)")
    parser.add_argument("--profile", type=str, default="balanced", help="Profil d'entraînement ML (quick, balanced, comprehensive, intensive)")
    parser.add_argument("--data", type=str, default=None, help="Chemin du fichier de données Keno")
    parser.add_argument("--silent", action="store_true", help="Mode silencieux")
    parser.add_argument("--retrain", action="store_true", help="Forcer le réentraînement du modèle ML")
    parser.add_argument("--save-top30-ml", action="store_true", help="Sauvegarder le TOP 30 ML dans un CSV")
    parser.add_argument("--test-grids", action="store_true", help="Évaluer les grilles générées avec le modèle ML")
    args = parser.parse_args()

    # Gestion de l'alias --grids
    num_grids = args.grids if args.grids is not None else args.n

    generator = KenoGeneratorAdvanced(
        data_path=args.data,
        silent=args.silent,
        training_profile=args.profile,
        grid_size=args.size
    )

    if args.retrain:
        generator.update_and_retrain()
    else:
        generator.run_full_pipeline(num_grids=num_grids, profile=args.profile)

    grids = generator.generate_optimized_grids(num_grids=num_grids)
    generator.save_results(grids)

    if args.save_top30_ml:
        generator.save_top30_ml_csv()

    if args.test_grids:
        grid_scores = generator.evaluate_grids_with_model(grids)
        print("Scores des grilles générées :")
        for i, score in enumerate(grid_scores, 1):
            print(f"Grille {i}: {score}")

    print(generator.generate_report(grids))
