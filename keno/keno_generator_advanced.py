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
    'min_selection': 2,         # Minimum de numéros sélectionnables
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
    
    def __init__(self, data_path: str = None, silent: bool = False, training_profile: str = "balanced"):
        """
        Initialise le générateur Keno
        
        Args:
            data_path: Chemin vers les données historiques
            silent: Mode silencieux pour réduire les sorties
            training_profile: Profil d'entraînement ('quick', 'balanced', 'comprehensive', 'intensive')
        """
        self.silent = silent
        self.data_path = data_path or str(DATA_DIR / "keno_202010.parquet")
        self.models_dir = MODELS_DIR
        self.output_dir = OUTPUT_DIR
        self.training_profile = training_profile
        
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
        
        self._log("🎲 Générateur Keno Avancé v2.0 initialisé")
        
    def _log(self, message: str, level: str = "INFO"):
        """Système de logging configuré"""
        if not self.silent or level == "ERROR":
            print(f"{message}")
    
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
                draw = [int(row[f'b{i}']) for i in range(1, 21)]
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
                'zone4_count': []
            }
            
            for idx, row in df_features.iterrows():
                draw_numbers = [int(row[f'boule{i}']) for i in range(1, 21)]
                zone_features['zone1_count'].append(sum(1 for n in draw_numbers if 1 <= n <= 17))
                zone_features['zone2_count'].append(sum(1 for n in draw_numbers if 18 <= n <= 35))
                zone_features['zone3_count'].append(sum(1 for n in draw_numbers if 36 <= n <= 52))
                zone_features['zone4_count'].append(sum(1 for n in draw_numbers if 53 <= n <= 70))
            
            # Ajout des features de zones
            for zone_name, zone_values in zone_features.items():
                df_features[zone_name] = zone_values
                feature_cols.append(zone_name)
            
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
            predictions = []
            model = self.ml_models['multilabel']
            
            # Préparation des features pour la prédiction
            current_date = pd.Timestamp.now()
            df_predict = pd.DataFrame({
                'date_de_tirage': [current_date],
                **{f'boule{i}': [0] for i in range(1, 21)}  # Valeurs dummy
            })
            
            # Ajout des features temporelles
            df_features = self.add_cyclic_features(df_predict)
            
            # Reconstruction des mêmes features que lors de l'entraînement (optimisé)
            feature_cols = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
            
            # Features d'historique (utiliser les derniers tirages disponibles)
            lag_features = {}
            
            if self.stats and self.stats.derniers_tirages:
                for lag in range(1, 6):
                    for ball_num in range(1, 21):
                        col_name = f'lag{lag}_boule{ball_num}'
                        # Utiliser les derniers tirages pour construire l'historique
                        if lag <= len(self.stats.derniers_tirages):
                            last_draw = self.stats.derniers_tirages[-(lag)]
                            if ball_num <= len(last_draw):
                                lag_features[col_name] = last_draw[ball_num - 1] if ball_num <= len(last_draw) else 0
                            else:
                                lag_features[col_name] = 0
                        else:
                            lag_features[col_name] = 0
                        feature_cols.append(col_name)
            else:
                # Valeurs par défaut si pas d'historique
                for lag in range(1, 6):
                    for ball_num in range(1, 21):
                        col_name = f'lag{lag}_boule{ball_num}'
                        lag_features[col_name] = 0
                        feature_cols.append(col_name)
            
            # Ajout des features d'historique en une fois pour éviter la fragmentation
            lag_df = pd.DataFrame(lag_features, index=df_features.index)
            
            # Features de zones (moyennes historiques)
            zone_features = {}
            if self.stats:
                total_draws = len(self.stats.derniers_tirages) if self.stats.derniers_tirages else 1
                zone_features['zone1_count'] = self.stats.zones_freq.get("zone1_17", 0) / total_draws
                zone_features['zone2_count'] = self.stats.zones_freq.get("zone18_35", 0) / total_draws 
                zone_features['zone3_count'] = self.stats.zones_freq.get("zone36_52", 0) / total_draws
                zone_features['zone4_count'] = self.stats.zones_freq.get("zone53_70", 0) / total_draws
            else:
                zone_features['zone1_count'] = 5  # Valeurs moyennes
                zone_features['zone2_count'] = 5
                zone_features['zone3_count'] = 5
                zone_features['zone4_count'] = 5
            
            # Création du DataFrame pour les zones
            zone_df = pd.DataFrame([zone_features], index=df_features.index)
            
            # Concaténation de toutes les nouvelles features en une seule fois
            df_features = pd.concat([df_features, lag_df, zone_df], axis=1)
                
            feature_cols.extend(['zone1_count', 'zone2_count', 'zone3_count', 'zone4_count'])
            
            # Préparation des features pour la prédiction
            X_pred = df_features[feature_cols].fillna(0)
            
            # Génération de grilles avec variation
            for grid_idx in range(num_grids):
                # Ajout de petites variations aléatoires pour diversifier les prédictions
                X_pred_variant = X_pred.copy()
                
                # Petites variations sur les features temporelles
                noise_factor = 0.1 * (grid_idx / max(1, num_grids - 1))  # Variation progressive
                X_pred_variant['day_sin'] += np.random.normal(0, noise_factor)
                X_pred_variant['day_cos'] += np.random.normal(0, noise_factor)
                
                # Prédiction des probabilités pour tous les numéros
                probabilities = model.predict_proba(X_pred_variant)
                
                # Extraction des probabilités pour la classe positive (numéro tiré)
                # probabilities est une liste de probabilités pour chaque output
                number_probs = {}
                for i in range(KENO_PARAMS['total_numbers']):
                    num = i + 1  # Numéros de 1 à 70
                    if len(probabilities[i][0]) > 1:  # Vérifier qu'on a bien les 2 classes
                        prob = probabilities[i][0][1]  # Probabilité de la classe positive
                    else:
                        prob = 0.5  # Valeur par défaut
                    number_probs[num] = prob
                
                # Simulation d'un tirage réaliste Keno
                # Dans un vrai tirage, on tire 20 numéros, mais le joueur en sélectionne 10
                
                # Sélection des numéros avec pondération par probabilité
                numbers = list(range(1, KENO_PARAMS['total_numbers'] + 1))
                probs = [number_probs[num] for num in numbers]
                
                # Normalisation des probabilités
                probs_array = np.array(probs)
                probs_normalized = probs_array / np.sum(probs_array)
                
                # Ajustement pour tenir compte des corrélations
                # Boost des numéros qui apparaissent souvent ensemble
                if self.stats and self.stats.paires_freq:
                    correlation_boost = {num: 0 for num in numbers}
                    
                    # Identifier les paires fréquentes
                    top_pairs = sorted(self.stats.paires_freq.items(), 
                                     key=lambda x: x[1], reverse=True)[:50]
                    
                    for (num1, num2), freq in top_pairs:
                        if num1 in number_probs and num2 in number_probs:
                            # Si les deux numéros ont des probabilités élevées, boost mutuel
                            if number_probs[num1] > 0.5 and number_probs[num2] > 0.5:
                                correlation_boost[num1] += 0.1 * (freq / 1000)
                                correlation_boost[num2] += 0.1 * (freq / 1000)
                    
                    # Application du boost
                    for num in numbers:
                        idx = num - 1
                        probs_normalized[idx] += correlation_boost[num]
                    
                    # Re-normalisation
                    probs_normalized = probs_normalized / np.sum(probs_normalized)
                
                # Sélection pondérée de 10 numéros (pour le joueur)
                try:
                    selected = np.random.choice(
                        numbers, 
                        size=10, 
                        replace=False, 
                        p=probs_normalized
                    )
                    grid_numbers = sorted(selected.tolist())
                except:
                    # Fallback en cas d'erreur
                    top_indices = np.argsort(probs_normalized)[-10:]
                    grid_numbers = sorted([numbers[i] for i in top_indices])
                
                predictions.append(grid_numbers)
            
            return predictions
            
        except Exception as e:
            self._log(f"❌ Erreur lors de la prédiction ML: {e}", "ERROR")
            import traceback
            self._log(f"Détails: {traceback.format_exc()}")
            return []
    
    def generate_frequency_based_grids(self, num_grids: int = 10) -> List[List[int]]:
        """
        Génère des grilles basées sur l'analyse complète des patterns
        
        Args:
            num_grids: Nombre de grilles à générer
            
        Returns:
            List[List[int]]: Liste des grilles générées optimisées
        """
        if not self.stats:
            self._log("❌ Statistiques non disponibles", "ERROR")
            return []
        
        self._log(f"🎯 Génération de {num_grids} grilles avec analyse complète des patterns")
        grids = []
        
        # 1. ANALYSE DES FRÉQUENCES SUR DIFFÉRENTES PÉRIODES
        freq_global = sorted(self.stats.frequences.items(), key=lambda x: x[1], reverse=True)
        freq_recente = sorted(self.stats.frequences_recentes.items(), key=lambda x: x[1], reverse=True)
        freq_50 = sorted(self.stats.frequences_50.items(), key=lambda x: x[1], reverse=True)
        freq_20 = sorted(self.stats.frequences_20.items(), key=lambda x: x[1], reverse=True)
        
        # 2. ANALYSE DES RETARDS (OVERDUE NUMBERS)
        retard_sorted = sorted(self.stats.retards.items(), key=lambda x: x[1], reverse=True)
        
        # 3. NUMÉROS CHAUDS ET FROIDS SELON DIFFÉRENTES PÉRIODES
        hot_global = [num for num, _ in freq_global[:25]]           # Top 25 historique
        hot_recent = [num for num, _ in freq_recente[:20]]          # Top 20 récent (100 tirages)
        hot_tendance = [num for num, _ in freq_20[:15]]             # Top 15 tendance (20 tirages)
        cold_retard = [num for num, _ in retard_sorted[:30]]        # Top 30 en retard
        
        # 4. ANALYSE DES TENDANCES
        tendances_positives = []
        for num, tendance in self.stats.tendances_50.items():
            if tendance > 1.2:  # Numéros en forte hausse
                tendances_positives.append((num, tendance))
        tendances_positives.sort(key=lambda x: x[1], reverse=True)
        hot_tendances = [num for num, _ in tendances_positives[:15]]
        
        # 5. PAIRES ET TRIOS FRÉQUENTS
        top_pairs = sorted(self.stats.paires_freq.items(), key=lambda x: x[1], reverse=True)[:100]
        if self.stats.trios_freq:
            top_trios = sorted(self.stats.trios_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        else:
            top_trios = []
        
        # 6. GÉNÉRATION DE GRILLES DIVERSIFIÉES
        strategies = [
            ("hot_global", "Numéros chauds historiques"),
            ("hot_recent", "Numéros chauds récents"),
            ("cold_retard", "Numéros en retard"),
            ("hot_tendances", "Tendances positives"),
            ("mixed_smart", "Mix intelligent"),
            ("pairs_based", "Basé sur les paires"),
            ("balanced_zones", "Équilibrage par zones"),
            ("pattern_parité", "Pattern parité optimisé")
        ]
        
        for i in range(num_grids):
            strategy_name, strategy_desc = strategies[i % len(strategies)]
            grid = []
            
            if strategy_name == "hot_global":
                # Grille basée sur les numéros historiquement fréquents
                grid = self._generate_hot_global_grid(hot_global)
                
            elif strategy_name == "hot_recent":
                # Grille basée sur les tendances récentes
                grid = self._generate_hot_recent_grid(hot_recent, hot_tendance)
                
            elif strategy_name == "cold_retard":
                # Grille basée sur les numéros en retard
                grid = self._generate_cold_retard_grid(cold_retard)
                
            elif strategy_name == "hot_tendances":
                # Grille basée sur les tendances positives
                grid = self._generate_tendance_grid(hot_tendances, hot_recent)
                
            elif strategy_name == "mixed_smart":
                # Mix intelligent de tous les critères
                grid = self._generate_mixed_smart_grid(hot_global, hot_recent, cold_retard, hot_tendances)
                
            elif strategy_name == "pairs_based":
                # Grille basée sur les paires fréquentes
                grid = self._generate_pairs_based_grid(top_pairs)
                
            elif strategy_name == "balanced_zones":
                # Grille équilibrée par zones
                grid = self._generate_balanced_zones_grid()
                
            elif strategy_name == "pattern_parité":
                # Grille optimisée pour la parité
                grid = self._generate_parite_optimized_grid(hot_global, hot_recent)
            
            # Validation et ajustement de la grille
            if len(grid) < 10:
                # Compléter avec des numéros manquants
                available = [n for n in range(1, 71) if n not in grid]
                weights = [self.stats.frequences.get(n, 1) for n in available]
                while len(grid) < 10 and available:
                    selected = random.choices(available, weights=weights, k=1)[0]
                    grid.append(selected)
                    idx = available.index(selected)
                    available.pop(idx)
                    weights.pop(idx)
            
            grid = sorted(grid[:10])
            grids.append(grid)
            self._log(f"   ✅ Grille {i+1}: {strategy_desc}")
        
        return grids
    
    def _generate_hot_global_grid(self, hot_global: List[int]) -> List[int]:
        """Génère une grille basée sur les numéros historiquement fréquents"""
        return random.sample(hot_global, min(10, len(hot_global)))
    
    def _generate_hot_recent_grid(self, hot_recent: List[int], hot_tendance: List[int]) -> List[int]:
        """Génère une grille basée sur les tendances récentes"""
        grid = []
        # 6 numéros récents + 4 tendances
        grid.extend(random.sample(hot_recent, min(6, len(hot_recent))))
        available_tendance = [n for n in hot_tendance if n not in grid]
        grid.extend(random.sample(available_tendance, min(4, len(available_tendance))))
        return grid
    
    def _generate_cold_retard_grid(self, cold_retard: List[int]) -> List[int]:
        """Génère une grille basée sur les numéros en retard"""
        # Stratégie: les numéros en retard ont plus de chances de sortir
        return random.sample(cold_retard, min(10, len(cold_retard)))
    
    def _generate_tendance_grid(self, hot_tendances: List[int], hot_recent: List[int]) -> List[int]:
        """Génère une grille basée sur les tendances positives"""
        grid = []
        # 7 tendances + 3 récents
        grid.extend(random.sample(hot_tendances, min(7, len(hot_tendances))))
        available_recent = [n for n in hot_recent if n not in grid]
        grid.extend(random.sample(available_recent, min(3, len(available_recent))))
        return grid
    
    def _generate_mixed_smart_grid(self, hot_global: List[int], hot_recent: List[int], 
                                  cold_retard: List[int], hot_tendances: List[int]) -> List[int]:
        """Génère un mix intelligent de tous les critères"""
        grid = []
        
        # 3 numéros historiquement chauds
        grid.extend(random.sample(hot_global, min(3, len(hot_global))))
        
        # 3 numéros récemment chauds (non déjà sélectionnés)
        available_recent = [n for n in hot_recent if n not in grid]
        grid.extend(random.sample(available_recent, min(3, len(available_recent))))
        
        # 2 numéros en retard
        available_cold = [n for n in cold_retard if n not in grid]
        grid.extend(random.sample(available_cold, min(2, len(available_cold))))
        
        # 2 numéros en tendance positive
        available_tendance = [n for n in hot_tendances if n not in grid]
        grid.extend(random.sample(available_tendance, min(2, len(available_tendance))))
        
        return grid
    
    def _generate_pairs_based_grid(self, top_pairs: List[Tuple[Tuple[int, int], int]]) -> List[int]:
        """Génère une grille basée sur les paires fréquentes"""
        grid = []
        used_numbers = set()
        
        # Sélectionner des paires fréquentes
        for (num1, num2), freq in top_pairs[:20]:  # Top 20 paires
            if len(grid) >= 8:  # Laisser de la place pour 2 autres numéros
                break
            if num1 not in used_numbers and num2 not in used_numbers:
                grid.extend([num1, num2])
                used_numbers.update([num1, num2])
        
        # Compléter si nécessaire
        while len(grid) < 10:
            candidates = [n for n in range(1, 71) if n not in used_numbers]
            if candidates:
                selected = random.choice(candidates)
                grid.append(selected)
                used_numbers.add(selected)
            else:
                break
        
        return grid
    
    def _generate_balanced_zones_grid(self) -> List[int]:
        """Génère une grille équilibrée par zones"""
        zones = {
            "zone1": list(range(1, 18)),      # 1-17
            "zone2": list(range(18, 36)),     # 18-35
            "zone3": list(range(36, 53)),     # 36-52
            "zone4": list(range(53, 71))      # 53-70
        }
        
        # Répartition équilibrée: 2-3 numéros par zone
        grid = []
        grid.extend(random.sample(zones["zone1"], 2))
        grid.extend(random.sample(zones["zone2"], 3))
        grid.extend(random.sample(zones["zone3"], 3))
        grid.extend(random.sample(zones["zone4"], 2))
        
        return grid
    
    def _generate_parite_optimized_grid(self, hot_global: List[int], hot_recent: List[int]) -> List[int]:
        """Génère une grille optimisée pour la parité (éviter tout pair/tout impair)"""
        # Statistiques montrent que <1% des grilles sont tout pair ou tout impair
        grid = []
        
        # Pool de numéros chauds
        hot_pool = list(set(hot_global + hot_recent))
        
        # Sélectionner 5-6 pairs et 4-5 impairs
        pairs = [n for n in hot_pool if n % 2 == 0]
        impairs = [n for n in hot_pool if n % 2 == 1]
        
        # Si pas assez dans le pool chaud, compléter
        if len(pairs) < 6:
            pairs.extend([n for n in range(2, 71, 2) if n not in pairs])
        if len(impairs) < 5:
            impairs.extend([n for n in range(1, 71, 2) if n not in impairs])
        
        # Sélection équilibrée
        grid.extend(random.sample(pairs, min(5, len(pairs))))
        grid.extend(random.sample(impairs, min(5, len(impairs))))
        
        return grid
    
    def calculate_grid_score(self, grid: List[int]) -> float:
        """
        Calcule un score de qualité pour une grille Keno
        
        Args:
            grid: Liste des numéros de la grille
            
        Returns:
            float: Score de qualité
        """
        if not self.stats:
            return 0.0
        
        score = 0.0
        
        # Score basé sur les fréquences normalisées
        total_freq = sum(self.stats.frequences.values())
        for num in grid:
            freq_score = self.stats.frequences[num] / total_freq
            score += freq_score
        
        # Bonus pour la répartition par zones
        zones = {
            "zone1_17": sum(1 for n in grid if 1 <= n <= 17),
            "zone18_35": sum(1 for n in grid if 18 <= n <= 35), 
            "zone36_52": sum(1 for n in grid if 36 <= n <= 52),
            "zone53_70": sum(1 for n in grid if 53 <= n <= 70)
        }
        
        # Pénalité pour déséquilibre extrême des zones
        zone_counts = list(zones.values())
        if max(zone_counts) <= 4 and min(zone_counts) >= 1:  # Répartition équilibrée
            score += 0.1
        
        # Bonus pour les paires fréquentes
        for i in range(len(grid)):
            for j in range(i + 1, len(grid)):
                pair = tuple(sorted([grid[i], grid[j]]))
                if pair in self.stats.paires_freq:
                    pair_freq = self.stats.paires_freq[pair]
                    score += pair_freq / 1000  # Normalisation
        
        # Score de dispersion (éviter les numéros trop consécutifs)
        consecutive_count = 0
        sorted_grid = sorted(grid)
        for i in range(len(sorted_grid) - 1):
            if sorted_grid[i + 1] - sorted_grid[i] == 1:
                consecutive_count += 1
        
        if consecutive_count <= 2:  # Maximum 2 paires consécutives acceptables
            score += 0.05
        
        return score
    
    def generate_optimized_grids(self, num_grids: int = 100) -> List[Tuple[List[int], float]]:
        """
        Génère des grilles optimisées en combinant ML et analyse fréquentielle
        
        Args:
            num_grids: Nombre de grilles à générer
            
        Returns:
            List[Tuple[List[int], float]]: Liste des grilles avec leurs scores
        """
        self._log(f"🎯 Génération de {num_grids} grilles Keno optimisées...")
        
        all_grids = []
        
        # Répartition selon les poids adaptatifs
        ml_count = int(num_grids * self.adaptive_weights['ml_weight'])
        freq_count = num_grids - ml_count
        
        self._log(f"   🤖 Poids adaptatifs: ML={self.adaptive_weights['ml_weight']:.1%}, Freq={self.adaptive_weights['freq_weight']:.1%}")
        
        # Génération ML
        if ml_count > 0 and self.ml_models:
            ml_grids = self.predict_numbers_ml(ml_count)
            all_grids.extend(ml_grids)
        
        # Génération basée sur les fréquences
        if freq_count > 0:
            freq_grids = self.generate_frequency_based_grids(freq_count)
            all_grids.extend(freq_grids)
        
        # Calcul des scores et tri
        scored_grids = []
        for grid in all_grids:
            if len(grid) == 10:  # Validation
                score = self.calculate_grid_score(grid)
                scored_grids.append((grid, score))
        
        # Tri par score décroissant
        scored_grids.sort(key=lambda x: x[1], reverse=True)
        
        # Suppression des doublons
        unique_grids = []
        seen = set()
        for grid, score in scored_grids:
            grid_tuple = tuple(sorted(grid))
            if grid_tuple not in seen:
                unique_grids.append((grid, score))
                seen.add(grid_tuple)
        
        self._log(f"✅ {len(unique_grids)} grilles uniques générées")
        return unique_grids[:num_grids]
    
    def save_results(self, grids_with_scores: List[Tuple[List[int], float]], 
                    filename: str = None) -> str:
        """
        Sauvegarde les résultats dans un fichier CSV
        
        Args:
            grids_with_scores: Liste des grilles avec scores
            filename: Nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        if not filename:
            # Utilisation d'un nom fixe qui remplace le fichier précédent
            filename = "grilles_keno.csv"
        
        filepath = self.output_dir / filename
        
        # Préparation des données
        data_rows = []
        for grid, score in grids_with_scores:
            row = {}
            for i, num in enumerate(grid, 1):
                row[f'numero_{i}'] = num
            row['score'] = score
            data_rows.append(row)
        
        # Sauvegarde
        df_results = pd.DataFrame(data_rows)
        df_results.to_csv(filepath, index=False)
        
        self._log(f"💾 Résultats sauvegardés: {filepath}")
        return str(filepath)
    
    def generate_report(self, grids_with_scores: List[Tuple[List[int], float]]) -> str:
        """
        Génère un rapport d'analyse détaillé
        
        Args:
            grids_with_scores: Grilles avec leurs scores
            
        Returns:
            str: Contenu du rapport
        """
        if not grids_with_scores:
            return "Aucune grille générée."
        
        report = []
        report.append("# 🎲 RAPPORT D'ANALYSE KENO")
        report.append("=" * 50)
        report.append(f"📅 **Date**: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        report.append(f"📊 **Grilles générées**: {len(grids_with_scores)}")
        report.append("")
        
        # Top 5 des grilles
        report.append("## 🏆 TOP 5 DES GRILLES RECOMMANDÉES")
        report.append("")
        for i, (grid, score) in enumerate(grids_with_scores[:5], 1):
            nums_str = " - ".join(f"{n:2d}" for n in grid)
            report.append(f"**{i}.** [{nums_str}] | Score: {score:.4f}")
        report.append("")
        
        # Statistiques
        scores = [score for _, score in grids_with_scores]
        report.append("## 📈 STATISTIQUES")
        report.append("")
        report.append(f"- **Score moyen**: {np.mean(scores):.4f}")
        report.append(f"- **Score médian**: {np.median(scores):.4f}")
        report.append(f"- **Meilleur score**: {max(scores):.4f}")
        report.append(f"- **Score minimum**: {min(scores):.4f}")
        report.append("")
        
        # Analyse des numéros les plus sélectionnés
        all_numbers = []
        for grid, _ in grids_with_scores:
            all_numbers.extend(grid)
        
        num_counts = {}
        for num in all_numbers:
            num_counts[num] = num_counts.get(num, 0) + 1
        
        top_numbers = sorted(num_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report.append("## 🔥 NUMÉROS LES PLUS SÉLECTIONNÉS")
        report.append("")
        for num, count in top_numbers:
            percentage = (count / len(grids_with_scores)) * 100
            report.append(f"- **{num:2d}**: {count} fois ({percentage:.1f}%)")
        report.append("")
        
        # Stratégie adaptative
        report.append("## 🤖 STRATÉGIE ADAPTATIVE")
        report.append("")
        report.append(f"- **Poids ML**: {self.adaptive_weights['ml_weight']:.1%}")
        report.append(f"- **Poids Fréquence**: {self.adaptive_weights['freq_weight']:.1%}")
        report.append("")
        
        return "\n".join(report)
    
    def run(self, num_grids: int = 100, save_file: str = None, 
           retrain: bool = False) -> bool:
        """
        Lance la génération complète de grilles Keno
        
        Args:
            num_grids: Nombre de grilles à générer
            save_file: Nom du fichier de sauvegarde
            retrain: Force le réentraînement des modèles
            
        Returns:
            bool: True si succès
        """
        start_time = time.time()
        
        try:
            # 1. Chargement des données
            if not self.load_data():
                return False
            
            # 2. Analyse des patterns
            self.analyze_patterns()
            
            # 3. Modèles ML
            if HAS_ML:
                if retrain or not self.load_ml_models():
                    if not self.train_xgboost_models(retrain=retrain):
                        self._log("⚠️  Utilisation du mode fréquence uniquement")
                        self.adaptive_weights['ml_weight'] = 0.0
                        self.adaptive_weights['freq_weight'] = 1.0
            else:
                self._log("⚠️  Mode fréquence uniquement")
                self.adaptive_weights['ml_weight'] = 0.0
                self.adaptive_weights['freq_weight'] = 1.0
            
            # 4. Génération des grilles
            grids_with_scores = self.generate_optimized_grids(num_grids)
            
            if not grids_with_scores:
                self._log("❌ Aucune grille générée", "ERROR")
                return False
            
            # 5. Sauvegarde
            result_file = self.save_results(grids_with_scores, save_file)
            
            # 6. Rapport
            report = self.generate_report(grids_with_scores)
            
            # Affichage des résultats
            self._log("\n" + "=" * 70)
            self._log("🎯 GÉNÉRATION KENO TERMINÉE")
            self._log("=" * 70)
            
            # Top 5
            self._log("\n🏆 Top 5 des grilles recommandées:")
            for i, (grid, score) in enumerate(grids_with_scores[:5], 1):
                nums_str = " - ".join(f"{n:2d}" for n in grid)
                self._log(f"   {i}. [{nums_str}] | Score: {score:.4f}")
            
            # Statistiques
            scores = [score for _, score in grids_with_scores]
            self._log(f"\n📊 Statistiques:")
            self._log(f"   - Grilles générées: {len(grids_with_scores):,}")
            self._log(f"   - Score moyen: {np.mean(scores):.4f}")
            self._log(f"   - Meilleur score: {max(scores):.4f}")
            
            # Temps d'exécution
            elapsed = time.time() - start_time
            self._log(f"\n⏱️  Temps d'exécution: {elapsed:.2f} secondes")
            self._log(f"📁 Résultats sauvegardés: {result_file}")
            
            # Sauvegarde du rapport
            report_file = self.output_dir / "rapport_keno.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self._log(f"📄 Rapport sauvegardé: {report_file}")
            
            return True
            
        except Exception as e:
            self._log(f"❌ Erreur lors de l'exécution: {e}", "ERROR")
            return False
    
    def calculate_and_export_top30(self, export_path: Optional[str] = None) -> List[int]:
        """
        Calcule le TOP 30 des numéros Keno avec scoring intelligent optimal et l'exporte en CSV
        
        Utilise un scoring multi-critères avancé basé sur :
        - Fréquences multi-périodes (30%) : Global + Récent + Moyen terme
        - Retard intelligent (25%) : Retard optimal avec bonus zones de retard
        - Tendances dynamiques (20%) : Analyse sur 10, 50 et 100 tirages
        - Popularité paires (15%) : Bonus pour numéros dans paires fréquentes
        - Équilibrage zones (10%) : Répartition géographique optimale
        
        Args:
            export_path: Chemin d'export personnalisé (optionnel)
            
        Returns:
            List[int]: Liste des 30 meilleurs numéros
        """
        self._log("🧠 Calcul du TOP 30 Keno avec profil intelligent optimal...")
        
        if not hasattr(self, 'stats') or self.stats is None:
            self._log("⚠️ Statistiques non disponibles, analyse en cours...")
            self.analyze_patterns()
        
        # Calcul des maximums pour normalisation
        max_freq_global = max(self.stats.frequences.values()) if self.stats.frequences else 1
        max_freq_recent = max(self.stats.frequences_recentes.values()) if self.stats.frequences_recentes else 1
        max_freq_50 = max(self.stats.frequences_50.values()) if self.stats.frequences_50 else 1
        max_freq_20 = max(self.stats.frequences_20.values()) if self.stats.frequences_20 else 1
        max_retard = max(self.stats.retards.values()) if self.stats.retards else 1
        
        # Calcul du scoring intelligent optimal multi-critères
        scores = {}
        
        for numero in range(1, 71):
            score_total = 0.0
            
            # 1. FRÉQUENCES MULTI-PÉRIODES (30%) - Pondération intelligente
            freq_global = self.stats.frequences.get(numero, 0)
            freq_recent = self.stats.frequences_recentes.get(numero, 0)
            freq_50 = self.stats.frequences_50.get(numero, 0)
            freq_20 = self.stats.frequences_20.get(numero, 0)
            
            # Pondération : récent > moyen terme > global pour détecter les tendances
            freq_score = (
                (freq_global / max_freq_global) * 0.10 +      # 10% fréquence globale
                (freq_recent / max_freq_recent) * 0.08 +      # 8% fréquence 100 tirages  
                (freq_50 / max_freq_50) * 0.08 +              # 8% fréquence 50 tirages
                (freq_20 / max_freq_20) * 0.04                # 4% fréquence 20 tirages (tendance immédiate)
            ) * 100  # Normalisation sur 30 points
            score_total += freq_score
            
            # 2. RETARD INTELLIGENT (25%) - Zones de retard optimales
            retard = self.stats.retards.get(numero, 0)
            retard_normalized = retard / max_retard if max_retard > 0 else 0
            
            # Bonus pour retards dans la zone optimale (ni trop faible, ni trop élevé)
            if 0.15 <= retard_normalized <= 0.45:      # Zone optimale (15-45% du retard max)
                retard_bonus = 1.3                      # Bonus fort pour retards optimaux
            elif 0.45 < retard_normalized <= 0.70:     # Zone de retard modéré
                retard_bonus = 1.2                      # Bonus modéré
            elif retard_normalized > 0.70:             # Très en retard
                retard_bonus = 1.15                     # Léger bonus pour les grands retards
            else:                                       # Peu en retard
                retard_bonus = 0.9                      # Légère pénalité
                
            retard_score = (1 - retard_normalized) * 25 * retard_bonus
            score_total += retard_score
            
            # 3. TENDANCES DYNAMIQUES (20%) - Analyse multi-périodes
            tendance_10 = self.stats.tendances_10.get(numero, 1.0) if hasattr(self.stats, 'tendances_10') else 1.0
            tendance_50 = self.stats.tendances_50.get(numero, 1.0) if hasattr(self.stats, 'tendances_50') else 1.0
            tendance_100 = self.stats.tendances_100.get(numero, 1.0) if hasattr(self.stats, 'tendances_100') else 1.0
            
            # Moyenne pondérée des tendances (court terme > moyen terme > long terme)
            tendance_moyenne = (tendance_10 * 0.5 + tendance_50 * 0.3 + tendance_100 * 0.2)
            
            # Score basé sur la force de la tendance positive
            if tendance_moyenne > 1.3:                 # Forte tendance positive
                tendance_score = 20
            elif tendance_moyenne > 1.1:               # Tendance positive modérée
                tendance_score = 15
            elif tendance_moyenne > 0.9:               # Stable
                tendance_score = 10
            elif tendance_moyenne > 0.7:               # Tendance négative modérée
                tendance_score = 5
            else:                                       # Forte tendance négative
                tendance_score = 2
                
            score_total += tendance_score
            
            # 4. POPULARITÉ DANS LES PAIRES (15%) - Bonus pour associations fréquentes
            pair_score = 0
            if hasattr(self.stats, 'paires_freq'):
                # Compter les paires fréquentes contenant ce numéro
                paires_importantes = 0
                total_freq_paires = 0
                
                for (n1, n2), freq in self.stats.paires_freq.items():
                    if n1 == numero or n2 == numero:
                        if freq > 50:  # Seuil pour paires significatives
                            paires_importantes += 1
                            total_freq_paires += freq
                
                # Score basé sur nombre et fréquence des paires importantes
                if paires_importantes > 0:
                    pair_score = min(
                        (paires_importantes * 3) + (total_freq_paires / 200),
                        15  # Maximum 15 points
                    )
            
            score_total += pair_score
            
            # 5. ÉQUILIBRAGE ZONES (10%) - Répartition géographique optimale
            zone_score = 0
            if hasattr(self.stats, 'patterns_zones'):
                # Déterminer la zone du numéro
                if 1 <= numero <= 17:
                    zone_key = 'zone_1_17'
                elif 18 <= numero <= 35:
                    zone_key = 'zone_18_35'
                elif 36 <= numero <= 52:
                    zone_key = 'zone_36_52'
                else:  # 53-70
                    zone_key = 'zone_53_70'
                
                # Score basé sur l'activité de la zone
                zone_freq = self.stats.patterns_zones.get(zone_key, 0)
                max_zone_freq = max(self.stats.patterns_zones.values()) if self.stats.patterns_zones else 1
                
                if max_zone_freq > 0:
                    zone_score = (zone_freq / max_zone_freq) * 10
                else:
                    zone_score = 5  # Score par défaut
            else:
                zone_score = 5  # Score par défaut si pas de données zones
            
            score_total += zone_score
            
            scores[numero] = round(score_total, 3)
        
        # Tri et sélection du TOP 30 avec scores détaillés
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top30_numbers = [num for num, score in sorted_numbers[:30]]
        
        # Préparation des données enrichies pour export
        top30_data = []
        for num, score in sorted_numbers[:30]:
            # Calcul des composantes du score pour traçabilité
            freq_global = self.stats.frequences.get(num, 0)
            freq_recent = self.stats.frequences_recentes.get(num, 0)
            freq_50 = self.stats.frequences_50.get(num, 0)
            freq_20 = self.stats.frequences_20.get(num, 0)
            retard = self.stats.retards.get(num, 0)
            
            # Tendances multi-périodes
            tendance_10 = self.stats.tendances_10.get(num, 1.0) if hasattr(self.stats, 'tendances_10') else 1.0
            tendance_50 = self.stats.tendances_50.get(num, 1.0) if hasattr(self.stats, 'tendances_50') else 1.0
            tendance_100 = self.stats.tendances_100.get(num, 1.0) if hasattr(self.stats, 'tendances_100') else 1.0
            
            # Nombre de paires fréquentes
            nb_paires_freq = 0
            if hasattr(self.stats, 'paires_freq'):
                for (n1, n2), freq in self.stats.paires_freq.items():
                    if (n1 == num or n2 == num) and freq > 50:
                        nb_paires_freq += 1
            
            # Zone géographique
            if 1 <= num <= 17:
                zone = 1
                zone_nom = "1-17"
            elif 18 <= num <= 35:
                zone = 2
                zone_nom = "18-35"
            elif 36 <= num <= 52:
                zone = 3
                zone_nom = "36-52"
            else:
                zone = 4
                zone_nom = "53-70"
            
            top30_data.append({
                'Numero': num,
                'Score_Total': round(score, 2),
                'Rang': len(top30_data) + 1,
                
                # Fréquences détaillées
                'Freq_Globale': freq_global,
                'Freq_100_Derniers': freq_recent,
                'Freq_50_Derniers': freq_50,
                'Freq_20_Derniers': freq_20,
                
                # Retard et tendances
                'Retard_Actuel': retard,
                'Tendance_10': round(tendance_10, 3),
                'Tendance_50': round(tendance_50, 3),
                'Tendance_100': round(tendance_100, 3),
                
                # Associations et répartition
                'Nb_Paires_Frequentes': nb_paires_freq,
                'Zone': zone,
                'Zone_Nom': zone_nom,
                'Parite': 'Pair' if num % 2 == 0 else 'Impair'
            })
        
        # Export enrichi en CSV
        if export_path is None:
            export_path = self.output_dir / f"keno_top30.csv"
        else:
            export_path = Path(export_path)
            
        # Créer le dossier parent si nécessaire
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde du TOP 30 enrichi
        top30_df = pd.DataFrame(top30_data)
        top30_df.to_csv(export_path, index=False)
        
        # Statistiques du TOP 30 pour validation
        total_pairs = sum(1 for num in top30_numbers if num % 2 == 0)
        repartition_zones = [0, 0, 0, 0]
        for num in top30_numbers:
            zone_idx = min((num - 1) // 18, 3)
            repartition_zones[zone_idx] += 1
        
        self._log(f"✅ TOP 30 optimisé calculé et exporté:")
        self._log(f"   📁 Fichier: {export_path}")
        self._log(f"   🎯 Top 10: {', '.join(map(str, top30_numbers[:10]))}")
        self._log(f"   📊 Répartition pairs/impairs: {total_pairs}/{30-total_pairs}")
        self._log(f"   🗺️  Répartition zones: {'-'.join(map(str, repartition_zones))}")
        self._log(f"   💯 Score moyen: {sum(scores[num] for num in top30_numbers)/30:.1f}")
        
        return top30_numbers

# ==============================================================================
# 🚀 POINT D'ENTRÉE PRINCIPAL
# ==============================================================================

def main():
    """Point d'entrée principal du script"""
    
    parser = argparse.ArgumentParser(
        description="🎲 Générateur Intelligent de Grilles Keno v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python keno_generator_advanced.py --quick                    # 10 grilles, entraînement rapide
  python keno_generator_advanced.py --balanced                 # 100 grilles, entraînement équilibré (défaut)
  python keno_generator_advanced.py --comprehensive            # 500 grilles, entraînement complet
  python keno_generator_advanced.py --intensive                # 1000 grilles, entraînement intensif
  python keno_generator_advanced.py --retrain --comprehensive  # Force ré-entraînement + mode complet
  python keno_generator_advanced.py --grids 50 --silent        # 50 grilles en mode silencieux

Profils d'entraînement:
  - quick:        Rapide (50 arbres, profondeur 8)
  - balanced:     Équilibré (100 arbres, profondeur 12) [DÉFAUT]
  - comprehensive: Complet (200 arbres, profondeur 15)
  - intensive:    Intensif (300 arbres, profondeur 20)
        """
    )
    
    parser.add_argument('--grids', type=int, default=100,
                       help='Nombre de grilles à générer (défaut: 100)')
    parser.add_argument('--output', type=str,
                       help='Nom du fichier de sortie (optionnel)')
    parser.add_argument('--retrain', action='store_true',
                       help='Force le réentraînement des modèles ML')
    parser.add_argument('--quick', action='store_true',
                       help='Mode rapide (10 grilles, entraînement ultra-rapide)')
    parser.add_argument('--balanced', action='store_true',
                       help='Mode équilibré (100 grilles, entraînement standard)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Mode complet (500 grilles, entraînement optimisé)')
    parser.add_argument('--intensive', action='store_true',
                       help='Mode intensif (1000 grilles, entraînement maximal)')
    parser.add_argument('--silent', action='store_true',
                       help='Mode silencieux')
    parser.add_argument('--data', type=str,
                       help='Chemin vers les données (optionnel)')
    
    args = parser.parse_args()
    
    # Gestion des profils d'entraînement mutuellement exclusifs
    profile_count = sum([args.quick, args.balanced, args.comprehensive, args.intensive])
    if profile_count > 1:
        print("❌ Erreur: Un seul profil d'entraînement peut être sélectionné à la fois")
        sys.exit(1)
    
    # Configuration selon le profil sélectionné
    if args.quick:
        args.grids = 10
        training_profile = "quick"
    elif args.balanced:
        args.grids = 100
        training_profile = "balanced"
    elif args.comprehensive:
        args.grids = 500
        training_profile = "comprehensive"
    elif args.intensive:
        args.grids = 1000
        training_profile = "intensive"
    else:
        # Mode par défaut (équilibré)
        training_profile = "balanced"
    
    # Banner
    if not args.silent:
        profile_names = {
            'quick': 'Ultra-rapide',
            'balanced': 'Équilibré', 
            'comprehensive': 'Complet',
            'intensive': 'Intensif'
        }
        profile_display = profile_names.get(training_profile, 'Équilibré')
        
        print("=" * 70)
        print("🎲 GÉNÉRATEUR INTELLIGENT DE GRILLES KENO v2.0 🎲")
        print("=" * 70)
        print("Machine Learning + Analyse Statistique")
        print("Optimisation des combinaisons Keno")
        print(f"📊 Profil: {profile_display} ({args.grids} grilles)")
        print("=" * 70)
    
    # Initialisation et exécution
    generator = KenoGeneratorAdvanced(
        data_path=args.data,
        silent=args.silent,
        training_profile=training_profile
    )
    
    success = generator.run(
        num_grids=args.grids,
        save_file=args.output,
        retrain=args.retrain
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
