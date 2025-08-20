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
    """Structure pour stocker les statistiques Keno"""
    frequences: Dict[int, int]
    retards: Dict[int, int]
    paires_freq: Dict[Tuple[int, int], int]
    zones_freq: Dict[str, int]
    derniers_tirages: List[List[int]]

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
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
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
        Analyse les patterns et statistiques des tirages Keno
        
        Returns:
            KenoStats: Statistiques calculées
        """
        self._log("🔍 Analyse des patterns Keno...")
        
        # Initialisation des statistiques
        frequences = {i: 0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
        retards = {i: 0 for i in range(1, KENO_PARAMS['total_numbers'] + 1)}
        paires_freq = {}
        zones_freq = {"zone1_17": 0, "zone18_35": 0, "zone36_52": 0, "zone53_70": 0}
        
        # Extraction des numéros de tous les tirages
        all_draws = []
        for _, row in self.data.iterrows():
            draw = [int(row[f'boule{i}']) for i in range(1, 21)]
            all_draws.append(sorted(draw))
            
            # Comptage des fréquences
            for num in draw:
                frequences[num] += 1
            
            # Analyse par zones
            for num in draw:
                if 1 <= num <= 17:
                    zones_freq["zone1_17"] += 1
                elif 18 <= num <= 35:
                    zones_freq["zone18_35"] += 1
                elif 36 <= num <= 52:
                    zones_freq["zone36_52"] += 1
                else:  # 53-70
                    zones_freq["zone53_70"] += 1
        
        # Calcul des retards
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            retard = 0
            for draw in reversed(all_draws):
                if num in draw:
                    break
                retard += 1
            retards[num] = retard
        
        # Analyse des paires fréquentes  
        for draw in all_draws:
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    paires_freq[pair] = paires_freq.get(pair, 0) + 1
        
        # Garder les derniers tirages pour l'analyse
        derniers_tirages = all_draws[-50:] if len(all_draws) >= 50 else all_draws
        
        self.stats = KenoStats(
            frequences=frequences,
            retards=retards, 
            paires_freq=paires_freq,
            zones_freq=zones_freq,
            derniers_tirages=derniers_tirages
        )
        
        self._log(f"✅ Analyse terminée - {len(all_draws)} tirages analysés")
        return self.stats
    
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
        Génère des grilles basées sur l'analyse de fréquence
        
        Args:
            num_grids: Nombre de grilles à générer
            
        Returns:
            List[List[int]]: Liste des grilles générées
        """
        if not self.stats:
            self._log("❌ Statistiques non disponibles", "ERROR")
            return []
        
        grids = []
        
        # Préparation des listes de numéros par critères
        freq_sorted = sorted(self.stats.frequences.items(), key=lambda x: x[1], reverse=True)
        retard_sorted = sorted(self.stats.retards.items(), key=lambda x: x[1], reverse=True)
        
        hot_numbers = [num for num, _ in freq_sorted[:35]]  # Top 35 numéros fréquents
        cold_numbers = [num for num, _ in retard_sorted[:35]]  # Top 35 numéros en retard
        
        for _ in range(num_grids):
            grid = []
            
            # Stratégie mixte: hot + cold + aléatoire
            # 5 numéros chauds
            hot_selection = random.sample(hot_numbers, min(5, len(hot_numbers)))
            grid.extend(hot_selection)
            
            # 3 numéros froids  
            available_cold = [n for n in cold_numbers if n not in grid]
            cold_selection = random.sample(available_cold, min(3, len(available_cold)))
            grid.extend(cold_selection)
            
            # 2 numéros aléatoires
            available_random = [n for n in range(1, KENO_PARAMS['total_numbers'] + 1) if n not in grid]
            random_selection = random.sample(available_random, min(2, len(available_random)))
            grid.extend(random_selection)
            
            # Assurer qu'on a exactement 10 numéros
            while len(grid) < 10:
                available = [n for n in range(1, KENO_PARAMS['total_numbers'] + 1) if n not in grid]
                if available:
                    grid.append(random.choice(available))
                else:
                    break
            
            grids.append(sorted(grid[:10]))
        
        return grids
    
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
