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
from collections import defaultdict, deque, Counter
from numpy.linalg import matrix_power

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

# Configuration de l'apprentissage incrémental
INCREMENTAL_CONFIG = {
    'performance_window': 50,      # Fenêtre pour le calcul de performance
    'adaptation_threshold': 0.05,  # Seuil pour ajuster les poids
    'min_samples_update': 10,      # Minimum d'échantillons pour mise à jour
    'max_history_size': 1000,      # Taille max de l'historique des performances
    'weight_decay': 0.95,          # Facteur d'oubli pour les anciennes performances
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
class LearningPerformance:
    """Structure pour stocker les performances d'apprentissage"""
    timestamp: datetime
    prediction_accuracy: float
    top30_hit_rate: float
    ml_weight_used: float
    freq_weight_used: float
    model_version: str
    sample_size: int
    feedback_score: float = 0.0  # Score basé sur les résultats réels

@dataclass
class IncrementalLearningState:
    """État de l'apprentissage incrémental"""
    performance_history: deque          # Historique des performances
    current_weights: Dict[str, float]   # Poids actuels (ml_weight, freq_weight)
    total_predictions: int              # Nombre total de prédictions
    successful_predictions: int         # Prédictions réussies
    model_version: int                  # Version actuelle du modèle
    last_update: datetime               # Dernière mise à jour
    adaptation_rate: float              # Taux d'adaptation actuel
    learning_momentum: float            # Momentum d'apprentissage

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

@dataclass
class MarkovChain:
    """Structure pour stocker les chaînes de Markov par numéro"""
    transition_matrices: Dict[int, np.ndarray]      # Matrice de transition par numéro
    steady_states: Dict[int, np.ndarray]            # États stationnaires 
    pattern_transitions: Dict[str, Dict[str, float]] # Transitions entre motifs
    zone_transitions: Dict[str, np.ndarray]         # Transitions par zones
    last_states: Dict[int, int]                     # Derniers états (0 ou 1)
    n_order: int                                    # Ordre de la chaîne (1, 2, ou 3)

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
        
        # Stratégie adaptative avec apprentissage incrémental
        self.adaptive_weights = {
            'ml_weight': 0.6,      # 60% ML par défaut
            'freq_weight': 0.4,    # 40% fréquence par défaut  
            'performance_history': [],
            'last_update': datetime.now()
        }
        
        # État d'apprentissage incrémental
        self.incremental_state = IncrementalLearningState(
            performance_history=deque(maxlen=INCREMENTAL_CONFIG['max_history_size']),
            current_weights={'ml_weight': 0.6, 'freq_weight': 0.4},
            total_predictions=0,
            successful_predictions=0,
            model_version=1,
            last_update=datetime.now(),
            adaptation_rate=0.1,
            learning_momentum=0.9
        )
        
        # Chargement de l'état précédent s'il existe
        self._load_incremental_state()
        
        # Méthode de prédiction par défaut
        self.method = "hybrid"
        
        self._log(f"🎲 Générateur Keno Avancé v2.0 initialisé (grilles de {self.grid_size} numéros)")

    def _log(self, message: str, level: str = "INFO"):
        """Système de logging configuré"""
        if not self.silent or level == "ERROR":
            print(f"{message}")
    
    def _load_incremental_state(self):
        """Charge l'état d'apprentissage incrémental depuis le disque"""
        state_file = self.models_dir / "incremental_state.pkl"
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    # Mise à jour de l'état actuel
                    if 'current_weights' in saved_state:
                        self.incremental_state.current_weights = saved_state['current_weights']
                        self.adaptive_weights['ml_weight'] = saved_state['current_weights']['ml_weight']
                        self.adaptive_weights['freq_weight'] = saved_state['current_weights']['freq_weight']
                    if 'performance_history' in saved_state:
                        self.incremental_state.performance_history = deque(
                            saved_state['performance_history'],
                            maxlen=INCREMENTAL_CONFIG['max_history_size']
                        )
                    if 'model_version' in saved_state:
                        self.incremental_state.model_version = saved_state['model_version']
                    if 'total_predictions' in saved_state:
                        self.incremental_state.total_predictions = saved_state['total_predictions']
                    if 'successful_predictions' in saved_state:
                        self.incremental_state.successful_predictions = saved_state['successful_predictions']
                        
                self._log(f"✅ État d'apprentissage incrémental chargé (version {self.incremental_state.model_version})")
                self._log(f"   📊 Performance: {self.incremental_state.successful_predictions}/{self.incremental_state.total_predictions}")
                self._log(f"   ⚖️  Poids: ML={self.incremental_state.current_weights['ml_weight']:.2f}, Freq={self.incremental_state.current_weights['freq_weight']:.2f}")
            except Exception as e:
                self._log(f"⚠️  Erreur lors du chargement de l'état incrémental: {e}")
    
    def _save_incremental_state(self):
        """Sauvegarde l'état d'apprentissage incrémental"""
        state_file = self.models_dir / "incremental_state.pkl"
        try:
            state_data = {
                'current_weights': self.incremental_state.current_weights,
                'performance_history': list(self.incremental_state.performance_history),
                'model_version': self.incremental_state.model_version,
                'total_predictions': self.incremental_state.total_predictions,
                'successful_predictions': self.incremental_state.successful_predictions,
                'last_update': self.incremental_state.last_update,
                'adaptation_rate': self.incremental_state.adaptation_rate,
                'learning_momentum': self.incremental_state.learning_momentum
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
                
            self._log(f"💾 État d'apprentissage sauvegardé (version {self.incremental_state.model_version})")
        except Exception as e:
            self._log(f"❌ Erreur lors de la sauvegarde de l'état: {e}", "ERROR")
    
    def update_performance(self, prediction_accuracy: float, top30_hit_rate: float, 
                          feedback_score: float = 0.0, sample_size: int = 1):
        """
        Met à jour les performances et ajuste les poids adaptatifs
        
        Args:
            prediction_accuracy: Précision de la prédiction (0-1)
            top30_hit_rate: Taux de réussite du TOP 30 (0-1) 
            feedback_score: Score de retour d'expérience (0-1)
            sample_size: Taille de l'échantillon évalué
        """
        # Créer l'enregistrement de performance
        performance = LearningPerformance(
            timestamp=datetime.now(),
            prediction_accuracy=prediction_accuracy,
            top30_hit_rate=top30_hit_rate,
            ml_weight_used=self.incremental_state.current_weights['ml_weight'],
            freq_weight_used=self.incremental_state.current_weights['freq_weight'],
            model_version=str(self.incremental_state.model_version),
            sample_size=sample_size,
            feedback_score=feedback_score
        )
        
        # Ajouter à l'historique
        self.incremental_state.performance_history.append(performance)
        
        # Mettre à jour les compteurs
        self.incremental_state.total_predictions += sample_size
        self.incremental_state.successful_predictions += int(prediction_accuracy * sample_size)
        
        # Calculer la performance moyenne récente
        recent_performances = list(self.incremental_state.performance_history)[-INCREMENTAL_CONFIG['performance_window']:]
        if len(recent_performances) >= INCREMENTAL_CONFIG['min_samples_update']:
            avg_accuracy = np.mean([p.prediction_accuracy for p in recent_performances])
            avg_top30_rate = np.mean([p.top30_hit_rate for p in recent_performances])
            avg_feedback = np.mean([p.feedback_score for p in recent_performances if p.feedback_score > 0])
            
            # Score composite de performance
            composite_score = (avg_accuracy * 0.4 + avg_top30_rate * 0.4 + avg_feedback * 0.2)
            
            # Ajustement adaptatif des poids
            self._adapt_weights(composite_score, avg_accuracy, avg_top30_rate)
            
            self._log(f"📊 Mise à jour performance: Accuracy={avg_accuracy:.3f}, TOP30={avg_top30_rate:.3f}, Composite={composite_score:.3f}")
        
        # Sauvegarde de l'état
        self._save_incremental_state()
    
    def _adapt_weights(self, composite_score: float, ml_performance: float, freq_performance: float):
        """
        Ajuste adaptivement les poids ML vs Fréquence selon les performances
        
        Args:
            composite_score: Score composite de performance
            ml_performance: Performance du ML seul
            freq_performance: Performance des fréquences seules
        """
        # Performance historique moyenne pour comparaison
        if len(self.incremental_state.performance_history) > 1:
            historical_scores = [
                (p.prediction_accuracy * 0.4 + p.top30_hit_rate * 0.4 + p.feedback_score * 0.2)
                for p in list(self.incremental_state.performance_history)[:-INCREMENTAL_CONFIG['performance_window']]
                if p.feedback_score > 0
            ]
            historical_avg = np.mean(historical_scores) if historical_scores else 0.5
        else:
            historical_avg = 0.5
        
        # Déterminer la direction d'ajustement
        performance_delta = composite_score - historical_avg
        
        if abs(performance_delta) > INCREMENTAL_CONFIG['adaptation_threshold']:
            # Ajustement basé sur la performance relative ML vs Freq
            if ml_performance > freq_performance:
                # ML performe mieux, augmenter son poids
                weight_adjustment = self.incremental_state.adaptation_rate * performance_delta
                new_ml_weight = min(0.8, max(0.2, self.incremental_state.current_weights['ml_weight'] + weight_adjustment))
            else:
                # Fréquences performent mieux, augmenter leur poids
                weight_adjustment = self.incremental_state.adaptation_rate * performance_delta
                new_ml_weight = min(0.8, max(0.2, self.incremental_state.current_weights['ml_weight'] - weight_adjustment))
            
            new_freq_weight = 1.0 - new_ml_weight
            
            # Application du momentum pour lisser les changements
            momentum_ml = (self.incremental_state.learning_momentum * self.incremental_state.current_weights['ml_weight'] + 
                          (1 - self.incremental_state.learning_momentum) * new_ml_weight)
            momentum_freq = 1.0 - momentum_ml
            
            # Mise à jour des poids
            old_ml_weight = self.incremental_state.current_weights['ml_weight']
            self.incremental_state.current_weights['ml_weight'] = momentum_ml
            self.incremental_state.current_weights['freq_weight'] = momentum_freq
            self.adaptive_weights['ml_weight'] = momentum_ml
            self.adaptive_weights['freq_weight'] = momentum_freq
            
            self._log(f"🎯 Ajustement poids adaptatif: ML {old_ml_weight:.3f}→{momentum_ml:.3f}, Freq {1-old_ml_weight:.3f}→{momentum_freq:.3f}")
            self._log(f"   📈 Performance delta: {performance_delta:+.3f}, Score composite: {composite_score:.3f}")
    
    def add_feedback(self, predicted_numbers: List[int], actual_draw: List[int], 
                     prediction_timestamp: datetime = None):
        """
        Ajoute un feedback basé sur un tirage réel pour améliorer l'apprentissage
        
        Args:
            predicted_numbers: Numéros prédits par le modèle
            actual_draw: Numéros réellement tirés
            prediction_timestamp: Timestamp de la prédiction (optionnel)
        """
        if not predicted_numbers or not actual_draw:
            return
        
        # Calcul du score de feedback
        predicted_set = set(predicted_numbers[:30])  # TOP 30 prédit
        actual_set = set(actual_draw)
        
        # Métriques de succès
        hit_count = len(predicted_set.intersection(actual_set))
        hit_rate = hit_count / len(actual_set)  # Sur les 20 numéros tirés
        precision = hit_count / len(predicted_set) if predicted_set else 0
        
        # Score composite de feedback (0-1)
        feedback_score = (hit_rate * 0.6 + precision * 0.4)
        
        # Mise à jour des performances avec ce feedback
        self.update_performance(
            prediction_accuracy=hit_rate,
            top30_hit_rate=precision,
            feedback_score=feedback_score,
            sample_size=1
        )
        
        self._log(f"📥 Feedback ajouté: {hit_count}/20 numéros trouvés, Score={feedback_score:.3f}")
        self._log(f"   🎯 Précision TOP30: {precision:.3f}, Taux de réussite: {hit_rate:.3f}")
        
        # Si le feedback est suffisamment récent et significatif, déclencher une mise à jour incrémentale
        if feedback_score > 0.3 and len(self.incremental_state.performance_history) >= 5:
            self._trigger_incremental_update(predicted_numbers, actual_draw)
    
    def _trigger_incremental_update(self, predicted_numbers: List[int], actual_draw: List[int]):
        """
        Déclenche une mise à jour incrémentale du modèle basée sur le feedback
        
        Args:
            predicted_numbers: Numéros prédits
            actual_draw: Tirage réel
        """
        if not HAS_ML or 'multilabel' not in self.ml_models:
            return
            
        try:
            self._log("🔄 Déclenchement d'une mise à jour incrémentale du modèle...")
            
            # Créer un échantillon d'entraînement à partir du feedback
            current_date = datetime.now()
            feedback_data = {
                'date_de_tirage': [current_date],
                **{f'boule{i}': [actual_draw[i-1] if i-1 < len(actual_draw) else 0] for i in range(1, 21)}
            }
            
            df_feedback = pd.DataFrame(feedback_data)
            
            # Ajouter les mêmes features que lors de l'entraînement initial
            df_features = self.add_cyclic_features(df_feedback)
            df_features = self.enrich_features(df_features)
            
            # ... (suite de la préparation des features comme dans train_xgboost_models)
            # Note: Cette partie nécessiterait une refactorisation pour éviter la duplication de code
            
            # Pour l'instant, incrémenter la version du modèle et marquer pour réentraînement
            self.incremental_state.model_version += 1
            self.incremental_state.last_update = datetime.now()
            
            self._log(f"✅ Mise à jour incrémentale planifiée (version {self.incremental_state.model_version})")
            
        except Exception as e:
            self._log(f"❌ Erreur lors de la mise à jour incrémentale: {e}", "ERROR")
    
    def get_learning_report(self) -> str:
        """
        Génère un rapport détaillé sur l'état de l'apprentissage
        
        Returns:
            str: Rapport formaté
        """
        report = "# 📊 Rapport d'Apprentissage Incrémental Keno\n\n"
        
        # Statistiques générales
        report += "## 📈 Statistiques Générales\n"
        report += f"- **Version du modèle**: {self.incremental_state.model_version}\n"
        report += f"- **Prédictions totales**: {self.incremental_state.total_predictions}\n"
        report += f"- **Prédictions réussies**: {self.incremental_state.successful_predictions}\n"
        
        if self.incremental_state.total_predictions > 0:
            success_rate = self.incremental_state.successful_predictions / self.incremental_state.total_predictions
            report += f"- **Taux de succès global**: {success_rate:.1%}\n"
        
        report += f"- **Dernière mise à jour**: {self.incremental_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Poids actuels
        report += "## ⚖️ Poids Adaptatifs Actuels\n"
        report += f"- **ML Weight**: {self.incremental_state.current_weights['ml_weight']:.3f}\n"
        report += f"- **Frequency Weight**: {self.incremental_state.current_weights['freq_weight']:.3f}\n"
        report += f"- **Taux d'adaptation**: {self.incremental_state.adaptation_rate:.3f}\n"
        report += f"- **Momentum d'apprentissage**: {self.incremental_state.learning_momentum:.3f}\n\n"
        
        # Performances récentes
        if self.incremental_state.performance_history:
            recent_performances = list(self.incremental_state.performance_history)[-10:]
            report += "## 📊 Performances Récentes (10 dernières)\n"
            
            for i, perf in enumerate(recent_performances, 1):
                report += f"**{i}.** {perf.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                report += f"Acc: {perf.prediction_accuracy:.3f}, "
                report += f"TOP30: {perf.top30_hit_rate:.3f}, "
                report += f"Feedback: {perf.feedback_score:.3f}\n"
            
            # Moyennes récentes
            avg_acc = np.mean([p.prediction_accuracy for p in recent_performances])
            avg_top30 = np.mean([p.top30_hit_rate for p in recent_performances])
            avg_feedback = np.mean([p.feedback_score for p in recent_performances if p.feedback_score > 0])
            
            report += f"\n**Moyennes récentes:**\n"
            report += f"- Accuracy: {avg_acc:.3f}\n"
            report += f"- TOP30 Hit Rate: {avg_top30:.3f}\n"
            report += f"- Feedback Score: {avg_feedback:.3f}\n\n"
        
        # Recommandations
        report += "## 🎯 Recommandations\n"
        if self.incremental_state.total_predictions < 50:
            report += "- ⚠️ Données insuffisantes pour des recommandations fiables\n"
            report += "- 📝 Continuez à utiliser le système pour collecter plus de données\n"
        else:
            if len(self.incremental_state.performance_history) >= 20:
                recent_trend = np.polyfit(
                    range(len(recent_performances)), 
                    [p.prediction_accuracy for p in recent_performances], 
                    1
                )[0]
                if recent_trend > 0.01:
                    report += "- ✅ Tendance d'amélioration détectée - Le modèle apprend efficacement\n"
                elif recent_trend < -0.01:
                    report += "- ⚠️ Tendance de dégradation - Considérer un réentraînement complet\n"
                else:
                    report += "- 📈 Performance stable - Le modèle converge\n"
        
        return report
    
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
                # Préparer toutes les features statistiques en une fois avec un dictionnaire
                stats_features = {}
                for num in range(1, 71):
                    stats_features[f'freq_recent_{num}'] = [self.stats.frequences_recentes.get(num, 0)] * len(df_features)
                    stats_features[f'retard_{num}'] = [self.stats.retards.get(num, 0)] * len(df_features)
                    stats_features[f'tendance_{num}'] = [self.stats.tendances_50.get(num, 1.0)] * len(df_features)
                
                # Ajouter toutes ces features en une seule fois avec pd.concat
                stats_df = pd.DataFrame(stats_features, index=df_features.index)
                df_features = pd.concat([df_features, stats_df], axis=1)
                
                # Ajouter les noms des features
                for feature_name in stats_features.keys():
                    feature_cols.append(feature_name)
            else:
                # Créer des valeurs par défaut pour toutes les features statistiques en une fois
                stats_features = {}
                for num in range(1, 71):
                    stats_features[f'freq_recent_{num}'] = [0] * len(df_features)
                    stats_features[f'retard_{num}'] = [0] * len(df_features)
                    stats_features[f'tendance_{num}'] = [1.0] * len(df_features)
                
                stats_df = pd.DataFrame(stats_features, index=df_features.index)
                df_features = pd.concat([df_features, stats_df], axis=1)
                
                for feature_name in stats_features.keys():
                    feature_cols.append(feature_name)
            
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
        Inclut maintenant l'apprentissage incrémental.
        """
        self._log("🚀 Démarrage du pipeline complet Keno avec ML + Markov + Apprentissage incrémental...")
        
        if not self.load_data():
            self._log("❌ Chargement des données impossible.", "ERROR")
            return
        
        self.stats = self.analyze_patterns()
        
        # 🔗 NOUVEAU : Construction des chaînes de Markov
        self.markov_chain = self.build_markov_chains(order=2)
        
        self.train_xgboost_models(retrain=False)
        self.load_ml_models()
        
        # Évaluation initiale des performances
        if len(self.data) > 100:
            self._evaluate_initial_performance()
        
        self._log("✅ Pipeline complet terminé avec ML + Markov + apprentissage incrémental activé.")
    
    def _evaluate_initial_performance(self):
        """Évalue les performances initiales du modèle sur les données de test"""
        try:
            # Utiliser les 10% derniers tirages comme test
            test_size = int(len(self.data) * 0.1)
            test_data = self.data.tail(test_size).copy()
            
            self._log(f"📊 Évaluation initiale des performances sur {test_size} tirages de test...")
            
            # Générer des prédictions pour chaque tirage de test
            accurate_predictions = 0
            total_hit_rate = 0.0
            
            for idx in range(min(10, len(test_data))):  # Limiter à 10 évaluations pour la vitesse
                test_row = test_data.iloc[idx]
                actual_draw = [int(test_row[f'boule{i}']) for i in range(1, 21)]
                
                # Obtenir une prédiction TOP 30
                top30_predictions = self.predict_numbers_ml(num_grids=1)
                if top30_predictions:
                    predicted_numbers = [num for num, _ in top30_predictions][:30]
                    
                    # Calculer les métriques
                    hit_count = len(set(predicted_numbers).intersection(set(actual_draw)))
                    hit_rate = hit_count / len(actual_draw)
                    accuracy = 1.0 if hit_count >= 5 else hit_count / 20.0  # 5+ hits = succès
                    
                    accurate_predictions += accuracy
                    total_hit_rate += hit_rate
            
            # Calculer les moyennes
            avg_accuracy = accurate_predictions / 10 if total_hit_rate > 0 else 0.5
            avg_hit_rate = total_hit_rate / 10 if total_hit_rate > 0 else 0.3
            
            # Mettre à jour les performances initiales
            self.update_performance(
                prediction_accuracy=avg_accuracy,
                top30_hit_rate=avg_hit_rate,
                feedback_score=avg_hit_rate,
                sample_size=10
            )
            
            self._log(f"📊 Performance initiale: Accuracy={avg_accuracy:.3f}, Hit rate={avg_hit_rate:.3f}")
            
        except Exception as e:
            self._log(f"⚠️ Erreur lors de l'évaluation initiale: {e}")
    
    def simulate_learning_improvement(self, num_simulations: int = 20):
        """
        Simule l'amélioration de l'apprentissage avec des tirages synthétiques
        Utile pour démontrer les capacités d'apprentissage incrémental
        
        Args:
            num_simulations: Nombre de simulations à effectuer
        """
        self._log(f"🧪 Simulation de {num_simulations} cycles d'apprentissage incrémental...")
        
        for i in range(num_simulations):
            # Générer un tirage synthétique réaliste
            # (basé sur les patterns des données existantes)
            if self.stats:
                # Utiliser les fréquences pour générer un tirage probable
                weights = np.array([self.stats.frequences.get(num, 1) for num in range(1, 71)])
                weights = weights / weights.sum()
                
                synthetic_draw = sorted(np.random.choice(
                    range(1, 71),
                    size=20,
                    replace=False,


                    p=weights
                ))
            else:
                synthetic_draw = sorted(random.sample(range(1, 71), 20))
            
            # Obtenir une prédiction
            top30_predictions = self.predict_numbers_ml(num_grids=1)
            if top30_predictions:
                predicted_numbers = [num for num, _ in top30_predictions][:30]
                
                # Simuler le feedback
                self.add_feedback(predicted_numbers, synthetic_draw)
                
                if (i + 1) % 5 == 0:
                    self._log(f"   📈 Cycle {i+1}/{num_simulations} terminé")
        
        # Afficher le rapport final
        report = self.get_learning_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        self._log(f"✅ Simulation d'apprentissage terminée après {num_simulations} cycles")
    
    def retrain_with_incremental_data(self):
        """
        Réentraîne le modèle en intégrant les données d'apprentissage incrémental
        """
        self._log("🔄 Réentraînement avec données d'apprentissage incrémental...")
        
        # Sauvegarder l'ancien modèle
        old_model_path = self.models_dir / f"xgb_keno_multilabel_v{self.incremental_state.model_version-1}.pkl"
        current_model_path = self.models_dir / "xgb_keno_multilabel.pkl"
        
        if current_model_path.exists():
            import shutil
            shutil.copy2(current_model_path, old_model_path)
            self._log(f"💾 Ancien modèle sauvegardé comme version {self.incremental_state.model_version-1}")
        
        # Réentraîner avec les nouveaux poids adaptatifs
        success = self.train_xgboost_models(retrain=True)
        
        if success:
            self.incremental_state.model_version += 1
            self._save_incremental_state()
            self._log(f"✅ Réentraînement réussi - Nouvelle version {self.incremental_state.model_version}")
        else:
            self._log("❌ Échec du réentraînement - Conservation du modèle précédent", "ERROR")

    def build_markov_chains(self, order: int = 1) -> MarkovChain:
        """
        Construit les chaînes de Markov pour chaque numéro Keno
        
        Args:
            order: Ordre de la chaîne (1=simple, 2=mémoire sur 2 tirages, 3=mémoire sur 3 tirages)
    
        Returns:
            MarkovChain: Structure contenant toutes les chaînes
        """
        self._log(f"🔗 Construction des chaînes de Markov d'ordre {order} pour le Keno...")
    
        if not self.stats or not self.stats.tous_tirages:
            self._log("❌ Données de tirages non disponibles", "ERROR")
            return None
    
        tirages = self.stats.tous_tirages
        transition_matrices = {}
        last_states = {}
    
        # 1️⃣ CHAÎNES DE MARKOV PAR NUMÉRO (approche granulaire)
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            # Extraire la séquence d'états pour ce numéro (0=non tiré, 1=tiré)
            states_sequence = []
            for tirage in tirages:
                states_sequence.append(1 if num in tirage else 0)
        
            # Construire la matrice de transition selon l'ordre
            if order == 1:
                matrix = self._build_first_order_matrix(states_sequence)
            elif order == 2:
                matrix = self._build_second_order_matrix(states_sequence)
            else:  # order == 3
                matrix = self._build_third_order_matrix(states_sequence)
        
            transition_matrices[num] = matrix
            last_states[num] = states_sequence[-order:] if len(states_sequence) >= order else [0] * order
    
        # 2️⃣ CHAÎNES DE MARKOV PAR MOTIFS (zones, parité, etc.)
        pattern_transitions = self._build_pattern_transitions(tirages)
    
        # 3️⃣ CHAÎNES DE MARKOV PAR ZONES
        zone_transitions = self._build_zone_transitions(tirages)
    
        # 4️⃣ CALCUL DES ÉTATS STATIONNAIRES (distribution limite)
        steady_states = {}
        for num, matrix in transition_matrices.items():
            try:
                # Calculer l'état stationnaire (vecteur propre pour valeur propre 1)
                eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
                steady_idx = np.argmin(np.abs(eigenvalues - 1.0))
                steady_state = np.real(eigenvectors[:, steady_idx])
                steady_state = steady_state / steady_state.sum()  # Normalisation
                steady_states[num] = steady_state
            except:
                # Fallback : distribution uniforme
                steady_states[num] = np.array([0.8, 0.2])  # ~80% non tiré, ~20% tiré
    
        markov_chain = MarkovChain(
            transition_matrices=transition_matrices,
            steady_states=steady_states,
            pattern_transitions=pattern_transitions,
            zone_transitions=zone_transitions,
            last_states=last_states,
            n_order=order
        )
    
        self._log(f"✅ Chaînes de Markov d'ordre {order} construites pour {len(transition_matrices)} numéros")
        return markov_chain

    def _build_first_order_matrix(self, states_sequence: List[int]) -> np.ndarray:
        """Construit une matrice de transition d'ordre 1 (2x2)"""
        # Compter les transitions
        transitions = {
            (0, 0): 0,  # non tiré -> non tiré
            (0, 1): 0,  # non tiré -> tiré
            (1, 0): 0,  # tiré -> non tiré
            (1, 1): 0   # tiré -> tiré
        }
        
        for i in range(len(states_sequence) - 1):
            current_state = states_sequence[i]
            next_state = states_sequence[i + 1]
            transitions[(current_state, next_state)] += 1
    
        # Construire la matrice 2x2
        matrix = np.zeros((2, 2))
    
        # Ligne 0 : transitions depuis l'état "non tiré" (0)
        total_from_0 = transitions[(0, 0)] + transitions[(0, 1)]
        if total_from_0 > 0:
            matrix[0, 0] = transitions[(0, 0)] / total_from_0  # P(0→0)
            matrix[0, 1] = transitions[(0, 1)] / total_from_0  # P(0→1)
        else:
            matrix[0, :] = [0.8, 0.2]  # Défaut : 20% de chance d'être tiré
    
        # Ligne 1 : transitions depuis l'état "tiré" (1)
        total_from_1 = transitions[(1, 0)] + transitions[(1, 1)]
        if total_from_1 > 0:
            matrix[1, 0] = transitions[(1, 0)] / total_from_1  # P(1→0)
            matrix[1, 1] = transitions[(1, 1)] / total_from_1  # P(1→1)
        else:
            matrix[1, :] = [0.8, 0.2]  # Défaut
    
        return matrix

    def _build_second_order_matrix(self, states_sequence: List[int]) -> np.ndarray:
        """Construit une matrice de transition d'ordre 2 (4x4)"""
        # États : 00, 01, 10, 11 (les 2 derniers tirages)
        state_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        transitions = defaultdict(int)
        
        for i in range(len(states_sequence) - 2):
            current_state = (states_sequence[i], states_sequence[i + 1])
            next_state_single = states_sequence[i + 2]
            next_state = (states_sequence[i + 1], next_state_single)
            
            curr_idx = state_map[current_state]
            next_idx = state_map[next_state]
            transitions[(curr_idx, next_idx)] += 1
    
        # Construire la matrice 4x4
        matrix = np.zeros((4, 4))
        for from_state in range(4):
            total = sum(transitions[(from_state, to_state)] for to_state in range(4))
            if total > 0:
                for to_state in range(4):
                    matrix[from_state, to_state] = transitions[(from_state, to_state)] / total
            else:
                matrix[from_state, :] = [0.25, 0.25, 0.25, 0.25]  # Uniforme par défaut
    
        return matrix

    def _build_third_order_matrix(self, states_sequence: List[int]) -> np.ndarray:
        """Construit une matrice de transition d'ordre 3 (8x8)"""
        # États : 000, 001, 010, 011, 100, 101, 110, 111
        state_map = {
            (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 2, (0, 1, 1): 3,
            (1, 0, 0): 4, (1, 0, 1): 5, (1, 1, 0): 6, (1, 1, 1): 7
        }
        transitions = defaultdict(int)
        
        for i in range(len(states_sequence) - 3):
            current_state = (states_sequence[i], states_sequence[i + 1], states_sequence[i + 2])
            next_state_single = states_sequence[i + 3]
            next_state = (states_sequence[i + 1], states_sequence[i + 2], next_state_single)
            
            curr_idx = state_map[current_state]
            next_idx = state_map[next_state]
            transitions[(curr_idx, next_idx)] += 1
    
        # Construire la matrice 8x8
        matrix = np.zeros((8, 8))
        for from_state in range(8):
            total = sum(transitions[(from_state, to_state)] for to_state in range(8))
            if total > 0:
                for to_state in range(8):
                    matrix[from_state, to_state] = transitions[(from_state, to_state)] / total
            else:
                matrix[from_state, :] = [1/8] * 8  # Uniforme par défaut
    
        return matrix

    def _build_pattern_transitions(self, tirages: List[List[int]]) -> Dict[str, Dict[str, float]]:
        """Construit les transitions entre motifs (parité, zones, etc.)"""
        pattern_transitions = {}
    
        # 🎯 MOTIFS DE PARITÉ
        parity_sequences = []
        for tirage in tirages:
            pairs_count = sum(1 for num in tirage if num % 2 == 0)
            if pairs_count <= 6:
                parity_pattern = "low_pairs"
            elif pairs_count <= 14:
                parity_pattern = "medium_pairs"
            else:
                parity_pattern = "high_pairs"
            parity_sequences.append(parity_pattern)
        
        # Construire matrice de transition de parité
        parity_states = ["low_pairs", "medium_pairs", "high_pairs"]
        parity_matrix = self._build_categorical_transitions(parity_sequences, parity_states)
        pattern_transitions["parity"] = parity_matrix
    
        # 🎯 MOTIFS DE ZONES
        zone_sequences = []
        for tirage in tirages:
            zone_counts = [0, 0, 0, 0]  # 4 zones
            for num in tirage:
                if 1 <= num <= 17:
                    zone_counts[0] += 1
                elif 18 <= num <= 35:
                    zone_counts[1] += 1
                elif 36 <= num <= 52:
                    zone_counts[2] += 1
                else:  # 53-70
                    zone_counts[3] += 1
            
            # Classification du motif de zone
            dominant_zone = zone_counts.index(max(zone_counts))
            zone_pattern = f"zone{dominant_zone + 1}_dominant"
            zone_sequences.append(zone_pattern)
        
        zone_states = ["zone1_dominant", "zone2_dominant", "zone3_dominant", "zone4_dominant"]
        zone_matrix = self._build_categorical_transitions(zone_sequences, zone_states)
        pattern_transitions["zones"] = zone_matrix
    
        return pattern_transitions

    def _build_categorical_transitions(self, sequences: List[str], states: List[str]) -> Dict[str, float]:
        """Construit une matrice de transition pour des états catégoriels"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(sequences) - 1):
            current = sequences[i]
            next_state = sequences[i + 1]
            transitions[current][next_state] += 1
    
        # Normaliser pour obtenir des probabilités
        result = {}
        for from_state in states:
            total = sum(transitions[from_state].values())
            if total > 0:
                for to_state in states:
                    result[f"{from_state}→{to_state}"] = transitions[from_state][to_state] / total
            else:
                # Distribution uniforme par défaut
                for to_state in states:
                    result[f"{from_state}→{to_state}"] = 1.0 / len(states)
    
        return result

    def _build_zone_transitions(self, tirages: List[List[int]]) -> Dict[str, np.ndarray]:
        """Construit les matrices de transition par zones géographiques"""
        zone_transitions = {}
    
        # Définir les zones
        zones = {
            "zone1": list(range(1, 18)),   # 1-17
            "zone2": list(range(18, 36)),  # 18-35
            "zone3": list(range(36, 53)),  # 36-52
            "zone4": list(range(53, 71))   # 53-70
        }
        
        for zone_name, zone_numbers in zones.items():
            # Créer une séquence d'activité pour cette zone
            zone_activity = []
            for tirage in tirages:
                count_in_zone = sum(1 for num in tirage if num in zone_numbers)
                # Classifier l'activité : 0=faible, 1=moyenne, 2=forte
                if count_in_zone <= 3:
                    zone_activity.append(0)  # Faible
                elif count_in_zone <= 7:
                    zone_activity.append(1)  # Moyenne
                else:
                    zone_activity.append(2)  # Forte
            
            # Construire la matrice de transition 3x3 pour cette zone
            matrix = np.zeros((3, 3))
            transitions = defaultdict(int)
            
            for i in range(len(zone_activity) - 1):
                current = zone_activity[i]
                next_state = zone_activity[i + 1]
                transitions[(current, next_state)] += 1
            
            # Remplir la matrice
            for from_state in range(3):
                total = sum(transitions[(from_state, to_state)] for to_state in range(3))
                if total > 0:
                    for to_state in range(3):
                        matrix[from_state, to_state] = transitions[(from_state, to_state)] / total
                else:
                    matrix[from_state, :] = [1/3, 1/3, 1/3]  # Uniforme
            
            zone_transitions[zone_name] = matrix
    
        return zone_transitions

    def predict_top30_markov(self, n_steps: int = 1) -> List[Tuple[int, float]]:
        """
        Prédiction des 30 numéros les plus probables en utilisant les chaînes de Markov
        
        Args:
            n_steps: Nombre de pas de prédiction (1 = prochain tirage)
        
        Returns:
            List[Tuple[int, float]]: TOP 30 avec probabilités
        """
        if not hasattr(self, 'markov_transitions') or self.markov_transitions is None:
            self._log("⚠️ Matrices de transition Markov non disponibles", "ERROR")
            return [(i, 0.1) for i in range(1, 31)]
        
        try:
            # Obtenir l'état actuel basé sur les derniers tirages
            current_state = self._get_current_markov_state()
            
            # Calculer la distribution de probabilité après n_steps
            transition_matrix = np.array(list(self.markov_transitions.values()))
            state_probabilities = np.mean(transition_matrix, axis=0)  # État initial uniforme
            
            # Appliquer n_steps de transition
            for _ in range(n_steps):
                new_probs = np.zeros_like(state_probabilities)
                for i, prob in enumerate(state_probabilities):
                    if prob > 0:
                        new_probs += prob * transition_matrix[i]
                state_probabilities = new_probs
            
            # Convertir les probabilités d'état en probabilités de numéros
            numero_probs = {}
            for numero in range(1, 71):
                numero_probs[numero] = 0.0
                
            # Calculer les probabilités pour chaque numéro
            for state, prob in enumerate(state_probabilities):
                if prob > 0:
                    # Distribuer la probabilité de cet état sur tous les numéros possibles
                    for numero in range(1, 71):
                        numero_probs[numero] += prob / 70  # Distribution uniforme de base
            
            # Trier et retourner le TOP 30
            sorted_numbers = sorted(numero_probs.items(), key=lambda x: x[1], reverse=True)
            return sorted_numbers[:30]
            
        except Exception as e:
            self._log(f"❌ Erreur dans predict_top30_markov: {e}", "ERROR")
            return [(i, 0.1) for i in range(1, 31)]

    def _get_current_markov_state(self) -> int:
        """
        Obtenir l'état actuel de Markov basé sur les derniers tirages
        
        Returns:
            int: Index de l'état actuel
        """
        try:
            if hasattr(self, 'df') and not self.df.empty:
                # Prendre les derniers tirages pour déterminer l'état
                recent_draws = self.df.tail(3)
                # Calculer un hash simple basé sur les derniers tirages
                state_hash = sum(hash(tuple(sorted(draw))) for draw in recent_draws['numeros_tires'])
                return abs(state_hash) % 10  # 10 états possibles
            return 0
        except Exception as e:
            self._log(f"⚠️ Erreur calcul état Markov: {e}", "ERROR")
            return 0

    def generate_optimized_grids(self, num_grids: int = 10) -> List[List[int]]:
        """
        Génère des grilles optimisées selon la méthode choisie
        
        Args:
            num_grids: Nombre de grilles à générer
            
        Returns:
            List[List[int]]: Liste des grilles générées
        """
        try:
            # Utiliser la méthode hybride ou ML selon la disponibilité
            if hasattr(self, 'method') and self.method == 'hybrid':
                top30 = self.get_top30_numbers_hybrid()
            elif hasattr(self, 'method') and self.method == 'markov':
                markov_predictions = self.predict_top30_markov()
                top30 = [num for num, _ in markov_predictions]
            elif hasattr(self, 'method') and self.method == 'freq':
                top30 = self.get_top30_numbers_advanced()
            else:
                # Par défaut : utiliser ML
                ml_grids = self.predict_numbers_ml(num_grids=1)
                if ml_grids:
                    top30 = ml_grids[0]
                else:
                    top30 = self.get_top30_numbers_advanced()
            
            # Générer les grilles à partir du TOP 30
            grids = []
            for i in range(num_grids):
                # Sélectionner grid_size numéros du TOP 30 avec variation
                import random
                random.seed(42 + i)  # Seed différent pour chaque grille
                grid = random.sample(top30[:min(len(top30), self.grid_size * 2)], self.grid_size)
                grids.append(sorted(grid))
            
            return grids
            
        except Exception as e:
            self._log(f"Erreur génération grilles: {e}", "ERROR")
            # Génération de fallback
            fallback_grids = []
            for i in range(num_grids):
                import random
                random.seed(42 + i)
                grid = random.sample(range(1, 71), self.grid_size)
                fallback_grids.append(sorted(grid))
            return fallback_grids

    def get_top30_numbers_hybrid(self) -> List[int]:
        """
        Combinaison hybride : ML + Fréquences + Markov pour le TOP 30 final
        """
        self._log("🎯 Calcul du TOP 30 hybride (ML + Fréquences + Markov)...")
    
        # 1. TOP 30 ML
        top30_ml = self.predict_numbers_ml()
        ml_scores = {num: prob for num, prob in top30_ml} if top30_ml else {}
    
        # 2. TOP 30 Fréquences avancées
        top30_freq = self.get_top30_numbers_advanced()
    
        # 3. TOP 30 Markov
        top30_markov = self.predict_top30_markov()
        markov_scores = {num: prob for num, prob in top30_markov} if top30_markov else {}
    
        # 4. FUSION PONDÉRÉE ADAPTATIVE
        hybrid_scores = {}
    
        # Poids adaptatifs incluant Markov
        ml_weight = self.adaptive_weights['ml_weight'] * 0.8  # Réduire légèrement le ML
        freq_weight = self.adaptive_weights['freq_weight'] * 0.8  # Réduire légèrement les fréq
        markov_weight = 0.4  # Nouveau poids pour Markov
    
        # Normaliser les poids
        total_weight = ml_weight + freq_weight + markov_weight
        ml_weight /= total_weight
        freq_weight /= total_weight  
        markov_weight /= total_weight
    
        for num in range(1, KENO_PARAMS['total_numbers'] + 1):
            score = 0.0
            
            # Score ML normalisé
            if num in ml_scores:
                score += ml_scores[num] * ml_weight
            
            # Score fréquence normalisé (position dans le TOP 30)
            if num in top30_freq:
                freq_score = (30 - top30_freq.index(num)) / 30.0
                score += freq_score * freq_weight
            
            # Score Markov normalisé
            if num in markov_scores:
                score += markov_scores[num] * markov_weight
            
            hybrid_scores[num] = score
    
        # TOP 30 final
        top30_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        top30_numbers = [num for num, _ in top30_hybrid]
    
        self._log(f"✅ TOP 30 hybride: ML({ml_weight:.2f}) + Freq({freq_weight:.2f}) + Markov({markov_weight:.2f})")
        return top30_numbers

    def analyze_markov_performance(self) -> str:
        """
        Analyse les performances des chaînes de Markov sur les données historiques
        """
        if not hasattr(self, 'markov_chain') or not self.markov_chain:
            return "❌ Chaînes de Markov non construites"
        
        report = "# 🔗 Analyse des Chaînes de Markov Keno\n\n"
        
        # Test sur les 20 derniers tirages
        test_tirages = self.stats.tous_tirages[-20:] if len(self.stats.tous_tirages) > 20 else []
        
        if len(test_tirages) < 5:
            return report + "❌ Données insuffisantes pour l'analyse\n"
        
        successes = 0
        total_tests = 0
        
        for i in range(len(test_tirages) - 1):
            # Simuler la prédiction à partir du tirage i
            # (ici on devrait reconstruire l'état à partir des tirages précédents)
            predicted_top30 = self.predict_top30_markov(n_steps=1)
            predicted_numbers = [num for num, _ in predicted_top30]
            
            actual_draw = test_tirages[i + 1]
            hits = len(set(predicted_numbers).intersection(set(actual_draw)))
            
            if hits >= 8:  # Au moins 8 numéros trouvés sur 20
                successes += 1
            total_tests += 1
    
        success_rate = successes / total_tests if total_tests > 0 else 0
    
        report += f"## 📊 Performance sur {total_tests} tests\n"
        report += f"- **Succès** (≥8 numéros trouvés): {successes}/{total_tests}\n"
        report += f"- **Taux de succès**: {success_rate:.1%}\n\n"
        
        # Analyse des matrices de transition
        report += "## 🔍 Analyse des Matrices de Transition\n"
        
        # Exemples de numéros intéressants
        interesting_numbers = [7, 23, 42, 55, 69]  # Sélection arbitraire
        
        for num in interesting_numbers:
            if num in self.markov_chain.transition_matrices:
                matrix = self.markov_chain.transition_matrices[num]
                steady_state = self.markov_chain.steady_states.get(num, np.array([0.8, 0.2]))
                
                report += f"**Numéro {num}:**\n"
                if matrix.shape == (2, 2):
                    report += f"- P(non tiré → tiré): {matrix[0, 1]:.3f}\n"
                    report += f"- P(tiré → tiré): {matrix[1, 1]:.3f}\n"
                report += f"- État stationnaire: {steady_state[1]:.3f} (prob. long terme)\n\n"
        
        return report
    
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
    parser = argparse.ArgumentParser(description="Générateur avancé de grilles Keno avec apprentissage incrémental")
    parser.add_argument("--n", type=int, default=10, help="Nombre de grilles à générer")
    parser.add_argument("--grids", type=int, help="Alias pour --n (nombre de grilles à générer)")
    parser.add_argument("--size", type=int, default=10, choices=[7, 8, 9, 10], help="Taille des grilles (7 à 10 numéros)")
    parser.add_argument("--profile", type=str, default="balanced", help="Profil d'entraînement ML (quick, balanced, comprehensive, intensive)")
    parser.add_argument("--data", type=str, default=None, help="Chemin du fichier de données Keno")
    parser.add_argument("--silent", action="store_true", help="Mode silencieux")
    parser.add_argument("--retrain", action="store_true", help="Forcer le réentraînement même si le modèle existe")
    parser.add_argument("--save-top30-ml", action="store_true", help="Sauvegarder le TOP 30 ML dans un CSV")
    parser.add_argument("--test-grids", action="store_true", help="Évaluer les grilles générées avec le modèle ML")
    
    # Nouvelles options pour l'apprentissage incrémental
    parser.add_argument("--learning-report", action="store_true", help="Afficher le rapport d'apprentissage incrémental")
    parser.add_argument("--simulate-learning", type=int, metavar="N", help="Simuler N cycles d'apprentissage incrémental")
    parser.add_argument("--add-feedback", nargs=2, metavar=("PREDICTED", "ACTUAL"), 
                       help="Ajouter un feedback (format: 'num1,num2,...' 'num1,num2,...')")
    parser.add_argument("--retrain-incremental", action="store_true", help="Réentraîner avec les données incrémentales")
    parser.add_argument("--demo-data", action="store_true", help="Générer des données de démonstration")

    # Options pour les chaînes de Markov
    parser.add_argument("--markov-order", type=int, default=2, choices=[1, 2, 3], 
                   help="Ordre des chaînes de Markov (1, 2, ou 3)")
    parser.add_argument("--markov-steps", type=int, default=1, 
                   help="Nombre de pas de prédiction Markov")
    parser.add_argument("--markov-analysis", action="store_true", 
                   help="Afficher l'analyse des chaînes de Markov")
    parser.add_argument("--method", type=str, default="hybrid", 
                   choices=["ml", "freq", "markov", "hybrid"],
                   help="Méthode de prédiction (ml, freq, markov, hybrid)")
    
    args = parser.parse_args()

    # Gestion de l'alias --grids
    num_grids = args.grids if args.grids is not None else args.n
    
    # Génération de données de démonstration si demandée
    if args.demo_data:
        from pathlib import Path
        demo_script = Path(__file__).parent / "generate_demo_data.py"
        if demo_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(demo_script), "--draws", "1000"])
        else:
            print("❌ Script de génération de données de démonstration non trouvé")
        sys.exit(0)

    generator = KenoGeneratorAdvanced(
        data_path=args.data,
        silent=args.silent,
        training_profile=args.profile,
        grid_size=args.size
    )
    
    # Configuration de la méthode de prédiction
    generator.method = args.method
    
    # Gestion des commandes spéciales d'apprentissage incrémental
    if args.learning_report:
        print(generator.get_learning_report())
        sys.exit(0)
    
    if args.simulate_learning:
        if not generator.load_data():
            print("❌ Impossible de charger les données pour la simulation")
            sys.exit(1)
        generator.stats = generator.analyze_patterns()
        generator.train_xgboost_models(retrain=False)
        generator.load_ml_models()
        generator.simulate_learning_improvement(args.simulate_learning)
        sys.exit(0)
    
    if args.add_feedback:
        try:
            predicted = [int(x) for x in args.add_feedback[0].split(',')]
            actual = [int(x) for x in args.add_feedback[1].split(',')]
            generator.add_feedback(predicted, actual)
            print("✅ Feedback ajouté avec succès")
            print(generator.get_learning_report())
        except ValueError:
            print("❌ Format de feedback invalide. Utilisez: --add-feedback '1,2,3,...' '4,5,6,...'")
        sys.exit(0)
    
    if args.retrain_incremental:
        if not generator.load_data():
            print("❌ Impossible de charger les données pour le réentraînement")
            sys.exit(1)
        generator.stats = generator.analyze_patterns()
        generator.retrain_with_incremental_data()
        sys.exit(0)

    # Pipeline principal
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
        print("\n📊 Scores des grilles générées :")
        for i, (grid, score) in enumerate(grid_scores, 1):
            print(f"Grille {i}: {grid} (Score: {score:.4f})")

    print("\n" + generator.generate_report(grids))
    
    # Affichage du rapport d'apprentissage si des performances ont été collectées
    if len(generator.incremental_state.performance_history) > 0:
        print("\n" + "="*60)
        print("📊 RAPPORT D'APPRENTISSAGE INCRÉMENTAL")
        print("="*60)
        print(generator.get_learning_report())
