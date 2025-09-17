import os
import duckdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import xgboost as xgb
from itertools import combinations
from pathlib import Path
from joblib import dump, load, Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import scipy.stats
from numba import njit
import random
import argparse
import sys
import redis
import stumpy
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq
import json
import time

warnings.filterwarnings("ignore")

# --- Configuration Générale et Reproductibilité ---
load_dotenv()

# Configuration des chemins pour loto/data
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Fonction pour convertir CSV en Parquet si nécessaire
def ensure_parquet_file():
    """S'assure qu'un fichier Parquet existe, en convertissant le CSV si nécessaire"""
    parquet_files = list(DATA_DIR.glob('*.parquet'))
    csv_files = list(DATA_DIR.glob('*.csv'))
    
    if parquet_files:
        # Vérifier si le Parquet est plus récent que le CSV
        parquet_path = parquet_files[0]
        if csv_files:
            csv_path = csv_files[0]
            if csv_path.stat().st_mtime > parquet_path.stat().st_mtime:
                print(f"� Le fichier CSV est plus récent, reconversion nécessaire...")
                convert_csv_to_parquet(csv_path, parquet_path)
        return parquet_path
    elif csv_files:
        # Pas de Parquet, mais CSV disponible - conversion automatique
        csv_path = csv_files[0]
        parquet_path = DATA_DIR / csv_path.with_suffix('.parquet').name
        print(f"📂 Fichier CSV trouvé : {csv_path.name}")
        print(f"🔄 Conversion automatique en Parquet pour de meilleures performances...")
        convert_csv_to_parquet(csv_path, parquet_path)
        return parquet_path
    else:
        # Aucun fichier trouvé - fallback
        fallback_path = SCRIPT_DIR / 'loto_data' / 'loto_201911.parquet'
        print(f"⚠️  Aucun fichier CSV/Parquet trouvé dans {DATA_DIR}")
        print(f"⚠️  Utilisation du fallback : {fallback_path}")
        return fallback_path

def convert_csv_to_parquet(csv_path, parquet_path):
    """Convertit un fichier CSV en Parquet en utilisant DuckDB"""
    import duckdb
    try:
        con = duckdb.connect()
        con.execute(f"COPY (SELECT * FROM read_csv_auto('{str(csv_path)}')) TO '{str(parquet_path)}' (FORMAT PARQUET);")
        
        # Vérifier la taille des fichiers pour information
        csv_size = csv_path.stat().st_size / (1024*1024)
        parquet_size = parquet_path.stat().st_size / (1024*1024)
        compression_ratio = (1 - parquet_size/csv_size) * 100 if csv_size > 0 else 0
        
        print(f"✅ Conversion terminée : {csv_path.name} → {parquet_path.name}")
        print(f"📊 Compression : {compression_ratio:.1f}% ({csv_size:.1f}MB → {parquet_size:.1f}MB)")
        con.close()
    except Exception as e:
        print(f"❌ Erreur lors de la conversion : {e}")
        raise

# Obtenir le fichier Parquet (avec conversion automatique si nécessaire)
parquet_path = ensure_parquet_file()
print(f"📂 Utilisation du fichier Parquet : {parquet_path}")

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Générateur Loto Avancé avec IA et Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python loto_generator_advanced_Version2.py                    # Configuration par défaut
  python loto_generator_advanced_Version2.py -s 5000          # 5000 simulations
  python loto_generator_advanced_Version2.py -c 4             # 4 processeurs
  python loto_generator_advanced_Version2.py -s 20000 -c 8    # 20k simulations sur 8 cœurs
  python loto_generator_advanced_Version2.py --quick          # Mode rapide (1000 simulations)
  python loto_generator_advanced_Version2.py --intensive      # Mode intensif (50000 simulations)
  python loto_generator_advanced_Version2.py --exclude 1,5,12 # Exclure les numéros 1, 5 et 12
  python loto_generator_advanced_Version2.py --exclude auto   # Exclure les 3 derniers tirages (défaut)
        """
    )
    
    parser.add_argument('-s', '--simulations', 
                        type=int, 
                        default=10000,
                        help='Nombre de simulations à effectuer (défaut: 10000)')
    
    parser.add_argument('-c', '--cores', 
                        type=int, 
                        default=None,
                        help=f'Nombre de processeurs à utiliser (défaut: auto = {mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1})')
    
    parser.add_argument('--quick', 
                        action='store_true',
                        help='Mode rapide : 1000 simulations')
    
    parser.add_argument('--intensive', 
                        action='store_true',
                        help='Mode intensif : 50000 simulations')
    
    parser.add_argument('--seed', 
                        type=int, 
                        default=None,
                        help='Graine pour la reproductibilité (défaut: aléatoire)')
    
    parser.add_argument('--silent', 
                        action='store_true',
                        help='Mode silencieux (pas de confirmation)')
    
    parser.add_argument('--exclude', 
                        type=str, 
                        default=None,
                        help='Numéros à exclure, séparés par des virgules (ex: --exclude 1,5,12,30) ou "auto" pour les 3 derniers tirages (défaut: auto)')
    
    parser.add_argument('--retrain', 
                        action='store_true',
                        help='Force le ré-entraînement des modèles ML pour compatibilité avec les nouvelles features')
    
    parser.add_argument('--fast-training',
                        action='store_true', 
                        help='Mode entraînement ultra-rapide (paramètres optimisés pour la vitesse)')
    
    args = parser.parse_args()
    
    # Gestion des modes prédéfinis
    if args.quick:
        args.simulations = 1000
    elif args.intensive:
        args.simulations = 50000
    
    # Validation des arguments
    if args.simulations < 100:
        print("❌ Erreur: Le nombre de simulations doit être d'au moins 100")
        sys.exit(1)
    
    if args.simulations > 100000:
        print("⚠️  Attention: Plus de 100,000 simulations peuvent prendre beaucoup de temps")
        if not args.silent:
            confirm = input("Continuer ? (o/N): ").strip().lower()
            if confirm != 'o':
                sys.exit(0)
    
    # Validation des numéros exclus
    if args.exclude and args.exclude.lower() != 'auto':
        try:
            excluded_nums = [int(x.strip()) for x in args.exclude.split(',')]
            # Vérifier que tous les numéros sont valides (1-49)
            invalid_nums = [num for num in excluded_nums if num < 1 or num > 49]
            if invalid_nums:
                print(f"❌ Erreur: Numéros invalides détectés: {invalid_nums}. Les numéros doivent être entre 1 et 49.")
                sys.exit(1)
            # Vérifier qu'il ne faut pas exclure trop de numéros
            if len(excluded_nums) > 44:  # Il faut au moins 5 numéros pour faire une grille
                print(f"❌ Erreur: Trop de numéros exclus ({len(excluded_nums)}). Maximum autorisé: 44.")
                sys.exit(1)
            args.excluded_numbers = set(excluded_nums)
        except ValueError:
            print("❌ Erreur: Format invalide pour --exclude. Utilisez des numéros séparés par des virgules (ex: 1,5,12)")
            sys.exit(1)
    else:
        args.excluded_numbers = None  # Sera défini plus tard avec les 3 derniers tirages
    
    # Configuration automatique des cores si non spécifié
    if args.cores is None:
        args.cores = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1
    elif args.cores < 1:
        args.cores = 1
    elif args.cores > mp.cpu_count():
        print(f"⚠️  Limitation: {args.cores} cœurs demandés, mais seulement {mp.cpu_count()} disponibles")
        args.cores = mp.cpu_count()
    
    return args

# Configuration globale (sera mise à jour par les arguments)
ARGS = parse_arguments()

# Gestion de la seed : aléatoire si non spécifiée
if ARGS.seed is None:
    import time
    GLOBAL_SEED = int(time.time()) % 2**31  # Seed basée sur le timestamp
else:
    GLOBAL_SEED = ARGS.seed

N_SIMULATIONS = ARGS.simulations
N_CORES = ARGS.cores
EXCLUDED_NUMBERS = ARGS.excluded_numbers

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BALLS = np.arange(1, 50)
CHANCE_BALLS = np.arange(1, 11)

print(f"Configuration : {N_CORES} processeurs, {N_SIMULATIONS:,} grilles, SEED={GLOBAL_SEED}.")

# --- Gestion des Dossiers & Connexion Redis ---
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = BASE_DIR / 'boost_models'
for directory in [OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)

# --- Stratégie Adaptative Améliorée ---
class AdaptiveStrategy:
    """Stratégie adaptative qui s'améliore en continu avec apprentissage par renforcement"""
    
    def __init__(self, history_file=None):
        self.history_file = history_file or (MODEL_DIR / 'performance_history.json')
        self.performance_history = self.load_history()
        self.ml_weight = 0.6  # Poids initial pour ML vs fréquences
        self.adaptation_rate = 0.05  # Vitesse d'adaptation réduite pour plus de stabilité
        self.confidence_threshold = 0.7  # Seuil de confiance pour les adaptations
        self.learning_rate_decay = 0.995  # Décroissance du taux d'apprentissage
        self.exploration_factor = 0.1  # Facteur d'exploration pour éviter les optima locaux
        self.performance_window = 20  # Fenêtre glissante pour l'évaluation
        
    def load_history(self):
        """Charge l'historique des performances avec validation"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                
                # Validation et migration des données si nécessaire
                required_keys = ['predictions', 'actuals', 'ml_scores', 'freq_scores', 
                               'dates', 'ml_weight_history', 'strategy_changes', 
                               'performance_trends', 'confidence_scores']
                
                for key in required_keys:
                    if key not in history:
                        history[key] = []
                
                # Limiter l'historique pour éviter une croissance excessive
                max_history = 500
                for key in history:
                    if isinstance(history[key], list) and len(history[key]) > max_history:
                        history[key] = history[key][-max_history:]
                
                return history
            except Exception as e:
                print(f"⚠️ Erreur chargement historique: {e}, initialisation nouveau")
                
        return {
            'predictions': [],
            'actuals': [],
            'ml_scores': [],
            'freq_scores': [],
            'dates': [],
            'ml_weight_history': [],
            'strategy_changes': [],
            'performance_trends': [],
            'confidence_scores': []
        }
    
    def save_history(self):
        """Sauvegarde l'historique avec gestion d'erreurs"""
        try:
            # Sauvegarde atomique pour éviter la corruption
            temp_file = self.history_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            temp_file.replace(self.history_file)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde historique: {e}")
    
    def calculate_performance_trend(self):
        """Calcule la tendance de performance récente"""
        if len(self.performance_history['ml_scores']) < 10:
            return 0.0
        
        recent_scores = self.performance_history['ml_scores'][-self.performance_window:]
        
        # Régression linéaire simple pour détecter la tendance
        x = np.arange(len(recent_scores))
        if len(recent_scores) > 1:
            slope, _ = np.polyfit(x, recent_scores, 1)
            return slope
        return 0.0
    
    def calculate_confidence(self):
        """Calcule le niveau de confiance dans les prédictions actuelles"""
        if len(self.performance_history['ml_scores']) < 5:
            return 0.5
        
        recent_scores = self.performance_history['ml_scores'][-self.performance_window:]
        
        # Confiance basée sur la stabilité et la performance
        stability = 1.0 - np.std(recent_scores)  # Plus stable = plus confiant
        performance = np.mean(recent_scores)     # Meilleure performance = plus confiant
        
        confidence = (stability + performance) / 2.0
        return np.clip(confidence, 0.1, 0.9)
    
    def adaptive_learning_rate(self):
        """Calcule un taux d'apprentissage adaptatif"""
        base_rate = self.adaptation_rate
        
        # Réduire le taux si on a beaucoup d'expérience
        experience_factor = max(0.1, 1.0 - len(self.performance_history['ml_scores']) / 1000)
        
        # Augmenter le taux si la performance est en baisse
        trend = self.calculate_performance_trend()
        trend_factor = 1.0 + max(0, -trend * 10)  # Augmenter si tendance négative
        
        adaptive_rate = base_rate * experience_factor * trend_factor
        return np.clip(adaptive_rate, 0.01, 0.2)
    
    def evaluate_prediction_accuracy(self, predicted_probs, actual_draw):
        """Évaluation améliorée de la précision des prédictions"""
        if len(actual_draw) != 5:
            return 0.0, 0.0
        
        # Score ML amélioré avec pondération
        if isinstance(predicted_probs, np.ndarray) and len(predicted_probs) == 49:
            # Score basé sur la probabilité moyenne des boules tirées
            ml_score = np.mean([predicted_probs[boule-1] for boule in actual_draw])
            
            # Bonus pour les prédictions très sûres (probabilités élevées)
            high_prob_bonus = np.sum([predicted_probs[boule-1] for boule in actual_draw if predicted_probs[boule-1] > 0.7])
            ml_score += high_prob_bonus * 0.1
        else:
            ml_score = 0.0
            
        # Score fréquences avec amélioration
        freq_weights = np.ones(49) / 49  # Uniforme en fallback
        freq_score = np.mean([freq_weights[boule-1] for boule in actual_draw])
        
        return ml_score, freq_score
    
    def detect_pattern_changes(self):
        """Détecte les changements dans les patterns de tirage"""
        if len(self.performance_history['ml_scores']) < 20:
            return False
        
        # Comparer les performances récentes vs historiques
        recent_perf = np.mean(self.performance_history['ml_scores'][-10:])
        historical_perf = np.mean(self.performance_history['ml_scores'][-20:-10])
        
        # Détection de changement significatif
        change_threshold = 0.1
        return abs(recent_perf - historical_perf) > change_threshold
    
    def update_performance(self, prediction, actual_draw, ml_probs=None):
        """Mise à jour améliorée des performances avec apprentissage adaptatif"""
        if ml_probs is not None:
            ml_score, freq_score = self.evaluate_prediction_accuracy(ml_probs, actual_draw)
            
            # Ajouter à l'historique
            self.performance_history['predictions'].append(prediction)
            self.performance_history['actuals'].append(actual_draw)
            self.performance_history['ml_scores'].append(float(ml_score))
            self.performance_history['freq_scores'].append(float(freq_score))
            self.performance_history['dates'].append(datetime.now().isoformat())
            
            # Calculer la confiance et la tendance
            confidence = self.calculate_confidence()
            trend = self.calculate_performance_trend()
            
            self.performance_history['confidence_scores'].append(float(confidence))
            self.performance_history['performance_trends'].append(float(trend))
            
            # Adaptation intelligente du poids ML
            old_weight = self.ml_weight
            self.adapt_ml_weight_intelligent(ml_score, freq_score, confidence, trend)
            
            # Enregistrer les changements de stratégie significatifs
            weight_change = abs(self.ml_weight - old_weight)
            if weight_change > 0.05:  # Changement significatif
                change_info = {
                    'timestamp': datetime.now().isoformat(),
                    'old_weight': float(old_weight),
                    'new_weight': float(self.ml_weight),
                    'reason': 'adaptive_learning',
                    'ml_score': float(ml_score),
                    'freq_score': float(freq_score),
                    'confidence': float(confidence),
                    'trend': float(trend)
                }
                self.performance_history['strategy_changes'].append(change_info)
            
            # Maintenir une taille d'historique raisonnable
            max_history = 200
            for key in self.performance_history:
                if isinstance(self.performance_history[key], list) and len(self.performance_history[key]) > max_history:
                    self.performance_history[key] = self.performance_history[key][-max_history:]
            
            self.save_history()
    
    def adapt_ml_weight_intelligent(self, ml_score, freq_score, confidence, trend):
        """Adaptation intelligente du poids ML avec multiple critères"""
        if len(self.performance_history['ml_scores']) < 5:
            return
        
        # Taux d'apprentissage adaptatif
        learning_rate = self.adaptive_learning_rate()
        
        # Calculer la performance relative avec fenêtre glissante
        window_size = min(self.performance_window, len(self.performance_history['ml_scores']))
        recent_ml = np.mean(self.performance_history['ml_scores'][-window_size:])
        recent_freq = np.mean(self.performance_history['freq_scores'][-window_size:])
        
        # Facteur de performance relative
        if recent_freq > 0:
            performance_ratio = recent_ml / recent_freq
        else:
            performance_ratio = 1.0
        
        # Ajustement basé sur multiple critères
        if performance_ratio > 1.1 and confidence > self.confidence_threshold:
            # ML performe bien avec confiance élevée -> augmenter poids ML
            adjustment = learning_rate * (1 + trend * 2)
            self.ml_weight = min(0.95, self.ml_weight + adjustment)
        elif performance_ratio < 0.9 or confidence < 0.3:
            # ML performe mal ou confiance faible -> réduire poids ML
            adjustment = learning_rate * (1 + abs(trend))
            self.ml_weight = max(0.05, self.ml_weight - adjustment)
        elif trend < -0.05:
            # Tendance négative -> réduction prudente
            adjustment = learning_rate * 0.5
            self.ml_weight = max(0.1, self.ml_weight - adjustment)
        
        # Exploration périodique pour éviter les optima locaux
        if len(self.performance_history['ml_scores']) % 50 == 0:
            exploration_noise = np.random.normal(0, self.exploration_factor)
            self.ml_weight = np.clip(self.ml_weight + exploration_noise, 0.05, 0.95)
        
        # Enregistrer le poids dans l'historique
        self.performance_history['ml_weight_history'].append(float(self.ml_weight))
        
        # Décroissance du taux d'apprentissage avec l'expérience
        self.adaptation_rate *= self.learning_rate_decay
        self.adaptation_rate = max(0.01, self.adaptation_rate)
    
    def get_adaptive_weights(self):
        """Retourne les poids adaptatifs optimisés"""
        return self.ml_weight, 1.0 - self.ml_weight
    
    def should_retrain_models(self):
        """Détermine si les modèles doivent être ré-entraînés"""
        # Critères pour déclencher un ré-entraînement
        pattern_change = self.detect_pattern_changes()
        low_confidence = self.calculate_confidence() < 0.3
        negative_trend = self.calculate_performance_trend() < -0.1
        
        # Éviter les ré-entraînements trop fréquents
        last_retrain = getattr(self, 'last_retrain', datetime.min)
        time_since_retrain = (datetime.now() - last_retrain).days
        min_days_between_retrains = 7
        
        should_retrain = (pattern_change or low_confidence or negative_trend) and \
                        time_since_retrain > min_days_between_retrains
        
        if should_retrain:
            self.last_retrain = datetime.now()
        
        return should_retrain
    
    def get_performance_summary(self):
        """Retourne un résumé détaillé des performances"""
        if len(self.performance_history['ml_scores']) < 5:
            return "Données insuffisantes pour l'analyse adaptative"
        
        window_size = min(self.performance_window, len(self.performance_history['ml_scores']))
        recent_ml = np.mean(self.performance_history['ml_scores'][-window_size:])
        recent_freq = np.mean(self.performance_history['freq_scores'][-window_size:])
        confidence = self.calculate_confidence()
        trend = self.calculate_performance_trend()
        
        # Stabilité des poids
        recent_weights = self.performance_history['ml_weight_history'][-window_size:] if self.performance_history['ml_weight_history'] else [self.ml_weight]
        weight_stability = 1.0 - np.std(recent_weights) if len(recent_weights) > 1 else 1.0
        
        return {
            'ml_score_recent': recent_ml,
            'freq_score_recent': recent_freq,
            'performance_ratio': recent_ml / recent_freq if recent_freq > 0 else 1.0,
            'current_ml_weight': self.ml_weight,
            'confidence': confidence,
            'trend': trend,
            'weight_stability': weight_stability,
            'total_predictions': len(self.performance_history['ml_scores']),
            'strategy_changes': len(self.performance_history['strategy_changes']),
            'should_retrain': self.should_retrain_models(),
            'learning_rate': self.adaptation_rate
        }

# Initialiser la stratégie adaptative
adaptive_strategy = AdaptiveStrategy()

def update_adaptive_strategy_with_recent_draws(con, strategy):
    """Met à jour la stratégie adaptative avec les tirages récents"""
    try:
        # Récupérer les 5 derniers tirages pour évaluation
        recent_draws = con.execute("""
            SELECT boule_1, boule_2, boule_3, boule_4, boule_5, date_de_tirage
            FROM loto_draws 
            ORDER BY date_de_tirage DESC 
            LIMIT 5
        """).fetchdf()
        
        if len(recent_draws) >= 2:
            print("   🎯 Mise à jour de la stratégie adaptative...")
            
            # Simuler des prédictions pour les tirages passés
            for i in range(1, min(len(recent_draws), 4)):
                actual_draw = recent_draws.iloc[i-1][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values
                previous_draw = recent_draws.iloc[i][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values
                
                # Mettre à jour la stratégie (sans probabilities ML pour simplifier)
                strategy.update_performance(
                    prediction=previous_draw.tolist(), 
                    actual_draw=actual_draw.tolist()
                )
            
            # Afficher le résumé des performances
            summary = strategy.get_performance_summary()
            if isinstance(summary, dict):
                print(f"      ML Weight: {summary['current_ml_weight']:.2f}")
                print(f"      Total Predictions: {summary['total_predictions']}")
    except Exception as e:
        print(f"   ⚠️ Erreur mise à jour stratégie: {e}")

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    print("✓ Connexion à Redis réussie.")
except redis.exceptions.ConnectionError:
    print("⚠️ Impossible de se connecter à Redis. Le caching sera désactivé.")
    redis_client = None

# --- Fonctions de Scoring Optimisées ---
@njit
def _count_consecutive_numba(grid: np.ndarray) -> int:
    if len(grid) < 2:
        return 0
    count = 0
    sorted_grid = np.sort(grid)
    for i in range(len(sorted_grid) - 1):
        if sorted_grid[i+1] - sorted_grid[i] == 1:
            count += 1
    return count

def count_consecutive_safe(row_values, balls_cols):
    grid = np.array([row_values[col] for col in balls_cols], dtype=np.int32)
    return _count_consecutive_numba(grid)

# --- Extraction des features cycliques ---
def add_cyclic_features(df, date_col='date_de_tirage'):
    df2 = df.copy()
    if date_col in df2.columns:
        df2[date_col] = pd.to_datetime(df2[date_col])
        df2['dayofweek'] = df2[date_col].dt.dayofweek
        df2['month'] = df2[date_col].dt.month
        df2['sin_day'] = np.sin(2 * np.pi * df2['dayofweek'] / 7)
        df2['cos_day'] = np.cos(2 * np.pi * df2['dayofweek'] / 7)
        df2['sin_month'] = np.sin(2 * np.pi * df2['month'] / 12)
        df2['cos_month'] = np.cos(2 * np.pi * df2['month'] / 12)
    for i in range(1, 6):
        col = f'boule_{i}'
        df2[f'sin_boule_{i}'] = np.sin(2 * np.pi * df2[col] / 49)
        df2[f'cos_boule_{i}'] = np.cos(2 * np.pi * df2[col] / 49)
    return df2


# --- Fonctions d'Analyse (DuckDB) ---
def analyze_criteria_duckdb(db_con: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    print("Début de l'analyse des critères avec DuckDB...")

    db_con.execute(f"""
        CREATE OR REPLACE TEMPORARY VIEW BaseData AS
        SELECT ROW_NUMBER() OVER (ORDER BY date_de_tirage) AS draw_index, 
               date_de_tirage::DATE as date_de_tirage,
               boule_1, boule_2, boule_3, boule_4, boule_5
        FROM {table_name};
    """)

    df_pandas = db_con.table('BaseData').fetchdf()
    balls_cols = [f'boule_{i}' for i in range(1, 6)]

    freq = pd.Series(df_pandas[balls_cols].values.flatten()).value_counts().reindex(BALLS, fill_value=0)
    last_3_draws = df_pandas.iloc[-3:][balls_cols].values.flatten()
    last_3_numbers_set = set(last_3_draws)
    
    # MODIFICATION : Ne plus exclure de numéros automatiquement
    # Filtre de fréquence : exclure les numéros avec fréquence < 80 ou > 100
    # freq_excluded = set(freq[(freq < 80) | (freq > 100)].index.tolist())
    freq_excluded = set()  # Aucun numéro exclu par défaut
    
    # MODIFICATION : Utiliser uniquement les exclusions manuelles de l'utilisateur
    if EXCLUDED_NUMBERS is not None:
        numbers_to_exclude = EXCLUDED_NUMBERS  # Seulement les exclusions manuelles
        exclusion_source = "paramètre utilisateur uniquement"
        print(f"   ✓ Numéros exclus (utilisateur): {sorted(numbers_to_exclude)}")
    else:
        numbers_to_exclude = set()  # Aucune exclusion automatique
        exclusion_source = "aucune exclusion"
        print(f"   ✓ Mode sans exclusion : tous les numéros 1-49 disponibles")
    
    # Garder l'info des anciens critères pour les statistiques
    freq_excluded_info = set(freq[(freq < 80) | (freq > 100)].index.tolist())
    if freq_excluded_info:
        print(f"   ℹ️ Numéros avec fréquence atypique (<80 ou >100) : {sorted(freq_excluded_info)} (non exclus)")

    last_appearance = df_pandas.melt(
        id_vars=['date_de_tirage'], 
        value_vars=balls_cols, 
        value_name='numero'
    ).groupby('numero')['date_de_tirage'].max()

    pair_counts_df = db_con.execute("""
        SELECT n1, n2, COUNT(*) as compte
        FROM (
            SELECT LEAST(boule_1, boule_2) AS n1, GREATEST(boule_1, boule_2) AS n2 FROM BaseData
            UNION ALL SELECT LEAST(boule_1, boule_3), GREATEST(boule_1, boule_3) FROM BaseData
            UNION ALL SELECT LEAST(boule_1, boule_4), GREATEST(boule_1, boule_4) FROM BaseData
            UNION ALL SELECT LEAST(boule_1, boule_5), GREATEST(boule_1, boule_5) FROM BaseData
            UNION ALL SELECT LEAST(boule_2, boule_3), GREATEST(boule_2, boule_3) FROM BaseData
            UNION ALL SELECT LEAST(boule_2, boule_4), GREATEST(boule_2, boule_4) FROM BaseData
            UNION ALL SELECT LEAST(boule_2, boule_5), GREATEST(boule_2, boule_5) FROM BaseData
            UNION ALL SELECT LEAST(boule_3, boule_4), GREATEST(boule_3, boule_4) FROM BaseData
            UNION ALL SELECT LEAST(boule_3, boule_5), GREATEST(boule_3, boule_5) FROM BaseData
            UNION ALL SELECT LEAST(boule_4, boule_5), GREATEST(boule_4, boule_5) FROM BaseData
        )
        WHERE n1 IS NOT NULL AND n2 IS NOT NULL
        GROUP BY n1, n2
        ORDER BY compte DESC
        LIMIT 20
    """).fetchdf()

    pair_counts = pd.Series(pair_counts_df.set_index(['n1', 'n2'])['compte']) if not pair_counts_df.empty else pd.Series(dtype='int64')

    gaps_query = """
        SELECT numero, AVG(draw_index - lag) as periodicite
        FROM (
            SELECT numero, draw_index, LAG(draw_index, 1) OVER (PARTITION BY numero ORDER BY draw_index) as lag
            FROM (
                SELECT draw_index, unnest(list_value(boule_1, boule_2, boule_3, boule_4, boule_5)) as numero
                FROM BaseData
            )
        )
        WHERE lag IS NOT NULL
        GROUP BY numero
    """
    gaps_df = db_con.execute(gaps_query).fetchdf()
    gaps = pd.Series(gaps_df.set_index('numero')['periodicite']) if not gaps_df.empty else pd.Series(dtype='float64', index=BALLS)

    pair_impair_dist = (df_pandas[balls_cols] % 2 == 0).sum(axis=1).value_counts().sort_index()

    consecutive_list = []
    for _, row in df_pandas.iterrows():
        grid = np.array([row[col] for col in balls_cols], dtype=np.int32)
        consecutive_list.append(_count_consecutive_numba(grid))
    consecutive_counts = pd.Series(consecutive_list).value_counts().sort_index()

    hot_numbers = freq.nlargest(10).index.tolist()
    cold_numbers = freq.nsmallest(10).index.tolist()

    correlation_matrix = df_pandas[balls_cols].corr()

    regression_results = {}
    for ball in balls_cols:
        result = db_con.execute(f"""
            SELECT draw_index, {ball} as value
            FROM BaseData
            ORDER BY draw_index
        """).fetchdf()
        X = result['draw_index'].values.reshape(-1, 1)
        y = result['value'].values
        slope, intercept = np.polyfit(X.flatten(), y, 1)
        regression_results[ball] = {'slope': slope, 'intercept': intercept}

    sums = df_pandas[balls_cols].sum(axis=1)
    stds = df_pandas[balls_cols].std(axis=1)

    # === NOUVEAUX CRITÈRES STATISTIQUES ===
    
    # 1. Distribution par dizaines (0-9, 10-19, 20-29, 30-39, 40-49)
    decades_dist = []
    for _, row in df_pandas.iterrows():
        grid = row[balls_cols].values
        decades = [(n-1)//10 for n in grid]  # 0,1,2,3,4 pour les dizaines
        decades_count = pd.Series(decades).value_counts()
        decades_dist.append(decades_count.reindex(range(5), fill_value=0).values)
    decades_dist = np.array(decades_dist)
    decades_entropy = -np.sum(decades_dist * np.log(decades_dist + 1e-10), axis=1)
    
    # 2. Espacement entre numéros consécutifs (gaps)
    gaps_between_numbers = []
    for _, row in df_pandas.iterrows():
        sorted_nums = sorted(row[balls_cols].values)
        local_gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(4)]
        gaps_between_numbers.append(np.std(local_gaps))  # Écart-type des espacements
    gaps_between_numbers = np.array(gaps_between_numbers)
    
    # 3. Répartition haute/basse (1-25 vs 26-49)
    high_low_ratio = []
    for _, row in df_pandas.iterrows():
        grid = row[balls_cols].values
        low_count = sum(1 for n in grid if n <= 25)
        high_count = 5 - low_count
        ratio = low_count / 5.0  # Proportion de numéros bas
        high_low_ratio.append(ratio)
    high_low_ratio = np.array(high_low_ratio)
    
    # 4. Nombres premiers vs composés
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    prime_ratio = []
    for _, row in df_pandas.iterrows():
        grid = row[balls_cols].values
        prime_count = sum(1 for n in grid if n in primes)
        prime_ratio.append(prime_count / 5.0)
    prime_ratio = np.array(prime_ratio)
    
    # 5. Variance des positions (mesure de dispersion)
    position_variance = []
    for _, row in df_pandas.iterrows():
        grid = sorted(row[balls_cols].values)
        positions = [(n-1)/48.0 for n in grid]  # Normalisation 0-1
        position_variance.append(np.var(positions))
    position_variance = np.array(position_variance)

    # Calcul des variances pour tous les critères
    sum_var = np.var(sums)
    std_var = np.var(stds)
    pair_impair_var = np.var((pair_impair_dist/pair_impair_dist.sum()).values)
    decades_entropy_var = np.var(decades_entropy)
    gaps_var = np.var(gaps_between_numbers)
    high_low_var = np.var(high_low_ratio)
    prime_var = np.var(prime_ratio)
    position_var = np.var(position_variance)
    
    # Poids équilibrés basés sur l'inverse des variances, mais avec normalisation
    raw_weights = {
        'sum': 1.0 / (1.0 + sum_var),
        'std': 1.0 / (1.0 + std_var), 
        'pair_impair': 1.0 / (1.0 + pair_impair_var),
        'decades_entropy': 1.0 / (1.0 + decades_entropy_var),
        'gaps': 1.0 / (1.0 + gaps_var),
        'high_low': 1.0 / (1.0 + high_low_var),
        'prime_ratio': 1.0 / (1.0 + prime_var),
        'position_variance': 1.0 / (1.0 + position_var)
    }
    
    # Normalisation pour que les poids soient plus équilibrés (pas de domination extrême)
    total_weight = sum(raw_weights.values())
    dynamic_weights = {key: value / total_weight for key, value in raw_weights.items()}
    
    # Ajustement pour éviter qu'un critère domine complètement (max 25% pour un critère maintenant)
    max_weight = 0.25
    for key in dynamic_weights:
        if dynamic_weights[key] > max_weight:
            excess = dynamic_weights[key] - max_weight
            dynamic_weights[key] = max_weight
            # Redistribuer l'excès sur les autres critères
            other_keys = [k for k in dynamic_weights.keys() if k != key]
            for other_key in other_keys:
                dynamic_weights[other_key] += excess / len(other_keys)
    
    # Re-normalisation finale
    total_weight = sum(dynamic_weights.values())
    dynamic_weights = {key: value / total_weight for key, value in dynamic_weights.items()}

    delta = pd.to_datetime('now').normalize() - pd.to_datetime(last_appearance.reindex(BALLS))
    numbers_analysis = pd.DataFrame({
        'Numero': BALLS,
        'Frequence': freq.reindex(BALLS, fill_value=0),
        'Dernier_Tirage': last_appearance.reindex(BALLS),
        'Jours_Depuis_Tirage': delta.dt.days,
        'Ecart_Moyen_Tirages': gaps.reindex(BALLS)
    }).sort_values('Frequence', ascending=False).reset_index(drop=True)

    print("   ✓ Analyse DuckDB terminée.")

    return {
        'freq': freq,
        'hot_numbers': hot_numbers,
        'cold_numbers': cold_numbers,
        'last_draw': df_pandas[balls_cols].iloc[-1].tolist(),
        'pair_impair_probs': pair_impair_dist / pair_impair_dist.sum(),
        'consecutive_counts': consecutive_counts,
        'pair_counts': pair_counts,
        'numbers_to_exclude': numbers_to_exclude,  # Maintenant peut être vide
        'freq_excluded': freq_excluded_info,  # Pour info seulement, pas d'exclusion
        'sums': sums,
        'stds': stds,
        'dynamic_weights': dynamic_weights,
        'last_appearance': last_appearance,
        'numbers_analysis': numbers_analysis,
        'correlation_matrix': correlation_matrix,
        'regression_results': regression_results,
        # Nouveaux critères statistiques
        'decades_entropy': decades_entropy,
        'gaps_between_numbers': gaps_between_numbers,
        'high_low_ratio': high_low_ratio,
        'prime_ratio': prime_ratio,
        'position_variance': position_variance
    }

# --- Stratégie Adaptative Améliorée ---
class AdaptiveStrategy:
    """Stratégie adaptative qui s'améliore en continu avec apprentissage par renforcement"""
    
    def __init__(self, history_file=None):
        self.history_file = history_file or (MODEL_DIR / 'performance_history.json')
        self.performance_history = self.load_history()
        self.ml_weight = 0.6  # Poids initial pour ML vs fréquences
        self.adaptation_rate = 0.05  # Vitesse d'adaptation réduite pour plus de stabilité
        self.confidence_threshold = 0.7  # Seuil de confiance pour les adaptations
        self.learning_rate_decay = 0.995  # Décroissance du taux d'apprentissage
        self.exploration_factor = 0.1  # Facteur d'exploration pour éviter les optima locaux
        self.performance_window = 20  # Fenêtre glissante pour l'évaluation
        
    def load_history(self):
        """Charge l'historique des performances avec validation"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                
                # Validation et migration des données si nécessaire
                required_keys = ['predictions', 'actuals', 'ml_scores', 'freq_scores', 
                               'dates', 'ml_weight_history', 'strategy_changes', 
                               'performance_trends', 'confidence_scores']
                
                for key in required_keys:
                    if key not in history:
                        history[key] = []
                
                # Limiter l'historique pour éviter une croissance excessive
                max_history = 500
                for key in history:
                    if isinstance(history[key], list) and len(history[key]) > max_history:
                        history[key] = history[key][-max_history:]
                
                return history
            except Exception as e:
                print(f"⚠️ Erreur chargement historique: {e}, initialisation nouveau")
                
        return {
            'predictions': [],
            'actuals': [],
            'ml_scores': [],
            'freq_scores': [],
            'dates': [],
            'ml_weight_history': [],
            'strategy_changes': [],
            'performance_trends': [],
            'confidence_scores': []
        }
    
    def save_history(self):
        """Sauvegarde l'historique avec gestion d'erreurs"""
        try:
            # Sauvegarde atomique pour éviter la corruption
            temp_file = self.history_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            temp_file.replace(self.history_file)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde historique: {e}")
    
    def calculate_performance_trend(self):
        """Calcule la tendance de performance récente"""
        if len(self.performance_history['ml_scores']) < 10:
            return 0.0
        
        recent_scores = self.performance_history['ml_scores'][-self.performance_window:]
        
        # Régression linéaire simple pour détecter la tendance
        x = np.arange(len(recent_scores))
        if len(recent_scores) > 1:
            slope, _ = np.polyfit(x, recent_scores, 1)
            return slope
        return 0.0
    
    def calculate_confidence(self):
        """Calcule le niveau de confiance dans les prédictions actuelles"""
        if len(self.performance_history['ml_scores']) < 5:
            return 0.5
        
        recent_scores = self.performance_history['ml_scores'][-self.performance_window:]
        
        # Confiance basée sur la stabilité et la performance
        stability = 1.0 - np.std(recent_scores)  # Plus stable = plus confiant
        performance = np.mean(recent_scores)     # Meilleure performance = plus confiant
        
        confidence = (stability + performance) / 2.0
        return np.clip(confidence, 0.1, 0.9)
    
    def adaptive_learning_rate(self):
        """Calcule un taux d'apprentissage adaptatif"""
        base_rate = self.adaptation_rate
        
        # Réduire le taux si on a beaucoup d'expérience
        experience_factor = max(0.1, 1.0 - len(self.performance_history['ml_scores']) / 1000)
        
        # Augmenter le taux si la performance est en baisse
        trend = self.calculate_performance_trend()
        trend_factor = 1.0 + max(0, -trend * 10)  # Augmenter si tendance négative
        
        adaptive_rate = base_rate * experience_factor * trend_factor
        return np.clip(adaptive_rate, 0.01, 0.2)
    
    def evaluate_prediction_accuracy(self, predicted_probs, actual_draw):
        """Évaluation améliorée de la précision des prédictions"""
        if len(actual_draw) != 5:
            return 0.0, 0.0
        
        # Score ML amélioré avec pondération
        if isinstance(predicted_probs, np.ndarray) and len(predicted_probs) == 49:
            # Score basé sur la probabilité moyenne des boules tirées
            ml_score = np.mean([predicted_probs[boule-1] for boule in actual_draw])
            
            # Bonus pour les prédictions très sûres (probabilités élevées)
            high_prob_bonus = np.sum([predicted_probs[boule-1] for boule in actual_draw if predicted_probs[boule-1] > 0.7])
            ml_score += high_prob_bonus * 0.1
        else:
            ml_score = 0.0
            
        # Score fréquences avec amélioration
        freq_weights = np.ones(49) / 49  # Uniforme en fallback
        freq_score = np.mean([freq_weights[boule-1] for boule in actual_draw])
        
        return ml_score, freq_score
    
    def detect_pattern_changes(self):
        """Détecte les changements dans les patterns de tirage"""
        if len(self.performance_history['ml_scores']) < 20:
            return False
        
        # Comparer les performances récentes vs historiques
        recent_perf = np.mean(self.performance_history['ml_scores'][-10:])
        historical_perf = np.mean(self.performance_history['ml_scores'][-20:-10])
        
        # Détection de changement significatif
        change_threshold = 0.1
        return abs(recent_perf - historical_perf) > change_threshold
    
    def update_performance(self, prediction, actual_draw, ml_probs=None):
        """Mise à jour améliorée des performances avec apprentissage adaptatif"""
        if ml_probs is not None:
            ml_score, freq_score = self.evaluate_prediction_accuracy(ml_probs, actual_draw)
            
            # Ajouter à l'historique
            self.performance_history['predictions'].append(prediction)
            self.performance_history['actuals'].append(actual_draw)
            self.performance_history['ml_scores'].append(float(ml_score))
            self.performance_history['freq_scores'].append(float(freq_score))
            self.performance_history['dates'].append(datetime.now().isoformat())
            
            # Calculer la confiance et la tendance
            confidence = self.calculate_confidence()
            trend = self.calculate_performance_trend()
            
            self.performance_history['confidence_scores'].append(float(confidence))
            self.performance_history['performance_trends'].append(float(trend))
            
            # Adaptation intelligente du poids ML
            old_weight = self.ml_weight
            self.adapt_ml_weight_intelligent(ml_score, freq_score, confidence, trend)
            
            # Enregistrer les changements de stratégie significatifs
            weight_change = abs(self.ml_weight - old_weight)
            if weight_change > 0.05:  # Changement significatif
                change_info = {
                    'timestamp': datetime.now().isoformat(),
                    'old_weight': float(old_weight),
                    'new_weight': float(self.ml_weight),
                    'reason': 'adaptive_learning',
                    'ml_score': float(ml_score),
                    'freq_score': float(freq_score),
                    'confidence': float(confidence),
                    'trend': float(trend)
                }
                self.performance_history['strategy_changes'].append(change_info)
            
            # Maintenir une taille d'historique raisonnable
            max_history = 200
            for key in self.performance_history:
                if isinstance(self.performance_history[key], list) and len(self.performance_history[key]) > max_history:
                    self.performance_history[key] = self.performance_history[key][-max_history:]
            
            self.save_history()
    
    def adapt_ml_weight_intelligent(self, ml_score, freq_score, confidence, trend):
        """Adaptation intelligente du poids ML avec multiple critères"""
        if len(self.performance_history['ml_scores']) < 5:
            return
        
        # Taux d'apprentissage adaptatif
        learning_rate = self.adaptive_learning_rate()
        
        # Calculer la performance relative avec fenêtre glissante
        window_size = min(self.performance_window, len(self.performance_history['ml_scores']))
        recent_ml = np.mean(self.performance_history['ml_scores'][-window_size:])
        recent_freq = np.mean(self.performance_history['freq_scores'][-window_size:])
        
        # Facteur de performance relative
        if recent_freq > 0:
            performance_ratio = recent_ml / recent_freq
        else:
            performance_ratio = 1.0
        
        # Ajustement basé sur multiple critères
        if performance_ratio > 1.1 and confidence > self.confidence_threshold:
            # ML performe bien avec confiance élevée -> augmenter poids ML
            adjustment = learning_rate * (1 + trend * 2)
            self.ml_weight = min(0.95, self.ml_weight + adjustment)
        elif performance_ratio < 0.9 or confidence < 0.3:
            # ML performe mal ou confiance faible -> réduire poids ML
            adjustment = learning_rate * (1 + abs(trend))
            self.ml_weight = max(0.05, self.ml_weight - adjustment)
        elif trend < -0.05:
            # Tendance négative -> réduction prudente
            adjustment = learning_rate * 0.5
            self.ml_weight = max(0.1, self.ml_weight - adjustment)
        
        # Exploration périodique pour éviter les optima locaux
        if len(self.performance_history['ml_scores']) % 50 == 0:
            exploration_noise = np.random.normal(0, self.exploration_factor)
            self.ml_weight = np.clip(self.ml_weight + exploration_noise, 0.05, 0.95)
        
        # Enregistrer le poids dans l'historique
        self.performance_history['ml_weight_history'].append(float(self.ml_weight))
        
        # Décroissance du taux d'apprentissage avec l'expérience
        self.adaptation_rate *= self.learning_rate_decay
        self.adaptation_rate = max(0.01, self.adaptation_rate)
    
    def get_adaptive_weights(self):
        """Retourne les poids adaptatifs optimisés"""
        return self.ml_weight, 1.0 - self.ml_weight
    
    def should_retrain_models(self):
        """Détermine si les modèles doivent être ré-entraînés"""
        # Critères pour déclencher un ré-entraînement
        pattern_change = self.detect_pattern_changes()
        low_confidence = self.calculate_confidence() < 0.3
        negative_trend = self.calculate_performance_trend() < -0.1
        
        # Éviter les ré-entraînements trop fréquents
        last_retrain = getattr(self, 'last_retrain', datetime.min)
        time_since_retrain = (datetime.now() - last_retrain).days
        min_days_between_retrains = 7
        
        should_retrain = (pattern_change or low_confidence or negative_trend) and \
                        time_since_retrain > min_days_between_retrains
        
        if should_retrain:
            self.last_retrain = datetime.now()
        
        return should_retrain
    
    def get_performance_summary(self):
        """Retourne un résumé détaillé des performances"""
        if len(self.performance_history['ml_scores']) < 5:
            return "Données insuffisantes pour l'analyse adaptative"
        
        window_size = min(self.performance_window, len(self.performance_history['ml_scores']))
        recent_ml = np.mean(self.performance_history['ml_scores'][-window_size:])
        recent_freq = np.mean(self.performance_history['freq_scores'][-window_size:])
        confidence = self.calculate_confidence()
        trend = self.calculate_performance_trend()
        
        # Stabilité des poids
        recent_weights = self.performance_history['ml_weight_history'][-window_size:] if self.performance_history['ml_weight_history'] else [self.ml_weight]
        weight_stability = 1.0 - np.std(recent_weights) if len(recent_weights) > 1 else 1.0
        
        return {
            'ml_score_recent': recent_ml,
            'freq_score_recent': recent_freq,
            'performance_ratio': recent_ml / recent_freq if recent_freq > 0 else 1.0,
            'current_ml_weight': self.ml_weight,
            'confidence': confidence,
            'trend': trend,
            'weight_stability': weight_stability,
            'total_predictions': len(self.performance_history['ml_scores']),
            'strategy_changes': len(self.performance_history['strategy_changes']),
            'should_retrain': self.should_retrain_models(),
            'learning_rate': self.adaptation_rate
        }

# Initialiser la stratégie adaptative
adaptive_strategy = AdaptiveStrategy()

def update_adaptive_strategy_with_recent_draws(con, strategy):
    """Met à jour la stratégie adaptative avec les tirages récents"""
    try:
        # Récupérer les 5 derniers tirages pour évaluation
        recent_draws = con.execute("""
            SELECT boule_1, boule_2, boule_3, boule_4, boule_5, date_de_tirage
            FROM loto_draws 
            ORDER BY date_de_tirage DESC 
            LIMIT 5
        """).fetchdf()
        
        if len(recent_draws) >= 2:
            print("   🎯 Mise à jour de la stratégie adaptative...")
            
            # Simuler des prédictions pour les tirages passés
            for i in range(1, min(len(recent_draws), 4)):
                actual_draw = recent_draws.iloc[i-1][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values
                previous_draw = recent_draws.iloc[i][['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values
                
                # Mettre à jour la stratégie (sans probabilities ML pour simplifier)
                strategy.update_performance(
                    prediction=previous_draw.tolist(), 
                    actual_draw=actual_draw.tolist()
                )
            
            # Afficher le résumé des performances
            summary = strategy.get_performance_summary()
            if isinstance(summary, dict):
                print(f"      ML Weight: {summary['current_ml_weight']:.2f}")
                print(f"      Total Predictions: {summary['total_predictions']}")
    except Exception as e:
        print(f"   ⚠️ Erreur mise à jour stratégie: {e}")

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    print("✓ Connexion à Redis réussie.")
except redis.exceptions.ConnectionError:
    print("⚠️ Impossible de se connecter à Redis. Le caching sera désactivé.")
    redis_client = None

# --- Fonctions de Scoring Optimisées ---
@njit
def _count_consecutive_numba(grid: np.ndarray) -> int:
    if len(grid) < 2:
        return 0
    count = 0
    sorted_grid = np.sort(grid)
    for i in range(len(sorted_grid) - 1):
        if sorted_grid[i+1] - sorted_grid[i] == 1:
            count += 1
    return count

def count_consecutive_safe(row_values, balls_cols):
    grid = np.array([row_values[col] for col in balls_cols], dtype=np.int32)
    return _count_consecutive_numba(grid)

# --- Extraction des features cycliques ---
def add_cyclic_features(df, date_col='date_de_tirage'):
    df2 = df.copy()
    if date_col in df2.columns:
        df2[date_col] = pd.to_datetime(df2[date_col])
        df2['dayofweek'] = df2[date_col].dt.dayofweek
        df2['month'] = df2[date_col].dt.month
        df2['sin_day'] = np.sin(2 * np.pi * df2['dayofweek'] / 7)
        df2['cos_day'] = np.cos(2 * np.pi * df2['dayofweek'] / 7)
        df2['sin_month'] = np.sin(2 * np.pi * df2['month'] / 12)
        df2['cos_month'] = np.cos(2 * np.pi * df2['month'] / 12)
    for i in range(1, 6):
        col = f'boule_{i}'
        df2[f'sin_boule_{i}'] = np.sin(2 * np.pi * df2[col] / 49)
        df2[f'cos_boule_{i}'] = np.cos(2 * np.pi * df2[col] / 49)
    return df2

# --- Fonctions d'Analyse (DuckDB) ---
def analyze_criteria_duckdb(db_con: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    print("Début de l'analyse des critères avec DuckDB...")

    db_con.execute(f"""
        CREATE OR REPLACE TEMPORARY VIEW BaseData AS
        SELECT ROW_NUMBER() OVER (ORDER BY date_de_tirage) AS draw_index, 
               date_de_tirage::DATE as date_de_tirage,
               boule_1, boule_2, boule_3, boule_4, boule_5
        FROM {table_name};
    """)

    df_pandas = db_con.table('BaseData').fetchdf()
    balls_cols = [f'boule_{i}' for i in range(1, 6)]

    freq = pd.Series(df_pandas[balls_cols].values.flatten()).value_counts().reindex(BALLS, fill_value=0)
    last_3_draws = df_pandas.iloc[-3:][balls_cols].values.flatten()
    last_3_numbers_set = set(last_3_draws)
    
    # MODIFICATION : Ne plus exclure de numéros automatiquement
    # Filtre de fréquence : exclure les numéros avec fréquence < 80 ou > 100
    # freq_excluded = set(freq[(freq < 80) | (freq > 100)].index.tolist())
    freq_excluded = set()  # Aucun numéro exclu par défaut
    
    # MODIFICATION : Utiliser uniquement les exclusions manuelles de l'utilisateur
    if EXCLUDED_NUMBERS is not None:
        numbers_to_exclude = EXCLUDED_NUMBERS  # Seulement les exclusions manuelles
        exclusion_source = "paramètre utilisateur uniquement"
        print(f"   ✓ Numéros exclus (utilisateur): {sorted(numbers_to_exclude)}")
    else:
        numbers_to_exclude = set()  # Aucune exclusion automatique
        exclusion_source = "aucune exclusion"
        print(f"   ✓ Mode sans exclusion : tous les numéros 1-49 disponibles")
    
    # Garder l'info des anciens critères pour les statistiques
    freq_excluded_info = set(freq[(freq < 80) | (freq > 100)].index.tolist())
    if freq_excluded_info:
        print(f"   ℹ️ Numéros avec fréquence atypique (<80 ou >100) : {sorted(freq_excluded_info)} (non exclus)")

    last_appearance = df_pandas.melt(
        id_vars=['date_de_tirage'], 
        value_vars=balls_cols, 
        value_name='numero'
    ).groupby('numero')['date_de_tirage'].max()

    pair_counts_df = db_con.execute("""
        SELECT n1, n2, COUNT(*) as compte
        FROM (
            SELECT LEAST(boule_1, boule_2) AS n1, GREATEST(boule_1, boule_2) AS n2 FROM BaseData
            UNION ALL SELECT LEAST(boule_1, boule_3), GREATEST(boule_1, boule_3) FROM BaseData
            UNION ALL SELECT LEAST(boule_1, boule_4), GREATEST(boule_1, boule_4) FROM BaseData
            UNION ALL SELECT LEAST(boule_1, boule_5), GREATEST(boule_1, boule_5) FROM BaseData
            UNION ALL SELECT LEAST(boule_2, boule_3), GREATEST(boule_2, boule_3) FROM BaseData
            UNION ALL SELECT LEAST(boule_2, boule_4), GREATEST(boule_2, boule_4) FROM BaseData
            UNION ALL SELECT LEAST(boule_2, boule_5), GREATEST(boule_2, boule_5) FROM BaseData
            UNION ALL SELECT LEAST(boule_3, boule_4), GREATEST(boule_3, boule_4) FROM BaseData
            UNION ALL SELECT LEAST(boule_3, boule_5), GREATEST(boule_3, boule_5) FROM BaseData
            UNION ALL SELECT LEAST(boule_4, boule_5), GREATEST(boule_4, boule_5) FROM BaseData
        )
        WHERE n1 IS NOT NULL AND n2 IS NOT NULL
        GROUP BY n1, n2
        ORDER BY compte DESC
        LIMIT 20
   
    """).fetchdf()

    pair_counts = pd.Series(pair_counts_df.set_index(['n1', 'n2'])['compte']) if not pair_counts_df.empty else pd.Series(dtype='int64')

    gaps_query = """
        SELECT numero, AVG(draw_index - lag) as periodicite
        FROM (
            SELECT numero, draw_index, LAG(draw_index, 1) OVER (PARTITION BY numero ORDER BY draw_index) as lag
            FROM (
                SELECT draw_index, unnest(list_value(boule_1, boule_2, boule_3, boule_4, boule_5)) as numero
                FROM BaseData
            )
        )
        WHERE lag IS NOT NULL
        GROUP BY numero
    """
    gaps_df = db_con.execute(gaps_query).fetchdf()
    gaps = pd.Series(gaps_df.set_index('numero')['periodicite']) if not gaps_df.empty else pd.Series(dtype='float64', index=BALLS)

    pair_impair_dist = (df_pandas[balls_cols] % 2 == 0).sum(axis=1).value_counts().sort_index()

    consecutive_list = []
    for _, row in df_pandas.iterrows():
        grid = np.array([row[col] for col in balls_cols], dtype=np.int32)
        consecutive_list.append(_count_consecutive_numba(grid))
    consecutive_counts = pd.Series(consecutive_list).value_counts().sort_index()

    hot_numbers = freq.nlargest(10).index.tolist()
    cold_numbers = freq.nsmallest(10).index.tolist()

    correlation_matrix = df_pandas[balls_cols].corr()

    regression_results = {}
    for ball in balls_cols:
        result = db_con.execute(f"""
            SELECT draw_index, {ball} as value
            FROM BaseData
            ORDER BY draw_index
        """).fetchdf()
        X = result['draw_index'].values.reshape(-1, 1)
        y = result['value'].values
        slope, intercept = np.polyfit(X.flatten(), y, 1)
        regression_results[ball] = {'slope': slope, 'intercept': intercept}

    sums = df_pandas[balls_cols].sum(axis=1)
    stds = df_pandas[balls_cols].std(axis=1)

    # === NOUVEAUX CRITÈRES STATISTIQUES ===
    
    # 1. Distribution par dizaines (0-9, 10-19, 20-29, 30-39, 40-49)
    decades_dist = []
    for _, row in df_pandas.iterrows():
        grid = row[balls_cols].values
        decades = [(n-1)//10 for n in grid]  # 0,1,2,3,4 pour les dizaines
        decades_count = pd.Series(decades).value_counts()
        decades_dist.append(decades_count.reindex(range(5), fill_value=0).values)
    decades_dist = np.array(decades_dist)
    decades_entropy = -np.sum(decades_dist * np.log(decades_dist + 1e-10), axis=1)
    
    # 2. Espacement entre numéros consécutifs (gaps)
    gaps_between_numbers = []
    for _, row in df_pandas.iterrows():
        sorted_nums = sorted(row[balls_cols].values)
        local_gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(4)]
        gaps_between_numbers.append(np.std(local_gaps))  # Écart-type des espacements
    gaps_between_numbers = np.array(gaps_between_numbers)
    
    # 3. Répartition haute/basse (1-25 vs 26-49)
    high_low_ratio = []
    for _, row in df_pandas.iterrows():
        grid = row[balls_cols].values
        low_count = sum(1 for n in grid if n <= 25)
        high_count = 5 - low_count
        ratio = low_count / 5.0  # Proportion de numéros bas
        high_low_ratio.append(ratio)
    high_low_ratio = np.array(high_low_ratio)
    
    # 4. Nombres premiers vs composés
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    prime_ratio = []
    for _, row in df_pandas.iterrows():
        grid = row[balls_cols].values
        prime_count = sum(1 for n in grid if n in primes)
        prime_ratio.append(prime_count / 5.0)
    prime_ratio = np.array(prime_ratio)
    
    # 5. Variance des positions (mesure de dispersion)
    position_variance = []
    for _, row in df_pandas.iterrows():
        grid = sorted(row[balls_cols].values)
        positions = [(n-1)/48.0 for n in grid]  # Normalisation 0-1
        position_variance.append(np.var(positions))
    position_variance = np.array(position_variance)

    # Calcul des variances pour tous les critères
    sum_var = np.var(sums)
    std_var = np.var(stds)
    pair_impair_var = np.var((pair_impair_dist/pair_impair_dist.sum()).values)
    decades_entropy_var = np.var(decades_entropy)
    gaps_var = np.var(gaps_between_numbers)
    high_low_var = np.var(high_low_ratio)
    prime_var = np.var(prime_ratio)
    position_var = np.var(position_variance)
    
    # Poids équilibrés basés sur l'inverse des variances, mais avec normalisation
    raw_weights = {
        'sum': 1.0 / (1.0 + sum_var),
        'std': 1.0 / (1.0 + std_var), 
        'pair_impair': 1.0 / (1.0 + pair_impair_var),
        'decades_entropy': 1.0 / (1.0 + decades_entropy_var),
        'gaps': 1.0 / (1.0 + gaps_var),
        'high_low': 1.0 / (1.0 + high_low_var),
        'prime_ratio': 1.0 / (1.0 + prime_var),
        'position_variance': 1.0 / (1.0 + position_var)
    }
    
    # Normalisation pour que les poids soient plus équilibrés (pas de domination extrême)
    total_weight = sum(raw_weights.values())
    dynamic_weights = {key: value / total_weight for key, value in raw_weights.items()}
    
    # Ajustement pour éviter qu'un critère domine complètement (max 25% pour un critère maintenant)
    max_weight = 0.25
    for key in dynamic_weights:
        if dynamic_weights[key] > max_weight:
            excess = dynamic_weights[key] - max_weight
            dynamic_weights[key] = max_weight
            # Redistribuer l'excès sur les autres critères
            other_keys = [k for k in dynamic_weights.keys() if k != key]
            for other_key in other_keys:
                dynamic_weights[other_key] += excess / len(other_keys)
    
    # Re-normalisation finale
    total_weight = sum(dynamic_weights.values())
    dynamic_weights = {key: value / total_weight for key, value in dynamic_weights.items()}

    delta = pd.to_datetime('now').normalize() - pd.to_datetime(last_appearance.reindex(BALLS))
    numbers_analysis = pd.DataFrame({
        'Numero': BALLS,
        'Frequence': freq.reindex(BALLS, fill_value=0),
        'Dernier_Tirage': last_appearance.reindex(BALLS),
        'Jours_Depuis_Tirage': delta.dt.days,
        'Ecart_Moyen_Tirages': gaps.reindex(BALLS)
    }).sort_values('Frequence', ascending=False).reset_index(drop=True)

    print("   ✓ Analyse DuckDB terminée.")

    return {
        'freq': freq,
        'hot_numbers': hot_numbers,
        'cold_numbers': cold_numbers,
        'last_draw': df_pandas[balls_cols].iloc[-1].tolist(),
        'pair_impair_probs': pair_impair_dist / pair_impair_dist.sum(),
        'consecutive_counts': consecutive_counts,
        'pair_counts': pair_counts,
        'numbers_to_exclude': numbers_to_exclude,  # Maintenant peut être vide
        'freq_excluded': freq_excluded_info,  # Pour info seulement, pas d'exclusion
        'sums': sums,
        'stds': stds,
        'dynamic_weights': dynamic_weights,
        'last_appearance': last_appearance,
        'numbers_analysis': numbers_analysis,
        'correlation_matrix': correlation_matrix,
        'regression_results': regression_results,
        # Nouveaux critères statistiques
        'decades_entropy': decades_entropy,
        'gaps_between_numbers': gaps_between_numbers,
        'high_low_ratio': high_low_ratio,
        'prime_ratio': prime_ratio,
        'position_variance': position_variance
    }

# --- Fonctions de Machine Learning ---
def train_xgboost_parallel(df: pd.DataFrame):
    print("Entraînement des modèles XGBoost...")
    
    # Paramètres selon le mode
    if hasattr(ARGS, 'fast_training') and ARGS.fast_training:
        print("  ⚡ Mode entraînement ultra-rapide activé")
        xgb_params = {
            'n_estimators': 25,       # Très réduit pour la vitesse
            'max_depth': 3,           # Profondeur minimale
            'learning_rate': 0.3,     # Taux d'apprentissage élevé
            'subsample': 0.7,         # Sous-échantillonnage agressif
            'colsample_bytree': 0.7,  # Sélection de features réduite
        }
    else:
        print("  🎯 Mode entraînement optimisé standard")
        xgb_params = {
            'n_estimators': 50,       # Équilibre vitesse/performance
            'max_depth': 4,           # Profondeur modérée
            'learning_rate': 0.2,     # Taux d'apprentissage modéré
            'subsample': 0.8,         # Sous-échantillonnage modéré
            'colsample_bytree': 0.8,  # Sélection de features standard
        }
    
    balls_cols = [f'boule_{i}' for i in range(1, 6)]

    df_features = add_cyclic_features(df)
    feature_cols = balls_cols + [col for col in df_features.columns if col.startswith('sin_') or col.startswith('cos_')]
    X = df_features[feature_cols].iloc[:-1].values
    
    def train_balls_multilabel_model():
        """Entraîne un modèle multi-label pour prédire les 5 boules principales (1-49)"""
        print("  📊 Entraînement du modèle multi-label pour les boules principales (1-49)...")
        
        # Créer les labels multi-label pour les 49 boules possibles
        y_balls = np.zeros((len(X), 49))
        for i, row in enumerate(df_features[balls_cols].iloc[1:].values):
            for ball_num in row:
                if 1 <= ball_num <= 49:  # Vérification de sécurité
                    y_balls[i, ball_num - 1] = 1  # -1 car indexage à partir de 0
        
        # Statistiques sur les données
        positive_samples_per_ball = np.sum(y_balls, axis=0)
        print(f"   📈 Moyenne d'occurrences par boule: {np.mean(positive_samples_per_ball):.1f}")
        print(f"   📊 Min/Max occurrences: {np.min(positive_samples_per_ball):.0f}/{np.max(positive_samples_per_ball):.0f}")
        
        # Utilisation de RandomForest avec MultiOutputClassifier pour gérer la multi-label
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        
        model = MultiOutputClassifier(base_estimator, n_jobs=-1)
        model.fit(X, y_balls)
        
        dump(model, MODEL_DIR / 'model_balls_multilabel.joblib')
        print("   ✓ Modèle multi-label boules principales sauvegardé.")
        return model
    
    def train_chance_multilabel_model():
        """Entraîne un modèle multi-label pour prédire le numéro chance (1-10)"""
        print("  🎯 Entraînement du modèle multi-label pour les numéros chance (1-10)...")
        
        # Créer les labels multi-label pour les 10 numéros chance possibles
        y_chance = np.zeros((len(X), 10))
        for i, chance_val in enumerate(df_features['numero_chance'].iloc[1:].values):
            if 1 <= chance_val <= 10:  # Vérification de sécurité
                y_chance[i, chance_val - 1] = 1  # -1 car indexage à partir de 0
        
        # Statistiques sur les données
        positive_samples_per_chance = np.sum(y_chance, axis=0)
        print(f"   📈 Moyenne d'occurrences par numéro chance: {np.mean(positive_samples_per_chance):.1f}")
        print(f"   📊 Min/Max occurrences: {np.min(positive_samples_per_chance):.0f}/{np.max(positive_samples_per_chance):.0f}")
        
        # Utilisation de RandomForest avec MultiOutputClassifier pour gérer la multi-label
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        
        model = MultiOutputClassifier(base_estimator, n_jobs=-1)
        model.fit(X, y_chance)
        
        dump(model, MODEL_DIR / 'model_chance_multilabel.joblib')
        print("   ✓ Modèle multi-label numéros chance sauvegardé.")
        return model
    
    # Entraînement des modèles multi-label
    balls_model = train_balls_multilabel_model()
    chance_model = train_chance_multilabel_model()
    
    models = [balls_model, chance_model]
    
    # Sauvegarder les métadonnées
    metadata = {
        'features_count': len(feature_cols),
        'model_type': 'randomforest_multilabel',
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'version': '4.0',
        'ball_models': 1,  # Un seul modèle multi-label pour les boules
        'chance_models': 1,  # Un seul modèle multi-label pour les chances
        'total_models': 2,  # Au lieu de 59 modèles individuels
        'balls_outputs': 49,  # 49 sorties pour les boules 1-49
        'chance_outputs': 10   # 10 sorties pour les chances 1-10
    }
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("   ✓ Entraînement terminé et modèles sauvegardés.")
    return models

# --- Auto-amélioration du Training ---
def train_xgboost_parallel_adaptive(df: pd.DataFrame, adaptive_strategy=None):
    """Entraînement des modèles XGBoost avec auto-amélioration"""
    print("Entraînement adaptatif des modèles XGBoost...")
    
    # Déterminer les paramètres optimaux selon la performance historique
    performance_summary = adaptive_strategy.get_performance_summary() if adaptive_strategy else None
    
    if performance_summary and isinstance(performance_summary, dict):
        confidence = performance_summary.get('confidence', 0.5)
        trend = performance_summary.get('trend', 0.0)
        
        # Ajuster les paramètres selon la confiance et la tendance
        if confidence > 0.7 and trend > 0:
            # Performance excellente -> paramètres conservateurs
            print("  🎯 Mode conservateur (performance élevée)")
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        elif confidence < 0.4 or trend < -0.1:
            # Performance faible -> exploration plus agressive
            print("  🔍 Mode exploratoire (amélioration nécessaire)")
            xgb_params = {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.05,
                'reg_lambda': 0.05
            }
        else:
            # Performance normale -> paramètres équilibrés
            print("  ⚖️ Mode équilibré (performance stable)")
            xgb_params = {
                'n_estimators': 75,
                'max_depth': 5,
                'learning_rate': 0.15,
                'subsample': 0.85,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.08,
                'reg_lambda': 0.08
            }
    elif hasattr(ARGS, 'fast_training') and ARGS.fast_training:
        print("  ⚡ Mode entraînement ultra-rapide activé")
        xgb_params = {
            'n_estimators': 25,
            'max_depth': 3,
            'learning_rate': 0.3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
    else:
        print("  🎯 Mode entraînement standard")
        xgb_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    
    balls_cols = [f'boule_{i}' for i in range(1, 6)]
    
    # Amélioration des features avec plus d'ingénierie
    df_features = add_advanced_features(df)
    feature_cols = balls_cols + [col for col in df_features.columns 
                                if col.startswith(('sin_', 'cos_', 'lag_', 'rolling_', 'trend_'))]
    
    X = df_features[feature_cols].iloc[:-1].values
    print(f"  📊 Features utilisées: {len(feature_cols)}")
    
    def train_balls_multilabel_model():
        """Entraîne un modèle multi-label amélioré pour les boules principales"""
        print("  🎯 Entraînement du modèle multi-label optimisé pour les boules principales...")
        
        # Créer les labels multi-label pour les 49 boules possibles
        y_balls = np.zeros((len(X), 49))
        for i, row in enumerate(df_features[balls_cols].iloc[1:].values):
            for ball_num in row:
                if 1 <= ball_num <= 49:
                    y_balls[i, ball_num - 1] = 1
        
        # Utilisation d'un ensemble de modèles pour plus de robustesse
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Ensemble de classifieurs
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=GLOBAL_SEED,
                n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=GLOBAL_SEED + 1,
                n_jobs=-1
            ))
        ]
        
        # Entraîner les modèles de l'ensemble
        models = {}
        for name, estimator in estimators:
            print(f"    Entraînement {name}...")
            model = MultiOutputClassifier(estimator, n_jobs=-1)
            model.fit(X, y_balls)
            models[name] = model
        
        # Sauvegarder l'ensemble
        ensemble_model = {
            'models': models,
            'feature_names': feature_cols,
            'training_params': xgb_params,
            'training_date': datetime.now().isoformat()
        }
        
        dump(ensemble_model, MODEL_DIR / 'model_balls_multilabel.joblib')
        print("   ✓ Ensemble de modèles multi-label boules sauvegardé.")
        return ensemble_model
    
    def train_chance_multilabel_model():
        """Entraîne un modèle multi-label amélioré pour les numéros chance"""
        print("  🎲 Entraînement du modèle multi-label optimisé pour les numéros chance...")
        
        # Créer les labels multi-label pour les 10 numéros chance possibles
        y_chance = np.zeros((len(X), 10))
        for i, chance_val in enumerate(df_features['numero_chance'].iloc[1:].values):
            if 1 <= chance_val <= 10:
                y_chance[i, chance_val - 1] = 1
        
        # Modèle spécialisé pour les numéros chance (plus simple)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        base_estimator = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        
        model = MultiOutputClassifier(base_estimator, n_jobs=-1)
        model.fit(X, y_chance)
        
        # Encapsuler dans un dictionnaire pour cohérence
        chance_model = {
            'model': model,
            'feature_names': feature_cols,
            'training_params': xgb_params,
            'training_date': datetime.now().isoformat()
        }
        
        dump(chance_model, MODEL_DIR / 'model_chance_multilabel.joblib')
        print("   ✓ Modèle multi-label numéros chance amélioré sauvegardé.")
        return chance_model
    
    # Entraînement des modèles
    balls_model = train_balls_multilabel_model()
    chance_model = train_chance_multilabel_model()
    
    models = [balls_model, chance_model]
    
    # Sauvegarder les métadonnées étendues
    metadata = {
        'features_count': len(feature_cols),
        'model_type': 'ensemble_multilabel_adaptive',
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '5.0',
        'ball_models': len(balls_model['models']),
        'chance_models': 1,
        'total_models': len(balls_model['models']) + 1,
        'balls_outputs': 49,
        'chance_outputs': 10,
        'training_params': xgb_params,
        'adaptive_strategy_used': adaptive_strategy is not None,
        'performance_summary': performance_summary
    }
    
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("   ✓ Entraînement adaptatif terminé et modèles sauvegardés.")
    return {'balls_multilabel': balls_model, 'chance_multilabel': chance_model}

def add_advanced_features(df, date_col='date_de_tirage'):
    """Ajoute des features avancées pour améliorer les prédictions"""
    df2 = df.copy()
    balls_cols = [f'boule_{i}' for i in range(1, 6)]
    
    # Features cycliques existantes
    if date_col in df2.columns:
        df2[date_col] = pd.to_datetime(df2[date_col])
        df2['dayofweek'] = df2[date_col].dt.dayofweek
        df2['month'] = df2[date_col].dt.month
        df2['sin_day'] = np.sin(2 * np.pi * df2['dayofweek'] / 7)
        df2['cos_day'] = np.cos(2 * np.pi * df2['dayofweek'] / 7)
        df2['sin_month'] = np.sin(2 * np.pi * df2['month'] / 12)
        df2['cos_month'] = np.cos(2 * np.pi * df2['month'] / 12)
    
    for i in range(1, 6):
        col = f'boule_{i}'
        df2[f'sin_boule_{i}'] = np.sin(2 * np.pi * df2[col] / 49)
        df2[f'cos_boule_{i}'] = np.cos(2 * np.pi * df2[col] / 49)
    
    # Features de lag (valeurs précédentes)
    for lag in [1, 2, 3]:
        for col in balls_cols:
            df2[f'lag_{lag}_{col}'] = df2[col].shift(lag)
    
    # Features de moyenne mobile
    for window in [3, 5, 7]:
        df2[f'rolling_mean_{window}'] = df2[balls_cols].mean(axis=1).rolling(window=window).mean()
        df2[f'rolling_std_{window}'] = df2[balls_cols].std(axis=1).rolling(window=window).mean()
    
    # Tendances
    df2['trend_sum'] = df2[balls_cols].sum(axis=1).diff()
    df2['trend_mean'] = df2[balls_cols].mean(axis=1).diff()
    
    # Combler les valeurs manquantes
    df2 = df2.fillna(method='bfill').fillna(0)
    
    return df2

def load_saved_models() -> dict:
    """Charge les modèles multi-label améliorés"""
    models = {'balls_multilabel': None, 'chance_multilabel': None}
    
    # Vérifier la compatibilité des features
    metadata_path = MODEL_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        expected_features = metadata.get('features_count', 19)
        model_type = metadata.get('model_type', 'ensemble_multilabel_adaptive')
        print(f"   📊 Modèles attendus: {model_type} avec {expected_features} features")
    
    # Charger le modèle ensemble pour les boules
    balls_model_path = MODEL_DIR / 'model_balls_multilabel.joblib'
    if balls_model_path.exists():
        try:
            balls_model = load(balls_model_path)
            if isinstance(balls_model, dict) and 'models' in balls_model:
                print(f"   ✓ Ensemble de modèles boules chargé ({len(balls_model['models'])} modèles)")
            else:
                print("   ✓ Modèle boules standard chargé")
            models['balls_multilabel'] = balls_model
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle boules: {e}")
    
    # Charger le modèle pour les numéros chance
    chance_model_path = MODEL_DIR / 'model_chance_multilabel.joblib'
    if chance_model_path.exists():
        try:
            chance_model = load(chance_model_path)
            models['chance_multilabel'] = chance_model
            print("   ✓ Modèle chance chargé")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle chance: {e}")
    
    loaded_models = sum(1 for model in models.values() if model is not None)
    print(f"   {'✅' if loaded_models == 2 else '⚠️'} {loaded_models}/2 modèles chargés")
    
    return models

def generate_grid_vectorized_adaptive(criteria: dict, models: dict, X_last: np.ndarray) -> list:
    """Génération de grille adaptative avec ensemble de modèles et privilège du TOP 25"""
    N_CANDIDATES, EXPLORATION_RATE, TOP_PERCENT_SELECTION = 500, 0.2, 0.05
    
    # Vérifier la disponibilité des modèles
    has_ball_model = models and 'balls_multilabel' in models and models['balls_multilabel'] is not None
    has_chance_model = models and 'chance_multilabel' in models and models['chance_multilabel'] is not None
    use_models = has_ball_model
    
    freq_weights = criteria['freq'].reindex(BALLS, fill_value=0).values / criteria['freq'].sum()
    
    # Obtenir les poids adaptatifs
    ml_weight, freq_weight = adaptive_strategy.get_adaptive_weights()
    
    # NOUVEAU : Charger et privilégier le TOP 25 ML
    top25_numbers = []
    top25_weights = None
    ml_top25_path = OUTPUT_DIR / 'ml_top25_predictions.csv'
    
    if ml_top25_path.exists():
        try:
            top25_df = pd.read_csv(ml_top25_path)
            top25_numbers = top25_df['numero'].head(25).tolist()
            
            # Créer des poids privilégiés pour le TOP 25
            top25_weights = np.zeros(49)
            for i, numero in enumerate(top25_numbers):
                if 1 <= numero <= 49:
                    # Poids dégressif : TOP 1 = poids max, TOP 25 = poids min mais supérieur à la normale
                    privilege_factor = 3.0 - (i / 25.0) * 2.0  # Entre 3.0 et 1.0
                    top25_weights[numero-1] = freq_weights[numero-1] * privilege_factor
            
            # Les numéros non TOP 25 gardent leurs poids originaux mais réduits
            for i in range(49):
                if top25_weights[i] == 0:  # Pas dans le TOP 25
                    top25_weights[i] = freq_weights[i] * 0.5  # Réduction du poids
            
            # Normalisation
            top25_weights = top25_weights / top25_weights.sum()
            
            print(f"   🎯 TOP 25 ML chargé et privilégié ({len(top25_numbers)} numéros)")
            if not ARGS.silent:
                print(f"      Top 5: {top25_numbers[:5]}")
                
        except Exception as e:
            print(f"   ⚠️ Erreur chargement TOP 25: {e}")
            top25_numbers = []
            top25_weights = None
    
    # Préparation des features avancées
    df_last = pd.DataFrame([X_last[0]], columns=[f'boule_{i}' for i in range(1, 6)])
    df_last['date_de_tirage'] = pd.Timestamp.now()
    df_last_features = add_advanced_features(df_last)
    
    # Sélectionner les mêmes features que l'entraînement
    available_features = df_last_features.columns.tolist()
    feature_cols = [f'boule_{i}' for i in range(1, 6)] + \
                   [col for col in available_features if col.startswith(('sin_', 'cos_', 'lag_', 'rolling_', 'trend_'))]
    
    # Combler les features manquantes avec des valeurs par défaut
    for col in feature_cols:
        if col not in df_last_features.columns:
            df_last_features[col] = 0
    
    X_last_features = df_last_features[feature_cols].values
    
    if use_models:
        try:
            ball_model = models['balls_multilabel']
            
            if isinstance(ball_model, dict) and 'models' in ball_model:
                # Ensemble de modèles - faire la moyenne des prédictions
                ensemble_predictions = []
                
                for model_name, model in ball_model['models'].items():
                    if hasattr(model, 'predict_proba'):
                        ball_probas_list = model.predict_proba(X_last_features)
                        predictions = np.array([probas[0][1] if len(probas[0]) > 1 else probas[0][0] 
                                               for probas in ball_probas_list])
                        ensemble_predictions.append(predictions)
                    elif hasattr(model, 'predict'):
                        # Si pas de predict_proba, utiliser predict
                        predictions = model.predict(X_last_features)[0]  # Premier échantillon
                        ensemble_predictions.append(predictions)
                
                if ensemble_predictions:
                    # Moyenne pondérée des prédictions de l'ensemble
                    ball_predictions = np.mean(ensemble_predictions, axis=0)
                    print(f"   🎯 Prédictions ensemble ({len(ensemble_predictions)} modèles)")
                else:
                    ball_predictions = freq_weights
            else:
                # Modèle unique
                if hasattr(ball_model, 'predict_proba'):
                    ball_probas_list = ball_model.predict_proba(X_last_features)
                    ball_predictions = np.array([probas[0][1] if len(probas[0]) > 1 else probas[0][0] 
                                               for probas in ball_probas_list])
                elif hasattr(ball_model, 'predict'):
                    ball_predictions = ball_model.predict(X_last_features)[0]
                else:
                    ball_predictions = freq_weights
            
            # Normalisation et combinaison adaptative
            ball_predictions = np.clip(ball_predictions, 0.001, 0.999)
            ball_predictions = ball_predictions / ball_predictions.sum()
            
            # MODIFICATION : Intégrer les poids du TOP 25
            if top25_weights is not None:
                # Combinaison : ML + Fréquence + TOP 25 privilégié
                exploitation_weights = (
                    ml_weight * 0.4 * ball_predictions + 
                    freq_weight * 0.3 * freq_weights + 
                    0.3 * top25_weights  # 30% de poids pour le privilège TOP 25
                )
                print(f"   🏆 Génération avec privilège TOP 25 (30% des poids)")
            else:
                exploitation_weights = (ml_weight * ball_predictions + freq_weight * freq_weights)
            
            exploitation_weights /= exploitation_weights.sum()
            
            if not ARGS.silent:
                ml_effective_weight = ml_weight * 0.4 if top25_weights is not None else ml_weight
                freq_effective_weight = freq_weight * 0.3 if top25_weights is not None else freq_weight
                top25_effective_weight = 0.3 if top25_weights is not None else 0.0
                print(f"   🎯 Poids effectifs: ML={ml_effective_weight:.3f}, Freq={freq_effective_weight:.3f}, TOP25={top25_effective_weight:.3f}")
                
        except Exception as e:
            print(f"⚠️ Erreur avec les modèles ML: {e}")
            exploitation_weights = top25_weights if top25_weights is not None else freq_weights
    else:
        exploitation_weights = top25_weights if top25_weights is not None else freq_weights
    
    # MODIFICATION MAJEURE : Plus d'exclusion de numéros
    excluded_numbers = criteria.get('numbers_to_exclude', set())
    available_numbers = [n for n in BALLS if n not in excluded_numbers]
    
    # MODIFICATION : Si aucune exclusion, utiliser tous les numéros
    if len(excluded_numbers) == 0:
        available_numbers = list(BALLS)  # Tous les numéros 1-49 disponibles
        print(f"   ✅ TOUS les 49 numéros disponibles (aucune exclusion)")
    else:
        print(f"   ⚠️ {len(excluded_numbers)} numéros exclus manuellement: {sorted(excluded_numbers)}")
    
    if len(available_numbers) < 5:
        print(f"⚠️ ATTENTION: Seulement {len(available_numbers)} numéros disponibles!")
        return []
    
    # Génération des candidats avec privilège TOP 25
    if use_models or top25_weights is not None:
        available_exploitation_weights = np.array([exploitation_weights[n-1] for n in available_numbers])
        available_exploitation_weights = available_exploitation_weights / available_exploitation_weights.sum()
    
    available_freq_weights = np.array([freq_weights[n-1] for n in available_numbers])
    available_freq_weights = available_freq_weights / available_freq_weights.sum()
    
    candidates = []
    max_attempts = N_CANDIDATES * 10
    attempts = 0
    
    while len(candidates) < N_CANDIDATES and attempts < max_attempts:
        attempts += 1
        if use_models or top25_weights is not None:
            p = available_freq_weights if random.random() < EXPLORATION_RATE else available_exploitation_weights
        else:
            p = available_freq_weights
        candidate = np.random.choice(available_numbers, size=5, replace=False, p=p)
        candidates.append(candidate)
    
    if not candidates:
        return []
    
    # Scoring et sélection avec bonus TOP 25
    candidates_matrix = np.array(candidates)
    scores = []
    
    for grid in candidates_matrix:
        base_score = score_grid(grid, criteria, diversity_factor=0.05)
        
        # NOUVEAU : Bonus pour les numéros du TOP 25
        top25_bonus = 0.0
        if top25_numbers:
            top25_count = sum(1 for num in grid if num in top25_numbers)
            if top25_count >= 3:  # Au moins 3 numéros du TOP 25
                top25_bonus = 0.1 * (top25_count / 5.0)  # Bonus jusqu'à 0.1
                
                # Bonus supplémentaire pour les TOP 5
                top5_count = sum(1 for num in grid if num in top25_numbers[:5])
                if top5_count >= 1:
                    top25_bonus += 0.05 * (top5_count / 5.0)  # Bonus supplémentaire pour TOP 5
        
        total_score = base_score + top25_bonus
        scores.append(total_score)
    
    scores = np.array(scores)
    top_n = max(1, int(len(candidates) * TOP_PERCENT_SELECTION))
    best_indices = np.argpartition(scores, -top_n)[-top_n:]
    chosen_index = np.random.choice(best_indices)
    
    best_grid_list = [int(b) for b in candidates_matrix[chosen_index]]
    
    # Affichage des statistiques TOP 25 pour la grille choisie
    if top25_numbers and not ARGS.silent:
        top25_in_grid = [num for num in best_grid_list if num in top25_numbers]
        top5_in_grid = [num for num in best_grid_list if num in top25_numbers[:5]]
        if top25_in_grid:
            print(f"   🏆 Grille contient {len(top25_in_grid)} numéros TOP 25: {top25_in_grid}")
            if top5_in_grid:
                print(f"   🥇 Dont {len(top5_in_grid)} du TOP 5: {top5_in_grid}")
    
    # Génération du numéro chance avec modèle amélioré (inchangé)
    if has_chance_model:
        try:
            chance_model = models['chance_multilabel']
            
            # Gérer les deux formats de modèle
            if isinstance(chance_model, dict) and 'model' in chance_model:
                model = chance_model['model']
            else:
                model = chance_model
            
            if hasattr(model, 'predict_proba'):
                chance_probas_list = model.predict_proba(X_last_features)
                chance_predictions = np.array([probas[0][1] if len(probas[0]) > 1 else probas[0][0] 
                                             for probas in chance_probas_list])
                chance_predictions = np.clip(chance_predictions, 0.001, 0.999)
                chance_predictions = chance_predictions / chance_predictions.sum()
                chance_ball = int(np.random.choice(CHANCE_BALLS, p=chance_predictions))
            else:
                chance_ball = int(np.random.choice(CHANCE_BALLS))
        except Exception as e:
            print(f"⚠️ Erreur modèle chance: {e}")
            chance_ball = int(np.random.choice(CHANCE_BALLS))
    else:
        chance_ball = int(np.random.choice(CHANCE_BALLS))
    
    return sorted(best_grid_list) + [chance_ball]

# ...existing code...

# Ajouter ces fonctions après la fonction generate_grid_vectorized_adaptive et avant la fonction main()

def generate_ml_top25_predictions(models: dict, X_last: np.ndarray, criteria: dict, output_dir: Path):
    """Génère le top 25 des numéros basés sur les prédictions ML et l'exporte en CSV"""
    print("Génération du top 25 des numéros selon le training ML...")
    
    if not models or 'balls_multilabel' not in models or models['balls_multilabel'] is None:
        print("   ⚠️ Pas de modèle ML disponible pour générer les prédictions")
        return None
    
    try:
        # Préparation des features avancées
        df_last = pd.DataFrame([X_last[0]], columns=[f'boule_{i}' for i in range(1, 6)])
        df_last['date_de_tirage'] = pd.Timestamp.now()
        df_last_features = add_advanced_features(df_last)
        
        # Sélectionner les mêmes features que l'entraînement
        available_features = df_last_features.columns.tolist()
        feature_cols = [f'boule_{i}' for i in range(1, 6)] + \
                       [col for col in available_features if col.startswith(('sin_', 'cos_', 'lag_', 'rolling_', 'trend_'))]
        
        # Combler les features manquantes avec des valeurs par défaut
        for col in feature_cols:
            if col not in df_last_features.columns:
                df_last_features[col] = 0
        
        X_last_features = df_last_features[feature_cols].values
        
        # Obtenir les prédictions ML
        ball_model = models['balls_multilabel']
        ml_predictions = np.zeros(49)  # Pour les 49 numéros possibles
        
        if isinstance(ball_model, dict) and 'models' in ball_model:
            # Ensemble de modèles - faire la moyenne des prédictions
            ensemble_predictions = []
            
            for model_name, model in ball_model['models'].items():
                if hasattr(model, 'predict_proba'):
                    ball_probas_list = model.predict_proba(X_last_features)
                    predictions = np.array([probas[0][1] if len(probas[0]) > 1 else probas[0][0] 
                                           for probas in ball_probas_list])
                    ensemble_predictions.append(predictions)
                elif hasattr(model, 'predict'):
                    # Si pas de predict_proba, utiliser predict
                    predictions = model.predict(X_last_features)[0]  # Premier échantillon
                    ensemble_predictions.append(predictions)
            
            if ensemble_predictions:
                # Moyenne pondérée des prédictions de l'ensemble
                ml_predictions = np.mean(ensemble_predictions, axis=0)
                print(f"   🎯 Prédictions générées via ensemble ({len(ensemble_predictions)} modèles)")
            else:
                print("   ⚠️ Aucune prédiction générée par l'ensemble")
                return None
        else:
            # Modèle unique
            if hasattr(ball_model, 'predict_proba'):
                ball_probas_list = ball_model.predict_proba(X_last_features)
                ml_predictions = np.array([probas[0][1] if len(probas[0]) > 1 else probas[0][0] 
                                          for probas in ball_probas_list])
            elif hasattr(ball_model, 'predict'):
                ml_predictions = ball_model.predict(X_last_features)[0]
            else:
                print("   ⚠️ Modèle sans méthode de prédiction reconnue")
                return None
        
        # Normaliser les prédictions
        ml_predictions = np.clip(ml_predictions, 0.001, 0.999)
        ml_predictions = ml_predictions / ml_predictions.sum()
        
        # Récupérer les données de fréquence pour comparaison
        freq = criteria.get('freq', pd.Series())
        freq_normalized = freq.reindex(BALLS, fill_value=0).values
        freq_normalized = freq_normalized / freq_normalized.sum() if freq_normalized.sum() > 0 else freq_normalized
        
        # Obtenir les poids adaptatifs
        ml_weight, freq_weight = adaptive_strategy.get_adaptive_weights()
        
        # Calculer le score composite (ML + Fréquence)
        composite_scores = ml_weight * ml_predictions + freq_weight * freq_normalized
        
        # Créer le DataFrame avec tous les numéros et leurs scores
        predictions_data = []
        for i, numero in enumerate(BALLS):
            predictions_data.append({
                'numero': int(numero),
                'score_ml': float(ml_predictions[i]),
                'frequence_historique': int(freq[numero]) if numero in freq.index else 0,
                'score_frequence': float(freq_normalized[i]),
                'score_composite': float(composite_scores[i]),
                'rang_ml': 0,  # Sera calculé après tri
                'rang_frequence': 0,  # Sera calculé après tri
                'rang_composite': 0,  # Sera calculé après tri
                'poids_ml_utilise': float(ml_weight),
                'poids_freq_utilise': float(freq_weight),
                'is_excluded': numero in criteria.get('numbers_to_exclude', set()),
                'is_hot_number': numero in criteria.get('hot_numbers', []),
                'is_cold_number': numero in criteria.get('cold_numbers', [])
            })
        
        # Convertir en DataFrame
        predictions_df = pd.DataFrame(predictions_data)
        
        # Calculer les rangs
        predictions_df['rang_ml'] = predictions_df['score_ml'].rank(ascending=False, method='min').astype(int)
        predictions_df['rang_frequence'] = predictions_df['score_frequence'].rank(ascending=False, method='min').astype(int)
        predictions_df['rang_composite'] = predictions_df['score_composite'].rank(ascending=False, method='min').astype(int)
        
        # Trier par score composite décroissant
        predictions_df = predictions_df.sort_values('score_composite', ascending=False).reset_index(drop=True)
        
        # Ajouter une colonne de position finale
        predictions_df['position_finale'] = range(1, len(predictions_df) + 1)
        
        # Sélectionner le top 25
        top25_df = predictions_df.head(25).copy()
        
        # Réorganiser les colonnes pour la lisibilité
        column_order = [
            'position_finale', 'numero', 'score_composite', 'score_ml', 'score_frequence',
            'rang_ml', 'rang_frequence', 'rang_composite', 'frequence_historique',
            'poids_ml_utilise', 'poids_freq_utilise', 'is_excluded', 'is_hot_number', 'is_cold_number'
        ]
        top25_df = top25_df[column_order]
        
        # Sauvegarder le top 25
        top25_csv_path = output_dir / 'ml_top25_predictions.csv'
        top25_df.to_csv(top25_csv_path, index=False, float_format='%.6f')
        
        # Sauvegarder aussi le classement complet des 49 numéros
        full_predictions_path = output_dir / 'ml_full_predictions.csv'
        predictions_df.to_csv(full_predictions_path, index=False, float_format='%.6f')
        
        # Statistiques
        top25_numbers = top25_df['numero'].tolist()
        excluded_in_top25 = sum(1 for num in top25_numbers if num in criteria.get('numbers_to_exclude', set()))
        hot_in_top25 = sum(1 for num in top25_numbers if num in criteria.get('hot_numbers', []))
        cold_in_top25 = sum(1 for num in top25_numbers if num in criteria.get('cold_numbers', []))
        
        print(f"   ✅ Top 25 des prédictions ML généré avec succès")
        print(f"   📊 Statistiques du top 25:")
        print(f"      - Numéros exclus présents: {excluded_in_top25}/25")
        print(f"      - Numéros chauds présents: {hot_in_top25}/25") 
        print(f"      - Numéros froids présents: {cold_in_top25}/25")
        print(f"      - Score composite moyen: {top25_df['score_composite'].mean():.6f}")
        print(f"      - Score ML moyen: {top25_df['score_ml'].mean():.6f}")
        print(f"   💾 Fichiers sauvegardés:")
        print(f"      - Top 25: '{top25_csv_path}'")
        print(f"      - Classement complet: '{full_predictions_path}'")
        
        return top25_df
        
    except Exception as e:
        print(f"   ❌ Erreur lors de la génération des prédictions ML: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_ml_predictions_summary(top25_df, criteria: dict, output_dir: Path):
    """Crée un résumé détaillé des prédictions ML"""
    if top25_df is None:
        return
    
    try:
        # Créer un rapport de synthèse
        summary_content = f"""# Rapport des Prédictions ML - Top 25 Numéros
*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}*

## 🎯 Configuration du Modèle
- **Poids ML utilisé**: {top25_df['poids_ml_utilise'].iloc[0]:.3f} ({top25_df['poids_ml_utilise'].iloc[0]*100:.1f}%)
- **Poids Fréquence utilisé**: {top25_df['poids_freq_utilise'].iloc[0]:.3f} ({top25_df['poids_freq_utilise'].iloc[0]*100:.1f}%)
- **Nombre total de numéros évalués**: 49
- **Numéros sélectionnés pour le top**: 25

## 🏆 Top 10 des Numéros Recommandés par l'IA

| Rang | Numéro | Score Composite | Score ML | Score Fréq. | Freq. Hist. | Statut |
|------|--------|----------------|----------|-------------|-------------|---------|
"""
        
        # Ajouter les 10 premiers numéros
        for _, row in top25_df.head(10).iterrows():
            statut_icons = []
            if row['is_hot_number']:
                statut_icons.append('🔥 Chaud')
            if row['is_cold_number']:
                statut_icons.append('❄️ Froid')
            if row['is_excluded']:
                statut_icons.append('🚫 Exclu')
            
            statut = ' '.join(statut_icons) if statut_icons else '✅ Normal'
            
            summary_content += f"| {int(row['position_finale'])} | **{int(row['numero'])}** | {row['score_composite']:.4f} | {row['score_ml']:.4f} | {row['score_frequence']:.4f} | {int(row['frequence_historique'])} | {statut} |\n"
        
        summary_content += f"""
## 📊 Analyse Statistique du Top 25

### Répartition par Catégorie
- **Numéros normaux**: {25 - sum([top25_df['is_hot_number'].sum(), top25_df['is_cold_number'].sum(), top25_df['is_excluded'].sum()])}
- **Numéros chauds (🔥)**: {int(top25_df['is_hot_number'].sum())}
- **Numéros froids (❄️)**: {int(top25_df['is_cold_number'].sum())}
- **Numéros exclus (🚫)**: {int(top25_df['is_excluded'].sum())}

### Scores Moyens
- **Score composite moyen**: {top25_df['score_composite'].mean():.6f}
- **Score ML moyen**: {top25_df['score_ml'].mean():.6f}
- **Score fréquence moyen**: {top25_df['score_frequence'].mean():.6f}
- **Fréquence historique moyenne**: {top25_df['frequence_historique'].mean():.1f}

### Distribution des Rangs
- **Meilleur rang ML**: {int(top25_df['rang_ml'].min())}
- **Moins bon rang ML**: {int(top25_df['rang_ml'].max())}
- **Rang ML médian**: {int(top25_df['rang_ml'].median())}

## 🎲 Top 25 Complet

| Pos | Numéro | Score Comp | ML | Freq | Rang ML | Rang Freq | Freq Hist | Statut |
|-----|--------|------------|----|----- |---------|-----------|-----------|--------|
"""
        
        # Ajouter tous les numéros du top 25
        for _, row in top25_df.iterrows():
            statut_short = []
            if row['is_hot_number']:
                statut_short.append('🔥')
            if row['is_cold_number']:
                statut_short.append('❄️')
            if row['is_excluded']:
                statut_short.append('🚫')
            
            statut = ''.join(statut_short) if statut_short else '✅'
            
            summary_content += f"| {int(row['position_finale'])} | {int(row['numero'])} | {row['score_composite']:.4f} | {row['score_ml']:.4f} | {row['score_frequence']:.4f} | {int(row['rang_ml'])} | {int(row['rang_frequence'])} | {int(row['frequence_historique'])} | {statut} |\n"
        
        summary_content += f"""
## 📈 Recommandations d'Usage

### Pour la Sélection de Grilles
1. **Priorité haute**: Numéros positions 1-5 (scores composites les plus élevés)
2. **Priorité moyenne**: Numéros positions 6-15 
3. **Priorité faible**: Numéros positions 16-25

### Stratégie Conseillée
- Utiliser 2-3 numéros du top 5
- Compléter avec 1-2 numéros du top 15
- Éviter de prendre plus de 4 numéros du top 10 (diversification)
- Tenir compte des numéros exclus (🚫) selon votre stratégie

### Notes Importantes
- Les scores ML reflètent les patterns appris sur l'historique des tirages
- La fréquence historique indique la popularité passée du numéro
- Le score composite combine intelligemment ML et fréquence selon les performances
- Les numéros exclus sont basés sur les paramètres utilisateur uniquement

---
*Fichiers détaillés disponibles: `ml_top25_predictions.csv` et `ml_full_predictions.csv`*
"""
        
        # Sauvegarder le rapport
        summary_path = output_dir / 'ml_predictions_summary.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"   📋 Résumé des prédictions ML sauvegardé: '{summary_path}'")
        
    except Exception as e:
        print(f"   ⚠️ Erreur création résumé ML: {e}")

# Ajouter aussi les fonctions de visualisation avancées manquantes
def plot_autocorrelation(series: pd.Series, output_dir: Path, max_lags: int = 50):
    """Trace l'autocorrélation de la série"""
    try:
        from statsmodels.tsa.stattools import acf
        
        plt.figure(figsize=(12, 6))
        autocorr = acf(series, nlags=max_lags, fft=True)
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(autocorr)), autocorr, 'b-', alpha=0.8)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Seuil 5%')
        plt.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
        plt.title('Fonction d\'Autocorrélation')
        plt.xlabel('Décalage')
        plt.ylabel('Autocorrélation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Corrélogramme en barres
        plt.subplot(1, 2, 2)
        plt.bar(range(len(autocorr)), autocorr, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Corrélogramme')
        plt.xlabel('Décalage')
        plt.ylabel('Autocorrélation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'autocorrelation_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Erreur création autocorrélation: {e}")

def plot_fft(series: pd.Series, output_dir: Path):
    """Trace le spectre FFT de la série"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Calcul FFT
        fft_values = fft(series.values)
        freqs = fftfreq(len(series), d=1)
        
        # Ne garder que les fréquences positives
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_values[:len(fft_values)//2])
        
        plt.subplot(1, 2, 1)
        plt.plot(positive_freqs, positive_fft, 'b-', alpha=0.8)
        plt.title('Spectre FFT - Amplitude')
        plt.xlabel('Fréquence')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Spectre de puissance (log)
        plt.subplot(1, 2, 2)
        power_spectrum = positive_fft**2
        plt.semilogy(positive_freqs, power_spectrum, 'r-', alpha=0.8)
        plt.title('Spectre de Puissance (échelle log)')
        plt.xlabel('Fréquence')
        plt.ylabel('Puissance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fft_spectrum_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Erreur création FFT: {e}")

def plot_seasonal_decomposition(series: pd.Series, output_dir: Path, period: int = 10):
    """Décomposition saisonnière STL"""
    try:
        if len(series) < 2 * period:
            print(f"⚠️ Série trop courte pour décomposition (min: {2*period}, actuel: {len(series)})")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Décomposition STL
        stl = STL(series, seasonal=period)
        result = stl.fit()
        
        # 4 sous-graphiques
        plt.subplot(4, 1, 1)
        plt.plot(result.observed, label='Série Originale', color='black')
        plt.title('Décomposition Saisonnière STL')
        plt.ylabel('Observé')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 2)
        plt.plot(result.trend, label='Tendance', color='blue')
        plt.ylabel('Tendance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 3)
        plt.plot(result.seasonal, label='Saisonnier', color='green')
        plt.ylabel('Saisonnier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 4)
        plt.plot(result.resid, label='Résidus', color='red')
        plt.ylabel('Résidus')
        plt.xlabel('Index')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stl_decomposition_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Erreur décomposition STL: {e}")

def plot_matrix_profile(series: pd.Series, output_dir: Path, window: int = 10):
    """Matrix Profile pour détecter les motifs récurrents"""
    try:
        if len(series) < 2 * window:
            print(f"⚠️ Série trop courte pour Matrix Profile (min: {2*window}, actuel: {len(series)})")
            return
            
        plt.figure(figsize=(15, 8))
        
        # Calcul du Matrix Profile
        matrix_profile = stumpy.stump(series.values, m=window)
        
        plt.subplot(2, 1, 1)
        plt.plot(series.values, color='black', alpha=0.7)
        plt.title(f'Série Temporelle (fenêtre={window})')
        plt.ylabel('Valeur')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(matrix_profile[:, 0], color='red', alpha=0.8)
        plt.title('Matrix Profile - Distance aux motifs les plus proches')
        plt.ylabel('Distance')
        plt.xlabel('Index')
        plt.grid(True, alpha=0.3)
        
        # Marquer les anomalies (pics élevés)
        threshold = np.percentile(matrix_profile[:, 0], 95)
        anomalies = np.where(matrix_profile[:, 0] > threshold)[0]
        if len(anomalies) > 0:
            plt.scatter(anomalies, matrix_profile[anomalies, 0], 
                       color='orange', s=50, zorder=5, alpha=0.8, 
                       label=f'Anomalies (>P95)')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'matrix_profile_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Erreur Matrix Profile: {e}")

# --- Simulation Parallèle ---
def simulate_chunk(args_tuple):
    n_sims_chunk, criteria, X_last_shared, models, chunk_seed = args_tuple
    random.seed(chunk_seed)
    np.random.seed(chunk_seed)
    results = []
    for _ in range(n_sims_chunk):
        grid = generate_grid_vectorized(criteria, models, X_last_shared)
        if grid and len(grid) >= 5:
            # Pas de facteur de diversité dans le scoring final pour garder la précision
            score = score_grid(np.array(grid[:-1]), criteria, diversity_factor=0.0)
            results.append({'grid': grid, 'score': score})
    return results

# Fonction de scoring à définir avant simulate_chunk_adaptive
def score_grid(grid, criteria, diversity_factor=0.0):
    """Score une grille selon les critères statistiques avec poids dynamiques optimisés"""
    try:
        if len(grid) != 5:
            return 0.0
        
        # Récupération des données nécessaires avec gestion des erreurs
        dynamic_weights = criteria.get('dynamic_weights', {})
        if not dynamic_weights:
            # Poids par défaut si les poids dynamiques ne sont pas disponibles
            dynamic_weights = {
                'sum': 0.15,
                'std': 0.15,
                'pair_impair': 0.15,
                'decades_entropy': 0.15,
                'gaps': 0.10,
                'high_low': 0.10,
                'prime_ratio': 0.10,
                'position_variance': 0.10
            }
        
        # 1. Score basé sur la somme
        grid_sum = float(np.sum(grid))
        target_sum_mean = criteria.get('sums', pd.Series([125])).mean()  # Fallback si pas disponible
        target_sum_std = criteria.get('sums', pd.Series([125])).std() if len(criteria.get('sums', [])) > 1 else 20
        sum_score = 1.0 - min(1.0, abs(grid_sum - target_sum_mean) / (2 * max(1, target_sum_std)))
        
        # 2. Score basé sur l'écart-type
        grid_std = float(np.std(grid))
        target_std_mean = criteria.get('stds', pd.Series([15])).mean()  # Fallback si pas disponible
        target_std_std = criteria.get('stds', pd.Series([15])).std() if len(criteria.get('stds', [])) > 1 else 5
        std_score = 1.0 - min(1.0, abs(grid_std - target_std_mean) / (2 * max(1, target_std_std)))
        
        # 3. Score pair/impair
        even_count = int(np.sum(grid % 2 == 0))
        pair_impair_probs = criteria.get('pair_impair_probs', pd.Series([0.2]*6))
        if even_count in pair_impair_probs.index:
            pair_impair_score = float(pair_impair_probs[even_count])
        else:
            pair_impair_score = 0.1
        
        # 4. Score entropie des décennies
        decades = [(n-1)//10 for n in grid]
        decades_counts = pd.Series(decades).value_counts()
        if len(decades_counts) > 0:
            decades_entropy = -sum([(count/5) * np.log(count/5) for count in decades_counts.values if count > 0])
            # Normaliser l'entropie (max = log(5) ≈ 1.6)
            decades_entropy_score = min(1.0, decades_entropy / np.log(5))
        else:
            decades_entropy_score = 0.5
        
        # 5. Score espacement (gaps)
        sorted_grid = np.sort(grid)
        gaps = [sorted_grid[i+1] - sorted_grid[i] for i in range(4)]
        gaps_std = np.std(gaps)
        
        # Fallback pour gaps si pas disponible dans criteria
        target_gaps = criteria.get('gaps_between_numbers', [10])
        if hasattr(target_gaps, '__iter__') and len(target_gaps) > 0:
            gaps_mean = np.mean(target_gaps)
        else:
            gaps_mean = 10  # Valeur par défaut
        
        gaps_score = 1.0 - min(1.0, abs(gaps_std - gaps_mean) / max(1.0, gaps_mean))
        
        # 6. Score haute/basse répartition
        low_count = sum(1 for n in grid if n <= 25)
        high_low_ratio = low_count / 5.0
        
        # Fallback pour high_low_ratio si pas disponible dans criteria
        target_high_low = criteria.get('high_low_ratio', [0.5])
        if hasattr(target_high_low, '__iter__') and len(target_high_low) > 0:
            high_low_mean = np.mean(target_high_low)
        else:
            high_low_mean = 0.5  # Valeur par défaut
        
        high_low_score = 1.0 - min(1.0, abs(high_low_ratio - high_low_mean) / 0.5)
        
        # 7. Score nombres premiers
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
        prime_count = sum(1 for n in grid if n in primes)
        prime_ratio = prime_count / 5.0
        
        # Fallback pour prime_ratio si pas disponible dans criteria
        target_prime = criteria.get('prime_ratio', [0.3])
        if hasattr(target_prime, '__iter__') and len(target_prime) > 0:
            prime_mean = np.mean(target_prime)
        else:
            prime_mean = 0.3  # Valeur par défaut
        
        prime_score = 1.0 - min(1.0, abs(prime_ratio - prime_mean) / 0.5)
        
        # 8. Score variance des positions
        positions = [(n-1)/48.0 for n in grid]
        position_var = np.var(positions)
        
        # Fallback pour position_variance si pas disponible dans criteria
        target_position_var = criteria.get('position_variance', [0.2])
        if hasattr(target_position_var, '__iter__') and len(target_position_var) > 0:
            position_var_mean = np.mean(target_position_var)
        else:
            position_var_mean = 0.2  # Valeur par défaut
        
        position_variance_score = 1.0 - min(1.0, abs(position_var - position_var_mean) / max(0.1, position_var_mean))
        
        # Score composite avec poids dynamiques
        composite_score = (
            dynamic_weights.get('sum', 0.15) * sum_score +
            dynamic_weights.get('std', 0.15) * std_score +
            dynamic_weights.get('pair_impair', 0.15) * pair_impair_score +
            dynamic_weights.get('decades_entropy', 0.15) * decades_entropy_score +
            dynamic_weights.get('gaps', 0.10) * gaps_score +
            dynamic_weights.get('high_low', 0.10) * high_low_score +
            dynamic_weights.get('prime_ratio', 0.10) * prime_score +
            dynamic_weights.get('position_variance', 0.10) * position_variance_score
        )
        
        # Bonus de diversité (optionnel)
        diversity_bonus = 0.0
        if diversity_factor > 0:
            # Simple diversité basée sur la variance des numéros
            diversity_bonus = diversity_factor * (np.var(grid) / 200.0)  # Normalisation approximative
        
        # Score de fréquence (bonus pour les numéros avec de bonnes fréquences)
        freq = criteria.get('freq', pd.Series())
        if not freq.empty:
            freq_bonus = 0.0
            for num in grid:
                if num in freq.index:
                    # Normaliser la fréquence entre 0 et 1
                    normalized_freq = freq[num] / freq.max() if freq.max() > 0 else 0
                    freq_bonus += normalized_freq
            freq_bonus = freq_bonus / len(grid)  # Moyenne
            composite_score += 0.05 * freq_bonus  # Petit bonus fréquence
        
        final_score = composite_score + diversity_bonus
        return max(0.0, min(1.0, final_score))  # Clamp entre 0 et 1
        
    except Exception as e:
        print(f"⚠️ Erreur dans score_grid pour grille {grid}: {e}")
        return 0.0

# Ajouter aussi une fonction de génération vectorisée standard qui manque
def generate_grid_vectorized(criteria: dict, models: dict, X_last: np.ndarray) -> list:
    """Génération de grille vectorisée standard (fallback vers adaptative)"""
    return generate_grid_vectorized_adaptive(criteria, models, X_last)

# --- Simulation Parallèle ---
def simulate_chunk_adaptive(args_tuple):
    """Fonction de simulation adaptative pour le multiprocessing"""
    n_sims_chunk, criteria, X_last_shared, models, chunk_seed = args_tuple
    random.seed(chunk_seed)
    np.random.seed(chunk_seed)
    results = []
    for _ in range(n_sims_chunk):
        grid = generate_grid_vectorized_adaptive(criteria, models, X_last_shared)
        if grid and len(grid) >= 5:
            score = score_grid(np.array(grid[:-1]), criteria, diversity_factor=0.0)
            results.append({'grid': grid, 'score': score})
    return results

def simulate_grids_parallel(n_simulations: int, criteria: dict, X_last: np.ndarray, models: dict):
    chunk_size = max(1, n_simulations // (N_CORES * 4))
    chunks_args = []
    sims_left, i = n_simulations, 0
    
    while sims_left > 0:
        size = min(chunk_size, sims_left)
        chunks_args.append((size, criteria, X_last, models, GLOBAL_SEED + i))
        sims_left -= size
        i += 1

    all_results = []
    print(f"Simulation de {n_simulations} grilles sur {N_CORES} coeurs...")
    print(f"Nombre de chunks: {len(chunks_args)}")
    
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(simulate_chunk, args) for args in chunks_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulation des grilles"):
            try:
                chunk_result = future.result()
                print(f"Chunk terminé avec {len(chunk_result)} grilles")
                all_results.extend(chunk_result)
            except Exception as e:
                print(f"❌ ERREUR dans un chunk de simulation: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"Total grilles générées: {len(all_results)}")
    return sorted(all_results, key=lambda x: x['score'], reverse=True)

def analyze_generated_grids(grids: list, criteria: dict):
    print("\n" + "="*50 + "\n🔬 Contrôle Qualité des Grilles Générées\n" + "="*50)
    if not grids:
        print("Aucune grille à analyser.")
        return
    
    grids_df = pd.DataFrame([g['grid'][:-1] for g in grids])
    sums = grids_df.sum(axis=1)
    print(f"Somme des boules :")
    print(f"  - Générée : Moy={sums.mean():.1f}, Std={sums.std():.1f}")
    print(f"  - Cible    : Moy={criteria['sums'].mean():.1f}, Std={criteria['sums'].std():.1f}")
    
    evens = (grids_df % 2 == 0).sum(axis=1)
    print("\nRépartition Pair/Impair:")
    print("  - Générée:")
    print(evens.value_counts(normalize=True).sort_index().to_string())

# --- Fonctions de Visualisation ---
def plot_frequencies_with_filter(criteria: dict, output_dir: Path):
    """Visualisation des fréquences avec mise en évidence des numéros filtrés"""
    plt.figure(figsize=(16, 10))
    
    # Préparer les données
    freq = criteria['freq']
    freq_excluded = criteria.get('freq_excluded', set())
    hot_numbers = set(criteria['hot_numbers'])
    cold_numbers = set(criteria['cold_numbers'])
    
    # Créer les couleurs selon les catégories
    colors = []
    labels = []
    for num in freq.index:
        if num in freq_excluded:
            colors.append('red')
            labels.append('Exclu (fréq < 80 ou > 100)')
        elif num in hot_numbers:
            colors.append('orange')
            labels.append('Numéro chaud')
        elif num in cold_numbers:
            colors.append('lightblue')
            labels.append('Numéro froid')
        else:
            colors.append('lightgreen')
            labels.append('Numéro normal')
    
    # Graphique principal
    plt.subplot(2, 2, 1)
    bars = plt.bar(freq.index, freq.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title("Fréquences des numéros avec filtre appliqué", fontsize=14, fontweight='bold')
    plt.xlabel('Numéro')
    plt.ylabel('Fréquence')
    plt.grid(True, alpha=0.3)
    
    # Ajouter les lignes de seuil
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Seuil min (80)')
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Seuil max (100)')
    
    # Légende personnalisée
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label=f'Exclus par fréquence ({len(freq_excluded)})'),
        Patch(facecolor='orange', alpha=0.8, label=f'Numéros chauds ({len(hot_numbers)})'),
        Patch(facecolor='lightblue', alpha=0.8, label=f'Numéros froids ({len(cold_numbers)})'),
        Patch(facecolor='lightgreen', alpha=0.8, label='Numéros normaux')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Histogramme des fréquences
    plt.subplot(2, 2, 2)
    plt.hist(freq.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='Seuil min (80)')
    plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Seuil max (100)')
    plt.title("Distribution des fréquences", fontsize=14)
    plt.xlabel('Fréquence')
    plt.ylabel('Nombre de numéros')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparaison avant/après filtre
    plt.subplot(2, 2, 3)
    available_nums = [n for n in freq.index if n not in freq_excluded]
    excluded_nums = list(freq_excluded);
    
    data_comparison = [
        len(available_nums),
        len(excluded_nums)
    ]
    labels_comparison = [f'Disponibles\n({len(available_nums)})', f'Exclus\n({len(excluded_nums)})']
    colors_comparison = ['lightgreen', 'red']
    
    wedges, texts, autotexts = plt.pie(data_comparison, labels=labels_comparison, 
                                      colors=colors_comparison, autopct='%1.1f%%', 
                                      startangle=90)
    plt.title("Répartition après filtre de fréquence", fontsize=14)
    
    # Tableau des numéros exclus
    plt.subplot(2, 2, 4)
    plt.axis('off')
    if freq_excluded:
        excluded_with_freq = [(num, freq[num]) for num in sorted(freq_excluded)]
        table_data = []
        for i in range(0, len(excluded_with_freq), 3):  # Grouper par 3
            row = []
            for j in range(3):
                if i + j < len(excluded_with_freq):
                    num, f = excluded_with_freq[i + j]
                    row.extend([f"N°{num}", f"{f}"])
                else:
                    row.extend(["", ""])
            table_data.append(row)
        
        table = plt.table(cellText=table_data,
                         colLabels=['Numéro', 'Fréq', 'Numéro', 'Fréq', 'Numéro', 'Fréq'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title("Numéros exclus par le filtre de fréquence", fontsize=14, pad=20)
    else:
        plt.text(0.5, 0.5, "Aucun numéro exclu\npar le filtre de fréquence", 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        plt.title("Numéros exclus par le filtre de fréquence", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequencies_with_filter.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_frequency_analysis(criteria: dict, output_dir: Path):
    plt.figure(figsize=(15, 7))
    sns.barplot(x=criteria['freq'].index, y=criteria['freq'].values, palette="viridis")
    plt.title("Fréquence de sortie des numéros", fontsize=16)
    plt.xlabel('Numéro')
    plt.ylabel('Fréquence')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'frequence_numeros.png', dpi=150)
    plt.close()

def plot_gap_analysis(criteria: dict, output_dir: Path):
    plt.figure(figsize=(15, 7))
    analysis_df = criteria['numbers_analysis']
    sns.barplot(x='Numero', y='Ecart_Moyen_Tirages', data=analysis_df.sort_values('Numero'), palette='YlOrRd')
    plt.title("Périodicité Moyenne par Numéro", fontsize=16)
    plt.xlabel('Numéro')
    plt.ylabel('Écart moyen entre tirages')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'periodicite_plot.png', dpi=150)
    plt.close()

def plot_odd_even_analysis(criteria: dict, output_dir: Path):
    pair_impair_probs = criteria['pair_impair_probs']
    plt.figure(figsize=(10, 5))
    sns.barplot(x=pair_impair_probs.index, y=pair_impair_probs.values, palette="coolwarm")
    plt.title("Répartition des numéros pairs et impairs", fontsize=16)
    plt.xlabel('Nombre de numéros pairs')
    plt.ylabel('Probabilité')
    plt.tight_layout()
    plt.savefig(output_dir / 'pair_impair_plot.png', dpi=150)
    plt.close()

def plot_consecutive_numbers_analysis(grids_df: pd.DataFrame, output_dir: Path):
    consecutive_counts = []
    for _, row in grids_df.iterrows():
        grid = np.array([row['boule_1'], row['boule_2'], row['boule_3'], row['boule_4'], row['boule_5']], dtype=np.int32)
        consecutive_counts.append(_count_consecutive_numba(grid))
    
    plt.figure(figsize=(10, 5))
    sns.countplot(x=consecutive_counts, palette="mako")
    plt.title("Fréquence des numéros consécutifs", fontsize=16)
    plt.xlabel('Nombre de paires consécutives')
    plt.ylabel('Fréquence')
    plt.tight_layout()
    plt.savefig(output_dir / 'consecutive_numbers_plot.png', dpi=150)
    plt.close()

def create_visualizations(criteria: dict, grids_df: pd.DataFrame, output_dir: Path):
    print("\nCréation des visualisations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Nouvelle visualisation des fréquences avec filtre
    plot_frequencies_with_filter(criteria, output_dir)
    
    plot_frequency_analysis(criteria, output_dir)
    plot_gap_analysis(criteria, output_dir)
    plot_odd_even_analysis(criteria, output_dir)
    plot_consecutive_numbers_analysis(grids_df, output_dir)

    # Ajout des visualisations de cycles et motifs
    sums_series = criteria['sums']
    plot_autocorrelation(sums_series, output_dir)
    plot_fft(sums_series, output_dir)
    plot_seasonal_decomposition(sums_series, output_dir, period=10)
    plot_matrix_profile(sums_series, output_dir, window=10)

    print(f"   ✓ Visualisations sauvegardées dans '{output_dir}'.")

# --- Génération de Rapports ---
def create_report(criteria: dict, best_grids: list, output_dir: Path, exec_time: float, adaptive_strategy=None):
    print("Génération du rapport Markdown...")
    
    top_5_grids_str = "\n".join([
        f"- **Grille {i+1}**: `{g['grid'][:-1]}` + Chance **{g['grid'][-1]}** (Score: {g['score']:.4f})" 
        for i, g in enumerate(best_grids[:5])
    ])
    
    hot_numbers_str = ", ".join(map(str, criteria['hot_numbers']))
    last_draw_str = ", ".join(map(str, criteria['last_draw']))
    
    top_pairs_str = "N/A"
    if not criteria['pair_counts'].empty:
        top_pairs_str = "\n".join([
            f"- `{int(p[0])}-{int(p[1])}`: {c} fois" 
            for (p, c) in criteria['pair_counts'].head(5).items()
        ])
    
    # MODIFICATION : Section exclusion adaptée
    excluded_numbers = criteria.get('numbers_to_exclude', set())
    if excluded_numbers:
        excluded_numbers_str = ", ".join(map(str, sorted(excluded_numbers)))
        exclusion_section = f"""
## 🚫 Numéros Exclus (Manuellement)
- **Numéros exclus des grilles générées** : `{excluded_numbers_str}`
"""
    else:
        exclusion_section = """
## ✅ Aucune Exclusion
- **Tous les 49 numéros sont utilisables** pour générer les grilles
"""

    # Vérifier si le fichier top 25 ML existe
    ml_top25_section = ""
    ml_top25_path = output_dir / 'ml_top25_predictions.csv'
    if ml_top25_path.exists():
        try:
            ml_df = pd.read_csv(ml_top25_path)
            top_10_ml = ml_df.head(10)
            top_10_ml_str = "\n".join([
                f"- **{int(row['numero'])}** (Score: {row['score_composite']:.4f}, ML: {row['score_ml']:.4f}, Rang ML: {int(row['rang_ml'])})" 
                for _, row in top_10_ml.iterrows()
            ])
            
            ml_top25_section = f"""
## 🤖 Top 10 des Numéros selon l'Intelligence Artificielle
{top_10_ml_str}

*📊 Fichier complet des 25 meilleurs disponible : `ml_top25_predictions.csv`*
*📋 Analyse détaillée disponible : `ml_predictions_summary.md`*
"""
        except Exception as e:
            ml_top25_section = f"\n## 🤖 Prédictions ML\n*Erreur lors du chargement: {e}*\n"
    else:
        ml_top25_section = "\n## 🤖 Prédictions ML\n*Non disponible - modèles ML non chargés*\n"

    # Informations sur la stratégie adaptative
    adaptive_info = ""
    if adaptive_strategy:
        history = adaptive_strategy.performance_history
        if history and 'ml_scores' in history and len(history['ml_scores']) > 0:
            recent_count = min(5, len(history['ml_scores']))
            avg_ml_acc = np.mean(history['ml_scores'][-recent_count:])  # Moyenne sur les dernières évaluations
            avg_freq_acc = np.mean(history['freq_scores'][-recent_count:])
            current_ml_weight = adaptive_strategy.ml_weight
            current_freq_weight = 1.0 - adaptive_strategy.ml_weight
            last_date = history['dates'][-1] if 'dates' in history and history['dates'] else "N/A"
            
            adaptive_info = f"""
## 🤖 Stratégie Adaptative Machine Learning
- **Poids ML actuel**: `{current_ml_weight:.3f}` ({current_ml_weight*100:.1f}%)
- **Poids Fréquence actuel**: `{current_freq_weight:.3f}` ({current_freq_weight*100:.1f}%)
- **Précision ML récente**: `{avg_ml_acc:.3f}` (moyenne {recent_count} dernières évaluations)
- **Précision Fréquence récente**: `{avg_freq_acc:.3f}` (moyenne {recent_count} dernières évaluations)
- **Évaluations effectuées**: `{len(history['ml_scores'])}`
- **Dernière évaluation**: `{last_date[:19] if last_date != "N/A" else "N/A"}`
"""
        else:
            adaptive_info = f"""
## 🤖 Stratégie Adaptative Machine Learning
- **Poids ML initial**: `{adaptive_strategy.ml_weight:.3f}` ({adaptive_strategy.ml_weight*100:.1f}%)
- **Poids Fréquence initial**: `{1.0 - adaptive_strategy.ml_weight:.3f}` ({(1.0 - adaptive_strategy.ml_weight)*100:.1f}%)
- **État**: Première utilisation - collecte des données de performance en cours
"""    

    report_content = f"""# Rapport d'Analyse Loto - {datetime.now().strftime('%d/%m/%Y %H:%M')}

## ⚙️ Configuration d'Exécution
- **Temps d'exécution**: `{exec_time:.2f} secondes`
- **Simulations**: `{N_SIMULATIONS:,}`
- **Seed reproductibilité**: `{GLOBAL_SEED}`
- **Processeurs utilisés**: `{N_CORES}`

## 🎯 Top 5 Grilles Recommandées
{top_5_grids_str}

{ml_top25_section}

{exclusion_section}

*📁 Toutes les grilles sont disponibles dans `grilles_conseillees.csv`.*

## 📊 Résumé de l'Analyse Statistique
- **Dernier tirage**: `{last_draw_str}`
- **Numéros Hot (Top 10)**: `{hot_numbers_str}`
- **Poids de scoring dynamiques**: `{dict((k, round(v, 3)) for k, v in criteria['dynamic_weights'].items())}`

### 🔗 Top 5 Paires Fréquentes
{top_pairs_str}
{adaptive_info}

## 📈 Fichiers Générés
- **Analyse détaillée**: `numbers_analysis.csv`
- **Grilles complètes**: `grilles_conseillees.csv`
- **Top 25 ML**: `ml_top25_predictions.csv`
- **Prédictions complètes ML**: `ml_full_predictions.csv`
- **Résumé ML**: `ml_predictions_summary.md`
- **Graphiques**: 
  - `frequence_numeros.png`
  - `periodicite_plot.png`
  - `pair_impair_plot.png`
  - `consecutive_numbers_plot.png`
  - `autocorrelation_plot.png`
  - `fft_spectrum_plot.png`
  - `stl_decomposition_plot.png`
  - `matrix_profile_plot.png`

## 📈 Visualisations Clés
![Fréquence des numéros](frequence_numeros.png)
![Autocorrélation des tirages](autocorrelation_plot.png)
![Spectre FFT](fft_spectrum_plot.png)
![Décomposition saisonnière STL](stl_decomposition_plot.png)
![Matrix Profile](matrix_profile_plot.png)

---
*Rapport généré automatiquement par le système d'analyse Loto*
"""
    report_path = output_dir / 'rapport_analyse.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"   ✓ Rapport sauvegardé dans '{report_path}'.")
    
def main():
    start_time = datetime.now()
    global adaptive_strategy
    
    if not parquet_path.exists():
        print(f"❌ ERREUR: Fichier Parquet '{parquet_path}' non trouvé.")
        return

    # Vérifier si un ré-entraînement automatique est nécessaire
    auto_retrain = adaptive_strategy.should_retrain_models() if adaptive_strategy else False
    force_retrain = (hasattr(ARGS, 'retrain') and ARGS.retrain) or auto_retrain
    
    if force_retrain:
        print("🔄 RE-TRAINING DÉCLENCHÉ:")
        if auto_retrain:
            print("  📊 Ré-entraînement automatique basé sur les performances")
        if hasattr(ARGS, 'retrain') and ARGS.retrain:
            print("  🔧 Ré-entraînement forcé par l'utilisateur")
        
        import shutil
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(exist_ok=True)
        print("   ✓ Anciens modèles supprimés")

    # Chargement et analyse des données
    criteria = None
    cache_key = f"loto_criteria:{parquet_path.stat().st_mtime}"
    
    if redis_client:
        cached_criteria = redis_client.get(cache_key)
        if cached_criteria:
            print("CACHE HIT: Chargement des critères depuis Redis...")
            criteria = pickle.loads(cached_criteria)
        else:
            print("CACHE MISS: Critères non trouvés dans Redis.")

    if criteria is None:
        con = None
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            print(f"1. Chargement et analyse des données depuis '{parquet_path}'...")
            con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_parquet('{str(parquet_path)}')")
            criteria = analyze_criteria_duckdb(con, 'loto_draws')
            
            # Mise à jour de la stratégie adaptative
            update_adaptive_strategy_with_recent_draws(con, adaptive_strategy)
            
            if redis_client:
                print("Sauvegarde des nouveaux critères dans Redis (expiration: 4h)...")
                ttl_seconds = int(timedelta(hours=4).total_seconds())
                redis_client.setex(cache_key, ttl_seconds, pickle.dumps(criteria))
        except Exception as e:
            print(f"❌ ERREUR lors du chargement des données: {e}")
            import traceback
            traceback.print_exc()
            return
        finally:
            if con:
                con.close()

    # Export des analyses
    analysis_csv_path = OUTPUT_DIR / 'numbers_analysis.csv'
    criteria['numbers_analysis'].to_csv(analysis_csv_path, index=False, float_format='%.2f')
    print(f"   ✓ Analyse détaillée exportée vers '{analysis_csv_path}'")

    frequencies_csv_path = OUTPUT_DIR / 'frequencies_analysis.csv'
    freq_df = pd.DataFrame({
        'numero': criteria['freq'].index,
        'frequence': criteria['freq'].values,
        'frequence_pourcentage': (criteria['freq'].values / criteria['freq'].sum() * 100),
        'is_hot': criteria['freq'].index.isin(criteria['hot_numbers']),
        'is_cold': criteria['freq'].index.isin(criteria['cold_numbers']),
        'is_excluded_by_frequency': criteria['freq'].index.isin(criteria.get('freq_excluded', set()))
    })
    freq_df = freq_df.sort_values('frequence', ascending=False)
    freq_df.to_csv(frequencies_csv_path, index=False, float_format='%.2f')
    print(f"   ✓ Analyse des fréquences exportée vers '{frequencies_csv_path}'")

    print("\n2. Gestion intelligente des modèles XGBoost...")
    
    if force_retrain:
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_parquet('{str(parquet_path)}')")
        df_full = con.table('loto_draws').fetchdf()
        con.close()
        models = train_xgboost_parallel_adaptive(df_full, adaptive_strategy)
    else:
        models = load_saved_models()
        
        # Vérification de la qualité des modèles chargés
        ball_models_available = 1 if models.get('balls_multilabel') is not None else 0
        chance_models_available = 1 if models.get('chance_multilabel') is not None else 0
        total_available = ball_models_available + chance_models_available
        
        if total_available < 2:
            print(f"  ⚠️ Modèles insuffisants ({total_available}/2), ré-entraînement...")
            con = duckdb.connect(database=':memory:', read_only=False)
            con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_parquet('{str(parquet_path)}')")
            df_full = con.table('loto_draws').fetchdf()
            con.close()
            models = train_xgboost_parallel_adaptive(df_full, adaptive_strategy)
        else:
            print(f"  ✅ {total_available}/2 modèles disponibles et prêts")

    print("\n3. Simulation intelligente adaptative des grilles...")
    X_last = np.array(criteria['last_draw']).reshape(1, -1)
    
    # Simulation parallèle adaptative
    chunk_size = max(1, N_SIMULATIONS // (N_CORES * 4))
    chunks_args = []
    sims_left, i = N_SIMULATIONS, 0
    
    while sims_left > 0:
        size = min(chunk_size, sims_left)
        chunks_args.append((size, criteria, X_last, models, GLOBAL_SEED + i))
        sims_left -= size
        i += 1

    all_results = []
    print(f"Simulation adaptative de {N_SIMULATIONS} grilles sur {N_CORES} coeurs...")
    print(f"Nombre de chunks: {len(chunks_args)}")
    
   
    
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(simulate_chunk_adaptive, args) for args in chunks_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulation adaptative"):
            try:
                chunk_result = future.result()
                all_results.extend(chunk_result)
            except Exception as e:
                print(f"❌ ERREUR dans un chunk adaptatif: {e}")
                import traceback
                traceback.print_exc()
    
    grids = sorted(all_results, key=lambda x: x['score'], reverse=True)
    print(f"Total grilles générées: {len(grids)}")

    # Sauvegarde et analyse
    grids_df = pd.DataFrame()
    if grids:
        grids_df = pd.DataFrame(grids)
        grid_cols = grids_df['grid'].apply(pd.Series)
        grid_cols.columns = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']
        grids_df = pd.concat([grid_cols, grids_df['score']], axis=1)
        
        grids_csv_path = OUTPUT_DIR / 'grilles_conseillees.csv'
        grids_df.to_csv(grids_csv_path, index=False, float_format='%.4f')
        print(f"   ✓ {len(grids)} grilles sauvegardées dans '{grids_csv_path}'")

    analyze_generated_grids(grids, criteria)

    # *** NOUVEAU: Génération du top 25 ML ***
    print("\n🤖 Génération des prédictions ML (Top 25)...")
    top25_ml_df = generate_ml_top25_predictions(models, X_last, criteria, OUTPUT_DIR)
    if top25_ml_df is not None:
        create_ml_predictions_summary(top25_ml_df, criteria, OUTPUT_DIR)

    print("\n4. Finalisation des rapports et visualisations...")
    if not grids_df.empty:
        create_visualizations(criteria, grids_df, OUTPUT_DIR)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    create_report(criteria, grids, OUTPUT_DIR, execution_time, adaptive_strategy)

    print(f"\n✅ Analyse adaptative terminée avec succès en {execution_time:.2f} secondes")
    print(f"📁 Tous les résultats sont disponibles dans : '{OUTPUT_DIR.resolve()}'")
    
    if grids:
        print("\n🎯 Top 5 des grilles recommandées (optimisation adaptative) :")
        for i, grid_info in enumerate(grids[:5]):
            balls_str = ', '.join(map(str, grid_info['grid'][:-1]))
            chance = grid_info['grid'][-1]
            score = grid_info['score']
            print(f"   {i+1}. Boules: [{balls_str}] | Chance: {chance} | Score: {score:.4f}")

        # Affichage des statistiques adaptatives
        print(f"\n📊 Statistiques de génération adaptative :")
        print(f"   - Grilles simulées : {N_SIMULATIONS:,}")
        print(f"   - Score moyen : {np.mean([g['score'] for g in grids]):.4f}")
        print(f"   - Meilleur score : {grids[0]['score']:.4f}")
        
        # Informations sur la stratégie adaptative
        if adaptive_strategy:
            summary = adaptive_strategy.get_performance_summary()
            if isinstance(summary, dict):
                print(f"\n🤖 Stratégie Adaptative - Résumé des performances :")
                print(f"   - Poids ML actuel : {summary['current_ml_weight']:.3f} ({summary['current_ml_weight']*100:.1f}%)")
                print(f"   - Confiance : {summary['confidence']:.3f}")
                print(f"   - Tendance : {summary['trend']:+.4f}")
                print(f"   - Stabilité des poids : {summary['weight_stability']:.3f}")
                print(f"   - Évaluations totales : {summary['total_predictions']}")
                print(f"   - Changements de stratégie : {summary['strategy_changes']}")
                print(f"   - Taux d'apprentissage : {summary['learning_rate']:.4f}")
                if summary['should_retrain']:
                    print(f"   - ⚠️ Ré-entraînement recommandé pour la prochaine session")
                
    print(f"\n🎲 Rappel : Ces grilles utilisent l'IA adaptative qui s'améliore en continu")
    print(f"   selon les performances passées, mais chaque tirage reste aléatoire !")

# AJOUT CRITIQUE : Appel de la fonction main() à la fin du script
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interruption par l'utilisateur (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)