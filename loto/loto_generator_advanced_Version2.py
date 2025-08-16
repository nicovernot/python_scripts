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
        fallback_path = Path(os.getenv('LOTO_PARQUET_PATH', '~/Téléchargements/loto_201911.parquet')).expanduser()
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

# --- Stratégie Adaptative ---
class AdaptiveStrategy:
    """Stratégie adaptative qui ajuste les paramètres selon les performances récentes"""
    
    def __init__(self, history_file=None):
        self.history_file = history_file or (MODEL_DIR / 'performance_history.json')
        self.performance_history = self.load_history()
        self.ml_weight = 0.6  # Poids initial pour ML vs fréquences
        self.adaptation_rate = 0.1  # Vitesse d'adaptation
        
    def load_history(self):
        """Charge l'historique des performances"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'predictions': [],
            'actuals': [],
            'ml_scores': [],
            'freq_scores': [],
            'dates': [],
            'ml_weight_history': []
        }
    
    def save_history(self):
        """Sauvegarde l'historique des performances"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde historique: {e}")
    
    def evaluate_prediction_accuracy(self, predicted_probs, actual_draw):
        """Évalue la précision des prédictions ML vs fréquences avec modèles binaires"""
        if len(actual_draw) != 5:
            return 0.0, 0.0
        
        # Avec les modèles binaires, predicted_probs contient les probabilités pour chaque boule (1-49)
        if isinstance(predicted_probs, np.ndarray) and len(predicted_probs) == 49:
            # Score ML : probabilité moyenne des boules tirées (predicted_probs[boule-1])
            ml_score = np.mean([predicted_probs[boule-1] for boule in actual_draw])
        else:
            # Fallback si format ancien ou erreur
            ml_score = 0.0
            
        # Score fréquences : score basé sur les fréquences historiques
        freq_weights = np.ones(49) / 49  # Uniforme en fallback
        freq_score = np.mean([freq_weights[boule-1] for boule in actual_draw])
        
        return ml_score, freq_score
    
    def update_performance(self, prediction, actual_draw, ml_probs=None):
        """Met à jour les performances et ajuste la stratégie"""
        if ml_probs is not None:
            ml_score, freq_score = self.evaluate_prediction_accuracy(ml_probs, actual_draw)
            
            # Ajouter à l'historique
            self.performance_history['predictions'].append(prediction)
            self.performance_history['actuals'].append(actual_draw)
            self.performance_history['ml_scores'].append(float(ml_score))
            self.performance_history['freq_scores'].append(float(freq_score))
            self.performance_history['dates'].append(datetime.now().isoformat())
            
            # Adapter le poids ML
            self.adapt_ml_weight(ml_score, freq_score)
            
            # Garder seulement les 100 dernières prédictions
            for key in self.performance_history:
                if len(self.performance_history[key]) > 100:
                    self.performance_history[key] = self.performance_history[key][-100:]
            
            self.save_history()
    
    def adapt_ml_weight(self, ml_score, freq_score):
        """Adapte le poids du ML selon les performances relatives"""
        if len(self.performance_history['ml_scores']) >= 10:  # Minimum 10 observations
            recent_ml = np.mean(self.performance_history['ml_scores'][-10:])
            recent_freq = np.mean(self.performance_history['freq_scores'][-10:])
            
            # Si ML performe mieux, augmenter son poids
            if recent_ml > recent_freq:
                self.ml_weight = min(0.9, self.ml_weight + self.adaptation_rate)
            else:
                self.ml_weight = max(0.1, self.ml_weight - self.adaptation_rate)
                
            self.performance_history['ml_weight_history'].append(float(self.ml_weight))
    
    def get_adaptive_weights(self):
        """Retourne les poids adaptés pour ML vs fréquences"""
        return self.ml_weight, 1.0 - self.ml_weight
    
    def get_performance_summary(self):
        """Retourne un résumé des performances"""
        if len(self.performance_history['ml_scores']) < 5:
            return "Données insuffisantes pour l'analyse adaptative"
        
        recent_ml = np.mean(self.performance_history['ml_scores'][-10:])
        recent_freq = np.mean(self.performance_history['freq_scores'][-10:])
        
        return {
            'ml_score_recent': recent_ml,
            'freq_score_recent': recent_freq,
            'current_ml_weight': self.ml_weight,
            'total_predictions': len(self.performance_history['ml_scores'])
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
    
    # Utiliser les numéros exclus configurables
    if EXCLUDED_NUMBERS is not None:
        numbers_to_exclude = EXCLUDED_NUMBERS
        exclusion_source = "paramètre utilisateur"
    else:
        numbers_to_exclude = last_3_numbers_set
        exclusion_source = "3 derniers tirages (auto)"
    
    print(f"   ✓ Numéros exclus ({exclusion_source}): {sorted(numbers_to_exclude)}")

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
        'numbers_to_exclude': numbers_to_exclude,
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

# --- Fonctions d'analyse de cycles et motifs ---
def autocorrelation_analysis(series, max_lag=50):
    autocorr = [series.autocorr(lag) for lag in range(1, max_lag + 1)]
    return pd.Series(autocorr, index=range(1, max_lag + 1))

def plot_autocorrelation(series, output_dir, max_lag=50):
    ac = autocorrelation_analysis(series, max_lag)
    plt.figure(figsize=(10, 5))
    plt.bar(ac.index, ac.values)
    plt.title("Autocorrélation des tirages (lags)")
    plt.xlabel("Décalage (lag)")
    plt.ylabel("Autocorrélation")
    plt.tight_layout()
    plt.savefig(output_dir / 'autocorrelation_plot.png', dpi=150)
    plt.close()

def fft_analysis(series):
    N = len(series)
    yf = fft(series - np.mean(series))
    xf = fftfreq(N, 1)[:N // 2]
    spectrum = np.abs(yf[:N // 2])
    return xf, spectrum

def plot_fft(series, output_dir):
    xf, spectrum = fft_analysis(series)
    plt.figure(figsize=(10, 5))
    plt.plot(xf, spectrum)
    plt.title("Spectre de fréquences (FFT)")
    plt.xlabel("Fréquence")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_dir / 'fft_spectrum_plot.png', dpi=150)
    plt.close()

def seasonal_decomposition(series, period=10):
    res = STL(series, period=period).fit()
    return res

def plot_seasonal_decomposition(series, output_dir, period=10):
    res = seasonal_decomposition(series, period)
    res.plot()
    plt.tight_layout()
    plt.savefig(output_dir / 'stl_decomposition_plot.png', dpi=150)
    plt.close()

def matrix_profile_motifs(series, window=10):
    mp = stumpy.stump(series.values, m=window)
    motifs = stumpy.motifs(series.values, mp, max_motifs=5)
    return motifs

def plot_matrix_profile(series, output_dir, window=10):
    mp = stumpy.stump(series.values.astype(np.float64), m=window)
    plt.figure(figsize=(10,5))
    plt.plot(mp[:,0], label="Matrix Profile")
    plt.title(f"Matrix Profile (window={window})")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'matrix_profile_plot.png', dpi=150)
    plt.close()

# --- Fonctions de Machine Learning ---
def train_xgboost_parallel(df: pd.DataFrame):
    print("Entraînement des modèles XGBoost...")
    balls_cols = [f'boule_{i}' for i in range(1, 6)]

    df_features = add_cyclic_features(df)
    feature_cols = balls_cols + [col for col in df_features.columns if col.startswith('sin_') or col.startswith('cos_')]
    X = df_features[feature_cols].iloc[:-1].values
    
    def train_ball_model(ball_num):
        """Entraîne un modèle binaire pour prédire si une boule sera tirée"""
        print(f"  - Entraînement modèle boule principale {ball_num}/49...")
        
        # Créer les labels binaires : 1 si la boule est tirée, 0 sinon
        y = np.zeros(len(X))
        for i, row in enumerate(df_features[balls_cols].iloc[1:].values):
            if ball_num in row:
                y[i] = 1
        
        # Vérifier qu'il y a assez d'occurrences positives
        positive_samples = np.sum(y)
        if positive_samples < 10:
            print(f"   ⚠️  Boule {ball_num}: seulement {positive_samples} occurrences")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=GLOBAL_SEED,
            use_label_encoder=False,
            objective='binary:logistic',
            n_jobs=1
        )
        model.fit(X, y)
        dump(model, MODEL_DIR / f'model_ball_{ball_num}.joblib')
        return model
    
    def train_chance_model(chance_num):
        """Entraîne un modèle binaire pour prédire si un numéro chance sera tiré"""
        print(f"  - Entraînement modèle numéro chance {chance_num}/10...")
        
        # Créer les labels binaires pour le numéro chance
        y = np.zeros(len(X))
        for i, chance_val in enumerate(df_features['numero_chance'].iloc[1:].values):
            if chance_val == chance_num:
                y[i] = 1
        
        positive_samples = np.sum(y)
        if positive_samples < 5:
            print(f"   ⚠️  Chance {chance_num}: seulement {positive_samples} occurrences")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=GLOBAL_SEED,
            use_label_encoder=False,
            objective='binary:logistic',
            n_jobs=1
        )
        model.fit(X, y)
        dump(model, MODEL_DIR / f'model_chance_{chance_num}.joblib')
        return model
    
    print("  📊 Entraînement de 49 modèles pour les boules principales...")
    ball_models = Parallel(n_jobs=min(8, N_CORES))(
        delayed(train_ball_model)(ball) for ball in range(1, 50)
    )
    
    print("  🎯 Entraînement de 10 modèles pour les numéros chance...")
    chance_models = Parallel(n_jobs=min(8, N_CORES))(
        delayed(train_chance_model)(chance) for chance in range(1, 11)
    )
    
    models = ball_models + chance_models
    
    # Sauvegarder les métadonnées
    metadata = {
        'features_count': len(feature_cols),
        'model_type': 'xgboost_binary',
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'version': '3.0',
        'ball_models': 49,
        'chance_models': 10,
        'total_models': 59
    }
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("   ✓ Entraînement terminé et modèles sauvegardés.")
    return models

def load_saved_models() -> dict:
    """Charge les 59 modèles binaires (49 boules + 10 numéros chance)"""
    models = {'balls': {}, 'chance': {}}
    
    # Vérifier la compatibilité des features
    metadata_path = MODEL_DIR / 'metadata.json'
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        expected_features = metadata.get('features_count', 19)
        total_models = metadata.get('total_models', 59)
        print(f"   📊 Modèles attendus avec {expected_features} features ({total_models} modèles)")
        
        # Si incompatibilité détectée, signaler
        if expected_features != 19:  # 19 = nouvelles features avec cycliques complètes
            print(f"   ⚠️ INCOMPATIBILITÉ: Modèles avec {expected_features} features, code actuel avec 19 features")
            print(f"   🔄 Recommandation: Ré-entraîner les modèles avec --retrain")
    
    # Charger les modèles pour les boules (1-49)
    loaded_balls = 0
    for ball in range(1, 50):
        model_path = MODEL_DIR / f'model_ball_{ball}.joblib'
        if model_path.exists():
            try:
                models['balls'][ball] = load(model_path)
                loaded_balls += 1
            except Exception as e:
                print(f"⚠️ Erreur chargement modèle boule {ball}: {e}")
                models['balls'][ball] = None
        else:
            models['balls'][ball] = None
    
    # Charger les modèles pour les numéros chance (1-10)
    loaded_chance = 0
    for chance in range(1, 11):
        model_path = MODEL_DIR / f'model_chance_{chance}.joblib'
        if model_path.exists():
            try:
                models['chance'][chance] = load(model_path)
                loaded_chance += 1
            except Exception as e:
                print(f"⚠️ Erreur chargement modèle chance {chance}: {e}")
                models['chance'][chance] = None
        else:
            models['chance'][chance] = None
    
    total_loaded = loaded_balls + loaded_chance
    if total_loaded == 59:
        print(f"   ✓ 59 modèles XGBoost binaires chargés depuis '{MODEL_DIR}' ({loaded_balls} boules + {loaded_chance} chance).")
    else:
        print(f"   ⚠️ Seulement {total_loaded}/59 modèles chargés ({loaded_balls}/49 boules + {loaded_chance}/10 chance)")
    
    return models

# --- Fonctions de Scoring et Génération ---
def score_grid(grid: np.ndarray, criteria: dict, diversity_factor: float = 0.0) -> float:
    W = criteria['dynamic_weights']
    W_DECADES, W_PAIRS, W_CONSECUTIVE, PENALTY_OVERLAP = 0.05, 0.05, 0.05, 0.1
    W_HOT_NUMBERS = 0.03
    PENALTY_TOO_MANY_HOT = 0.08

    score = 0
    
    # Critères principaux avec poids dynamiques
    score += W.get('sum', 0.125) * scipy.stats.norm.pdf(
        np.sum(grid), 
        loc=criteria['sums'].mean(), 
        scale=criteria['sums'].std()
    )

    score += W.get('std', 0.125) * scipy.stats.norm.pdf(
        np.std(grid), 
        loc=criteria['stds'].mean(), 
        scale=criteria['stds'].std()
    )

    score += W.get('pair_impair', 0.125) * criteria['pair_impair_probs'].get(np.sum(grid % 2 == 0), 0)
    
    # === NOUVEAUX CRITÈRES STATISTIQUES ===
    
    # 1. Entropie des dizaines
    grid_decades = [(n-1)//10 for n in grid]
    grid_decades_count = pd.Series(grid_decades).value_counts()
    grid_decades_dist = grid_decades_count.reindex(range(5), fill_value=0).values
    grid_decades_entropy = -np.sum(grid_decades_dist * np.log(grid_decades_dist + 1e-10))
    score += W.get('decades_entropy', 0.125) * scipy.stats.norm.pdf(
        grid_decades_entropy,
        loc=criteria['decades_entropy'].mean(),
        scale=criteria['decades_entropy'].std() + 1e-10
    )
    
    # 2. Espacement entre numéros
    sorted_grid = sorted(grid)
    grid_gaps = [sorted_grid[i+1] - sorted_grid[i] for i in range(4)]
    grid_gaps_std = np.std(grid_gaps)
    score += W.get('gaps', 0.125) * scipy.stats.norm.pdf(
        grid_gaps_std,
        loc=criteria['gaps_between_numbers'].mean(),
        scale=criteria['gaps_between_numbers'].std() + 1e-10
    )
    
    # 3. Répartition haute/basse
    low_count = sum(1 for n in grid if n <= 25)
    grid_high_low_ratio = low_count / 5.0
    score += W.get('high_low', 0.125) * scipy.stats.norm.pdf(
        grid_high_low_ratio,
        loc=criteria['high_low_ratio'].mean(),
        scale=criteria['high_low_ratio'].std() + 1e-10
    )
    
    # 4. Ratio de nombres premiers
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    prime_count = sum(1 for n in grid if n in primes)
    grid_prime_ratio = prime_count / 5.0
    score += W.get('prime_ratio', 0.125) * scipy.stats.norm.pdf(
        grid_prime_ratio,
        loc=criteria['prime_ratio'].mean(),
        scale=criteria['prime_ratio'].std() + 1e-10
    )
    
    # 5. Variance des positions
    grid_positions = [(n-1)/48.0 for n in grid]
    grid_position_variance = np.var(grid_positions)
    score += W.get('position_variance', 0.125) * scipy.stats.norm.pdf(
        grid_position_variance,
        loc=criteria['position_variance'].mean(),
        scale=criteria['position_variance'].std() + 1e-10
    )

    # Critères traditionnels avec poids réduits
    score += W_DECADES * (len(set((n - 1) // 10 for n in grid)) / 5.0)

    grid_pairs = set(combinations(sorted(grid), 2))
    score += W_PAIRS * (len(grid_pairs.intersection(criteria['pair_counts'].index)) / 2.0)

    consecutive_count = _count_consecutive_numba(grid)
    if consecutive_count == 0:
        score += W_CONSECUTIVE * 1.0
    elif consecutive_count == 1:
        score += W_CONSECUTIVE * 0.5
    else:
        score += W_CONSECUTIVE * 0.0

    score -= PENALTY_OVERLAP * len(set(grid).intersection(set(criteria['last_draw'][:5])))

    hot_count = len(set(grid).intersection(set(criteria['hot_numbers'])))
    score += W_HOT_NUMBERS * hot_count

    if hot_count > 3:
        score -= PENALTY_TOO_MANY_HOT * (hot_count - 3)

    # Ajout d'un facteur de diversité pour éviter la convergence vers des grilles identiques
    if diversity_factor > 0:
        # Ajouter une petite perturbation aléatoire basée sur le hash de la grille
        grid_hash = hash(tuple(sorted(grid))) 
        np.random.seed(grid_hash % 2**31)  # Seed basé sur le hash pour reproductibilité
        diversity_bonus = np.random.normal(0, diversity_factor)
        score += diversity_bonus

    return max(0, score)

def generate_grid_vectorized(criteria: dict, models: dict, X_last: np.ndarray) -> list:
    """Génération de grille avec 59 modèles binaires (49 boules + 10 chance)"""
    N_CANDIDATES, EXPLORATION_RATE, TOP_PERCENT_SELECTION = 500, 0.2, 0.05
    
    # Vérifier si les modèles sont disponibles
    has_ball_models = models and 'balls' in models and any(models['balls'].values())
    has_chance_models = models and 'chance' in models and any(models['chance'].values())
    use_models = has_ball_models
    
    freq_weights = criteria['freq'].reindex(BALLS, fill_value=0).values / criteria['freq'].sum()

    # Obtenir les poids adaptatifs
    ml_weight, freq_weight = adaptive_strategy.get_adaptive_weights()

    # Ajout des features cycliques pour la grille à prédire
    # Créer un DataFrame avec une date fictive pour générer toutes les features
    df_last = pd.DataFrame([X_last[0]], columns=[f'boule_{i}' for i in range(1, 6)])
    df_last['date_de_tirage'] = pd.Timestamp.now()  # Date fictive pour générer les features cycliques
    df_last_features = add_cyclic_features(df_last)
    feature_cols = [f'boule_{i}' for i in range(1, 6)] + [col for col in df_last_features.columns if col.startswith('sin_') or col.startswith('cos_')]
    X_last_features = df_last_features[feature_cols].values

    if use_models:
        try:
            # Prédictions pour chaque boule (1-49) avec modèles binaires
            ball_predictions = np.zeros(49)
            successful_predictions = 0
            
            for ball in range(1, 50):
                model = models['balls'].get(ball)
                if model is not None:
                    try:
                        if hasattr(model, 'predict_proba'):
                            # Classification binaire : probabilité de la classe 1 (boule tirée)
                            prob = model.predict_proba(X_last_features)[0][1]
                        elif hasattr(model, 'predict'):
                            # Si pas de predict_proba, utiliser predict et convertir en probabilité
                            prediction = model.predict(X_last_features)[0]
                            prob = max(0.01, min(0.99, prediction))  # Borner entre 0.01 et 0.99
                        else:
                            prob = freq_weights[ball-1]  # Fallback sur fréquence historique
                        
                        ball_predictions[ball-1] = prob
                        successful_predictions += 1
                    except Exception as e:
                        # En cas d'erreur, utiliser la fréquence historique
                        ball_predictions[ball-1] = freq_weights[ball-1]
                else:
                    # Modèle manquant, utiliser fréquence historique
                    ball_predictions[ball-1] = freq_weights[ball-1]
            
            if successful_predictions > 0:
                # Normaliser les probabilités
                ball_predictions = ball_predictions / ball_predictions.sum()
                # Combiner avec les fréquences historiques selon les poids adaptatifs
                exploitation_weights = (ml_weight * ball_predictions + freq_weight * freq_weights)
                exploitation_weights /= exploitation_weights.sum()
                
                # Affichage des poids adaptatifs si en mode debug
                if not ARGS.silent:
                    print(f"   🎯 Poids adaptatifs: ML={ml_weight:.2f}, Freq={freq_weight:.2f} ({successful_predictions}/49 modèles)")
            else:
                exploitation_weights = freq_weights
        except Exception as e:
            print(f"⚠️ Erreur avec les modèles ML: {e}. Utilisation des fréquences historiques.")
            exploitation_weights = freq_weights
    else:
        exploitation_weights = freq_weights

    excluded_numbers = criteria.get('numbers_to_exclude', set())
    print(f"DEBUG: Numéros exclus: {excluded_numbers} (total: {len(excluded_numbers)})")

    available_numbers = [n for n in BALLS if n not in excluded_numbers]
    print(f"DEBUG: Numéros disponibles: {len(available_numbers)}/49")

    if len(available_numbers) < 5:
        print(f"⚠️ ATTENTION: Seulement {len(available_numbers)} numéros disponibles, génération impossible!")
        return []

    candidates = []
    max_attempts = N_CANDIDATES * 10
    attempts = 0

    while len(candidates) < N_CANDIDATES and attempts < max_attempts:
        attempts += 1
        p = freq_weights if random.random() < EXPLORATION_RATE else exploitation_weights
        candidate = np.random.choice(BALLS, size=5, replace=False, p=p)
        if not any(num in excluded_numbers for num in candidate):
            candidates.append(candidate)

    print(f"DEBUG: Candidats générés: {len(candidates)}/{N_CANDIDATES} en {attempts} tentatives")

    if not candidates:
        print("❌ ERREUR: Aucun candidat généré!")
        return []

    candidates_matrix = np.array(candidates)
    scores = np.zeros(len(candidates))
    
    # Ajouter un facteur de diversité (5% de perturbation) pour éviter la convergence
    diversity_factor = 0.05
    for i, grid in enumerate(candidates_matrix):
        scores[i] = score_grid(grid, criteria, diversity_factor)

    top_n = max(1, int(len(candidates) * TOP_PERCENT_SELECTION))
    best_indices = np.argpartition(scores, -top_n)[-top_n:]
    chosen_index = np.random.choice(best_indices)

    best_grid_list = [int(b) for b in candidates_matrix[chosen_index]]
    
    # Génération du numéro chance avec modèles si disponibles
    if has_chance_models:
        try:
            chance_predictions = np.zeros(10)
            successful_chance_predictions = 0
            
            for chance in range(1, 11):
                model = models['chance'].get(chance)
                if model is not None:
                    try:
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(X_last_features)[0][1]
                        elif hasattr(model, 'predict'):
                            prediction = model.predict(X_last_features)[0]
                            prob = max(0.01, min(0.99, prediction))
                        else:
                            prob = 1.0 / 10  # Probabilité uniforme
                        
                        chance_predictions[chance-1] = prob
                        successful_chance_predictions += 1
                    except Exception as e:
                        chance_predictions[chance-1] = 1.0 / 10
                else:
                    chance_predictions[chance-1] = 1.0 / 10
            
            if successful_chance_predictions > 0:
                chance_predictions = chance_predictions / chance_predictions.sum()
                chance_ball = int(np.random.choice(CHANCE_BALLS, p=chance_predictions))
                if not ARGS.silent:
                    print(f"   🎲 Numéro chance prédit avec {successful_chance_predictions}/10 modèles")
            else:
                chance_ball = int(np.random.choice(CHANCE_BALLS))
        except Exception as e:
            print(f"⚠️ Erreur avec les modèles chance: {e}")
            chance_ball = int(np.random.choice(CHANCE_BALLS))
    else:
        chance_ball = int(np.random.choice(CHANCE_BALLS))
    
    return sorted(best_grid_list) + [chance_ball]
    max_attempts = N_CANDIDATES * 10
    attempts = 0

    while len(candidates) < N_CANDIDATES and attempts < max_attempts:
        attempts += 1
        p = freq_weights if random.random() < EXPLORATION_RATE else exploitation_weights
        candidate = np.random.choice(BALLS, size=5, replace=False, p=p)
        if not any(num in excluded_numbers for num in candidate):
            candidates.append(candidate)

    print(f"DEBUG: Candidats générés: {len(candidates)}/{N_CANDIDATES} en {attempts} tentatives")

    if not candidates:
        print("❌ ERREUR: Aucun candidat généré!")
        return []

    candidates_matrix = np.array(candidates)
    scores = np.zeros(len(candidates))
    
    # Ajouter un facteur de diversité (5% de perturbation) pour éviter la convergence
    diversity_factor = 0.05
    for i, grid in enumerate(candidates_matrix):
        scores[i] = score_grid(grid, criteria, diversity_factor)

    top_n = max(1, int(len(candidates) * TOP_PERCENT_SELECTION))
    best_indices = np.argpartition(scores, -top_n)[-top_n:]
    chosen_index = np.random.choice(best_indices)

    best_grid_list = [int(b) for b in candidates_matrix[chosen_index]]
    chance_ball = int(np.random.choice(CHANCE_BALLS))
    return sorted(best_grid_list) + [chance_ball]

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
        
    excluded_numbers_str = ", ".join(map(str, sorted(criteria.get('numbers_to_exclude', []))))    

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


...
## 🚫 Numéros Exclu (3 Derniers Tirages)
- **Numéros exclus des grilles générées** : `{excluded_numbers_str}`

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

# --- Fonction Principale ---
def main():
    start_time = datetime.now()
    if not parquet_path.exists():
        print(f"❌ ERREUR: Fichier Parquet '{parquet_path}' non trouvé.")
        return

    # Gestion du ré-entraînement forcé
    if hasattr(parse_arguments(), 'retrain') and parse_arguments().retrain:
        print("🔄 FORCE RE-TRAINING: Suppression des anciens modèles...")
        import shutil
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(exist_ok=True)
        print("   ✓ Dossier modèles nettoyé, entraînement forcé.")

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
            
            # Mise à jour de la stratégie adaptative avec les données récentes
            update_adaptive_strategy_with_recent_draws(con, adaptive_strategy)
            
            if redis_client:
                print("Sauvegarde des nouveaux critères dans Redis (expiration: 4h)...")
                ttl_seconds = int(timedelta(hours=4).total_seconds())
                redis_client.setex(cache_key, ttl_seconds, pickle.dumps(criteria))
        finally:
            if con:
                con.close()

    analysis_csv_path = OUTPUT_DIR / 'numbers_analysis.csv'
    criteria['numbers_analysis'].to_csv(analysis_csv_path, index=False, float_format='%.2f', date_format='%Y-%m-%d')
    print(f"   ✓ Analyse détaillée des numéros exportée vers '{analysis_csv_path}'.")

    print("\n2. Gestion des modèles XGBoost...")
    
    if ARGS.retrain:
        print("  ⚠️ FLAG --retrain détecté : Nettoyage et ré-entraînement forcé...")
        # Nettoyage complet des modèles existants
        import shutil
        boost_dir = Path("boost_models")
        if boost_dir.exists():
            print("  🗑️ Suppression du répertoire boost_models existant...")
            shutil.rmtree(boost_dir)
        # Recréer le répertoire
        boost_dir.mkdir(exist_ok=True)
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_parquet('{str(parquet_path)}')")
        df_full = con.table('loto_draws').fetchdf()
        con.close()
        models = train_xgboost_parallel(df_full)
    else:
        models = load_saved_models()
        # Vérifier si on a suffisamment de modèles fonctionnels (au moins 70% des modèles)
        ball_models_available = sum(1 for m in models.get('balls', {}).values() if m is not None)
        chance_models_available = sum(1 for m in models.get('chance', {}).values() if m is not None)
        total_available = ball_models_available + chance_models_available
        
        if total_available < 42:  # Au moins 70% des 59 modèles (42/59)
            print(f"  Insuffisamment de modèles disponibles ({total_available}/59), ré-entraînement complet...")
            con = duckdb.connect(database=':memory:', read_only=False)
            con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_parquet('{str(parquet_path)}')")
            df_full = con.table('loto_draws').fetchdf()
            con.close()
            models = train_xgboost_parallel(df_full)
        else:
            print(f"  ✓ {total_available}/59 modèles disponibles ({ball_models_available} boules + {chance_models_available} chance)")

    print("\n3. Simulation intelligente des grilles...")
    X_last = np.array(criteria['last_draw']).reshape(1, -1)
    grids = simulate_grids_parallel(N_SIMULATIONS, criteria, X_last, models)
    print(f"   ✓ {len(grids)} grilles générées et classées par score de conformité.")

    grids_df = pd.DataFrame()
    if grids:
        grids_df = pd.DataFrame(grids)
        grid_cols = grids_df['grid'].apply(pd.Series)
        grid_cols.columns = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']
        grids_df = pd.concat([grid_cols, grids_df['score']], axis=1)
        
        grids_csv_path = OUTPUT_DIR / 'grilles_conseillees.csv'
        grids_df.to_csv(grids_csv_path, index=False, float_format='%.4f')
        print(f"   ✓ {len(grids)} grilles sauvegardées dans '{grids_csv_path}'.")
    else:
        print("⚠️ Aucune grille générée, création d'un DataFrame vide.")

    analyze_generated_grids(grids, criteria)

    print("\n4. Finalisation des rapports et visualisations...")
    if not grids_df.empty:
        create_visualizations(criteria, grids_df, OUTPUT_DIR)
    else:
        print("⚠️ Pas de visualisations créées car aucune grille n'a été générée.")
    execution_time = (datetime.now() - start_time).total_seconds()
    create_report(criteria, grids, OUTPUT_DIR, execution_time, adaptive_strategy)

    print(f"\n✅ Analyse terminée avec succès en {execution_time:.2f} secondes.")
    print(f"📁 Tous les résultats sont disponibles dans : '{OUTPUT_DIR.resolve()}'")
    
    if grids:
        print("\n🎯 Top 5 des grilles recommandées (score de conformité statistique) :")
        for i, grid_info in enumerate(grids[:5]):
            balls_str = ', '.join(map(str, grid_info['grid'][:-1]))
            chance = grid_info['grid'][-1]
            score = grid_info['score']
            print(f"   {i+1}. Boules: [{balls_str}] | Chance: {chance} | Score: {score:.4f}")

        print(f"\n📊 Statistiques de génération :")
        print(f"   - Grilles simulées : {N_SIMULATIONS:,}")
        print(f"   - Score moyen : {np.mean([g['score'] for g in grids]):.4f}")
        print(f"   - Score médian : {np.median([g['score'] for g in grids]):.4f}")
        print(f"   - Meilleur score : {grids[0]['score']:.4f}")
        print(f"   - Pire score : {grids[-1]['score']:.4f}")
        
        # Affichage des informations de stratégie adaptative
        if adaptive_strategy:
            freq_weight = 1.0 - adaptive_strategy.ml_weight
            print(f"\n🤖 Stratégie Adaptative Machine Learning :")
            print(f"   - Poids ML actuel : {adaptive_strategy.ml_weight:.3f} ({adaptive_strategy.ml_weight*100:.1f}%)")
            print(f"   - Poids Fréquence actuel : {freq_weight:.3f} ({freq_weight*100:.1f}%)")
            
            history = adaptive_strategy.performance_history
            if history and 'ml_scores' in history and len(history['ml_scores']) > 0:
                recent_count = min(5, len(history['ml_scores']))
                avg_ml_acc = np.mean(history['ml_scores'][-recent_count:])
                avg_freq_acc = np.mean(history['freq_scores'][-recent_count:])
                print(f"   - Précision ML récente : {avg_ml_acc:.3f} (moyenne sur {recent_count} évaluations)")
                print(f"   - Précision Fréquence récente : {avg_freq_acc:.3f}")
                print(f"   - Total évaluations : {len(history['ml_scores'])}")
            else:
                print(f"   - État : Première utilisation - collecte des données en cours")
    else:
        print("\n⚠️ Aucune grille n'a été générée.")
        print("Vérifiez les critères d'exclusion et la configuration.")

    print(f"\n🎲 Rappel : Ces grilles sont optimisées selon des critères statistiques")
    print(f"   basés sur l'historique, mais chaque tirage reste entièrement aléatoire !")

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("🎲 " + "="*60)
    print("   GÉNÉRATEUR INTELLIGENT DE GRILLES LOTO")
    print("   Analyse statistique + Machine Learning + Optimisation")
    print("="*64)
    
    main()
    
    print("\n" + "="*64)
    print("🏁 Fin d'exécution du générateur de grilles Loto")
    print("="*64)