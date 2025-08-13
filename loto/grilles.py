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
import redis

warnings.filterwarnings("ignore")

# --- Configuration Générale et Reproductibilité ---
load_dotenv()
# Utiliser le CSV téléchargé automatiquement
csv_path = Path("/home/nvernot/projets/loto_keno/loto/loto_data/loto_201911.csv")
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

N_SIMULATIONS = 10000
N_CORES = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1
BALLS = np.arange(1, 50)
CHANCE_BALLS = np.arange(1, 11)

print(f"Configuration : {N_CORES} processeurs, {N_SIMULATIONS} grilles, SEED={GLOBAL_SEED}.")

# --- Gestion des Dossiers & Connexion Redis ---
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = BASE_DIR / 'boost_models'
for directory in [OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)

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
    """Compte les numéros consécutifs dans une grille - Version Numba optimisée"""
    if len(grid) < 2:
        return 0
    count = 0
    sorted_grid = np.sort(grid)
    for i in range(len(sorted_grid) - 1):
        if sorted_grid[i+1] - sorted_grid[i] == 1:
            count += 1
    return count

def count_consecutive_safe(row_values, balls_cols):
    """Wrapper sécurisé pour Numba qui gère la conversion de types"""
    grid = np.array([row_values[col] for col in balls_cols], dtype=np.int32)
    return _count_consecutive_numba(grid)

# --- Fonctions d'Analyse (DuckDB) ---
def analyze_criteria_duckdb(db_con: duckdb.DuckDBPyConnection, table_name: str) -> dict:
    print("Début de l'analyse des critères avec DuckDB...")

    # Création d'une vue temporaire pour simplifier les requêtes
    db_con.execute(f"""
        CREATE OR REPLACE TEMPORARY VIEW BaseData AS
        SELECT ROW_NUMBER() OVER (ORDER BY date_de_tirage) AS draw_index, 
               date_de_tirage::DATE as date_de_tirage,
               boule_1, boule_2, boule_3, boule_4, boule_5
        FROM {table_name};
    """)

    # Récupération des données dans un DataFrame pandas
    df_pandas = db_con.table('BaseData').fetchdf()
    balls_cols = [f'boule_{i}' for i in range(1, 6)]

    # Fréquence des numéros
    freq = pd.Series(df_pandas[balls_cols].values.flatten()).value_counts().reindex(BALLS, fill_value=0)
    
    last_3_draws = df_pandas.iloc[-3:][[f'boule_{i}' for i in range(1, 6)]].values.flatten()
    last_3_numbers_set = set(last_3_draws)
    
    # Dernière apparition de chaque numéro
    last_appearance = df_pandas.melt(
        id_vars=['date_de_tirage'], 
        value_vars=balls_cols, 
        value_name='numero'
    ).groupby('numero')['date_de_tirage'].max()

    # Fréquence des paires
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

    # Écarts entre les tirages
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
    gaps = pd.Series(gaps_df.set_index('numero')['periodicite']) if not gaps_df.empty else pd.Series(dtype='float64')

    # Répartition pairs/impairs
    pair_impair_dist = (df_pandas[balls_cols] % 2 == 0).sum(axis=1).value_counts().sort_index()

    # CORRECTION : Séquences consécutives avec conversion sécurisée
    consecutive_list = []
    for _, row in df_pandas.iterrows():
        grid = np.array([row[col] for col in balls_cols], dtype=np.int32)
        consecutive_list.append(_count_consecutive_numba(grid))
    consecutive_counts = pd.Series(consecutive_list).value_counts().sort_index()

    # Numéros "chauds" et "froids"
    hot_numbers = freq.nlargest(10).index.tolist()
    cold_numbers = freq.nsmallest(10).index.tolist()

    # Corrélation entre numéros
    correlation_matrix = df_pandas[balls_cols].corr()

    # Analyse de régression linéaire
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

    # Calcul des sommes et écarts types
    sums = df_pandas[balls_cols].sum(axis=1)
    stds = df_pandas[balls_cols].std(axis=1)

    # Calcul des poids dynamiques
    variances = {
        'sum': 1/np.var(sums),
        'std': 1/np.var(stds),
        'pair_impair': 1/np.var((pair_impair_dist/pair_impair_dist.sum()).values)
    }
    dynamic_weights = {key: value / sum(variances.values()) for key, value in variances.items()}

    # Calcul du temps écoulé depuis le dernier tirage
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
        'numbers_to_exclude': last_3_numbers_set,
        'sums': sums,
        'stds': stds,
        'dynamic_weights': dynamic_weights,
        'last_appearance': last_appearance,
        'numbers_analysis': numbers_analysis,
        'correlation_matrix': correlation_matrix,
        'regression_results': regression_results
    }

# --- Fonctions de Machine Learning ---
def train_xgboost_parallel(df: pd.DataFrame):
    print("Entraînement des modèles XGBoost...")
    balls_cols = [f'boule_{i}' for i in range(1, 6)]
    X = df[balls_cols].iloc[:-1].values
    y = df[balls_cols].iloc[1:].values - 1

    def train_single_model(i):
        print(f"  - Entraînement pour la position de boule {i+1}...")
        model = xgb.XGBClassifier(
            n_estimators=100, 
            random_state=GLOBAL_SEED, 
            use_label_encoder=False, 
            objective='multi:softprob', 
            num_class=len(BALLS), 
            n_jobs=1
        )
        model.fit(X, y[:, i])
        dump(model, MODEL_DIR / f'model_boule_{i+1}.joblib')
        return model
    
    models = Parallel(n_jobs=min(5, N_CORES))(delayed(train_single_model)(i) for i in range(5))
    print("   ✓ Entraînement terminé et modèles sauvegardés.")
    return models

def load_saved_models() -> list:
    models = []
    for i in range(1, 6):
        model_path = MODEL_DIR / f'model_boule_{i}.joblib'
        if model_path.exists():
            try:
                models.append(load(model_path))
            except Exception as e:
                print(f"⚠️ Erreur chargement modèle {i}: {e}. Il sera ignoré.")
                models.append(None)
        else:
            models.append(None)
    
    if all(m is not None for m in models):
        print(f"   ✓ 5 modèles XGBoost chargés depuis '{MODEL_DIR}'.")
    return models

# --- Fonctions de Scoring et Génération ---

def score_grid(grid: np.ndarray, criteria: dict) -> float:
    """Score une grille selon les critères statistiques - VERSION AVEC DÉCOTE HOT NUMBERS"""
    W = criteria['dynamic_weights']
    W_DECADES, W_PAIRS, W_CONSECUTIVE, PENALTY_OVERLAP = 0.1, 0.1, 0.1, 0.1
    
    # OPTION 1: Réduire le poids des numéros hot
    W_HOT_NUMBERS = 0.05  # Réduit de 0.1 à 0.05
    
    # OPTION 2: Ajouter une pénalité pour trop de numéros hot
    PENALTY_TOO_MANY_HOT = 0.08  # Nouveau paramètre

    score = 0

    # Score basé sur la somme et l'écart type
    score += W.get('sum', 0.2) * scipy.stats.norm.pdf(
        np.sum(grid), 
        loc=criteria['sums'].mean(), 
        scale=criteria['sums'].std()
    )

    score += W.get('std', 0.2) * scipy.stats.norm.pdf(
        np.std(grid), 
        loc=criteria['stds'].mean(), 
        scale=criteria['stds'].std()
    )

    # Score basé sur la répartition pairs/impairs
    score += W.get('pair_impair', 0.15) * criteria['pair_impair_probs'].get(np.sum(grid % 2 == 0), 0)

    # Score basé sur les décennies
    score += W_DECADES * (len(set((n - 1) // 10 for n in grid)) / 5.0)

    # Score basé sur les paires fréquentes
    grid_pairs = set(combinations(sorted(grid), 2))
    score += W_PAIRS * (len(grid_pairs.intersection(criteria['pair_counts'].index)) / 2.0)

    # Score basé sur les numéros consécutifs
    consecutive_count = _count_consecutive_numba(grid)
    if consecutive_count == 0:
        score += W_CONSECUTIVE * 1.0
    elif consecutive_count == 1:
        score += W_CONSECUTIVE * 0.5
    else:
        score += W_CONSECUTIVE * 0.0

    # Pénalisation pour les numéros en double avec le dernier tirage
    score -= PENALTY_OVERLAP * len(set(grid).intersection(set(criteria['last_draw'][:5])))

    # MODIFICATION 1: Bonus réduit pour les numéros chauds
    hot_count = len(set(grid).intersection(set(criteria['hot_numbers'])))
    score += W_HOT_NUMBERS * hot_count

    # MODIFICATION 2: Pénalité progressive si trop de numéros hot
    if hot_count > 3:  # Si plus de 3 numéros hot dans la grille
        score -= PENALTY_TOO_MANY_HOT * (hot_count - 3)  # Pénalité croissante

    return max(0, score)

def generate_grid_vectorized(criteria: dict, models: list, X_last: np.ndarray) -> list:
    """Génère une grille optimisée avec approche vectorisée"""
    N_CANDIDATES, EXPLORATION_RATE, TOP_PERCENT_SELECTION = 500, 0.2, 0.05
    use_models = all(m is not None for m in models)
    
    # Poids basés sur les fréquences historiques
    freq_weights = criteria['freq'].reindex(BALLS, fill_value=0).values / criteria['freq'].sum()
    
    if use_models:
        # Prédictions des modèles ML
        probs_model = np.sum([m.predict_proba(X_last)[0] for m in models], axis=0) / len(models)
        exploitation_weights = (0.6 * probs_model + 0.4 * freq_weights)
        exploitation_weights /= exploitation_weights.sum()
    else:
        exploitation_weights = freq_weights
    
    # Génération de candidats
    excluded_numbers = criteria.get('numbers_to_exclude', set())
    candidates = []

    while len(candidates) < N_CANDIDATES:
        p = freq_weights if random.random() < EXPLORATION_RATE else exploitation_weights
        candidate = np.random.choice(BALLS, size=5, replace=False, p=p)

        if not any(num in excluded_numbers for num in candidate):
            candidates.append(candidate)

    candidates_matrix = np.array(candidates)
    
    # Scoring vectorisé
    scores = np.zeros(N_CANDIDATES)
    for i, grid in enumerate(candidates_matrix):
        scores[i] = score_grid(grid, criteria)

    # Sélection du meilleur candidat
    top_n = max(1, int(N_CANDIDATES * TOP_PERCENT_SELECTION))
    best_indices = np.argpartition(scores, -top_n)[-top_n:]
    chosen_index = np.random.choice(best_indices)
    
    best_grid_list = [int(b) for b in candidates_matrix[chosen_index]]
    chance_ball = int(np.random.choice(CHANCE_BALLS))
    return sorted(best_grid_list) + [chance_ball]

# --- Simulation Parallèle ---
def simulate_chunk(args_tuple):
    """Simule un chunk de grilles"""
    n_sims_chunk, criteria, X_last_shared, models, chunk_seed = args_tuple
    random.seed(chunk_seed)
    np.random.seed(chunk_seed)
    results = []
    for _ in range(n_sims_chunk):
        grid = generate_grid_vectorized(criteria, models, X_last_shared)
        score = score_grid(np.array(grid[:-1]), criteria)
        results.append({'grid': grid, 'score': score})
    return results

def simulate_grids_parallel(n_simulations: int, criteria: dict, X_last: np.ndarray, models: list):
    """Lance la simulation parallèle de grilles"""
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
    
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(simulate_chunk, args) for args in chunks_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulation des grilles"):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"Erreur dans un chunk de simulation: {e}")
    
    return sorted(all_results, key=lambda x: x['score'], reverse=True)

def analyze_generated_grids(grids: list, criteria: dict):
    """Analyse la qualité des grilles générées"""
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
    """Graphique des fréquences des numéros"""
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
    """Graphique de la périodicité moyenne"""
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
    """Graphique de répartition pair/impair"""
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
    """Graphique des numéros consécutifs - VERSION CORRIGÉE"""
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
    """Crée toutes les visualisations"""
    print("\nCréation des visualisations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    plot_frequency_analysis(criteria, output_dir)
    plot_gap_analysis(criteria, output_dir)
    plot_odd_even_analysis(criteria, output_dir)
    plot_consecutive_numbers_analysis(grids_df, output_dir)

    print(f"   ✓ Visualisations sauvegardées dans '{output_dir}'.")

# --- Génération de Rapports ---
def create_report(criteria: dict, best_grids: list, output_dir: Path, exec_time: float):
    """Génère le rapport Markdown final"""
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

## 📈 Fichiers Générés
- **Analyse détaillée**: `numbers_analysis.csv`
- **Grilles complètes**: `grilles_conseillees.csv`
- **Graphiques**: 
  - `frequence_numeros.png`
  - `periodicite_plot.png`
  - `pair_impair_plot.png`
  - `consecutive_numbers_plot.png`

## 📈 Visualisations Clés
![Fréquence des numéros](frequence_numeros.png)

![Périodicité Moyenne](periodicite_plot.png)

---
*Rapport généré automatiquement par le système d'analyse Loto*
"""
    
    report_path = output_dir / 'rapport_analyse.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"   ✓ Rapport sauvegardé dans '{report_path}'.")

# --- Fonction Principale ---
def main():
    """Fonction principale d'exécution"""
    start_time = datetime.now()
    
    if not csv_path.exists():
        print(f"❌ ERREUR: Fichier CSV '{csv_path}' non trouvé.")
        print("💡 Exécutez d'abord : python loto/result.py")
        return

    # 1. Analyse des données avec cache Redis
    criteria = None
    cache_key = f"loto_criteria:{csv_path.stat().st_mtime}"
    
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
            print(f"1. Chargement et analyse des données depuis '{csv_path}'...")
            con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_csv_auto('{str(csv_path)}', delim=';', header=true)")
            criteria = analyze_criteria_duckdb(con, 'loto_draws')
            
            if redis_client:
                print("Sauvegarde des nouveaux critères dans Redis (expiration: 4h)...")
                ttl_seconds = int(timedelta(hours=4).total_seconds())
                redis_client.setex(cache_key, ttl_seconds, pickle.dumps(criteria))
        finally:
            if con:
                con.close()

    # Export de l'analyse détaillée
    analysis_csv_path = OUTPUT_DIR / 'numbers_analysis.csv'
    criteria['numbers_analysis'].to_csv(analysis_csv_path, index=False, float_format='%.2f', date_format='%Y-%m-%d')
    print(f"   ✓ Analyse détaillée des numéros exportée vers '{analysis_csv_path}'.")

    # 2. Gestion des modèles XGBoost
    print("\n2. Gestion des modèles XGBoost...")
    models = load_saved_models()
    if not all(m is not None for m in models):
        print("  Certains modèles sont manquants, ré-entraînement complet...")
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute(f"CREATE TABLE loto_draws AS SELECT * FROM read_csv_auto('{str(csv_path)}', delim=';', header=true)")
        df_full = con.table('loto_draws').fetchdf()
        con.close()
        models = train_xgboost_parallel(df_full)

    # 3. Simulation des grilles
    print("\n3. Simulation intelligente des grilles...")
    X_last = np.array(criteria['last_draw']).reshape(1, -1)
    grids = simulate_grids_parallel(N_SIMULATIONS, criteria, X_last, models)
    print(f"   ✓ {len(grids)} grilles générées et classées par score de conformité.")

    # Export des grilles
    grids_df = pd.DataFrame()  # Initialisation par défaut
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

    # 4. Contrôle qualité
    analyze_generated_grids(grids, criteria)

    # 5. Finalisation des rapports et visualisations
    print("\n4. Finalisation des rapports et visualisations...")
    if not grids_df.empty:
        create_visualizations(criteria, grids_df, OUTPUT_DIR)
    else:
        print("⚠️ Pas de visualisations créées car aucune grille n'a été générée.")
    execution_time = (datetime.now() - start_time).total_seconds()
    create_report(criteria, grids, OUTPUT_DIR, execution_time)

    # Résumé final
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