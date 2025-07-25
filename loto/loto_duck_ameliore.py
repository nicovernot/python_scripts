import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
import time
import gc
import os
import psutil
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import itertools
from scipy.stats import norm

# ==============================================================================
# === 1. CONFIGURATIONS CENTRALISÉES
# ==============================================================================
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_CPUS = psutil.cpu_count(logical=True)
print(f"🚀 Utilisation de {N_CPUS} CPU disponibles")

# --- Paramètres du Loto ---
NUM_MAIN_BALLS = 49
NUM_CHANCE_BALLS = 10

# --- Chemins ---
DATA_PATH = os.getenv('DATA_PATH', './loto_data')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './loto_output_optimized')
INPUT_PATH = os.path.join(DATA_PATH, 'loto_201911.csv')
OUTPUT_DIR = Path(OUTPUT_PATH)
GRAPHS_DIR = OUTPUT_DIR / 'graphs'
MODELS_DIR = OUTPUT_DIR / 'models'

# --- Paramètres du Modèle ---
N_TRIALS_OPTUNA = 25
N_SPLITS_CV = 3
POOL_SIZE = 15
TOP_N_GRIDS_TO_SHOW = 10

# --- Configuration du Mode d'Exécution ---
BACKTEST_MODE = True 
BACKTEST_PERIOD_IN_DRAWS = 100 
WARMUP_PERIOD_IN_DRAWS = 500

# ==============================================================================
# === 2. FONCTIONS DE BASE (DATA PREP, FEATURE ENGINEERING)
# ==============================================================================

def setup_database_and_paths(con):
    """Crée les dossiers de sortie et charge les données dans DuckDB."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads TO {N_CPUS}; SET memory_limit = '8GB';")
    
    try:
        query = f"""
        CREATE OR REPLACE TABLE loto_raw AS SELECT * FROM read_csv_auto('{INPUT_PATH}', sep=';', header=True, ignore_errors=true) WHERE annee_numero_de_tirage IS NOT NULL;
        CREATE OR REPLACE TABLE loto_base AS SELECT
            CAST(annee_numero_de_tirage AS INTEGER) as tirage_id,
            CAST(date_de_tirage AS TIMESTAMP) as date_tirage,
            CAST(boule_1 AS TINYINT) as boule_1, CAST(boule_2 AS TINYINT) as boule_2,
            CAST(boule_3 AS TINYINT) as boule_3, CAST(boule_4 AS TINYINT) as boule_4,
            CAST(boule_5 AS TINYINT) as boule_5, CAST(numero_chance AS TINYINT) as numero_chance
        FROM loto_raw WHERE boule_1 IS NOT NULL AND numero_chance IS NOT NULL ORDER BY tirage_id;
        """
        con.execute(query)
        num_tirages_valid = con.execute("SELECT COUNT(*) FROM loto_base").fetchone()[0]
        if num_tirages_valid == 0:
            raise ValueError("Aucune donnée valide n'a été chargée.")
        print(f"✅ {num_tirages_valid} tirages valides chargés.")
    except Exception as e:
        print(f"❌ Erreur critique lors de la préparation des données : {e}")
        exit()

def create_advanced_features_with_duckdb(con, pool_name, num_total_boules, source_table='loto_base', unpivot_table_name='UnpivotedMain'):
    # Optimisation: Réduit les prints en mode backtest
    if not BACKTEST_MODE or pool_name == "Principaux":
        print(f"🦆 Création de features pour {pool_name} (Source: {source_table})...")
    
    sql_query = f"""
    WITH AllPossibleRows AS (
        SELECT t.tirage_id, b.boule FROM (SELECT DISTINCT tirage_id FROM {source_table}) t, (SELECT unnest(range(1, {num_total_boules + 1})) as boule) b
    ), JoinedData AS (
        SELECT apr.tirage_id, apr.boule, (lf.boule IS NOT NULL)::INT AS is_out FROM AllPossibleRows apr LEFT JOIN {unpivot_table_name} lf ON apr.tirage_id = lf.tirage_id AND apr.boule = lf.boule
    ), IslandGrouping AS (
        SELECT *, SUM(is_out) OVER (PARTITION BY boule ORDER BY tirage_id) as island_id FROM JoinedData
    ), BaseFeatures AS (
        SELECT g.tirage_id, g.boule, g.is_out, ROW_NUMBER() OVER (PARTITION BY g.boule, g.island_id ORDER BY g.tirage_id) - 1 AS tirages_sans_sortie FROM IslandGrouping g
    ), GapStats AS (
        SELECT boule, AVG(tirages_sans_sortie) as avg_gap, STDDEV_SAMP(tirages_sans_sortie) as stddev_gap FROM BaseFeatures WHERE is_out = 1 GROUP BY boule
    ), FinalFeatures AS (
        SELECT
            bf.tirage_id, bf.boule, bf.is_out, bf.tirages_sans_sortie,
            COALESCE((bf.tirages_sans_sortie - gs.avg_gap) / NULLIF(gs.stddev_gap, 0), 0) as zscore_gap,
            SUM(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) as freq_100,
            AVG(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as ewma_10,
            bf.boule % 2 AS is_even, bf.boule % 10 AS last_digit,
            LAG(bf.is_out, 1, 0) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id) as was_out_last_draw
        FROM BaseFeatures bf JOIN GapStats gs ON bf.boule = gs.boule
    )
    SELECT * FROM FinalFeatures WHERE tirage_id > (SELECT MIN(tirage_id) + 100 FROM {source_table}) ORDER BY tirage_id, boule
    """
    return con.execute(sql_query).df()

# ==============================================================================
# === 3. CŒUR DU MACHINE LEARNING (OPTIMISATION & ENTRAÎNEMENT)
# ==============================================================================

def objective(trial, X, y):
    params = {
        'objective': 'binary:logistic','eval_metric': 'auc','n_jobs': -1,'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0), 'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),'eta': trial.suggest_loguniform('eta', 0.01, 0.3)
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        model = XGBClassifier(**params, use_label_encoder=False, early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    return np.mean(scores)

def run_analysis_and_optimization(con, pool_name, num_total_boules, source_table, unpivot_table_name, run_optuna=True, best_params=None):
    full_df = create_advanced_features_with_duckdb(con, pool_name, num_total_boules, source_table, unpivot_table_name)
    if full_df.empty or 'tirage_id' not in full_df.columns: return None, None, None

    last_tirage_id = full_df['tirage_id'].max()
    df_predict = full_df[full_df['tirage_id'] == last_tirage_id].copy()
    X_predict = df_predict.drop(columns=['tirage_id', 'is_out'])
    df_train = full_df[full_df['tirage_id'] < last_tirage_id]
    X_train = df_train.drop(columns=['tirage_id', 'is_out'])
    y_train = df_train['is_out']
    del full_df, df_train
    gc.collect()

    study = None
    if run_optuna:
        print(f"🤖 Lancement de l'optimisation Optuna pour {pool_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS_OPTUNA, show_progress_bar=not BACKTEST_MODE)
        best_params = study.best_params
    
    final_model = XGBClassifier(**best_params, use_label_encoder=False, n_jobs=-1)
    final_model.fit(X_train, y_train, verbose=False)
    
    predictions_proba = final_model.predict_proba(X_predict)[:, 1]
    df_final = X_predict[['boule', 'tirages_sans_sortie', 'zscore_gap']].copy()
    df_final['proba_estimee'] = predictions_proba
    
    # Calcul optimisé du score composite
    for col, weight in [('proba_estimee', 0.5), ('tirages_sans_sortie', 0.3), ('zscore_gap', 0.2)]:
        min_val, max_val = df_final[col].min(), df_final[col].max()
        range_val = max_val - min_val
        norm_score = ((df_final[col] - min_val) / range_val) if range_val > 0 else 0
        df_final[f'score_{col}'] = norm_score * weight
    
    df_final['score_composite'] = df_final[['score_proba_estimee', 'score_tirages_sans_sortie', 'score_zscore_gap']].sum(axis=1)
    df_final = df_final.sort_values(by='score_composite', ascending=False).reset_index(drop=True)
    
    return df_final, final_model, study

# ==============================================================================
# === 4. FONCTIONS D'ANALYSE DE GRILLES & POST-TRAITEMENT
# ==============================================================================

def apply_hot_number_penalty(df, penalty_factor=0.25):
    if df is None or df.empty: return df
    rows_to_penalize_mask = (df['tirages_sans_sortie'] == 0)
    if rows_to_penalize_mask.any():
        df.loc[rows_to_penalize_mask, 'score_composite'] *= penalty_factor
        df = df.sort_values(by='score_composite', ascending=False).reset_index(drop=True)
    return df

def analyze_pairs_and_composition(con, source_table='UnpivotedMain'):
    pairs_query = f"WITH Pairs AS (SELECT t1.tirage_id, t1.boule as b1, t2.boule as b2 FROM {source_table} t1 JOIN {source_table} t2 ON t1.tirage_id = t2.tirage_id AND t1.boule < t2.boule) SELECT b1, b2, COUNT(*) as freq FROM Pairs GROUP BY b1, b2 ORDER BY freq DESC LIMIT 200;"
    top_pairs_df = con.execute(pairs_query).df()
    
    comp_query = f"WITH Props AS (SELECT tirage_id, boule, boule % 2 AS is_even FROM {source_table}) SELECT SUM(boule) as somme_grille, SUM(is_even) as count_even FROM Props GROUP BY tirage_id;"
    composition_stats = con.execute(comp_query).df().describe()
    
    return top_pairs_df, composition_stats

# REMPLACEZ l'ancienne version de cette fonction par celle-ci

def generate_and_score_grids(top_main_numbers_df, top_chance_numbers, top_pairs_df, grid_composition_stats):
    NUM_GRIDS_TO_GENERATE = 30000 
    candidate_pool = top_main_numbers_df.head(POOL_SIZE)
    candidate_numbers = candidate_pool['boule'].tolist()
    
    # Sécurité : si le pool de candidats est trop petit, on ne peut pas former de grilles de 5
    if len(candidate_numbers) < 5:
        return pd.DataFrame()

    candidate_scores = pd.Series(candidate_pool.score_composite.values, index=candidate_pool.boule).to_dict()

    all_combinations = list(itertools.combinations(candidate_numbers, 5))
    if len(all_combinations) > NUM_GRIDS_TO_GENERATE:
        indices = np.random.choice(len(all_combinations), NUM_GRIDS_TO_GENERATE, replace=False)
        all_combinations = [all_combinations[i] for i in indices]
    
    if not all_combinations: return pd.DataFrame()

    grids_df = pd.DataFrame(all_combinations, columns=[f'b{i+1}' for i in range(5)])

    # Préparation des données pour le scoring
    top_pairs_df['pair_key'] = top_pairs_df.apply(lambda r: tuple(sorted((r['b1'], r['b2']))), axis=1)
    pair_scores = pd.Series(top_pairs_df.freq.values, index=top_pairs_df.pair_key).to_dict()
    
    sum_mean, sum_std = grid_composition_stats.loc['mean', 'somme_grille'], grid_composition_stats.loc['std', 'somme_grille']
    even_mean, even_std = grid_composition_stats.loc['mean', 'count_even'], grid_composition_stats.loc['std', 'count_even']

    # Calcul des scores bruts
    scores = grids_df.apply(lambda r: pd.Series({
        'model_score': sum(candidate_scores.get(n, 0) for n in r),
        'pair_score': sum(pair_scores.get(p, 0) for p in itertools.combinations(sorted(r), 2)),
        # CORRECTION : Nom harmonisé en 'composition_score'
        'composition_score': norm.pdf(sum(r), sum_mean, sum_std) * norm.pdf(sum(1 for n in r if n%2==0), even_mean, even_std)
    }), axis=1)
    grids_df = pd.concat([grids_df, scores], axis=1)

    # Normalisation et pondération en une seule étape
    # CORRECTION : Utilisation de noms de colonnes cohérents
    final_score_components = []
    for col, weight in [('model_score', 0.6), ('pair_score', 0.2), ('composition_score', 0.2)]:
        min_val, max_val = grids_df[col].min(), grids_df[col].max()
        range_val = max_val - min_val
        norm_score = ((grids_df[col] - min_val) / range_val) if range_val > 0 else 0
        final_score_components.append(norm_score * weight)
    
    # Assemblage final
    grids_df['score_final'] = pd.concat(final_score_components, axis=1).sum(axis=1).fillna(0)

    # Tri et sélection des meilleures grilles
    top_grids = grids_df.sort_values(by='score_final', ascending=False).head(TOP_N_GRIDS_TO_SHOW)
    
    # Gestion du cas où il n'y a pas assez de numéros chance
    if not top_chance_numbers.empty:
        top_grids['chance_1'] = top_chance_numbers.iloc[0]['boule']
        top_grids['chance_2'] = top_chance_numbers.iloc[1]['boule'] if len(top_chance_numbers) > 1 else '-'
    else:
        top_grids['chance_1'] = '-'
        top_grids['chance_2'] = '-'

    return top_grids

# ==============================================================================
# === 5. FONCTION PRINCIPALE DE BACKTESTING
# ==============================================================================

# REMPLACEZ l'ancienne version de cette fonction par celle-ci

# REMPLACEZ l'ancienne version de cette fonction par celle-ci

def run_backtesting(con):
    start_time_backtest = time.time()
    print("="*80 + "\n🚀 DÉMARRAGE DU BACKTESTING DE LA STRATÉGIE\n" + "="*80)
    all_tirage_ids = con.execute("SELECT DISTINCT tirage_id FROM loto_base ORDER BY tirage_id").df()['tirage_id'].tolist()
    
    if len(all_tirage_ids) < WARMUP_PERIOD_IN_DRAWS + BACKTEST_PERIOD_IN_DRAWS:
        return print("❌ Données insuffisantes pour le backtest.")

    warmup_end_index = len(all_tirage_ids) - BACKTEST_PERIOD_IN_DRAWS
    warmup_ids, backtest_ids = tuple(all_tirage_ids[:warmup_end_index]), all_tirage_ids[warmup_end_index:]
    
    print(f"🔥 Phase 1: Optimisation des hyperparamètres sur {len(warmup_ids)} tirages de chauffe...")
    con.execute(f"CREATE OR REPLACE TEMP VIEW loto_warmup AS SELECT * FROM loto_base WHERE tirage_id IN {warmup_ids}")
    
    # CORRECTION DÉFINITIVE : Retour au UNION ALL
    con.execute("""
        CREATE OR REPLACE TEMP VIEW UnpivotedMain_warmup AS 
        SELECT tirage_id, boule_1 AS boule FROM loto_warmup UNION ALL
        SELECT tirage_id, boule_2 AS boule FROM loto_warmup UNION ALL
        SELECT tirage_id, boule_3 AS boule FROM loto_warmup UNION ALL
        SELECT tirage_id, boule_4 AS boule FROM loto_warmup UNION ALL
        SELECT tirage_id, boule_5 AS boule FROM loto_warmup
    """)
    con.execute("CREATE OR REPLACE TEMP VIEW UnpivotedChance_warmup AS SELECT tirage_id, numero_chance as boule FROM loto_warmup")

    _, _, study_main = run_analysis_and_optimization(con, "Principaux", NUM_MAIN_BALLS, 'loto_warmup', 'UnpivotedMain_warmup', True)
    _, _, study_chance = run_analysis_and_optimization(con, "Chance", NUM_CHANCE_BALLS, 'loto_warmup', 'UnpivotedChance_warmup', True)
    best_params_main, best_params_chance = study_main.best_params, study_chance.best_params
    print("✅ Hyperparamètres optimisés et prêts pour le backtest.")

    print("\n" + "="*80 + "\n⏳ Phase 2: Lancement de la simulation historique...\n" + "="*80)
    backtest_results = []
    
    for i, current_tirage_id in enumerate(backtest_ids):
        print(f"\n--- Simulation {i+1}/{len(backtest_ids)}: Prédiction pour tirage ID {current_tirage_id} ---")
        history_ids = tuple(all_tirage_ids[:warmup_end_index + i])
        con.execute(f"CREATE OR REPLACE TEMP VIEW loto_history AS SELECT * FROM loto_base WHERE tirage_id IN {history_ids}")
        
        # CORRECTION DÉFINITIVE : Retour au UNION ALL
        con.execute("""
            CREATE OR REPLACE TEMP VIEW UnpivotedMain_history AS 
            SELECT tirage_id, boule_1 AS boule FROM loto_history UNION ALL
            SELECT tirage_id, boule_2 AS boule FROM loto_history UNION ALL
            SELECT tirage_id, boule_3 AS boule FROM loto_history UNION ALL
            SELECT tirage_id, boule_4 AS boule FROM loto_history UNION ALL
            SELECT tirage_id, boule_5 AS boule FROM loto_history
        """)
        con.execute("CREATE OR REPLACE TEMP VIEW UnpivotedChance_history AS SELECT tirage_id, numero_chance as boule FROM loto_history")

        stats_main, _, _ = run_analysis_and_optimization(con, "Principaux", NUM_MAIN_BALLS, 'loto_history', 'UnpivotedMain_history', False, best_params_main)
        stats_chance, _, _ = run_analysis_and_optimization(con, "Chance", NUM_CHANCE_BALLS, 'loto_history', 'UnpivotedChance_history', False, best_params_chance)
        
        if stats_main is None or stats_chance is None: continue
        
        stats_main = apply_hot_number_penalty(stats_main)
        top_pairs, composition_stats = analyze_pairs_and_composition(con, 'UnpivotedMain_history')

        if top_pairs is None or composition_stats is None: continue
        
        recommended_grids = generate_and_score_grids(stats_main, stats_chance, top_pairs, composition_stats)
        if recommended_grids.empty: continue
        
        actual_result_df = con.execute(f"SELECT boule_1, boule_2, boule_3, boule_4, boule_5, numero_chance FROM loto_base WHERE tirage_id = {current_tirage_id}").df()
        actual_main_balls = set(actual_result_df.iloc[0, :5].astype(int))
        actual_chance_ball = actual_result_df.iloc[0, 5]
        
        for rank in range(min(3, len(recommended_grids))):
            grid_details = recommended_grids.iloc[rank]
            predicted_grid = set(grid_details[[f'b{i+1}' for i in range(5)]].astype(int))
            predicted_chance = grid_details['chance_1']
            main_matches = len(predicted_grid.intersection(actual_main_balls))
            chance_match = 1 if predicted_chance == actual_chance_ball else 0
            backtest_results.append({'tirage_id': current_tirage_id, 'rank': rank + 1, 'matches': main_matches, 'chance_match': chance_match, 'gain_str': f"{main_matches}+{chance_match}"})
        
        top_rank_result = next(item for item in backtest_results if item['tirage_id'] == current_tirage_id and item['rank'] == 1)
        print(f"🎯 Résultat (Grille N°1): {top_rank_result['matches']} bons numéros, {'Chance OK' if top_rank_result['chance_match'] else 'Chance KO'}")

    # --- Rapport final du Backtest ---
    # ... (le reste de la fonction est inchangé et correct) ...
    print("\n" + "="*80 + "\n📈 RAPPORT FINAL DU BACKTEST\n" + "="*80)
    if not backtest_results: return print("Aucun résultat de backtest à afficher.")
    
    results_df = pd.DataFrame(backtest_results)
    print("--- Performance moyenne par rang de grille ---\n", results_df.groupby('rank')['matches'].mean().to_string())
    print("\n--- Répartition des gains pour la grille N°1 ---\n", results_df[results_df['rank'] == 1]['gain_str'].value_counts().sort_index().to_string())
    
    results_path = OUTPUT_DIR / 'backtest_results_detailed.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n📁 Résultats détaillés du backtest sauvegardés : {results_path}")
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='gain_str', data=results_df[results_df['rank']==1], order=sorted(results_df[results_df['rank']==1]['gain_str'].unique()), palette='crest')
    plt.title(f'Distribution des Gains (Grille N°1) sur {len(backtest_ids)} Tirages Simulés')
    plt.xlabel('Bons Numéros + Numéro Chance'); plt.ylabel('Nombre de Grilles')
    plt.savefig(GRAPHS_DIR / "7_backtest_summary.png"); plt.close()
    print(f"🎨 Graphique du rapport de backtest sauvegardé.")
    backtest_duration = time.time() - start_time_backtest 
    generate_backtest_report(results_df, backtest_duration)

def generate_backtest_report(results_df, backtest_duration_seconds):
    """Génère un rapport texte complet à partir des résultats du backtest."""
    
    print("\n" + "="*80)
    print("📊 RAPPORT D'ANALYSE FINAL DU BACKTEST")
    print("="*80)

    num_tirages = len(results_df['tirage_id'].unique())
    print(f"Simulation réalisée sur {num_tirages} tirages en {backtest_duration_seconds:.1f} secondes.\n")

    # 1. Performance globale par rang
    avg_performance = results_df.groupby('rank')['matches'].mean().reset_index()
    best_rank_perf = avg_performance.sort_values(by='matches', ascending=False).iloc[0]
    
    print("--- 1. Performance Moyenne par Rang de Grille ---")
    print(avg_performance.to_string(index=False))
    print(f"\n✅ Le rang N°{int(best_rank_perf['rank'])} est le plus performant avec une moyenne de {best_rank_perf['matches']:.3f} bons numéros.")
    
    # 2. Analyse de la meilleure grille trouvée
    best_overall_grid = results_df.sort_values(by=['matches', 'chance_match'], ascending=False).iloc[0]
    tirage_id_best = best_overall_grid['tirage_id']
    gain_best = best_overall_grid['gain_str']
    rank_best = best_overall_grid['rank']
    
    print(f"\n--- 2. Meilleur Résultat Obtenu ---")
    print(f"🏆 Le meilleur coup a été un '{gain_best}' lors du tirage {tirage_id_best}.")
    print(f"   Ce jour-là, c'était la grille classée au rang N°{rank_best} qui a obtenu ce score.")

    # 3. Comparaison au hasard
    random_benchmark = 5 * (5 / NUM_MAIN_BALLS)
    best_strategy_perf = best_rank_perf['matches']
    improvement = ((best_strategy_perf - random_benchmark) / random_benchmark) * 100

    print(f"\n--- 3. Comparaison au Hasard ---")
    print(f"Moyenne du hasard pur : {random_benchmark:.3f} bons numéros.")
    print(f"Moyenne de votre meilleure stratégie (Rang {int(best_rank_perf['rank'])}) : {best_strategy_perf:.3f} bons numéros.")
    if improvement > 0:
        print(f"🟢 Votre stratégie surpasse le hasard de {improvement:.2f}%.")
    else:
        print(f"🔴 Votre stratégie est moins performante que le hasard de {-improvement:.2f}%.")
        
    # 4. Conclusion
    print("\n--- 4. Conclusion & Prochaines Étapes ---")
    if best_rank_perf['rank'] != 1:
        print("ACTION REQUISE : Votre scoring de grille n'est pas optimal car le rang 1 n'est pas le plus performant.")
        print("Pistes d'amélioration :")
        print("  - Augmenter le poids du 'model_score' dans le calcul du 'score_final'.")
        print("  - Analyser les scores détaillés du tirage où vous avez eu le meilleur résultat pour comprendre ce qui a fonctionné.")
    else:
        print("EXCELLENT : Votre scoring de grille est bien calibré, le rang 1 est le plus performant.")
        print("Pistes d'amélioration :")
        print("  - Continuer à affiner les features du modèle pour augmenter encore la performance.")
        print("  - Lancer le backtest sur une période plus longue pour valider ces bons résultats.")

    print("="*80)

# ==============================================================================
# === 6. SCRIPT PRINCIPAL (ORCHESTRATEUR)
# ==============================================================================

if __name__ == '__main__':
    start_time = time.time()
    con = duckdb.connect()
    
    setup_database_and_paths(con)

    if BACKTEST_MODE:
        run_backtesting(con)
    else:
        print("\n" + "="*80 + "\n🔮 LANCEMENT EN MODE PRÉDICTION FUTURE\n" + "="*80)
        
        # CORRECTION DÉFINITIVE : Retour au UNION ALL
        con.execute("""
            CREATE OR REPLACE TEMP VIEW UnpivotedMain AS 
            SELECT tirage_id, boule_1 AS boule FROM loto_base UNION ALL
            SELECT tirage_id, boule_2 AS boule FROM loto_base UNION ALL
            SELECT tirage_id, boule_3 AS boule FROM loto_base UNION ALL
            SELECT tirage_id, boule_4 AS boule FROM loto_base UNION ALL
            SELECT tirage_id, boule_5 AS boule FROM loto_base
        """)
        con.execute("CREATE OR REPLACE TEMP VIEW UnpivotedChance AS SELECT tirage_id, numero_chance as boule FROM loto_base")
        
        stats_main, _, _ = run_analysis_and_optimization(con, "Principaux", NUM_MAIN_BALLS, 'loto_base', 'UnpivotedMain', True)
        stats_chance, _, _ = run_analysis_and_optimization(con, "Chance", NUM_CHANCE_BALLS, 'loto_base', 'UnpivotedChance', True)
        
        if stats_main is not None and stats_chance is not None:
            stats_main = apply_hot_number_penalty(stats_main)
            stats_chance = apply_hot_number_penalty(stats_chance)
            
            top_pairs_df, grid_composition_stats = analyze_pairs_and_composition(con)
            
            if top_pairs_df is not None and grid_composition_stats is not None:
                recommended_grids = generate_and_score_grids(stats_main, stats_chance, top_pairs_df, grid_composition_stats)
                print("\n" + "="*80 + f"\n🏆 Top {TOP_N_GRIDS_TO_SHOW} des Grilles Recommandées\n" + "="*80)
                display_cols = [f'b{i+1}' for i in range(5)] + ['chance_1', 'chance_2', 'score_final']
                print(recommended_grids[display_cols].to_string(index=False))
                grids_path = OUTPUT_DIR / 'top_10_grilles_recommandees.csv'
                recommended_grids[display_cols].to_csv(grids_path, index=False, float_format='%.4f')
                print(f"\n📁 Fichier des grilles recommandées sauvegardé : {grids_path}")
    
    con.close()
    print(f"\n✅ Script terminé en {time.time() - start_time:.2f} secondes.")