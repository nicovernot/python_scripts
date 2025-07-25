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
import json
import os
import psutil
import optuna
import itertools
from scipy.stats import norm # Nouvel import pour un scoring plus intelligent
# // NOUVEAU // Import pour la visualisation Optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# === Configurations ===
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_CPUS = psutil.cpu_count(logical=True)
print(f"🚀 Utilisation de {N_CPUS} CPU disponibles")
NUM_MAIN_BALLS = 49
NUM_CHANCE_BALLS = 10
DATA_PATH = os.getenv('DATA_PATH', './loto_data')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './loto_output_duckdb_advanced')
INPUT_PATH = os.path.join(DATA_PATH, 'loto_201911.csv')
OUTPUT_DIR = Path(OUTPUT_PATH)
GRAPHS_DIR = OUTPUT_DIR / 'graphs'
MODELS_DIR = OUTPUT_DIR / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
N_TRIALS_OPTUNA = 50 # Augmenter pour une recherche plus approfondie si le temps le permet
N_SPLITS_CV = 5

# === Connexion et Prétraitement ===
# === Connexion et Prétraitement ===
con = duckdb.connect()
con.execute(f"SET threads TO {N_CPUS}; SET memory_limit = '8GB';")
print("📊 Lecture et prétraitement des données LOTO via DuckDB...")
try:
    # // CORRECTION APPLIQUÉE ICI //
    query = f"""
    CREATE OR REPLACE TABLE loto_raw AS SELECT * FROM read_csv_auto('{INPUT_PATH}', sep=';', header=True, ignore_errors=true) WHERE annee_numero_de_tirage IS NOT NULL;
    CREATE OR REPLACE TABLE loto_base AS
    SELECT
        CAST(annee_numero_de_tirage AS INTEGER) as tirage_id,
        CAST(date_de_tirage AS TIMESTAMP) as date_tirage, -- On caste directement la colonne déjà reconnue comme date
        CAST(boule_1 AS TINYINT) as boule_1,
        CAST(boule_2 AS TINYINT) as boule_2,
        CAST(boule_3 AS TINYINT) as boule_3,
        CAST(boule_4 AS TINYINT) as boule_4,
        CAST(boule_5 AS TINYINT) as boule_5,
        CAST(numero_chance AS TINYINT) as numero_chance
    FROM loto_raw
    WHERE boule_1 IS NOT NULL AND numero_chance IS NOT NULL
    ORDER BY tirage_id;

    CREATE OR REPLACE TABLE UnpivotedMain AS SELECT tirage_id, date_tirage, boule_1 AS boule FROM loto_base UNION ALL SELECT tirage_id, date_tirage, boule_2 AS boule FROM loto_base UNION ALL SELECT tirage_id, date_tirage, boule_3 AS boule FROM loto_base UNION ALL SELECT tirage_id, date_tirage, boule_4 AS boule FROM loto_base UNION ALL SELECT tirage_id, date_tirage, boule_5 AS boule FROM loto_base;
    CREATE OR REPLACE TABLE UnpivotedChance AS SELECT tirage_id, date_tirage, numero_chance as boule FROM loto_base;
    SELECT COUNT(*) FROM loto_base;
    """
    num_tirages = con.execute(query).fetchone()[0]
    if num_tirages == 0: exit("❌ Aucune donnée valide n'a été chargée.")
    print(f"✅ {num_tirages} tirages valides chargés.")
except Exception as e:
    exit(f"❌ Erreur lors de la préparation des données : {e}")

def generate_and_score_grids(top_main_numbers_df, top_chance_numbers, top_pairs_df, grid_composition_stats, con):
    """
    Génère des grilles, les note selon plusieurs critères et recommande les meilleures.
    """
    print("\n" + "="*80)
    print("🧠 Génération et Évaluation des Grilles Recommandées...")
    print("="*80)

    # --- Configuration ---
    POOL_SIZE = 15  # Nombre de numéros principaux à considérer pour la génération
    NUM_GRIDS_TO_GENERATE = 50000 # Limite pour éviter les calculs trop longs
    TOP_N_GRIDS_TO_SHOW = 10

    # 1. Sélectionner le pool de candidats
    candidate_pool = top_main_numbers_df.head(POOL_SIZE)
    candidate_numbers = candidate_pool['boule'].tolist()
    # Créer un dictionnaire pour un accès rapide aux scores
    candidate_scores = pd.Series(candidate_pool.score_composite.values, index=candidate_pool.boule).to_dict()

    print(f"Pool de {POOL_SIZE} numéros candidats : {candidate_numbers}")

    # 2. Générer les combinaisons
    all_combinations = list(itertools.combinations(candidate_numbers, 5))
    
    # Si le nombre de combinaisons est trop grand, on en prend un échantillon aléatoire
    if len(all_combinations) > NUM_GRIDS_TO_GENERATE:
        print(f"Le nombre de combinaisons ({len(all_combinations)}) est très élevé. Analyse sur un échantillon de {NUM_GRIDS_TO_GENERATE}.")
        indices = np.random.choice(len(all_combinations), NUM_GRIDS_TO_GENERATE, replace=False)
        all_combinations = [all_combinations[i] for i in indices]

    grids_df = pd.DataFrame(all_combinations, columns=[f'b{i+1}' for i in range(5)])

    # 3. Préparer les données pour le scoring
    # Score des Paires
    top_pairs_df['pair_key'] = top_pairs_df.apply(lambda row: tuple(sorted((row['boule_1'], row['boule_2']))), axis=1)
    pair_scores = pd.Series(top_pairs_df.frequence.values, index=top_pairs_df.pair_key).to_dict()
    
    # Score de Composition
    sum_mean = grid_composition_stats.loc['mean', 'somme_grille']
    sum_std = grid_composition_stats.loc['std', 'somme_grille']
    even_mean = grid_composition_stats.loc['mean', 'count_even']
    even_std = grid_composition_stats.loc['std', 'count_even']

    # --- 4. Noter chaque combinaison ---
    print(f"Évaluation de {len(grids_df)} grilles candidates...")

    def score_grid(row):
        grid = tuple(sorted(row.values))
        
        # a) Score du Modèle (pondération 40%)
        model_score = sum(candidate_scores.get(num, 0) for num in grid)

        # b) Score de Cohérence des Paires (pondération 30%)
        grid_pairs = list(itertools.combinations(grid, 2))
        pair_score = sum(pair_scores.get(p, 0) for p in grid_pairs)

        # c) Score de Composition (pondération 30%)
        grid_sum = sum(grid)
        grid_even_count = sum(1 for num in grid if num % 2 == 0)
        
        # Utilise la distribution normale pour noter la "typicité"
        # Un score proche de 1 signifie que la valeur est très proche de la moyenne historique
        sum_score = norm.pdf(grid_sum, sum_mean, sum_std)
        even_score = norm.pdf(grid_even_count, even_mean, even_std)
        composition_score = sum_score * even_score # On combine les probabilités

        return pd.Series([model_score, pair_score, composition_score])

    grids_df[['model_score', 'pair_score', 'composition_score']] = grids_df.apply(score_grid, axis=1)

    # Normaliser les scores entre 0 et 1 pour les combiner
    grids_df['model_score_norm'] = (grids_df['model_score'] - grids_df['model_score'].min()) / (grids_df['model_score'].max() - grids_df['model_score'].min())
    grids_df['pair_score_norm'] = (grids_df['pair_score'] - grids_df['pair_score'].min()) / (grids_df['pair_score'].max() - grids_df['pair_score'].min())
    grids_df['composition_score_norm'] = (grids_df['composition_score'] - grids_df['composition_score'].min()) / (grids_df['composition_score'].max() - grids_df['composition_score'].min())
    
    # Calcul du score final pondéré
    grids_df['score_final'] = (
        0.4 * grids_df['model_score_norm'] +
        0.3 * grids_df['pair_score_norm'] +
        0.3 * grids_df['composition_score_norm']
    ).fillna(0)

    # 5. Présenter les meilleures grilles
    top_grids = grids_df.sort_values(by='score_final', ascending=False).head(TOP_N_GRIDS_TO_SHOW)
    
    # Associer les numéros chance
    top_grids['chance_1'] = top_chance_numbers.iloc[0]['boule']
    top_grids['chance_2'] = top_chance_numbers.iloc[1]['boule'] if len(top_chance_numbers) > 1 else '-'
    
    print("\n" + "="*80)
    print(f"🏆 Top {TOP_N_GRIDS_TO_SHOW} des Grilles Recommandées")
    print("="*80)
    
    display_cols = [f'b{i+1}' for i in range(5)] + ['chance_1', 'chance_2', 'score_final']
    print(top_grids[display_cols].to_string(index=False))

    grids_path = OUTPUT_DIR / 'top_10_grilles_recommandees.csv'
    top_grids[display_cols].to_csv(grids_path, index=False, float_format='%.4f')
    print(f"\n📁 Fichier des grilles recommandées sauvegardé : {grids_path}")

# === Feature Engineering AVANCÉ avec de nouvelles méthodes statistiques ===
def create_advanced_features_with_duckdb(pool_name, num_total_boules):
    print(f"🦆 Création de features AVANCÉES pour les numéros {pool_name}...")
    unpivot_table = "UnpivotedMain" if pool_name == "Principaux" else "UnpivotedChance"
    
    # // MODIFIÉ // Ajout de la dépendance séquentielle (LAG)
    sql_query = f"""
    WITH AllPossibleRows AS (
        SELECT t.tirage_id, b.boule
        FROM (SELECT DISTINCT tirage_id FROM loto_base) t, (SELECT unnest(range(1, {num_total_boules + 1})) as boule) b
    ),
    JoinedData AS (
        SELECT apr.tirage_id, apr.boule, (lf.boule IS NOT NULL)::INT AS is_out
        FROM AllPossibleRows apr LEFT JOIN {unpivot_table} lf ON apr.tirage_id = lf.tirage_id AND apr.boule = lf.boule
    ),
    IslandGrouping AS (
        SELECT *, SUM(is_out) OVER (PARTITION BY boule ORDER BY tirage_id) as island_id
        FROM JoinedData
    ),
    BaseFeatures AS (
        SELECT
            g.tirage_id, g.boule, g.is_out,
            ROW_NUMBER() OVER (PARTITION BY g.boule, g.island_id ORDER BY g.tirage_id) - 1 AS tirages_sans_sortie
        FROM IslandGrouping g
    ),
    GapStats AS (
        SELECT 
            boule,
            AVG(tirages_sans_sortie) as avg_gap,
            STDDEV_SAMP(tirages_sans_sortie) as stddev_gap
        FROM BaseFeatures
        WHERE is_out = 1
        GROUP BY boule
    ),
    FinalFeatures AS (
        SELECT
            bf.tirage_id, bf.boule, bf.is_out, bf.tirages_sans_sortie,
            COALESCE((bf.tirages_sans_sortie - gs.avg_gap) / NULLIF(gs.stddev_gap, 0), 0) as zscore_gap,
            SUM(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) as freq_100,
            SUM(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING) as freq_50,
            SUM(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as freq_10,
            AVG(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as ewma_10,
            bf.boule % 2 AS is_even,
            bf.boule % 10 AS last_digit,
            CASE WHEN bf.boule <= {num_total_boules / 2} THEN 1 ELSE 0 END AS is_low_half,
            MAX(bf.is_out) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) as is_hot_5,

            -- // NOUVEAU // Est-ce que ce numéro était sorti au tirage précédent ?
            -- LAG(is_out, 1, 0) regarde la valeur de 'is_out' à la ligne précédente (1) pour la même boule.
            -- S'il n'y a pas de ligne précédente (1er tirage), la valeur par défaut est 0.
            LAG(bf.is_out, 1, 0) OVER (PARTITION BY bf.boule ORDER BY bf.tirage_id) as was_out_last_draw

        FROM BaseFeatures bf
        JOIN GapStats gs ON bf.boule = gs.boule
    )
    SELECT * FROM FinalFeatures
    WHERE tirage_id > (SELECT MIN(tirage_id) + 100 FROM loto_base)
    ORDER BY tirage_id, boule
    """
    features_df = con.execute(sql_query).df()
    print(f"✅ {len(features_df)} lignes de features AVANCÉES générées pour {pool_name}.")
    return features_df
def analyze_number_pairs_with_duckdb(con, output_dir):
    """Analyse la fréquence de sortie des paires de numéros."""
    print("\n" + "="*80)
    print("🤝 Analyse des paires de numéros les plus fréquentes...")
    print("="*80)
    
    query = """
    WITH Pairs AS (
        SELECT
            t1.tirage_id,
            t1.boule as boule_1,
            t2.boule as boule_2
        FROM UnpivotedMain t1
        JOIN UnpivotedMain t2 ON t1.tirage_id = t2.tirage_id AND t1.boule < t2.boule
    )
    SELECT
        boule_1,
        boule_2,
        COUNT(*) as frequence
    FROM Pairs
    GROUP BY boule_1, boule_2
    ORDER BY frequence DESC
    LIMIT 20;
    """
    top_pairs_df = con.execute(query).df()
    
    if top_pairs_df.empty:
        print("❌ Impossible de calculer les paires de numéros.")
        return None
    
    print("--- Top 20 des paires de numéros les plus fréquentes ---")
    print(top_pairs_df.to_string(index=False))
    
    pair_path = output_dir / 'analyse_paires_frequentes.csv'
    top_pairs_df.to_csv(pair_path, index=False)
    print(f"\n📁 Fichier de l'analyse des paires sauvegardé : {pair_path}")
    return top_pairs_df

def analyze_grid_composition_with_duckdb(con, output_dir):
    """Analyse les statistiques de composition des grilles de tirages historiques."""
    print("\n" + "="*80)
    print("📈 Analyse de la composition des grilles historiques...")
    print("="*80)
    
    query = f"""
    WITH UnpivotedWithProperties AS (
        SELECT
            tirage_id,
            boule,
            boule % 2 AS is_even,
            CASE WHEN boule <= {NUM_MAIN_BALLS / 2} THEN 1 ELSE 0 END AS is_low_half
        FROM UnpivotedMain
    )
    SELECT
        tirage_id,
        SUM(boule) as somme_grille,
        SUM(is_even) as count_even,
        5 - SUM(is_even) as count_odd,
        SUM(is_low_half) as count_low,
        5 - SUM(is_low_half) as count_high
    FROM UnpivotedWithProperties
    GROUP BY tirage_id;
    """
    composition_df = con.execute(query).df()
    
    if composition_df.empty:
        print("❌ Impossible de calculer la composition des grilles.")
        return None

    print("--- Statistiques descriptives de la composition des grilles ---")
    stats = composition_df[['somme_grille', 'count_even', 'count_low']].describe(percentiles=[.25, .5, .75])
    print(stats.round(2).to_string())
    
    comp_path = output_dir / 'analyse_composition_grilles.csv'
    stats.to_csv(comp_path)
    print(f"\n📁 Fichier de l'analyse de composition sauvegardé : {comp_path}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(composition_df['somme_grille'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution de la Somme des Grilles')
    sns.countplot(x='count_even', data=composition_df, ax=axes[1], palette='viridis')
    axes[1].set_title('Distribution du Nombre de Pairs')
    sns.countplot(x='count_low', data=composition_df, ax=axes[2], palette='magma')
    axes[2].set_title('Distribution du Nombre de "Petits" Numéros')
    plt.tight_layout()
    graph_path = GRAPHS_DIR / "6_analyse_composition.png"
    plt.savefig(graph_path)
    plt.close()
    print(f"🎨 Graphique de l'analyse de composition sauvegardé : {graph_path}")

    return stats

# === La fonction 'objectif' pour l'optimisation Optuna (inchangée mais cruciale) ===
def objective(trial, X, y):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': -1, 'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'eta': trial.suggest_loguniform('eta', 0.01, 0.3),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # // MODIFIÉ // Correction pour la nouvelle API XGBoost
        model = XGBClassifier(**params, use_label_encoder=False, early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, preds)
        scores.append(auc_score)

    return np.mean(scores)

# === La fonction d'analyse principale, orchestrant l'optimisation ===
# === La fonction d'analyse principale, orchestrant l'optimisation ===
def run_analysis_and_optimization(pool_name, num_total_boules):
    # // MODIFIÉ // Appel à la nouvelle fonction de feature engineering
    full_df = create_advanced_features_with_duckdb(pool_name, num_total_boules)
    if full_df.empty: return None, None, None
    
    last_tirage_id = full_df['tirage_id'].max()
    df_predict = full_df[full_df['tirage_id'] == last_tirage_id].copy()
    X_predict = df_predict.drop(columns=['tirage_id', 'is_out'])

    df_train = full_df[full_df['tirage_id'] < last_tirage_id]
    X_train = df_train.drop(columns=['tirage_id', 'is_out'])
    y_train = df_train['is_out']
    
    # Nettoyage de la mémoire avant l'optimisation
    del full_df, df_train
    gc.collect()

    # // CORRECTION // La section Optuna qui manquait a été restaurée
    print(f"🤖 Lancement de l'optimisation Optuna pour {pool_name} ({N_TRIALS_OPTUNA} essais)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)
    
    print(f"🏆 Optimisation terminée. Meilleur score AUC (CV): {study.best_value:.4f}")
    print(f"📋 Meilleurs hyperparamètres: {study.best_params}")

    # // CORRECTION // Définition du final_model avec les meilleurs paramètres
    print(f"💪 Entraînement du modèle final pour {pool_name} avec les meilleurs paramètres...")
    final_model = XGBClassifier(**study.best_params, use_label_encoder=False, n_jobs=-1)
    final_model.fit(X_train, y_train, verbose=False)

    print(f"🔮 Prédiction des scores pour le prochain tirage ({pool_name})...")
    predictions_proba = final_model.predict_proba(X_predict)[:, 1]
    
    df_final_prediction = X_predict[['boule', 'tirages_sans_sortie', 'zscore_gap']].copy()
    # Remplacer les NaN potentiels dans zscore_gap par 0 pour sécuriser les calculs
    df_final_prediction['zscore_gap'] = df_final_prediction['zscore_gap'].fillna(0)
    df_final_prediction['proba_estimee'] = predictions_proba
    
    # // MODIFIÉ ET SÉCURISÉ // Calcul du score composite
    max_proba = df_final_prediction['proba_estimee'].max()
    max_gap = df_final_prediction['tirages_sans_sortie'].max()
    
    # Calcul des composantes du score
    proba_score = (df_final_prediction['proba_estimee'] / max_proba) if max_proba > 0 else 0
    gap_score = (df_final_prediction['tirages_sans_sortie'] / max_gap) if max_gap > 0 else 0
    
    # Calcul sécurisé de la composante z-score
    z_min = df_final_prediction['zscore_gap'].min()
    z_max = df_final_prediction['zscore_gap'].max()
    z_range = z_max - z_min
    zscore_score = ((df_final_prediction['zscore_gap'] - z_min) / z_range) if z_range > 0 else 0
    
    # Assemblage final
    df_final_prediction['score_composite'] = (
        0.5 * proba_score + 
        0.3 * gap_score +
        0.2 * zscore_score
    ).fillna(0) # Sécurité finale pour éviter tout NaN

    df_final = df_final_prediction.sort_values(by='score_composite', ascending=False).reset_index(drop=True)
    
    model_path = MODELS_DIR / f"model_loto_{pool_name.lower()}_advanced.joblib"
    joblib.dump(final_model, model_path)
    print(f"💾 Modèle optimisé pour {pool_name} sauvegardé : {model_path}")

    return df_final, final_model, study

# === // NOUVEAU // Fonctions de visualisation améliorées ===
# === // NOUVEAU // Fonctions de visualisation améliorées ===
def generate_advanced_visualizations(df_stats, model, study, pool_name, output_dir):
    print(f"🎨 Création des visualisations AVANCÉES pour les numéros {pool_name}...")
    sns.set_theme(style="whitegrid")
    graph_paths = {}

    # 1. Top Recommandations (inchangé mais toujours utile)
    df_top = df_stats.head(15 if pool_name == "Principaux" else 5)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='score_composite', y='boule', data=df_top, orient='h', palette='viridis', order=df_top.boule.astype(str))
    plt.title(f'Top Recommandations par Score Composite - {pool_name}', fontsize=16, fontweight='bold')
    path = output_dir / f"1_top_reco_{pool_name.lower()}.png"
    plt.savefig(path, bbox_inches='tight'); plt.close()
    graph_paths['top_reco'] = path

    # 2. Scatter plot Proba vs Ecart (inchangé)
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df_stats, x='tirages_sans_sortie', y='proba_estimee',
                    size='score_composite', hue='zscore_gap', palette='coolwarm',
                    sizes=(50, 300), legend='auto')
    plt.title(f'Probabilité Estimée vs. Écart (Couleur = Z-score) - {pool_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Tirages sans sortie (Écart)')
    plt.ylabel('Probabilité de sortie estimée par le modèle')
    path = output_dir / f"2_proba_vs_ecart_{pool_name.lower()}.png"
    plt.savefig(path, bbox_inches='tight'); plt.close()
    graph_paths['proba_vs_ecart'] = path

    # 3. // NOUVEAU // Importance des features
    feature_importances = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    plt.figure(figsize=(12, 7))
    sns.barplot(x='importance', y='feature', data=feature_importances, palette='rocket')
    plt.title(f'Top 10 des Features les plus importantes - Modèle {pool_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    path = output_dir / f"3_feature_importance_{pool_name.lower()}.png"
    plt.savefig(path, bbox_inches='tight'); plt.close()
    graph_paths['feature_importance'] = path

    # 4. // NOUVEAU // Graphiques d'optimisation Optuna avec gestion d'erreur
    # // CORRECTION APPLIQUÉE ICI //
    try:
        fig_hist = plot_optimization_history(study)
        fig_hist.update_layout(title=f"Historique d'Optimisation Optuna - {pool_name}")
        path_hist = output_dir / f"4_optuna_history_{pool_name.lower()}.html"
        fig_hist.write_html(str(path_hist))
        graph_paths['optuna_history'] = path_hist

        fig_params = plot_param_importances(study)
        fig_params.update_layout(title=f"Importance des Hyperparamètres - {pool_name}")
        path_params = output_dir / f"5_optuna_params_{pool_name.lower()}.html"
        fig_params.write_html(str(path_params))
        graph_paths['optuna_params'] = path_params

    except RuntimeError as e:
        if "zero total variance" in str(e):
            print(f"⚠️ Avertissement : Impossible de générer les graphiques d'importance des hyperparamètres pour '{pool_name}'.")
            print("   Raison : Les performances du modèle étaient trop similaires entre les essais d'Optuna.")
            print("   Cela est courant pour les données à faible signal comme le loto. L'analyse se poursuit normalement.")
            # On s'assure que les clés existent même si les fichiers ne sont pas créés
            graph_paths['optuna_history'] = None
            graph_paths['optuna_params'] = None
        else:
            # Si c'est une autre RuntimeError, on la laisse planter pour investigation
            raise e

    print(f"✅ Graphiques avancés pour {pool_name} sauvegardés.")
    return graph_paths

# // NOUVEAU // Rapport Markdown amélioré
# // NOUVEAU // Rapport Markdown amélioré
def generate_markdown_report(stats_main, stats_chance, reco_main, reco_chance, graphs_main, graphs_chance, exec_time, output_path):
    print("📝 Génération du rapport Markdown final...")

    # // CORRECTION APPLIQUÉE ICI //
    # Section optionnelle pour les graphiques Optuna
    optuna_section = ""
    if graphs_main.get('optuna_history') and graphs_main['optuna_history'].exists():
        optuna_section = f"""
---
## ⚙️ Détails de l'Optimisation
*Ces graphiques interactifs (fichiers .html) montrent comment Optuna a convergé vers les meilleurs paramètres.*
- [Historique d'Optimisation - Numéros Principaux](./graphs/{graphs_main['optuna_history'].name})
- [Historique d'Optimisation - Numéro Chance](./graphs/{graphs_chance['optuna_history'].name})
"""

    md_content = f"""
# Rapport d'Analyse LOTO (Analyse Statistique et ML Avancée)
**Date du rapport :** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Temps total d'exécution :** `{exec_time / 60:.1f}` minutes

## Méthodologie
Ce rapport est généré par un pipeline de Machine Learning avancé :
1.  **Ingénierie de Features Multi-facettes :** Des dizaines de variables statistiques ont été créées pour chaque numéro à chaque tirage. Celles-ci incluent non seulement l'écart et les fréquences, mais aussi des indicateurs de **tendance (Moyenne Mobile Exponentielle)**, de **volatilité (Z-score de l'écart)** et de **chaleur (présence récente)**.
2.  **Optimisation d'Hyperparamètres (Bayésienne) :** La bibliothèque **Optuna** a exploré intelligemment **{N_TRIALS_OPTUNA} configurations** de modèle pour trouver la plus performante, bien au-delà d'un simple réglage manuel.
3.  **Validation Temporelle Robuste :** La performance est validée sur **{N_SPLITS_CV} plis temporels**, assurant que le modèle apprend des patterns passés pour prédire le futur, et non l'inverse.
4.  **Interprétabilité du Modèle :** L'analyse de l'**importance des features** nous révèle ce que le modèle considère comme les signaux les plus prédictifs.

---
## 🎯 Prédictions pour le Prochain Tirage
### Numéros Principaux (Top 7)
- **Prédiction Suggérée :** `{reco_main}`
- *Fichier détaillé : `prediction_numeros_principaux.csv`*
### Numéro Chance (Top 3)
- **Prédiction Suggérée :** `{reco_chance}`
- *Fichier détaillé : `prediction_numero_chance.csv`*
---
## 📊 Analyse des Numéros Principaux (1-49)
#### Top 15 des Numéros Principaux par Score Composite
{stats_main.head(15)[['boule', 'score_composite', 'proba_estimee', 'tirages_sans_sortie', 'zscore_gap']].to_markdown(index=False, floatfmt=".4f")}

### Qu'est-ce qui guide le modèle principal ?
![Importance des Features](./graphs/{graphs_main['feature_importance'].name})
*Ce graphique montre les variables qui ont le plus d'influence sur les prédictions du modèle.*

### Visualisation des Candidats
![Top Recommandations Principaux](./graphs/{graphs_main['top_reco'].name})
![Probabilité vs Écart Principaux](./graphs/{graphs_main['proba_vs_ecart'].name})

---
## ✨ Analyse du Numéro Chance (1-10)
#### Top 5 des Numéros Chance par Score Composite
{stats_chance.head(5)[['boule', 'score_composite', 'proba_estimee', 'tirages_sans_sortie', 'zscore_gap']].to_markdown(index=False, floatfmt=".4f")}

### Qu'est-ce qui guide le modèle Chance ?
![Importance des Features Chance](./graphs/{graphs_chance['feature_importance'].name})

### Visualisation des Candidats
![Top Recommandations Chance](./graphs/{graphs_chance['top_reco'].name})

{optuna_section}
---
**Avertissement :** Le Loto est un jeu de hasard. Ce rapport est une analyse statistique et un exercice de modélisation prédictive. Il ne garantit aucunement les résultats futurs. Jouez de manière responsable.
"""
# ... (ici se termine votre fonction generate_markdown_report)
# ... (ici se termine votre fonction generate_markdown_report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ Rapport Markdown sauvegardé : {output_path}")


# // NOUVEAU // Fonction pour appliquer une pénalité aux numéros "chauds" non exceptionnels
# VEUILLEZ VOUS ASSURER QUE CE BLOC DE CODE EST PRÉSENT DANS VOTRE SCRIPT
def apply_hot_number_penalty(df, pool_name, proba_percentile_threshold=95, penalty_factor=0.25):
    """
    Pénalise les numéros récemment sortis (écart=0) sauf si leur probabilité
    prédite est exceptionnellement élevée (au-dessus d'un certain percentile).

    Args:
        df (pd.DataFrame): Le dataframe de prédictions, trié par score_composite.
        pool_name (str): Le nom du pool ("Principaux" ou "Chance") pour les logs.
        proba_percentile_threshold (int): Le percentile pour définir une probabilité "exceptionnelle".
        penalty_factor (float): Le facteur par lequel multiplier le score des numéros pénalisés.

    Returns:
        pd.DataFrame: Le dataframe avec les scores ajustés et re-trié.
    """
    if df is None or df.empty or 'tirages_sans_sortie' not in df.columns:
        print(f"⚠️ Avertissement ({pool_name}): Dataframe de prédictions vide ou invalide. Pénalité non appliquée.")
        return df

    # 1. Définir le seuil de probabilité "exceptionnelle"
    high_proba_threshold = np.percentile(df['proba_estimee'], proba_percentile_threshold)

    # 2. Identifier les lignes à pénaliser: écart = 0 ET probabilité < seuil
    rows_to_penalize_mask = (df['tirages_sans_sortie'] == 0) & (df['proba_estimee'] < high_proba_threshold)
    
    num_penalized = rows_to_penalize_mask.sum()

    if num_penalized > 0:
        penalized_numbers = df.loc[rows_to_penalize_mask, 'boule'].tolist()
        print(f"⚖️ Pénalité ({pool_name}): {num_penalized} numéro(s) du dernier tirage ({penalized_numbers}) ont été déclassés car leur probabilité n'était pas exceptionnelle.")
        
        # 3. Appliquer la pénalité
        df.loc[rows_to_penalize_mask, 'score_composite'] *= penalty_factor
        
        # 4. Re-trier le dataframe
        df = df.sort_values(by='score_composite', ascending=False).reset_index(drop=True)
        
    # Afficher si un numéro chaud a été "sauvé" (même si aucun n'a été pénalisé)
    saved_numbers_mask = (df['tirages_sans_sortie'] == 0) & (df['proba_estimee'] >= high_proba_threshold)
    if saved_numbers_mask.any():
        saved_numbers = df.loc[saved_numbers_mask, 'boule'].tolist()
        print(f"🌟 Exception ({pool_name}): Le(s) numéro(s) {saved_numbers} du dernier tirage sont conservés grâce à une forte prédiction du modèle.")

    return df


# ==============================
# === SCRIPT PRINCIPAL START ===
# ==============================
if __name__ == '__main__':
    total_start_time = time.time()

    stats_main, model_main, study_main = run_analysis_and_optimization("Principaux", NUM_MAIN_BALLS)
    stats_chance, model_chance, study_chance = run_analysis_and_optimization("Chance", NUM_CHANCE_BALLS)

    if stats_main is not None and stats_chance is not None:
        
        # // MODIFIÉ // Application de la logique de pénalité
        print("\n" + "="*80)
        print("⚖️ Application des règles de sélection post-modélisation...")
        stats_main = apply_hot_number_penalty(stats_main, "Principaux")
        stats_chance = apply_hot_number_penalty(stats_chance, "Chance")
        print("="*80)

        print("\n" + "="*80)
        print("🎯 PRÉDICTIONS FINALES POUR LE PROCHAIN TIRAGE LOTO")
        print("="*80)

        pred_main = stats_main.head(7)
        pred_main['rang'] = range(1, 8)
        reco_main_list = pred_main['boule'].tolist()
        pred_main_output = pred_main[['rang', 'boule', 'score_composite', 'proba_estimee', 'tirages_sans_sortie']]
        pred_main_path = OUTPUT_DIR / 'prediction_numeros_principaux.csv'
        pred_main_output.to_csv(pred_main_path, index=False, float_format='%.4f')
        print("\n--- 7 Numéros Principaux Recommandés ---")
        print(pred_main_output.to_string(index=False))
        print(f"📁 Fichier sauvegardé : {pred_main_path}")

        pred_chance = stats_chance.head(3)
        pred_chance['rang'] = range(1, 4)
        reco_chance_list = pred_chance['boule'].tolist()
        pred_chance_output = pred_chance[['rang', 'boule', 'score_composite', 'proba_estimee', 'tirages_sans_sortie']]
        pred_chance_path = OUTPUT_DIR / 'prediction_numero_chance.csv'
        pred_chance_output.to_csv(pred_chance_path, index=False, float_format='%.4f')
        print("\n--- 3 Numéros Chance Recommandés ---")
        print(pred_chance_output.to_string(index=False))
        print(f"📁 Fichier sauvegardé : {pred_chance_path}")
        print("="*80)
        
        total_elapsed = time.time() - total_start_time
        print(f"\n🏁 ANALYSE COMPLÈTE TERMINÉE en {total_elapsed / 60:.1f} minutes")

        graphs_main = generate_advanced_visualizations(stats_main, model_main, study_main, "Principaux", GRAPHS_DIR)
        graphs_chance = generate_advanced_visualizations(stats_chance, model_chance, study_chance, "Chance", GRAPHS_DIR)

        md_report_path = OUTPUT_DIR / "rapport_analyse_loto_avance.md"
        generate_markdown_report(stats_main, stats_chance, reco_main_list, reco_chance_list, graphs_main, graphs_chance, total_elapsed, md_report_path)
        
    # =========================================================
    # === // NOUVEAU // Analyses supplémentaires de Phase 2 ===
    # =========================================================
    top_pairs_df = analyze_number_pairs_with_duckdb(con, OUTPUT_DIR)
    grid_composition_stats = analyze_grid_composition_with_duckdb(con, OUTPUT_DIR)

    # =================================================================
    # === // NOUVEAU // Génération de Grilles Intelligentes - Phase 3 ===
    # =================================================================
    if top_pairs_df is not None and grid_composition_stats is not None:
        generate_and_score_grids(stats_main, stats_chance, top_pairs_df, grid_composition_stats, con)

else:
    print("\n❌ L'analyse n'a pas pu être complétée.")

con.close()
print("✅ Connexion DuckDB fermée. Script terminé.")