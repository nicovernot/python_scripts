import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import psutil
import time
from numba import jit, prange
import gc
import json
import os

warnings.filterwarnings('ignore')

# === Configuration CPU optimisée ===
N_CPUS = psutil.cpu_count(logical=True)
N_WORKERS = min(N_CPUS, 16)
print(f"🚀 Utilisation de {N_WORKERS} workers sur {N_CPUS} CPU disponibles")

# === NOUVEAU : Paramètres spécifiques au Loto ===
# 49 is the maximum number in the main ball range for LOTO (main balls are numbered from 1 to 49)
NUM_MAIN_BALLS = 49
NUM_CHANCE_BALLS = 10
MAIN_BALL_COLS = [f"boule_{i}" for i in range(1, 6)]
CHANCE_BALL_COL = 'numero_chance'

# === Configurations des chemins ===
DATA_PATH_LOTO = os.getenv('DATA_PATH_LOTO', './loto_data')
OUTPUT_PATH_LOTO = os.getenv('OUTPUT_PATH_LOTO', './loto_output')
INPUT_PATH = os.path.join(DATA_PATH_LOTO, 'loto_201911.csv')

OUTPUT_DIR = Path(OUTPUT_PATH_LOTO)
MODELS_DIR = OUTPUT_DIR / 'models'
GRAPHS_DIR = OUTPUT_DIR / 'graphs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres d'analyse
N_PAST_TIRAGES = [25, 50, 100, 200]
MIN_SAMPLES_FOR_TRAINING = 500
CHUNK_SIZE = 1000

# === Lecture optimisée avec DuckDB ===
print("📊 Lecture des données LOTO avec DuckDB...")
try:
    con = duckdb.connect()
    con.execute(f"SET threads TO {N_CPUS}")
    con.execute("SET memory_limit = '8GB'")

    # Le séparateur est maintenant un point-virgule
    query = f'''
    SELECT * FROM read_csv_auto('{INPUT_PATH}', sep=';', header=True, ignore_errors=true)
    WHERE annee_numero_de_tirage IS NOT NULL
    ORDER BY CAST(annee_numero_de_tirage AS INTEGER)
    '''
    df = con.execute(query).df()
    con.close()
except Exception as e:
    print(f"❌ Erreur lors de la lecture du fichier CSV : {e}")
    print("Veuillez vérifier que le fichier 'loto_201911.csv' existe et est correctement formaté.")
    exit()


# === Prétraitement vectorisé ===
print("⚡ Prétraitement vectorisé...")
df['tirage_id'] = df['annee_numero_de_tirage'].astype('int32')
df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], dayfirst=True)

# Conversion des colonnes de boules en numérique, gestion des erreurs
for col in MAIN_BALL_COLS + [CHANCE_BALL_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=MAIN_BALL_COLS + [CHANCE_BALL_COL], inplace=True)
for col in MAIN_BALL_COLS:
    df[col] = df[col].astype('int8')
df[CHANCE_BALL_COL] = df[CHANCE_BALL_COL].astype('int8')

df['day_of_week'] = df['date_de_tirage'].dt.dayofweek.astype('int8')
df['month'] = df['date_de_tirage'].dt.month.astype('int8')
df['quarter'] = df['date_de_tirage'].dt.quarter.astype('int8')
df['is_weekend'] = df['jour_de_tirage'].str.upper().isin(['SAMEDI']).astype('int8')

df_meta = df[['tirage_id', 'date_de_tirage', 'day_of_week', 'month', 'quarter', 'is_weekend']].copy()

# === Transformation en format long (séparément pour les boules et le numéro chance) ===
print("🔄 Transformation vectorisée...")

# 1. Pour les numéros principaux (1-49)
df_boules_main = df[['tirage_id'] + MAIN_BALL_COLS]
main_data = []
for col in MAIN_BALL_COLS:
    temp_df = df_boules_main[['tirage_id', col]].copy()
    temp_df.rename(columns={col: 'boule'}, inplace=True)
    main_data.append(temp_df)
df_long_main = pd.concat(main_data, ignore_index=True)
df_long_main = df_long_main.sort_values('tirage_id').reset_index(drop=True)
print(f"✅ {len(df_long_main)} enregistrements créés pour les numéros principaux.")

# 2. Pour le numéro chance (1-10)
df_long_chance = df[['tirage_id', CHANCE_BALL_COL]].copy()
df_long_chance.rename(columns={CHANCE_BALL_COL: 'boule'}, inplace=True)
df_long_chance = df_long_chance.sort_values('tirage_id').reset_index(drop=True)
print(f"✅ {len(df_long_chance)} enregistrements créés pour le numéro chance.")


# === Fonctions d'analyse (Numba) - inchangées, mais seront appelées avec des paramètres différents ===
@jit(nopython=True)
def count_consecutive_pairs(boules):
    count = 0
    if len(boules) < 2: return 0
    sorted_boules = np.sort(boules)
    for i in range(len(sorted_boules) - 1):
        if sorted_boules[i+1] - sorted_boules[i] == 1: count += 1
    return count

@jit(nopython=True)
def count_repetitions(current_boules, previous_boules):
    if len(current_boules) == 0 or len(previous_boules) == 0: return 0
    count = 0
    previous_set = set(previous_boules)
    for boule in current_boules:
        if boule in previous_set: count += 1
    return count

@jit(nopython=True, parallel=True)
def calculate_last_seen_numba(boules_in_window, window_size, num_total_boules):
    result = np.full(num_total_boules, window_size, dtype=np.int32)
    for boule in prange(1, num_total_boules + 1):
        indices = np.where(boules_in_window == boule)[0]
        if len(indices) > 0:
            last_pos = indices[-1]
            result[boule-1] = len(boules_in_window) - 1 - last_pos
    return result

# === Fonctions de création de features et d'entraînement (adaptées) ===
def create_features_chunk(args):
    chunk_indices, df_long_subset, sorted_ids, n_past_tirages, num_total_boules = args
    features_chunk = []
    labels_chunk = {n: [] for n in range(1, num_total_boules + 1)}
    tirages_array = df_long_subset['tirage_id'].values
    boules_array = df_long_subset['boule'].values

    for i in chunk_indices:
        window_ids = sorted_ids[i - n_past_tirages:i]
        target_id = sorted_ids[i]
        previous_id = sorted_ids[i-1]

        window_mask = np.isin(tirages_array, window_ids)
        target_mask = tirages_array == target_id
        previous_mask = tirages_array == previous_id

        window_boules = boules_array[window_mask]
        target_boules = boules_array[target_mask]
        previous_boules = boules_array[previous_mask]

        features = {}
        unique_boules, counts = np.unique(window_boules, return_counts=True)
        count_dict = dict(zip(unique_boules, counts))
        for n in range(1, num_total_boules + 1):
            features[f"count_{n}"] = count_dict.get(n, 0)

        last_seen = calculate_last_seen_numba(window_boules, n_past_tirages, num_total_boules)
        for n in range(1, num_total_boules + 1):
            features[f"last_seen_{n}"] = last_seen[n-1]

        features['consecutive_pairs_count'] = count_consecutive_pairs(target_boules)
        features['repetition_count'] = count_repetitions(target_boules, previous_boules)

        target_meta_row = df_meta[df_meta['tirage_id'] == target_id].iloc[0]
        features.update({
            'tirage_id': target_id, 'weekday': target_meta_row['day_of_week'],
            'month': target_meta_row['month'], 'quarter': target_meta_row['quarter'],
            'is_weekend': target_meta_row['is_weekend']
        })
        features_chunk.append(features)

        target_boules_set = set(target_boules)
        for n in range(1, num_total_boules + 1):
            labels_chunk[n].append(1 if n in target_boules_set else 0)

    return features_chunk, labels_chunk

def create_parallel_dataset(df_long_data, n_past_tirages, num_total_boules, pool_name):
    print(f"🔧 Création dataset parallèle pour {pool_name} (fenêtre {n_past_tirages})...")
    start_time = time.time()
    sorted_ids = sorted(df['tirage_id'].unique())
    valid_indices = list(range(n_past_tirages, len(sorted_ids)))
    chunk_indices_list = [valid_indices[i:i + CHUNK_SIZE] for i in range(0, len(valid_indices), CHUNK_SIZE)]
    all_features, all_labels = [], {n: [] for n in range(1, num_total_boules + 1)}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        args_list = [(chunk, df_long_data, sorted_ids, n_past_tirages, num_total_boules) for chunk in chunk_indices_list]
        futures = [executor.submit(create_features_chunk, args) for args in args_list]
        for i, future in enumerate(as_completed(futures)):
            features_chunk, labels_chunk = future.result()
            all_features.extend(features_chunk)
            for n in range(1, num_total_boules + 1): all_labels[n].extend(labels_chunk[n])

    elapsed = time.time() - start_time
    print(f"✅ Dataset {pool_name} créé en {elapsed:.1f}s ({len(all_features)} échantillons)")
    return pd.DataFrame(all_features), all_labels

def train_single_model(args):
    boule_num, X_train, y_train, X_test, y_test, X_latest = args
    if sum(y_train) < 5: return boule_num, None, 0, 0
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 100, 'max_depth': 4,
              'learning_rate': 0.1, 'n_jobs': 1, 'tree_method': 'hist', 'predictor': 'cpu_predictor', 'early_stopping_rounds': 10}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    auc_score = 0
    if len(set(y_test)) > 1:
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    prediction = model.predict_proba(X_latest.values.reshape(1, -1))[0][1]
    return boule_num, model, auc_score, prediction

def train_models_parallel(X, y_dict, num_total_boules, pool_name):
    print(f"🤖 Entraînement parallèle des modèles {pool_name}...")
    start_time = time.time()
    X_processed = X.drop(columns=['tirage_id'])
    split_idx = int(len(X_processed) * 0.8)
    X_train, X_test = X_processed.iloc[:split_idx], X_processed.iloc[split_idx:]
    X_latest = X_processed.iloc[-1]
    y_train_dict, y_test_dict = {}, {}
    for n in range(1, num_total_boules + 1):
        y = y_dict[n]
        y_train_dict[n], y_test_dict[n] = y[:split_idx], y[split_idx:]

    train_args = [(n, X_train, y_train_dict[n], X_test, y_test_dict[n], X_latest) for n in range(1, num_total_boules + 1)]
    models, predictions, scores = {}, {}, []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(train_single_model, args) for args in train_args]
        for i, future in enumerate(as_completed(futures)):
            boule_num, model, auc_score, prediction = future.result()
            if model is not None:
                models[boule_num], predictions[boule_num] = model, prediction
                if auc_score > 0: scores.append(auc_score)

    elapsed = time.time() - start_time
    avg_score = np.mean(scores) if scores else 0
    print(f"✅ {len(models)} modèles {pool_name} entraînés en {elapsed:.1f}s (AUC moyen: {avg_score:.4f})")
    return models, predictions, avg_score

def calculate_stats_parallel(boule_nums, df_long_data, best_window_size, best_predictions):
    stats_chunk = []
    tirage_ids = df['tirage_id'].values
    derniers_tirages = tirage_ids[-best_window_size:]
    total_tirages = len(tirage_ids)

    for n in boule_nums:
        boule_data = df_long_data[df_long_data['boule'] == n]
        total_sorties = len(boule_data)
        tirages_sans_sortie, derniere_date_str = total_tirages, 'jamais'
        if not boule_data.empty:
            derniere_sortie_id = boule_data['tirage_id'].max()
            derniere_sortie_index = np.where(tirage_ids == derniere_sortie_id)[0]
            if len(derniere_sortie_index) > 0:
                tirages_sans_sortie = total_tirages - derniere_sortie_index[0] - 1
            derniere_date = df_meta[df_meta['tirage_id'] == derniere_sortie_id]['date_de_tirage'].iloc[0]
            derniere_date_str = pd.to_datetime(derniere_date).strftime('%d/%m/%Y')

        freq_glissante = np.sum((np.isin(df_long_data['tirage_id'], derniers_tirages)) & (df_long_data['boule'] == n))

        stats_chunk.append({
            'boule': n, 'proba_estimee': best_predictions.get(n, 0), 'total_sorties': total_sorties,
            'derniere_sortie': derniere_date_str, 'tirages_sans_sortie': tirages_sans_sortie,
            'freq_glissante': freq_glissante
        })
    return stats_chunk


# === NOUVEAU : Boucle d'analyse principale encapsulée ===
def run_full_analysis(df_long_data, num_total_boules, pool_name):
    """Exécute l'analyse complète pour une poule de numéros (Principaux ou Chance)."""
    print(f"\n{'='*80}\n🚀 Démarrage de l'analyse pour les Numéros {pool_name} (1 à {num_total_boules})\n{'='*80}")
    
    best_window_size, best_overall_score = None, -np.inf
    best_models, best_predictions = {}, {}

    for window_size in N_PAST_TIRAGES:
        print(f"\n🔍 Test fenêtre {window_size} tirages pour les numéros {pool_name}...")
        X, y_dict = create_parallel_dataset(df_long_data, window_size, num_total_boules, pool_name)

        if len(X) < MIN_SAMPLES_FOR_TRAINING:
            print(f"❌ Pas assez d'échantillons ({len(X)}), skip.")
            continue

        models, predictions, avg_score = train_models_parallel(X, y_dict, num_total_boules, pool_name)
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_window_size = window_size
            best_models = models.copy()
            best_predictions = predictions.copy()
            print(f"🏆 NOUVELLE MEILLEURE FENÊTRE TROUVÉE pour {pool_name}: {best_window_size} (AUC: {avg_score:.4f})")
        
        del X, y_dict, models, predictions; gc.collect()

    if best_window_size is None:
        print(f"\n❌ Analyse pour {pool_name} terminée prématurément. Aucun modèle n'a pu être entraîné.")
        return None, None, None

    # Calcul des statistiques finales
    print(f"\n📈 Calcul des statistiques finales pour les numéros {pool_name}...")
    num_chunks = (num_total_boules + 9) // 10
    boule_chunks = [list(range(1 + i*10, min(num_total_boules + 1, 1 + (i+1)*10))) for i in range(num_chunks)]
    all_stats = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        func = partial(calculate_stats_parallel, df_long_data=df_long_data, best_window_size=best_window_size, best_predictions=best_predictions)
        futures = [executor.submit(func, chunk) for chunk in boule_chunks]
        for future in as_completed(futures):
            all_stats.extend(future.result())

    df_stats = pd.DataFrame(all_stats).sort_values(by='boule').reset_index(drop=True)
    df_stats['proba_x100'] = df_stats['proba_estimee'] * 100
    max_proba = df_stats['proba_estimee'].max()
    max_gap = df_stats['tirages_sans_sortie'].max()
    df_stats['score_composite'] = (0.6 * (df_stats['proba_estimee'] / max_proba if max_proba > 0 else 0) +
                                   0.4 * (df_stats['tirages_sans_sortie'] / max_gap if max_gap > 0 else 0))
    df_final = df_stats.sort_values(by='score_composite', ascending=False)
    
    # Sauvegarde des modèles
    print(f"\n💾 Sauvegarde des meilleurs modèles {pool_name}...")
    for n, model in best_models.items():
        joblib.dump(model, MODELS_DIR / f"model_{pool_name.lower()}_boule_{n:02d}.joblib")

    return df_final, best_window_size, best_overall_score


# === Fonctions de rapport et de visualisation (adaptées) ===
def generate_visualizations(df_stats, df_long_all_data, pool_name, output_dir):
    print(f"🎨 Création des visualisations pour les numéros {pool_name}...")
    sns.set_theme(style="whitegrid")
    graph_paths = {}

    # 1. Fréquence globale
    plt.figure(figsize=(18, 6))
    sns.countplot(x='boule', data=df_long_all_data, palette='viridis', order=sorted(df_long_all_data['boule'].unique()))
    plt.title(f'Fréquence de sortie - Numéros {pool_name}', fontsize=16)
    path = output_dir / f"1_frequence_{pool_name.lower()}.png"
    plt.savefig(path); plt.close()
    graph_paths['frequence'] = path

    # 2. Top Recommandations
    df_top = df_stats.head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='score_composite', y='boule', data=df_top, orient='h', palette='magma', order=df_top.boule.astype(str))
    plt.title(f'Top Recommandations - Numéros {pool_name}', fontsize=16)
    path = output_dir / f"2_top_reco_{pool_name.lower()}.png"
    plt.savefig(path); plt.close()
    graph_paths['top_reco'] = path
    
    return graph_paths

def generate_markdown_report(stats_main, stats_chance, window_main, window_chance, score_main, score_chance, reco_main, reco_chance, graphs_main, graphs_chance, exec_time, output_path):
    print("📝 Génération du rapport Markdown final...")
    
    md_content = f"""
# Rapport d'Analyse LOTO

**Date du rapport :** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Temps total d'exécution :** `{exec_time / 60:.1f}` minutes

---

## 🎯 Prédictions pour le Prochain Tirage

Les numéros les plus probables, classés par leur score composite.

### Numéros Principaux (Top 7)
- **Prédiction :** `{reco_main}`
- *Fichier détaillé : `prediction_numeros_principaux.csv`*

### Numéro Chance (Top 3)
- **Prédiction :** `{reco_chance}`
- *Fichier détaillé : `prediction_numero_chance.csv`*

---

## 📊 Analyse des Numéros Principaux (1-49)

- **Fenêtre d'analyse optimale :** `{window_main}` tirages
- **Performance moyenne des modèles (AUC) :** `{score_main:.4f}`

#### Top 10 des Numéros Principaux par Score
{stats_main.head(10)[['boule', 'score_composite', 'proba_x100', 'tirages_sans_sortie']].to_markdown(index=False)}

![Top Recommandations Principaux](./graphs/{graphs_main['top_reco'].name})
![Fréquence Principaux](./graphs/{graphs_main['frequence'].name})

---

## ✨ Analyse du Numéro Chance (1-10)

- **Fenêtre d'analyse optimale :** `{window_chance}` tirages
- **Performance moyenne des modèles (AUC) :** `{score_chance:.4f}`

#### Top 5 des Numéros Chance par Score
{stats_chance.head(5)[['boule', 'score_composite', 'proba_x100', 'tirages_sans_sortie']].to_markdown(index=False)}

![Top Recommandations Chance](./graphs/{graphs_chance['top_reco'].name})
![Fréquence Chance](./graphs/{graphs_chance['frequence'].name})

---
*Rapport généré automatiquement.*
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ Rapport Markdown sauvegardé : {output_path}")

# ==============================
# === SCRIPT PRINCIPAL START ===
# ==============================
total_start_time = time.time()

# --- Analyse des Numéros Principaux ---
stats_main, window_main, score_main = run_full_analysis(df_long_main, NUM_MAIN_BALLS, "Principaux")

# --- Analyse du Numéro Chance ---
stats_chance, window_chance, score_chance = run_full_analysis(df_long_chance, NUM_CHANCE_BALLS, "Chance")

if stats_main is not None and stats_chance is not None:
    # --- Consolidation et Prédictions Finales ---
    print("\n" + "="*80)
    print("🎯 PRÉDICTIONS FINALES POUR LE PROCHAIN TIRAGE LOTO")
    print("="*80)

    # Prédiction pour les numéros principaux
    pred_main = stats_main.head(7).copy()
    pred_main['rang'] = range(1, 8)
    pred_main_output = pred_main[['rang', 'boule', 'score_composite', 'proba_x100', 'tirages_sans_sortie']]
    pred_main_path = OUTPUT_DIR / 'prediction_numeros_principaux.csv'
    pred_main_output.to_csv(pred_main_path, index=False, float_format='%.4f')
    print("\n--- 7 Numéros Principaux Recommandés ---")
    print(pred_main_output.to_string(index=False))
    print(f"📁 Fichier sauvegardé : {pred_main_path}")

    # Prédiction pour le numéro chance
    pred_chance = stats_chance.head(3).copy()
    pred_chance['rang'] = range(1, 4)
    pred_chance_output = pred_chance[['rang', 'boule', 'score_composite', 'proba_x100', 'tirages_sans_sortie']]
    pred_chance_path = OUTPUT_DIR / 'prediction_numero_chance.csv'
    pred_chance_output.to_csv(pred_chance_path, index=False, float_format='%.4f')
    print("\n--- 3 Numéros Chance Recommandés ---")
    print(pred_chance_output.to_string(index=False))
    print(f"📁 Fichier sauvegardé : {pred_chance_path}")
    print("="*80)

    # --- Génération des Rapports ---
    total_elapsed = time.time() - total_start_time
    print(f"\n🏁 ANALYSE COMPLÈTE TERMINÉE en {total_elapsed / 60:.1f} minutes")

    graphs_main = generate_visualizations(stats_main, df_long_main, "Principaux", GRAPHS_DIR)
    graphs_chance = generate_visualizations(stats_chance, df_long_chance, "Chance", GRAPHS_DIR)

    md_report_path = OUTPUT_DIR / "rapport_analyse_loto.md"
    generate_markdown_report(
        stats_main, stats_chance,
        window_main, window_chance,
        score_main, score_chance,
        pred_main['boule'].tolist(),
        pred_chance['boule'].tolist(),
        graphs_main, graphs_chance,
        total_elapsed, md_report_path
    )
    
    # Sauvegarde des infos de session
    session_info_path = OUTPUT_DIR / "session_info_loto.json"
    with open(session_info_path, "w") as f:
        last_processed_id = df['tirage_id'].max() if not df.empty else -1
        info = {
            'last_processed_id': int(last_processed_id),
            'best_window_main': int(window_main),
            'best_window_chance': int(window_chance),
            'last_run_date': pd.Timestamp.now().isoformat()
        }
        json.dump(info, f, indent=4)
    print(f"ℹ️ Infos de session sauvegardées : {session_info_path}")
else:
    print("\n❌ L'analyse n'a pas pu être complétée. Veuillez vérifier les données d'entrée et les paramètres.")