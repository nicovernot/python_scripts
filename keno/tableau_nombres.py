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

# === Configurations améliorées des chemins ===
DATA_PATH = os.getenv('DATA_PATH', './')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './keno_output')

INPUT_PATH = os.path.join(DATA_PATH, 'keno_data/keno_202010.csv')
OUTPUT_DIR = Path(OUTPUT_PATH)
MODELS_DIR = OUTPUT_DIR / 'models'
GRAPHS_DIR = OUTPUT_DIR / 'graphs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


# Paramètres optimisés
N_PAST_TIRAGES = [50, 100, 200, 500]
MIN_SAMPLES_FOR_TRAINING = 1000
CHUNK_SIZE = 1000

# === Lecture optimisée avec DuckDB ===
print("📊 Lecture des données avec DuckDB...")
con = duckdb.connect()
con.execute(f"SET threads TO {N_CPUS}")
con.execute("SET memory_limit = '8GB'")

query = f'''
SELECT * FROM read_csv_auto('{INPUT_PATH}', sep=';', header=True)
ORDER BY CAST(annee_numero_de_tirage AS INTEGER)
'''
df = con.execute(query).df()
con.close()

# === Prétraitement vectorisé ===
print("⚡ Prétraitement vectorisé...")
df['tirage_id'] = df['annee_numero_de_tirage'].astype('int32')
df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], dayfirst=True)

df['day_of_week'] = df['date_de_tirage'].dt.dayofweek.astype('int8')
df['month'] = df['date_de_tirage'].dt.month.astype('int8')
df['quarter'] = df['date_de_tirage'].dt.quarter.astype('int8')
df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
df['hour_type'] = (df['heure_de_tirage'] != 'midi').astype('int8')

df_meta = df[['tirage_id', 'date_de_tirage', 'heure_de_tirage', 'day_of_week', 'month', 'quarter', 'is_weekend', 'hour_type']].copy()

boule_cols = [f"boule{i}" for i in range(1, 21)]

# === Transformation vectorisée en format long ===
print("🔄 Transformation vectorisée...")
df_boules = df[['tirage_id'] + boule_cols]

boules_data = []
for col in boule_cols:
    temp_df = df_boules[['tirage_id', col]].copy()
    temp_df.rename(columns={col: 'boule'}, inplace=True)
    boules_data.append(temp_df)

df_long = pd.concat(boules_data, ignore_index=True)
df_long['boule'] = df_long['boule'].astype('int8')
df_long = df_long.sort_values('tirage_id')

print(f"✅ {len(df_long)} enregistrements créés")


# === Fonctions d'analyse (avec Numba pour la performance) ===
@jit(nopython=True)
def count_consecutive_pairs(boules):
    count = 0
    if len(boules) < 2: return 0
    sorted_boules = np.sort(boules)
    for i in range(len(sorted_boules) - 1):
        if sorted_boules[i+1] - sorted_boules[i] == 1:
            count += 1
    return count

@jit(nopython=True)
def count_repetitions(current_boules, previous_boules):
    if len(current_boules) == 0 or len(previous_boules) == 0: return 0
    previous_set = set(previous_boules)
    count = 0
    for boule in current_boules:
        if boule in previous_set:
            count += 1
    return count
    
@jit(nopython=True, parallel=True)
def calculate_last_seen_numba(all_tirages_in_window, boules_in_window, window_size):
    result = np.full(70, window_size, dtype=np.int32)
    for boule in prange(1, 71):
        indices = np.where(boules_in_window == boule)[0]
        if len(indices) > 0:
            last_pos = indices[-1]
            result[boule-1] = len(boules_in_window) - 1 - last_pos
    return result

@jit(nopython=True)
def analyze_number_properties(boules):
    odd_count = 0
    prime_count = 0
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67}
    for b in boules:
        if b % 2 != 0: odd_count += 1
        if b in primes: prime_count += 1
    even_count = len(boules) - odd_count
    return odd_count, even_count, prime_count

# === Création de features parallélisée ===
def create_features_chunk(args):
    chunk_indices, df_long_subset, sorted_ids, n_past_tirages = args
    features_chunk = []
    labels_chunk = {n: [] for n in range(1, 71)}
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
        unique_boules_counts, counts = np.unique(window_boules, return_counts=True)
        count_dict = dict(zip(unique_boules_counts, counts))
        for n in range(1, 71):
            features[f"count_{n}"] = count_dict.get(n, 0)

        last_seen = calculate_last_seen_numba(window_ids, window_boules, n_past_tirages)
        for n in range(1, 71):
            features[f"last_seen_{n}"] = last_seen[n-1]
        
        features['consecutive_pairs_count'] = count_consecutive_pairs(target_boules)
        features['repetition_count'] = count_repetitions(target_boules, previous_boules)
        
        odd_count, even_count, prime_count = analyze_number_properties(target_boules)
        features['odd_count'] = odd_count
        features['even_count'] = even_count
        features['prime_count'] = prime_count

        target_meta_row = df_meta[df_meta['tirage_id'] == target_id].iloc[0]
        features.update({
            'tirage_id': target_id, 'weekday': target_meta_row['day_of_week'],
            'month': target_meta_row['month'], 'quarter': target_meta_row['quarter'],
            'is_weekend': target_meta_row['is_weekend'], 'hour_type': target_meta_row['hour_type']
        })
        features_chunk.append(features)
        
        target_boules_set = set(target_boules)
        for n in range(1, 71):
            labels_chunk[n].append(1 if n in target_boules_set else 0)
    
    return features_chunk, labels_chunk

def create_parallel_dataset(n_past_tirages):
    print(f"🔧 Création dataset parallèle (fenêtre {n_past_tirages})...")
    start_time = time.time()
    sorted_ids = sorted(df['tirage_id'].unique())
    valid_indices = list(range(n_past_tirages, len(sorted_ids)))
    chunk_indices_list = [valid_indices[i:i + CHUNK_SIZE] for i in range(0, len(valid_indices), CHUNK_SIZE)]
    all_features, all_labels = [], {n: [] for n in range(1, 71)}
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        args_list = [(chunk, df_long, sorted_ids, n_past_tirages) for chunk in chunk_indices_list]
        futures = [executor.submit(create_features_chunk, args) for args in args_list]
        for i, future in enumerate(as_completed(futures)):
            features_chunk, labels_chunk = future.result()
            all_features.extend(features_chunk)
            for n in range(1, 71): all_labels[n].extend(labels_chunk[n])
            if (i + 1) % 5 == 0: print(f"   Chunks traités: {i + 1}/{len(futures)}")
    
    elapsed = time.time() - start_time
    print(f"✅ Dataset créé en {elapsed:.1f}s ({len(all_features)} échantillons)")
    return pd.DataFrame(all_features), all_labels


# === Entraînement parallèle des modèles ===
def train_single_model(args):
    boule_num, X_train, y_train, X_test, y_test, X_latest = args
    if sum(y_train) < 10: return boule_num, None, 0, 0
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0,
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'n_jobs': 1, 'tree_method': 'hist', 'predictor': 'cpu_predictor',
        'early_stopping_rounds': 10
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    auc_score = 0
    if len(set(y_test)) > 1:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    prediction = model.predict_proba(X_latest.values.reshape(1, -1))[0][1]
    return boule_num, model, auc_score, prediction

def train_models_parallel(X, y_dict):
    print("🤖 Entraînement parallèle des modèles...")
    start_time = time.time()
    X_processed = X.drop(columns=['tirage_id'])
    split_idx = int(len(X_processed) * 0.8)
    X_train, X_test = X_processed.iloc[:split_idx], X_processed.iloc[split_idx:]
    X_latest = X_processed.iloc[-1]
    y_train_dict, y_test_dict = {}, {}
    for n in range(1, 71):
        y = y_dict[n]
        y_train_dict[n], y_test_dict[n] = y[:split_idx], y[split_idx:]

    train_args = [(n, X_train, y_train_dict[n], X_test, y_test_dict[n], X_latest) for n in range(1, 71)]
    models, predictions, scores = {}, {}, []
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(train_single_model, args) for args in train_args]
        for i, future in enumerate(as_completed(futures)):
            boule_num, model, auc_score, prediction = future.result()
            if model is not None:
                models[boule_num], predictions[boule_num] = model, prediction
                if auc_score > 0: scores.append(auc_score)
            if (i + 1) % 10 == 0: print(f"   Modèles entraînés: {i + 1}/70")
    
    elapsed = time.time() - start_time
    avg_score = np.mean(scores) if scores else 0
    print(f"✅ {len(models)} modèles entraînés en {elapsed:.1f}s (AUC moyen: {avg_score:.4f})")
    return models, predictions, avg_score

# === Calcul parallèle des statistiques ===
def calculate_stats_parallel(boule_nums, best_window_size, best_predictions):
    stats_chunk = []
    tirage_ids = df['tirage_id'].values
    derniers_tirages = tirage_ids[-best_window_size:]
    total_tirages = len(tirage_ids)
    
    for n in boule_nums:
        boule_data = df_long[df_long['boule'] == n]
        total_sorties = len(boule_data)
        tirages_sans_sortie, derniere_date_str = total_tirages, 'jamais'
        if not boule_data.empty:
            derniere_sortie_id = boule_data['tirage_id'].max()
            derniere_sortie_index = np.where(tirage_ids == derniere_sortie_id)[0]
            if len(derniere_sortie_index) > 0:
                tirages_sans_sortie = total_tirages - derniere_sortie_index[0] - 1
            derniere_date = df_meta[df_meta['tirage_id'] == derniere_sortie_id]['date_de_tirage'].iloc[0]
            derniere_date_str = pd.to_datetime(derniere_date).strftime('%d/%m/%Y')
        
        freq_glissante = np.sum((np.isin(df_long['tirage_id'], derniers_tirages)) & (df_long['boule'] == n))
        mid_point_index = len(derniers_tirages) // 2
        freq_recent = np.sum((np.isin(df_long['tirage_id'], derniers_tirages[mid_point_index:])) & (df_long['boule'] == n))
        freq_older = np.sum((np.isin(df_long['tirage_id'], derniers_tirages[:mid_point_index])) & (df_long['boule'] == n))
        tendance = "hausse" if freq_recent > freq_older else "baisse" if freq_recent < freq_older else "stable"
        
        stats_chunk.append({
            'boule': n, 'proba_estimee': best_predictions.get(n, 0), 'total_sorties': total_sorties,
            'derniere_sortie': derniere_date_str, 'tirages_sans_sortie': tirages_sans_sortie,
            'freq_glissante': freq_glissante, 'moyenne_glissante': round(freq_glissante / best_window_size, 4),
            'ratio_total': round(total_sorties / total_tirages, 4) if total_tirages > 0 else 0, 'tendance': tendance
        })
    return stats_chunk


# === Création des graphiques ===
def generate_visualizations(df_stats, df_long_all_data, output_dir):
    print("🎨 Création des visualisations...")
    sns.set_theme(style="whitegrid")
    graph_paths = {}

    # 1. Fréquence globale
    plt.figure(figsize=(18, 6))
    sns.countplot(x='boule', data=df_long_all_data, palette='viridis')
    plt.title('Fréquence de sortie de chaque numéro (Historique complet)', fontsize=16)
    path = output_dir / "1_frequence_numeros.png"
    plt.savefig(path)
    plt.close()
    graph_paths['frequence'] = path

    # 2. Distribution des écarts
    plt.figure(figsize=(12, 6))
    sns.histplot(df_stats['tirages_sans_sortie'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution des écarts (Tirages depuis la dernière sortie)', fontsize=16)
    path = output_dir / "2_distribution_ecarts.png"
    plt.savefig(path)
    plt.close()
    graph_paths['ecarts'] = path

    # 3. Probabilité vs Écart
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=df_stats, x='tirages_sans_sortie', y='proba_estimee', hue='tendance',
                    palette={'hausse': 'green', 'baisse': 'red', 'stable': 'blue'}, size='score_composite', sizes=(50, 250))
    plt.title('Probabilité estimée vs. Écart actuel', fontsize=16)
    path = output_dir / "3_proba_vs_ecart.png"
    plt.savefig(path)
    plt.close()
    graph_paths['proba_vs_ecart'] = path
    
    # 4. Top 15 recommandations
    df_top15 = df_stats.nlargest(15, 'score_composite')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='score_composite', y='boule', data=df_top15, orient='h', palette='magma', order=df_top15.boule.astype(str))
    plt.title('Top 15 des recommandations (par Score Composite)', fontsize=16)
    plt.ylabel('Numéro de la boule')
    path = output_dir / "4_top15_recommandations.png"
    plt.savefig(path)
    plt.close()
    graph_paths['top15'] = path

    print("✅ Graphiques sauvegardés dans", output_dir)
    return graph_paths


# === Génération du rapport Markdown ===
def generate_markdown_report(df_final_stats, best_window, best_score, exec_time, graph_paths, reco_dict, output_path):
    print("📝 Génération du rapport Markdown...")
    top_20_table = df_final_stats.head(20)[['boule', 'score_composite', 'proba_x100', 'tirages_sans_sortie', 'tendance']].to_markdown(index=False)

    md_content = f"""
# Rapport d'Analyse Keno

**Date du rapport :** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé de l'Analyse

- **Fenêtre d'analyse optimale :** `{best_window}` tirages
- **Performance moyenne des modèles (AUC) :** `{best_score:.4f}`
- **Temps total d'exécution :** `{exec_time / 60:.1f}` minutes

## Prédiction pour le Prochain Tirage

Les 12 numéros les plus susceptibles de sortir au prochain tirage, classés par leur score composite, ont été sauvegardés dans un fichier séparé pour une utilisation facile.

- **Fichier de prédiction : `prediction_prochain_tirage.csv`**

## Stratégies de Jeu Suggérées

- **🔥 Haute Probabilité (Top 5) :** `{reco_dict['haute_proba']}`
- **⏰ Écart Maximum (Top 5) :** `{reco_dict['long_gap']}`
- **⚖️ Équilibré / Score élevé (5 suivants) :** `{reco_dict['equilibre']}`

---

## Top 20 des Numéros Recommandés

Ce tableau classe les numéros selon un **score composite** qui pondère la probabilité de sortie (60%) et l'écart (40%).

{top_20_table}

---

## Visualisations des Données

### Top 15 des Recommandations
![Top 15 des Recommandations](./graphs/{graph_paths['top15'].name})

### Probabilité vs. Écart
![Probabilité vs. Écart](./graphs/{graph_paths['proba_vs_ecart'].name})

### Fréquence Globale des Numéros
![Fréquence des numéros](./graphs/{graph_paths['frequence'].name})

### Distribution des Écarts
![Distribution des Écarts](./graphs/{graph_paths['ecarts'].name})

---
*Rapport généré automatiquement.*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ Rapport Markdown sauvegardé : {output_path}")


# === Boucle principale optimisée ===
print("\n🚀 Démarrage de l'analyse optimisée multi-CPU...")
total_start_time = time.time()

best_window_size = None
best_overall_score = -np.inf
best_models = {}
best_predictions = {}

for window_size in N_PAST_TIRAGES:
    print(f"\n{'='*60}\n🔍 Test fenêtre {window_size} tirages\n{'='*60}")
    
    X, y_dict = create_parallel_dataset(window_size)
    
    if len(X) < MIN_SAMPLES_FOR_TRAINING:
        print(f"❌ Pas assez d'échantillons ({len(X)}) pour fenêtre {window_size}, skip.")
        continue
    
    models, predictions, avg_score = train_models_parallel(X, y_dict)
    
    print(f"📊 Score AUC moyen pour la fenêtre {window_size}: {avg_score:.4f}")
    
    if avg_score > best_overall_score:
        best_overall_score = avg_score
        best_window_size = window_size
        best_models = models.copy()
        best_predictions = predictions.copy()
        print(f"🏆 NOUVELLE MEILLEURE FENÊTRE TROUVÉE!")
    
    del X, y_dict, models, predictions
    gc.collect()

if best_window_size is None:
    print("\n❌ Analyse terminée prématurément. Aucun modèle n'a pu être entraîné.")
else:
    print(f"\n🎯 RÉSULTAT OPTIMAL:")
    print(f"   Meilleure Fenêtre: {best_window_size} tirages")
    print(f"   Meilleur Score AUC moyen: {best_overall_score:.4f}")

    # === Sauvegarde parallèle des modèles ===
    print("\n💾 Sauvegarde des meilleurs modèles...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        save_args = list(best_models.items())
        list(executor.map(lambda args: joblib.dump(args[1], MODELS_DIR / f"model_boule_{args[0]:02d}_optimized.joblib"), save_args))

    # === Calcul parallèle des statistiques finales ===
    print("\n📈 Calcul des statistiques finales...")
    boule_chunks = [list(range(1 + i*10, min(71, 1 + (i+1)*10))) for i in range(7)]
    all_stats = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        func = partial(calculate_stats_parallel, best_window_size=best_window_size, best_predictions=best_predictions)
        futures = [executor.submit(func, chunk) for chunk in boule_chunks]
        for future in as_completed(futures):
            all_stats.extend(future.result())

    # === Score composite et finalisation ===
    df_stats = pd.DataFrame(all_stats).sort_values(by='boule').reset_index(drop=True)
    df_stats['proba_x100'] = df_stats['proba_estimee'] * 100
    
    max_proba = df_stats['proba_estimee'].max()
    max_gap = df_stats['tirages_sans_sortie'].max()
    df_stats['score_composite'] = (
        0.6 * (df_stats['proba_estimee'] / max_proba if max_proba > 0 else 0) +
        0.4 * (df_stats['tirages_sans_sortie'] / max_gap if max_gap > 0 else 0)
    )

    # === Résultats finaux ===
    csv_output_path = OUTPUT_DIR / 'keno_stats_final_report.csv'
    df_final = df_stats.sort_values(by='score_composite', ascending=False)
    df_final.to_csv(csv_output_path, index=False, float_format='%.4f')

    total_elapsed = time.time() - total_start_time
    print(f"\n🏁 ANALYSE TERMINÉE en {total_elapsed / 60:.1f} minutes")
    print(f"📁 Rapport CSV détaillé sauvegardé : {csv_output_path}")

    # === Recommandations par groupes ===
    print(f"\n📋 STRATÉGIES DE JEU SUGGÉRÉES:")
    print("-" * 50)
    reco = {
        'haute_proba': df_final.nlargest(5, 'proba_estimee')['boule'].tolist(),
        'long_gap': df_final.nlargest(5, 'tirages_sans_sortie')['boule'].tolist(),
        'equilibre': df_final.iloc[5:10]['boule'].tolist()
    }
    print(f"🔥 Haute Probabilité (5): {reco['haute_proba']}")
    print(f"⏰ Écart Maximum (5): {reco['long_gap']}")
    print(f"⚖️ Équilibré / Score élevé (5): {reco['equilibre']}")
    
    # === NOUVEAU : Prédiction pour le prochain tirage (sortie CSV) ===
    df_prediction = df_final.head(12).copy()
    df_prediction['rang'] = range(1, 13)
    # Sélection et réorganisation des colonnes pour le fichier de prédiction
    df_prediction = df_prediction[['rang', 'boule', 'score_composite', 'proba_x100', 'tirages_sans_sortie', 'tendance']]
    
    prediction_csv_path = OUTPUT_DIR / 'prediction_prochain_tirage.csv'
    df_prediction.to_csv(prediction_csv_path, index=False, float_format='%.4f')
    
    print("\n" + "="*80)
    print("🎯 PRÉDICTION DES 12 NUMÉROS POUR LE PROCHAIN TIRAGE (Classés par score)")
    print("="*80)
    # Rendre les colonnes numériques entières pour un affichage plus propre
    df_prediction['boule'] = df_prediction['boule'].astype(int)
    df_prediction['tirages_sans_sortie'] = df_prediction['tirages_sans_sortie'].astype(int)
    print(df_prediction.to_string(index=False))
    print(f"\n📁 Prédiction également sauvegardée dans : {prediction_csv_path}")
    print("="*80)


    # === Génération des visualisations et du rapport final ===
    graph_paths = generate_visualizations(df_final, df_long, GRAPHS_DIR)
    
    md_report_path = OUTPUT_DIR / "rapport_analyse_keno.md"
    generate_markdown_report(df_final, best_window_size, best_overall_score, total_elapsed, graph_paths, reco, md_report_path)
    
    # Sauvegarde des informations de la session pour la prochaine exécution
    session_info_path = OUTPUT_DIR / "session_info.json"
    with open(session_info_path, "w") as f:
        last_processed_id = df['tirage_id'].max() if not df.empty else -1
        info = {
            'last_processed_id': int(last_processed_id),
            'best_window_size': int(best_window_size),
            'last_run_date': pd.Timestamp.now().isoformat()
        }
        json.dump(info, f, indent=4)
    print(f"ℹ️ Infos de session sauvegardées : {session_info_path}")