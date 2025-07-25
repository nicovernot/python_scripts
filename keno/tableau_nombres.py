import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from scipy import stats
from pathlib import Path
import warnings
import multiprocessing as mp
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
N_CPUS = psutil.cpu_count(logical=True)  # Tous les CPU logiques
N_WORKERS = min(N_CPUS, 16)  # Limite pour éviter l'over-subscription
print(f"🚀 Utilisation de {N_WORKERS} workers sur {N_CPUS} CPU disponibles")

# === Configurations améliorées ===
DATA_PATH = os.getenv('DATA_PATH', '/home/nico/projets/keno/keno_data')
LOG_PATH = os.getenv('LOG_PATH', '/home/nico/projets/keno/logs')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', '/home/nico/projets/keno/keno_output')

INPUT_PATH = os.path.join(DATA_PATH, 'keno_202010.csv')
OUTPUT_DIR = Path(OUTPUT_PATH)
MODELS_DIR = OUTPUT_DIR / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres optimisés
N_PAST_TIRAGES = [50, 100, 200, 500]
MIN_SAMPLES_FOR_TRAINING = 5000
CHUNK_SIZE = 100  # Pour traitement par blocs

# === Lecture optimisée avec DuckDB ===
print("📊 Lecture des données avec DuckDB...")
con = duckdb.connect()
# Optimisation DuckDB
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

# Vectorisation des features temporelles
df['day_of_week'] = df['date_de_tirage'].dt.dayofweek.astype('int8')
df['month'] = df['date_de_tirage'].dt.month.astype('int8')
df['quarter'] = df['date_de_tirage'].dt.quarter.astype('int8')
df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
df['hour_type'] = (df['heure_de_tirage'] != 'midi').astype('int8')

df_meta = df[['tirage_id', 'date_de_tirage', 'heure_de_tirage', 'day_of_week', 'month', 'quarter', 'is_weekend', 'hour_type']].copy()

# Colonnes de boules
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


# === NOUVELLES FONCTIONS D'ANALYSE (SUITES ET RÉPÉTITIONS) ===
@jit(nopython=True)
def count_consecutive_pairs(boules):
    """Compte les paires de numéros consécutifs dans un tirage."""
    count = 0
    if len(boules) < 2:
        return 0
    sorted_boules = np.sort(boules)
    for i in range(len(sorted_boules) - 1):
        if sorted_boules[i+1] - sorted_boules[i] == 1:
            count += 1
    return count

@jit(nopython=True)
def count_repetitions(current_boules, previous_boules):
    """Compte combien de numéros du tirage courant étaient dans le tirage précédent."""
    if len(current_boules) == 0 or len(previous_boules) == 0:
        return 0
    previous_set = set(previous_boules)
    count = 0
    for boule in current_boules:
        if boule in previous_set:
            count += 1
    return count
    
@jit(nopython=True, parallel=True)
def calculate_last_seen_numba(all_tirages_in_window, boules_in_window, window_size):
    """Calcul vectorisé de la dernière apparition pour chaque boule de 1 à 70."""
    # Par défaut, le gap est la taille de la fenêtre si la boule n'est pas trouvée
    result = np.full(70, window_size, dtype=np.int32)
    
    # Pour chaque boule possible (de 1 à 70)
    for boule in prange(1, 71):
        # Trouver les indices où la boule apparait dans la fenêtre
        indices = np.where(boules_in_window == boule)[0]
        
        if len(indices) > 0:
            # La dernière apparition est au plus grand indice
            last_pos = indices[-1]
            # Le gap est la distance depuis la fin de la fenêtre
            result[boule-1] = len(boules_in_window) - 1 - last_pos
            
    return result

@jit(nopython=True)
def analyze_number_properties(boules):
    odd_count = 0
    prime_count = 0
    # Liste des nombres premiers jusqu'à 70
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67}
    
    for b in boules:
        if b % 2 != 0:
            odd_count += 1
        if b in primes:
            prime_count += 1
            
    even_count = len(boules) - odd_count
    return odd_count, even_count, prime_count

# === Création de features parallélisée (MODIFIÉE) ===
def create_features_chunk(args):
    """Traite un chunk de tirages en parallèle avec les nouvelles features."""
    chunk_indices, df_long_subset, sorted_ids, n_past_tirages = args
    
    features_chunk = []
    labels_chunk = {n: [] for n in range(1, 71)}
    
    # Conversion en arrays numpy pour la performance
    tirages_array = df_long_subset['tirage_id'].values
    boules_array = df_long_subset['boule'].values
    
    for i in chunk_indices:
        # Définir la fenêtre de tirages et le tirage cible
        window_ids = sorted_ids[i - n_past_tirages:i]
        target_id = sorted_ids[i]
        previous_id = sorted_ids[i-1] # ID du tirage juste avant le target
        
        # Masques pour filtrer les données rapidement
        window_mask = np.isin(tirages_array, window_ids)
        target_mask = tirages_array == target_id
        previous_mask = tirages_array == previous_id

        # Données de la fenêtre
        window_boules = boules_array[window_mask]
        
        # Données du tirage cible et précédent
        target_boules = boules_array[target_mask]
        previous_boules = boules_array[previous_mask]
        
        features = {}
        
        # Comptages simples sur la fenêtre
        unique_boules_counts, counts = np.unique(window_boules, return_counts=True)
        count_dict = dict(zip(unique_boules_counts, counts))
        for n in range(1, 71):
            features[f"count_{n}"] = count_dict.get(n, 0)

        # Dernière apparition (gap) sur la fenêtre
        # NOTE : On passe les boules de la fenêtre, pas tous les tirages
        last_seen = calculate_last_seen_numba(window_ids, window_boules, n_past_tirages)
        for n in range(1, 71):
            features[f"last_seen_{n}"] = last_seen[n-1]
        
        # --- NOUVEAU: Calcul des features de suites et de répétitions ---
        features['consecutive_pairs_count'] = count_consecutive_pairs(target_boules)
        features['repetition_count'] = count_repetitions(target_boules, previous_boules)
        
        # --- NOUVELLE AMÉLIORATION : PROPRIÉTÉS DES NOMBRES ---
        odd_count, even_count, prime_count = analyze_number_properties(target_boules)
        features['odd_count'] = odd_count
        features['even_count'] = even_count
        features['prime_count'] = prime_count

        # Features temporelles du tirage cible
        target_meta_row = df_meta[df_meta['tirage_id'] == target_id].iloc[0]
        
        features.update({
            'tirage_id': target_id,
            'weekday': target_meta_row['day_of_week'],
            'month': target_meta_row['month'],
            'quarter': target_meta_row['quarter'],
            'is_weekend': target_meta_row['is_weekend'],
            'hour_type': target_meta_row['hour_type']
        })
        
        features_chunk.append(features)
        
        # Labels (résultats) du tirage cible
        target_boules_set = set(target_boules)
        for n in range(1, 71):
            labels_chunk[n].append(1 if n in target_boules_set else 0)
    
    return features_chunk, labels_chunk

def create_parallel_dataset(n_past_tirages):
    """Crée le dataset en parallèle"""
    print(f"🔧 Création dataset parallèle (fenêtre {n_past_tirages})...")
    start_time = time.time()
    
    sorted_ids = sorted(df['tirage_id'].unique())
    valid_indices = list(range(n_past_tirages, len(sorted_ids)))
    
    chunk_indices_list = [valid_indices[i:i + CHUNK_SIZE] for i in range(0, len(valid_indices), CHUNK_SIZE)]
    
    all_features = []
    all_labels = {n: [] for n in range(1, 71)}
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        args_list = [(chunk, df_long, sorted_ids, n_past_tirages) for chunk in chunk_indices_list]
        
        futures = [executor.submit(create_features_chunk, args) for args in args_list]
        
        for i, future in enumerate(as_completed(futures)):
            features_chunk, labels_chunk = future.result()
            all_features.extend(features_chunk)
            
            for n in range(1, 71):
                all_labels[n].extend(labels_chunk[n])
            
            if (i + 1) % 5 == 0:
                print(f"   Chunks traités: {i + 1}/{len(futures)}")
    
    elapsed = time.time() - start_time
    print(f"✅ Dataset créé en {elapsed:.1f}s ({len(all_features)} échantillons)")
    
    return pd.DataFrame(all_features), all_labels


# === Entraînement parallèle des modèles ===
def train_single_model(args):
    """Entraîne un modèle pour une boule donnée"""
    boule_num, X_train, y_train, X_test, y_test, X_latest = args
    
    if sum(y_train) < 10:
        return boule_num, None, 0, 0
    
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0,
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': 1,
        'tree_method': 'hist', 'predictor': 'cpu_predictor', 'early_stopping_rounds': 10
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    auc_score = 0
    if len(set(y_test)) > 1:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Utiliser .values pour passer un array numpy, plus sûr
    prediction = model.predict_proba(X_latest.values.reshape(1, -1))[0][1]
    
    return boule_num, model, auc_score, prediction

def train_models_parallel(X, y_dict):
    """Entraîne tous les modèles en parallèle"""
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

    train_args = []
    for n in range(1, 71):
        args = (n, X_train, y_train_dict[n], X_test, y_test_dict[n], X_latest)
        train_args.append(args)
    
    models = {}
    predictions = {}
    scores = []
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(train_single_model, args) for args in train_args]
        
        for i, future in enumerate(as_completed(futures)):
            boule_num, model, auc_score, prediction = future.result()
            
            if model is not None:
                models[boule_num] = model
                predictions[boule_num] = prediction
                if auc_score > 0: scores.append(auc_score)
            
            if (i + 1) % 10 == 0: print(f"   Modèles entraînés: {i + 1}/70")
    
    elapsed = time.time() - start_time
    avg_score = np.mean(scores) if scores else 0
    print(f"✅ {len(models)} modèles entraînés en {elapsed:.1f}s (AUC moyen: {avg_score:.4f})")
    
    return models, predictions, avg_score

# === Calcul parallèle des statistiques ===
def calculate_stats_parallel(boule_nums, best_window_size, best_predictions):
    """Calcule les statistiques pour un groupe de boules"""
    stats_chunk = []
    
    tirage_ids = df['tirage_id'].values
    derniers_tirages = tirage_ids[-best_window_size:]
    total_tirages = len(tirage_ids)
    
    for n in boule_nums:
        boule_data = df_long[df_long['boule'] == n]
        total_sorties = len(boule_data)
        
        tirages_sans_sortie = total_tirages
        derniere_date_str = 'jamais'
        if not boule_data.empty:
            derniere_sortie_id = boule_data['tirage_id'].max()
            # Utiliser np.where pour trouver l'index de la dernière sortie
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
            'ratio_total': round(total_sorties / total_tirages, 4) if total_tirages > 0 else 0,
            'tendance': tendance
        })
    
    return stats_chunk

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
    def save_model(args):
        n, model = args
        joblib.dump(model, MODELS_DIR / f"model_boule_{n:02d}_optimized.joblib")
        return n

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        save_args = list(best_models.items())
        list(executor.map(save_model, save_args))

    # === Calcul parallèle des statistiques finales ===
    print("\n📈 Calcul des statistiques finales...")
    boule_chunks = [list(range(1 + i*10, min(71, 1 + (i+1)*10))) for i in range(7)]

    all_stats = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Utiliser partial pour fixer les arguments qui ne changent pas
        func = partial(calculate_stats_parallel, best_window_size=best_window_size, best_predictions=best_predictions)
        futures = [executor.submit(func, chunk) for chunk in boule_chunks]
        
        for future in as_completed(futures):
            all_stats.extend(future.result())

    # === Score composite et finalisation ===
    df_stats = pd.DataFrame(all_stats).sort_values(by='boule').reset_index(drop=True)
    df_stats['proba_x100'] = df_stats['proba_estimee'] * 100

    # Normalisation pour le score composite (pour éviter les divisions par zéro)
    max_proba = df_stats['proba_estimee'].max()
    max_gap = df_stats['tirages_sans_sortie'].max()

    df_stats['score_composite'] = (
        0.6 * (df_stats['proba_estimee'] / max_proba if max_proba > 0 else 0) +
        0.4 * (df_stats['tirages_sans_sortie'] / max_gap if max_gap > 0 else 0)
    )

    # === Résultats finaux ===
    output_path = OUTPUT_DIR / 'keno_stats_final_report.csv'
    df_final = df_stats.sort_values(by='score_composite', ascending=False)
    df_final.to_csv(output_path, index=False, float_format='%.4f')

    total_elapsed = time.time() - total_start_time
    print(f"\n🏁 ANALYSE TERMINÉE en {total_elapsed / 60:.1f} minutes")
    print(f"📁 Rapport final sauvegardé : {output_path}")

    # === Top recommandations ===
    print(f"\n🎯 TOP 15 BOULES RECOMMANDÉES (Basé sur le score composite):")
    print("="*80)
    top_15 = df_final.head(15)
    for i, (_, row) in enumerate(top_15.iterrows(), 1):
        print(f"{i:2d}. Boule {int(row['boule']):2d} | Score: {row['score_composite']:.3f} | "
              f"Proba: {row['proba_x100']:.1f}% | "
              f"Gap: {int(row['tirages_sans_sortie']):>3d} tirages | "
              f"Tendance: {row['tendance']}")

    # === Recommandations par groupes ===
    print(f"\n📋 STRATÉGIES DE JEU SUGGÉRÉES:")
    print("-" * 50)

    haute_proba = df_final.nlargest(5, 'proba_estimee')['boule'].tolist()
    print(f"🔥 Haute Probabilité (5): {haute_proba}")

    long_gap = df_final.nlargest(5, 'tirages_sans_sortie')['boule'].tolist()
    print(f"⏰ Écart Maximum (5): {long_gap}")

    equilibre = df_final.iloc[5:10]['boule'].tolist()
    print(f"⚖️ Équilibré / Score élevé (5): {equilibre}")

    # Sauvegarde recommandations texte
    reco_path = OUTPUT_DIR / "/home/nico/projets/keno/keno_output/recommandations_finales.txt"
    with open(reco_path, "w") as f:
        f.write(f"KENO - RAPPORT D'ANALYSE\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Temps d'exécution: {total_elapsed / 60:.1f} minutes\n")
        f.write(f"Fenêtre optimale: {best_window_size} tirages\n")
        f.write(f"Score AUC moyen: {best_overall_score:.4f}\n\n")
        
        f.write("--- STRATÉGIES RECOMMANDÉES ---\n")
        f.write(f"Haute probabilité: {haute_proba}\n")
        f.write(f"Écart Maximum: {long_gap}\n")
        f.write(f"Équilibré: {equilibre}\n\n")
        
        f.write("--- TOP 20 DÉTAILLÉ ---\n")
        for i, (_, row) in enumerate(df_final.head(20).iterrows(), 1):
            f.write(f"{i:2d}. Boule {int(row['boule']):<2d} - Score: {row['score_composite']:.3f} | "
                    f"Proba: {row['proba_x100']:.1f}% | Gap: {int(row['tirages_sans_sortie'])}\n")
        
        # Vérifiez si la variable est définie, sinon attribuez une valeur par défaut
        try:
            last_draw_id_in_file
        except NameError:
            last_draw_id_in_file = -1  # Valeur par défaut ou une valeur appropriée

        # Sauvegarde des informations dans le fichier JSON
        json.dump({'last_processed_id': int(last_draw_id_in_file), 'best_window_size': int(best_window_size)}, f)
    
    print(f"💡 Recommandations texte sauvegardées : {reco_path}")