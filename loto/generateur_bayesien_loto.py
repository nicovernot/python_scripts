#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GÉNÉRATEUR DE GRILLES LOTO - APPROCHE BAYÉSIENNE
=================================================
Pipeline:
  1. Chargement des données CSV via DuckDB
  2. Calcul des statistiques fréquentielles (prior Bayésien)
  3. Ingénierie de features temporelles et structurelles
  4. Entraînement du modèle XGBoost (vraisemblance)
  5. Calcul du posterior Bayésien par numéro
  6. Sélection du TOP 10 (numéros avec le plus de chances)
  7. Optimisation PuLP → 6 grilles couvrant au mieux le TOP 10
"""

import sys
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import scipy.stats as stats
import pulp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
CSV_PATH     = SCRIPT_DIR / "loto_data" / "loto_201911.csv"
MODEL_DIR    = SCRIPT_DIR.parent / "boost_models"
MODEL_PATH   = MODEL_DIR / "xgb_bayesian_loto.json"
SCALER_PATH  = MODEL_DIR / "scaler_bayesian_loto.joblib"

MODEL_DIR.mkdir(exist_ok=True)

MAX_BALL     = 49       # numéros de 1 à 49
MAX_CHANCE   = 10       # numéro chance de 1 à 10
N_DRAW       = 5        # boules par tirage
N_TOP        = 10       # numéros retenus pour le TOP
N_GRILLES    = 6        # grilles à générer via PuLP

ZONE_LOW  = range(1,  18)   # 1-17
ZONE_MID  = range(18, 35)   # 18-34
ZONE_HIGH = range(35, 50)   # 35-49

SEPARATOR = "=" * 65


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES VIA DUCKDB
# ---------------------------------------------------------------------------

def load_data_duckdb(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV et retourne un DataFrame propre (boules + numéro chance)."""
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 1 — Chargement des données via DuckDB")
    print(SEPARATOR)

    con = duckdb.connect()

    # Lecture brute
    df_raw = con.execute(f"""
        SELECT *
        FROM read_csv_auto('{csv_path}', delim=';', header=true, ignore_errors=true)
    """).df()

    print(f"  Colonnes disponibles: {list(df_raw.columns[:10])} ...")
    print(f"  Tirages bruts: {len(df_raw)}")

    # Nettoyage & renommage
    col_map = {}
    for c in df_raw.columns:
        cl = c.lower().strip()
        if cl == "boule_1":          col_map[c] = "b1"
        elif cl == "boule_2":        col_map[c] = "b2"
        elif cl == "boule_3":        col_map[c] = "b3"
        elif cl == "boule_4":        col_map[c] = "b4"
        elif cl == "boule_5":        col_map[c] = "b5"
        elif cl == "numero_chance":  col_map[c] = "chance"
        elif cl in ("date_de_tirage", "date_tirage"): col_map[c] = "date"

    df_raw.rename(columns=col_map, inplace=True)

    required = ["b1", "b2", "b3", "b4", "b5", "chance"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après mapping: {missing}. Colonnes: {list(df_raw.columns)}")

    # Conversion numérique
    for col in required:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw.dropna(subset=required, inplace=True)

    # Parsing de la date
    if "date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["date"], dayfirst=True, errors="coerce")
        df_raw.dropna(subset=["date"], inplace=True)
        df_raw.sort_values("date", inplace=True)
        df_raw.reset_index(drop=True, inplace=True)
    else:
        df_raw.reset_index(drop=True, inplace=True)
        df_raw["date"] = pd.date_range(end=datetime.today(), periods=len(df_raw), freq="3D")

    for col in ["b1", "b2", "b3", "b4", "b5", "chance"]:
        df_raw[col] = df_raw[col].astype(int)

    print(f"  Tirages valides: {len(df_raw)}")
    print(f"  Période: {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")

    con.close()
    return df_raw


# ---------------------------------------------------------------------------
# 2. STATISTIQUES BAYÉSIENNES AVEC DUCKDB
# ---------------------------------------------------------------------------

def compute_bayesian_stats_duckdb(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques fréquentielles via DuckDB (prior Bayésien).
    Retourne un dict avec fréquence, retard, momentum pour chaque numéro 1-49.
    """
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 2 — Statistiques Bayésiennes (DuckDB)")
    print(SEPARATOR)

    con = duckdb.connect()
    con.register("draws", df)

    total_draws = len(df)

    # ---- Fréquence absolue (prior) ----------------------------------------
    freq_sql = con.execute("""
        WITH unnested AS (
            SELECT b1 AS num FROM draws
            UNION ALL SELECT b2 FROM draws
            UNION ALL SELECT b3 FROM draws
            UNION ALL SELECT b4 FROM draws
            UNION ALL SELECT b5 FROM draws
        )
        SELECT num, COUNT(*) AS freq
        FROM unnested
        WHERE num BETWEEN 1 AND 49
        GROUP BY num
        ORDER BY num
    """).df()

    # ---- Retard (écart au dernier tirage) -----------------------------------
    # Index de la dernière apparition de chaque numéro
    last_idx_sql = con.execute("""
        WITH unnested AS (
            SELECT row_number() OVER () AS idx, b1 AS num FROM draws
            UNION ALL SELECT row_number() OVER (), b2 FROM draws
            UNION ALL SELECT row_number() OVER (), b3 FROM draws
            UNION ALL SELECT row_number() OVER (), b4 FROM draws
            UNION ALL SELECT row_number() OVER (), b5 FROM draws
        )
        SELECT num, MAX(idx) AS last_idx
        FROM unnested
        WHERE num BETWEEN 1 AND 49
        GROUP BY num
    """).df()

    # ---- Momentum (fréquence récente vs globale) ----------------------------
    recent_n = min(50, total_draws)
    df_recent = df.tail(recent_n).copy()
    con.register("draws_recent", df_recent)

    recent_freq_sql = con.execute("""
        WITH unnested AS (
            SELECT b1 AS num FROM draws_recent
            UNION ALL SELECT b2 FROM draws_recent
            UNION ALL SELECT b3 FROM draws_recent
            UNION ALL SELECT b4 FROM draws_recent
            UNION ALL SELECT b5 FROM draws_recent
        )
        SELECT num, COUNT(*) AS recent_freq
        FROM unnested
        WHERE num BETWEEN 1 AND 49
        GROUP BY num
    """).df()

    # ---- Numéro chance -------------------------------------------------------
    chance_freq_sql = con.execute("""
        SELECT chance AS num, COUNT(*) AS freq
        FROM draws
        WHERE chance BETWEEN 1 AND 10
        GROUP BY chance
        ORDER BY num
    """).df()

    # ---- Co-occurrence (paires les plus fréquentes) -------------------------
    cooc_sql = con.execute("""
        WITH pairs AS (
            SELECT LEAST(b1,b2) AS a, GREATEST(b1,b2) AS b FROM draws
            UNION ALL SELECT LEAST(b1,b3), GREATEST(b1,b3) FROM draws
            UNION ALL SELECT LEAST(b1,b4), GREATEST(b1,b4) FROM draws
            UNION ALL SELECT LEAST(b1,b5), GREATEST(b1,b5) FROM draws
            UNION ALL SELECT LEAST(b2,b3), GREATEST(b2,b3) FROM draws
            UNION ALL SELECT LEAST(b2,b4), GREATEST(b2,b4) FROM draws
            UNION ALL SELECT LEAST(b2,b5), GREATEST(b2,b5) FROM draws
            UNION ALL SELECT LEAST(b3,b4), GREATEST(b3,b4) FROM draws
            UNION ALL SELECT LEAST(b3,b5), GREATEST(b3,b5) FROM draws
            UNION ALL SELECT LEAST(b4,b5), GREATEST(b4,b5) FROM draws
        )
        SELECT a, b, COUNT(*) AS cooc
        FROM pairs
        WHERE a BETWEEN 1 AND 49 AND b BETWEEN 1 AND 49
        GROUP BY a, b
        ORDER BY cooc DESC
        LIMIT 200
    """).df()

    con.close()

    # ---- Assemblage des statistiques par numéro ----------------------------
    freq_dict        = dict(zip(freq_sql["num"],      freq_sql["freq"]))
    last_idx_dict    = dict(zip(last_idx_sql["num"],  last_idx_sql["last_idx"]))
    recent_dict      = dict(zip(recent_freq_sql["num"], recent_freq_sql["recent_freq"]))
    chance_freq_dict = dict(zip(chance_freq_sql["num"], chance_freq_sql["freq"]))

    stats_per_num = {}
    for n in range(1, MAX_BALL + 1):
        freq        = freq_dict.get(n, 0)
        last_idx    = last_idx_dict.get(n, 0)
        recent_freq = recent_dict.get(n, 0)

        # Probabilité empirique (fréquence / tirages)
        p_empirique = freq / (total_draws * N_DRAW) if total_draws > 0 else 1 / MAX_BALL

        # Retard normalisé (0 = apparu au dernier tirage, 1 = très en retard)
        retard = (total_draws - last_idx) / total_draws if total_draws > 0 else 0.5

        # Momentum = rapport fréquence récente / fréquence globale
        expected_recent = recent_n * N_DRAW / MAX_BALL
        momentum = recent_freq / expected_recent if expected_recent > 0 else 1.0

        stats_per_num[n] = {
            "freq"       : freq,
            "p_empirique": p_empirique,
            "retard"     : retard,
            "momentum"   : momentum,
            "recent_freq": recent_freq,
        }

    # ---- Statistiques de co-occurrence par numéro --------------------------
    cooc_score = {n: 0 for n in range(1, MAX_BALL + 1)}
    for _, row in cooc_sql.iterrows():
        cooc_score[int(row["a"])] += row["cooc"]
        cooc_score[int(row["b"])] += row["cooc"]
    max_cooc = max(cooc_score.values()) or 1
    for n in cooc_score:
        cooc_score[n] /= max_cooc

    # Affichage résumé
    top5_freq = sorted(stats_per_num.items(), key=lambda x: x[1]["freq"], reverse=True)[:5]
    top5_ret  = sorted(stats_per_num.items(), key=lambda x: x[1]["retard"], reverse=True)[:5]
    print(f"  Top 5 fréquence  : {[n for n,_ in top5_freq]}")
    print(f"  Top 5 retard     : {[n for n,_ in top5_ret]}")

    return {
        "stats"            : stats_per_num,
        "cooc_score"       : cooc_score,
        "chance_freq"      : chance_freq_dict,
        "total_draws"      : total_draws,
        "recent_n"         : recent_n,
        "top5_cooc"        : cooc_sql.head(10),
    }


# ---------------------------------------------------------------------------
# 3. INGÉNIERIE DE FEATURES POUR LE MODÈLE XGBOOST
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple:
    """
    Construit la matrice X et le vecteur y pour entraîner le modèle XGBoost.

    Pour chaque tirage t, chaque numéro n (1-49) est un exemple:
      - Features: statistiques basées sur l'historique avant t
      - Target: 1 si n apparaît au tirage t, 0 sinon

    Retourne (X, y, feature_names)
    """
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 3 — Construction des features (XGBoost)")
    print(SEPARATOR)

    n_draws = len(df)
    warmup  = 30  # premiers tirages ignorés (pas assez d'historique)

    balls_matrix = df[["b1","b2","b3","b4","b5"]].values  # (n_draws, 5)

    records = []
    windows = [10, 30, 60, 100]

    for t in range(warmup, n_draws):
        history = balls_matrix[:t]                  # tirages passés
        appeared_now = set(balls_matrix[t])         # boules au tirage t

        for n in range(1, MAX_BALL + 1):
            # --- Features de fréquence multi-fenêtres ---
            feats = {}
            for w in windows:
                hist_w = history[-w:] if len(history) >= w else history
                cnt = np.sum(hist_w == n)
                feats[f"freq_{w}"]     = cnt
                feats[f"freq_{w}_pct"] = cnt / (len(hist_w) * N_DRAW) if len(hist_w) > 0 else 0

            # --- Retard (écart depuis dernière apparition) ---
            appearances = np.where(np.any(history == n, axis=1))[0]
            retard = t - appearances[-1] if len(appearances) > 0 else t
            feats["retard"]        = retard
            feats["retard_norm"]   = retard / t if t > 0 else 0

            # --- Momentum (tendance court terme) ---
            freq_10 = feats["freq_10_pct"]
            freq_60 = feats["freq_60_pct"]
            feats["momentum_10_60"] = freq_10 - freq_60  # positif = en hausse

            # --- Zone (low/mid/high) ---
            feats["zone_low"]  = int(n in ZONE_LOW)
            feats["zone_mid"]  = int(n in ZONE_MID)
            feats["zone_high"] = int(n in ZONE_HIGH)

            # --- Parité ---
            feats["is_odd"]    = int(n % 2 == 1)
            feats["is_even"]   = int(n % 2 == 0)

            # --- Numéro normalisé ---
            feats["num_norm"]  = n / MAX_BALL

            # --- Fréquence de co-occurrence avec numéros chauds récents ---
            hot_threshold = 2
            recent_10 = history[-10:] if len(history) >= 10 else history
            hot_nums = set()
            for ball in range(1, MAX_BALL + 1):
                if np.sum(recent_10 == ball) >= hot_threshold:
                    hot_nums.add(ball)
            # Combien de fois n apparaît avec les numéros chauds dans les 30 derniers
            hist_30 = history[-30:] if len(history) >= 30 else history
            cooc_hot = 0
            for row in hist_30:
                if n in row:
                    cooc_hot += len(set(row) & hot_nums)
            feats["cooc_hot"] = cooc_hot

            # --- Target ---
            target = int(n in appeared_now)

            records.append((list(feats.values()), target))

    feature_names = list(records[0][0].__class__.__name__ and
                         {k: None for k in feats.keys()}.keys())
    # Rebuild properly
    feature_names = list(feats.keys())

    X = np.array([r[0] for r in records], dtype=np.float32)
    y = np.array([r[1] for r in records], dtype=np.float32)

    pos_rate = y.mean() * 100
    print(f"  Exemples: {len(X):,} ({pos_rate:.1f}% positifs)")
    print(f"  Features: {len(feature_names)}")

    return X, y, feature_names


# ---------------------------------------------------------------------------
# 4. ENTRAÎNEMENT DU MODÈLE XGBOOST
# ---------------------------------------------------------------------------

def train_xgboost_model(X: np.ndarray, y: np.ndarray,
                        feature_names: list,
                        retrain: bool = False) -> xgb.XGBClassifier:
    """
    Entraîne ou charge le modèle XGBoost (boost model).
    Utilise TimeSeriesSplit pour une validation temporellement correcte.
    """
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 4 — Modèle XGBoost (Boost Model)")
    print(SEPARATOR)

    if MODEL_PATH.exists() and not retrain:
        print(f"  Modèle existant chargé: {MODEL_PATH.name}")
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        return model

    print("  Entraînement du modèle XGBoost...")

    # Validation temporelle (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=3)

    best_params = {
        "n_estimators"     : 400,
        "max_depth"        : 5,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_weight" : 10,
        "gamma"            : 0.1,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "scale_pos_weight" : (y == 0).sum() / max((y == 1).sum(), 1),
        "objective"        : "binary:logistic",
        "eval_metric"      : "logloss",
        "tree_method"      : "hist",
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbosity"        : 0,
    }

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = xgb.XGBClassifier(**best_params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

        preds = m.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, preds)
        cv_scores.append(score)
        print(f"    Fold {fold+1}: log_loss = {score:.4f}")

    print(f"  CV log_loss moyen: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Entraînement final sur tout le dataset
    model = xgb.XGBClassifier(**best_params)
    model.fit(X, y, verbose=False)
    model.save_model(str(MODEL_PATH))
    print(f"  Modèle sauvegardé: {MODEL_PATH}")

    # Importance des features
    importances = model.feature_importances_
    top_feats = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:8]
    print("  Top features:")
    for fname, imp in top_feats:
        bar = "█" * int(imp * 100)
        print(f"    {fname:25s} {bar} {imp:.4f}")

    return model


# ---------------------------------------------------------------------------
# 5. CALCUL DU POSTERIOR BAYÉSIEN
# ---------------------------------------------------------------------------

def compute_bayesian_posterior(
        df: pd.DataFrame,
        model: xgb.XGBClassifier,
        feature_names: list,
        bayes_stats: dict) -> np.ndarray:
    """
    Calcule le posterior Bayésien pour chaque numéro (1-49).

    P(n | history) ∝ P(history | n) × P(n)

    - Prior P(n)       = fréquence empirique lissée (lissage de Laplace)
    - Vraisemblance    = prédiction XGBoost sur les features du DERNIER tirage
    - Posterior        = prior × vraisemblance (normalisé)

    Retourne un vecteur de 49 probabilités (index 0 = numéro 1).
    """
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 5 — Posterior Bayésien")
    print(SEPARATOR)

    n_draws      = len(df)
    stats_per_num = bayes_stats["stats"]
    total_draws   = bayes_stats["total_draws"]
    cooc_score    = bayes_stats["cooc_score"]

    balls_matrix = df[["b1","b2","b3","b4","b5"]].values

    # ---- Prior Bayésien (fréquence + lissage de Laplace) -------------------
    alpha = 1.0  # hyper-paramètre de lissage
    prior = np.zeros(MAX_BALL)
    for n in range(1, MAX_BALL + 1):
        freq = stats_per_num[n]["freq"]
        prior[n-1] = (freq + alpha) / (total_draws * N_DRAW + alpha * MAX_BALL)

    # ---- Vraisemblance : prédiction du modèle sur les features actuelles ---
    windows = [10, 30, 60, 100]
    history = balls_matrix  # tout l'historique = "état actuel"

    feature_rows = []
    for n in range(1, MAX_BALL + 1):
        feats = {}
        for w in windows:
            hist_w = history[-w:] if len(history) >= w else history
            cnt = np.sum(hist_w == n)
            feats[f"freq_{w}"]     = cnt
            feats[f"freq_{w}_pct"] = cnt / (len(hist_w) * N_DRAW) if len(hist_w) > 0 else 0

        appearances = np.where(np.any(history == n, axis=1))[0]
        retard = n_draws - appearances[-1] if len(appearances) > 0 else n_draws
        feats["retard"]        = retard
        feats["retard_norm"]   = retard / n_draws if n_draws > 0 else 0

        freq_10 = feats["freq_10_pct"]
        freq_60 = feats["freq_60_pct"]
        feats["momentum_10_60"] = freq_10 - freq_60

        feats["zone_low"]  = int(n in ZONE_LOW)
        feats["zone_mid"]  = int(n in ZONE_MID)
        feats["zone_high"] = int(n in ZONE_HIGH)
        feats["is_odd"]    = int(n % 2 == 1)
        feats["is_even"]   = int(n % 2 == 0)
        feats["num_norm"]  = n / MAX_BALL

        hot_threshold = 2
        recent_10 = history[-10:] if len(history) >= 10 else history
        hot_nums = set()
        for ball in range(1, MAX_BALL + 1):
            if np.sum(recent_10 == ball) >= hot_threshold:
                hot_nums.add(ball)
        hist_30 = history[-30:] if len(history) >= 30 else history
        cooc_hot = 0
        for row in hist_30:
            if n in row:
                cooc_hot += len(set(row) & hot_nums)
        feats["cooc_hot"] = cooc_hot

        feature_rows.append(list(feats.values()))

    X_pred = np.array(feature_rows, dtype=np.float32)

    # Vraisemblance = proba XGBoost que le numéro sorte
    likelihood = model.predict_proba(X_pred)[:, 1]  # shape (49,)

    # ---- Posterior non normalisé -------------------------------------------
    # Intégration du score de co-occurrence comme terme supplémentaire
    cooc_arr = np.array([cooc_score.get(n, 0) for n in range(1, MAX_BALL + 1)])

    # Posterior = prior × likelihood × (1 + cooc_bonus)
    cooc_weight = 0.15
    posterior_raw = prior * likelihood * (1.0 + cooc_weight * cooc_arr)

    # ---- Normalisation (distribution de probabilité) -----------------------
    posterior = posterior_raw / posterior_raw.sum()

    # Affichage diagnostique
    top10_idx = np.argsort(posterior)[::-1][:10]
    print("  Posterior Bayésien — TOP 10 numéros:")
    print(f"  {'Num':>4} | {'Prior':>8} | {'Vraisemb.':>10} | {'Posterior':>10}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    for idx in top10_idx:
        n = idx + 1
        print(f"  {n:>4} | {prior[idx]:>8.5f} | {likelihood[idx]:>10.5f} | {posterior[idx]:>10.5f}")

    return posterior


# ---------------------------------------------------------------------------
# 6. SÉLECTION DU TOP 10
# ---------------------------------------------------------------------------

def select_top10(posterior: np.ndarray,
                 bayes_stats: dict) -> list:
    """
    Sélectionne les 10 numéros avec le posterior le plus élevé.
    Affiche un rapport complet incluant prior, likelihood, momentum, retard.
    """
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 6 — Sélection du TOP 10")
    print(SEPARATOR)

    top10_idx  = np.argsort(posterior)[::-1][:N_TOP]
    top10_nums = [idx + 1 for idx in top10_idx]

    stats_per_num = bayes_stats["stats"]
    chance_freq   = bayes_stats["chance_freq"]

    print(f"\n  ★ TOP 10 numéros recommandés : {sorted(top10_nums)}")
    print()
    print(f"  {'#':>2}  {'Num':>4} | {'Post.(%)':>9} | {'Fréq.':>7} | {'Retard':>7} | {'Momentum':>9} | Zone")
    print(f"  {'─'*2}  {'─'*4}-+-{'─'*9}-+-{'─'*7}-+-{'─'*7}-+-{'─'*9}-+─────")
    for rank, n in enumerate(top10_nums, 1):
        s     = stats_per_num[n]
        p     = posterior[n-1] * 100
        zone  = "BAS" if n in ZONE_LOW else ("MID" if n in ZONE_MID else "HAUT")
        print(f"  {rank:>2}. {n:>4} | {p:>8.3f}% | {s['freq']:>7} | {s['retard']:>6.2f} | {s['momentum']:>9.3f} | {zone}")

    # Meilleur numéro chance
    best_chance = max(chance_freq.items(), key=lambda x: x[1])[0] if chance_freq else 7
    print(f"\n  Numéro chance recommandé : {best_chance} "
          f"(sorti {chance_freq.get(best_chance, 0)} fois)")

    return top10_nums, best_chance


# ---------------------------------------------------------------------------
# 7. OPTIMISATION PULP — 6 GRILLES OPTIMISÉES
# ---------------------------------------------------------------------------

def _solve_single_grid(posterior: np.ndarray,
                        all_nums: list,
                        low_nums: list,
                        mid_nums: list,
                        high_nums: list,
                        odd_nums: list,
                        previous_grilles: list,
                        grid_id: int) -> list | None:
    """
    Résout un problème PuLP pour UNE seule grille.
    Applique les contraintes de diversité par rapport aux grilles précédentes.
    """
    prob = pulp.LpProblem(f"LotoGrid_{grid_id}", pulp.LpMaximize)

    x = {n: pulp.LpVariable(f"x_{grid_id}_{n}", cat="Binary") for n in all_nums}

    # Objectif : maximiser la somme des posteriors
    prob += pulp.lpSum(posterior[n - 1] * x[n] for n in all_nums)

    # Exactement 5 numéros
    prob += pulp.lpSum(x[n] for n in all_nums) == N_DRAW

    # Équilibre des zones (au moins 1 de chaque zone)
    prob += pulp.lpSum(x[n] for n in low_nums)  >= 1
    prob += pulp.lpSum(x[n] for n in mid_nums)  >= 1
    prob += pulp.lpSum(x[n] for n in high_nums) >= 1

    # Parité : entre 2 et 4 impairs
    prob += pulp.lpSum(x[n] for n in odd_nums) >= 2
    prob += pulp.lpSum(x[n] for n in odd_nums) <= 4

    # Diversité : partage au plus 3 numéros avec chaque grille précédente
    # (= au moins 2 numéros distincts par rapport à chaque grille précédente)
    for prev in previous_grilles:
        prob += pulp.lpSum(x[n] for n in prev) <= N_DRAW - 2

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    if prob.status == 1:
        grille = sorted([n for n in all_nums if pulp.value(x[n]) > 0.5])
        if len(grille) == N_DRAW:
            return grille
    return None


def optimize_grilles_pulp(top10: list,
                           posterior: np.ndarray,
                           best_chance: int,
                           n_grilles: int = N_GRILLES) -> list:
    """
    Génère n_grilles optimisées via PuLP (résolution séquentielle).

    Stratégie séquentielle :
      - Chaque grille est optimisée indépendamment
      - Chaque nouvelle grille doit différer d'au moins 2 numéros des précédentes
      - Contraintes structurelles : zone (BAS/MID/HAUT), parité (2-4 impairs)
      - Objectif : maximiser la somme des posteriors Bayésiens
    """
    print(f"\n{SEPARATOR}")
    print("  ÉTAPE 7 — Optimisation PuLP (6 grilles, séquentiel)")
    print(SEPARATOR)

    all_nums  = list(range(1, MAX_BALL + 1))
    low_nums  = [n for n in all_nums if n in ZONE_LOW]
    mid_nums  = [n for n in all_nums if n in ZONE_MID]
    high_nums = [n for n in all_nums if n in ZONE_HIGH]
    odd_nums  = [n for n in all_nums if n % 2 == 1]

    grilles = []
    for g in range(n_grilles):
        print(f"  Grille {g+1}/{n_grilles}...", end=" ", flush=True)
        grille = _solve_single_grid(
            posterior, all_nums, low_nums, mid_nums, high_nums,
            odd_nums, grilles, g
        )
        if grille:
            grilles.append(grille)
            p_sum = sum(posterior[n-1] for n in grille) * 100
            print(f"OK → {grille}  (posterior {p_sum:.3f}%)")
        else:
            print("échec, fallback glouton")
            grille = _greedy_single(posterior, all_nums, odd_nums,
                                    low_nums, mid_nums, high_nums, grilles)
            grilles.append(grille)

    return grilles


def _greedy_single(posterior: np.ndarray,
                   all_nums: list,
                   odd_nums: list,
                   low_nums: list,
                   mid_nums: list,
                   high_nums: list,
                   previous: list) -> list:
    """Génère une grille gloutonne respectant les contraintes de base."""
    sorted_nums = sorted(all_nums, key=lambda n: -posterior[n-1])
    # Assure au moins 1 par zone
    candidates = []
    zones_covered = {"low": False, "mid": False, "high": False}

    for n in sorted_nums:
        if len(candidates) == N_DRAW:
            break
        zone = "low" if n in low_nums else ("mid" if n in mid_nums else "high")
        candidates.append(n)
        zones_covered[zone] = True

    # Correction zones manquantes
    for zone, nums in [("low", low_nums), ("mid", mid_nums), ("high", high_nums)]:
        if not zones_covered[zone]:
            best_z = max(nums, key=lambda n: posterior[n-1])
            if best_z not in candidates:
                candidates[-1] = best_z  # remplace le moins bon

    return sorted(candidates[:N_DRAW])


# ---------------------------------------------------------------------------
# 8. AFFICHAGE FINAL ET RAPPORT
# ---------------------------------------------------------------------------

def print_final_report(grilles: list,
                        top10: list,
                        best_chance: int,
                        posterior: np.ndarray,
                        bayes_stats: dict):
    """Affiche un rapport formaté des 6 grilles optimisées."""
    print(f"\n{SEPARATOR}")
    print("  RÉSULTAT FINAL — 6 GRILLES OPTIMISÉES")
    print(SEPARATOR)

    total_draws = bayes_stats["total_draws"]
    stats_per_num = bayes_stats["stats"]

    top10_set = set(top10)

    for i, grille in enumerate(grilles, 1):
        top_in = [n for n in grille if n in top10_set]
        p_sum  = sum(posterior[n-1] for n in grille) * 100
        zones  = []
        for n in grille:
            if n in ZONE_LOW:   zones.append("B")
            elif n in ZONE_MID: zones.append("M")
            else:                zones.append("H")
        n_odd = sum(1 for n in grille if n % 2 == 1)

        nums_str = "  ".join(f"{n:>2}" for n in grille)
        zone_str = "/".join(zones)

        print(f"\n  ┌─ Grille {i} {'─'*47}┐")
        print(f"  │  Numéros : {nums_str}  +  Chance: {best_chance:>2}          │")
        print(f"  │  Score posterior  : {p_sum:>6.3f}%                           │")
        print(f"  │  TOP10 couverts   : {top_in}  ({len(top_in)}/5)              │")
        print(f"  │  Zones (B/M/H)    : {zone_str:<8}  |  Impairs: {n_odd}/5        │")
        print(f"  └{'─'*60}┘")

    # Couverture des top10
    print(f"\n  Couverture du TOP 10 sur les 6 grilles:")
    coverage = {n: 0 for n in top10}
    for grille in grilles:
        for n in grille:
            if n in coverage:
                coverage[n] += 1

    for n in sorted(top10):
        bar  = "█" * coverage[n]
        p    = posterior[n-1] * 100
        print(f"    Numéro {n:>2}: {bar:<8} ({coverage[n]}/{N_GRILLES} grilles)  — posterior {p:.3f}%")

    print(f"\n{SEPARATOR}")
    print(f"  Analyse basée sur {total_draws} tirages historiques")
    print(f"  Date d'exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main(retrain: bool = False):
    print(f"\n{'#'*65}")
    print("  GÉNÉRATEUR BAYÉSIEN DE GRILLES LOTO")
    print("  Bayesian Statistics + XGBoost Boost Model + PuLP Optimizer")
    print(f"{'#'*65}")

    # 1. Chargement
    df = load_data_duckdb(CSV_PATH)

    # 2. Statistiques Bayésiennes via DuckDB
    bayes_stats = compute_bayesian_stats_duckdb(df)

    # 3. Features pour XGBoost
    X, y, feature_names = build_features(df)

    # 4. Modèle XGBoost
    model = train_xgboost_model(X, y, feature_names, retrain=retrain)

    # 5. Posterior Bayésien
    posterior = compute_bayesian_posterior(df, model, feature_names, bayes_stats)

    # 6. TOP 10
    top10, best_chance = select_top10(posterior, bayes_stats)

    # 7. Optimisation PuLP → 6 grilles
    grilles = optimize_grilles_pulp(top10, posterior, best_chance, n_grilles=N_GRILLES)

    # 8. Rapport final
    print_final_report(grilles, top10, best_chance, posterior, bayes_stats)

    # Sauvegarde JSON (conversion explicite en types Python natifs)
    output = {
        "date"        : datetime.now().isoformat(),
        "top10"       : [int(n) for n in sorted(top10)],
        "best_chance" : int(best_chance),
        "grilles"     : [[int(n) for n in g] for g in grilles],
        "posteriors"  : {str(n): float(posterior[n-1]) for n in range(1, 50)},
    }
    out_path = SCRIPT_DIR.parent / "loto_output" / "bayesian_grilles.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Résultats sauvegardés : {out_path}")

    return grilles, top10, posterior


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Générateur Bayésien de grilles Loto")
    parser.add_argument("--retrain", action="store_true",
                        help="Force le ré-entraînement du modèle XGBoost")
    args = parser.parse_args()
    main(retrain=args.retrain)
