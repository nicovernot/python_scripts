#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ANALYSEUR KENO STRATÉGIQUE AVEC DUCKDB
==========================================

Script complet pour analyser les données Keno en utilisant DuckDB pour des 
performances optimales. Génère des analyses approfondies, visualisations 
et recommandations stratégiques.

Utilisation:
    python keno/duckdb_keno.py --csv keno_data/extracted/keno_202010.csv --plots --export-stats
    python keno/duckdb_keno.py --auto-consolidated  # Utilise automatiquement le fichier consolidé
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from itertools import combinations
from collections import Counter, defaultdict
import pickle
from functools import lru_cache

# --- Import utilitaires consolidation ---
try:
    from utils_consolidation import load_keno_data, is_consolidated_available, recommend_consolidation
except ImportError:
    print("⚠️  Module utils_consolidation non trouvé - fonctionnalité de consolidation désactivée")
    load_keno_data = None
    is_consolidated_available = None
    recommend_consolidation = None

# --- Dépendances ---
try:
    import pandas as pd
    import numpy as np
    import duckdb
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from scipy import stats
    from scipy.stats import zscore
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les dépendances: pip install pandas numpy duckdb matplotlib seaborn scikit-learn scipy")
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning, module='duckdb')

class KenoAnalyzer:
    """Analyseur de données Keno utilisant DuckDB pour des performances optimales."""
    
    PLOTS_DIR = "keno_analyse_plots"
    STATS_DIR = "keno_stats_exports"
    CACHE_DIR = "cache"
    OUTPUT_DIR = "keno_output"

    def __init__(self, max_number: int = 70, numbers_per_draw: int = 20):
        self.max_number = max_number
        self.numbers_per_draw = numbers_per_draw
        self.NUMBERS = list(range(1, max_number + 1))
        
        # Colonnes des boules Keno (format unifié: b1, b2, ..., b20)
        self.balls_cols = [f'b{i}' for i in range(1, numbers_per_draw + 1)]
        
        # Cache pour optimiser les performances
        self._features_cache = {}
        self._analysis_cache = {}
        
        # Créer les répertoires nécessaires
        for directory in [self.PLOTS_DIR, self.STATS_DIR, self.CACHE_DIR, self.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def load_data_from_csv(self, csv_path: str, db_con: duckdb.DuckDBPyConnection) -> str:
        """Charge et valide les données Keno depuis un fichier CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Fichier non trouvé: {csv_path}")
        
        self.logger.info(f"📁 Chargement et validation de {csv_path}")
        
        try:
            # Essayer le nouveau format d'abord (b1, b2, ..., b20)
            sample_df = pd.read_csv(csv_path, nrows=5)
            new_format_cols = [f'b{i}' for i in range(1, 21)]
            
            if all(col in sample_df.columns for col in new_format_cols):
                # Nouveau format détecté
                self.balls_cols = new_format_cols
                self.logger.info("✅ Format unifié détecté (b1, b2, ..., b20)")
            else:
                # Ancien format - détecter le délimiteur
                sample_df = pd.read_csv(csv_path, nrows=5, sep=';')
                if len(sample_df.columns) < 10:  # Essayer avec ','
                    sample_df = pd.read_csv(csv_path, nrows=5, sep=',')
                
                # Vérifier les colonnes boule1, boule2, etc.
                old_format_cols = [f'boule{i}' for i in range(1, 21)]
                if all(col in sample_df.columns for col in old_format_cols):
                    self.balls_cols = old_format_cols
                    self.logger.info("✅ Ancien format détecté (boule1, boule2, ...)")
                else:
                    # Essayer de détecter automatiquement
                    ball_columns = [col for col in sample_df.columns if 'boule' in col.lower() or col.startswith('b')]
                    if len(ball_columns) >= 20:
                        self.balls_cols = ball_columns[:20]
                        self.logger.info(f"✅ Colonnes boules détectées: {self.balls_cols[:5]}...")
                    else:
                        raise ValueError(f"Impossible de trouver 20 colonnes de boules. Trouvé: {ball_columns}")
            
            # Créer la table dans DuckDB avec détection correcte du délimiteur
            table_name = "keno_historical_data"
            
            # Détecter le bon délimiteur en vérifiant le nombre de colonnes
            if len(sample_df.columns) >= 20:  # Format correct attendu (date + 20 boules)
                delimiter = ','
                self.logger.info(f"✅ Délimiteur ',' détecté ({len(sample_df.columns)} colonnes)")
            else:
                # Essayer avec point-virgule
                test_df = pd.read_csv(csv_path, nrows=1, sep=';')
                if len(test_df.columns) >= 20:
                    delimiter = ';'
                    self.logger.info(f"✅ Délimiteur ';' détecté ({len(test_df.columns)} colonnes)")
                else:
                    delimiter = ','
                    self.logger.warning(f"⚠️ Délimiteur par défaut ',' utilisé ({len(sample_df.columns)} colonnes)")
            
            db_con.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS 
                SELECT * FROM read_csv('{csv_path}', delim='{delimiter}', header=true, auto_detect=true)
            """)
            
            count = db_con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            self.logger.info(f"✅ {count} tirages Keno chargés avec succès")
            
            return table_name
            
        except Exception as e:
            raise Exception(f"❌ Erreur lors du chargement CSV: {e}")

    def _create_base_view(self, db_con, table_name):
        """Crée une vue de données propres et la charge dans un DataFrame."""
        # Créer la vue avec numérotation des tirages
        date_col = self._detect_date_column(db_con, table_name)
        
        balls_list = ', '.join(self.balls_cols)
        null_checks = ' AND '.join(f'{col} IS NOT NULL' for col in self.balls_cols[:5])
        
        sql_query = f"""
            CREATE OR REPLACE TEMPORARY VIEW BaseData AS
            SELECT ROW_NUMBER() OVER (ORDER BY {date_col}) AS draw_index,
                   {date_col} as date_tirage,
                   {balls_list}
            FROM {table_name}
            WHERE {date_col} IS NOT NULL 
            AND {null_checks}
        """
        
        db_con.execute(sql_query)
        
        df_pandas = db_con.table('BaseData').fetchdf()
        
        if df_pandas.empty:
            raise ValueError("❌ Aucune donnée valide trouvée après filtrage")
        
        self.logger.info(f"📊 Analyse de {len(df_pandas)} tirages Keno valides")
        return df_pandas

    def _detect_date_column(self, db_con, table_name):
        """Détecte automatiquement la colonne de date."""
        try:
            # Essayer d'abord avec DESCRIBE car PRAGMA table_info peut être problématique avec CSV
            columns = db_con.execute(f"DESCRIBE {table_name}").fetchall()
            
            for col_info in columns:
                col_name = col_info[0]  # Première colonne = nom
                if any(keyword in col_name.lower() for keyword in ['date', 'jour', 'time']):
                    return col_name
            
            # Par défaut, utiliser la première colonne
            first_col = columns[0][0] if columns else "date"
            return first_col
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de colonne de date: {e}")
            # Fallback vers la colonne date standard
            return "date"

    def analyze_frequencies(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse la fréquence d'apparition de chaque numéro."""
        self.logger.info("📊 Analyse des fréquences des numéros...")
        
        # Créer une requête UNION pour toutes les boules
        union_query = " UNION ALL ".join([
            f"SELECT {col} as numero FROM BaseData WHERE {col} IS NOT NULL"
            for col in self.balls_cols
        ])
        
        query = f"""
            SELECT numero, 
                   COUNT(*) as frequence,
                   COUNT(*) * 100.0 / (SELECT SUM(count) FROM (
                       SELECT COUNT(*) as count FROM ({union_query})
                   )) as pourcentage
            FROM ({union_query})
            WHERE numero BETWEEN 1 AND {self.max_number}
            GROUP BY numero
            ORDER BY frequence DESC
        """
        
        return db_con.execute(query).fetchdf()

    def analyze_delays(self, db_con: duckdb.DuckDBPyConnection, recent_draws: int = 30) -> pd.DataFrame:
        """Analyse les retards (numéros non sortis récemment)."""
        self.logger.info(f"⏰ Analyse des retards sur les {recent_draws} derniers tirages...")
        
        # Pour chaque numéro, trouver sa dernière apparition
        delays = []
        
        for numero in range(1, self.max_number + 1):
            # Trouver la dernière apparition de ce numéro avec la date du tirage
            union_query = " UNION ALL ".join([
                f"SELECT draw_index, date_tirage, {col} as numero FROM BaseData WHERE {col} = {numero}"
                for col in self.balls_cols
            ])
            
            query = f"""
                SELECT MAX(draw_index) as last_appearance_index, 
                       MAX(date_tirage) as last_appearance_date
                FROM ({union_query})
                WHERE draw_index >= (SELECT MAX(draw_index) - {recent_draws} FROM BaseData)
            """
            
            result = db_con.execute(query).fetchone()
            last_appearance_index = result[0] if result[0] else 0
            last_appearance_date = result[1] if result[1] else None
            
            max_index = db_con.execute("SELECT MAX(draw_index) FROM BaseData").fetchone()[0]
            delay = max_index - last_appearance_index if last_appearance_index else recent_draws
            
            delays.append({
                'numero': numero,
                'retard': delay,
                'derniere_apparition_date': last_appearance_date,
                'derniere_apparition_index': last_appearance_index
            })
        
        return pd.DataFrame(delays).sort_values('retard', ascending=False)

    def _generate_zscore_strategy(self, frequencies_df: pd.DataFrame) -> List[int]:
        """Stratégie Z-Score : numéros avec écarts statistiques significatifs."""
        mean_freq = frequencies_df['frequence'].mean()
        std_freq = frequencies_df['frequence'].std()
        
        # Calculer les Z-scores
        frequencies_df = frequencies_df.copy()
        frequencies_df['zscore'] = (frequencies_df['frequence'] - mean_freq) / std_freq
        
        # Sélectionner les numéros avec Z-score extrême (positif ou négatif)
        extreme_scores = frequencies_df[abs(frequencies_df['zscore']) > 1.0]
        return extreme_scores.head(15)['numero'].tolist()
    
    def _generate_fibonacci_strategy(self, frequencies_df: pd.DataFrame) -> List[int]:
        """Stratégie Fibonacci : sélection basée sur la suite de Fibonacci."""
        # Générer la suite de Fibonacci jusqu'à 70
        fib_sequence = [1, 2]  # Commencer par [1, 2] pour éviter les doublons
        while fib_sequence[-1] < 70:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        
        # Filtrer les numéros valides (1-70) et éliminer les doublons
        fib_numbers = list(set([f for f in fib_sequence if 1 <= f <= 70]))
        fib_numbers.sort()  # Trier pour garder l'ordre
        
        # Compléter avec les numéros les plus proches de Fibonacci dans les fréquences
        fib_based = []
        for num in fib_numbers:
            if num in frequencies_df['numero'].values:
                fib_based.append(num)
        
        # Compléter jusqu'à 15 numéros si nécessaire
        remaining = [n for n in frequencies_df['numero'].tolist() if n not in fib_based]
        fib_based.extend(remaining[:15-len(fib_based)])
        
        return fib_based[:15]
    
    def _generate_sectors_strategy(self, frequencies_df: pd.DataFrame) -> List[int]:
        """Stratégie Secteurs : répartition géographique optimale (grille 7x10)."""
        # Diviser en 4 secteurs : haut-gauche, haut-droite, bas-gauche, bas-droite
        sectors = {
            'top_left': list(range(1, 36)),      # Numéros 1-35
            'top_right': list(range(36, 71)),     # Numéros 36-70
            'low': list(range(1, 24)),           # Numéros bas (1-23)
            'high': list(range(47, 71))          # Numéros hauts (47-70)
        }
        
        sector_picks = []
        for sector_name, sector_nums in sectors.items():
            # Prendre les 2-3 meilleurs de chaque secteur
            sector_freq = frequencies_df[frequencies_df['numero'].isin(sector_nums)]
            sector_best = sector_freq.head(3)['numero'].tolist()
            sector_picks.extend(sector_best)
        
        # Éliminer les doublons et limiter à 15
        unique_picks = list(dict.fromkeys(sector_picks))
        return unique_picks[:15]
    
    def _generate_trend_strategy(self, db_con: duckdb.DuckDBPyConnection) -> List[int]:
        """Stratégie Tendance : analyse des évolutions récentes (20 derniers tirages)."""
        # Analyser les 20 derniers tirages
        union_query = " UNION ALL ".join([
            f"SELECT draw_index, {col} as numero FROM BaseData WHERE {col} IS NOT NULL"
            for col in self.balls_cols
        ])
        
        query = f"""
            SELECT numero, COUNT(*) as recent_freq
            FROM ({union_query})
            WHERE draw_index > (SELECT MAX(draw_index) - 20 FROM BaseData)
            AND numero BETWEEN 1 AND {self.max_number}
            GROUP BY numero
            ORDER BY recent_freq DESC, numero ASC
            LIMIT 15
        """
        
        result = db_con.execute(query).fetchdf()
        return result['numero'].tolist() if not result.empty else list(range(1, 16))
    
    def _generate_montecarlo_strategy(self, frequencies_df: pd.DataFrame, delays_df: pd.DataFrame) -> List[int]:
        """Stratégie Monte Carlo : simulation probabiliste."""
        # Normaliser les probabilités
        total_freq = frequencies_df['frequence'].sum()
        frequencies_df = frequencies_df.copy()
        frequencies_df['probability'] = frequencies_df['frequence'] / total_freq
        
        # Ajuster les probabilités selon les retards
        delays_dict = dict(zip(delays_df['numero'], delays_df['retard']))
        
        def calculate_adjusted_prob(row):
            base_prob = row['probability']
            delay = delays_dict.get(row['numero'], 0)
            # Plus le retard est important, plus on augmente la probabilité
            delay_factor = 1 + (delay / 100)  # Facteur d'ajustement
            return base_prob * delay_factor
        
        frequencies_df['adjusted_prob'] = frequencies_df.apply(calculate_adjusted_prob, axis=1)
        
        # Simulation Monte Carlo (10000 tirages virtuels)
        np.random.seed(42)  # Pour la reproductibilité
        monte_carlo_counts = Counter()
        
        for _ in range(10000):
            # Sélectionner selon les probabilités ajustées
            selected = np.random.choice(
                frequencies_df['numero'].values,
                size=min(10, len(frequencies_df)),
                replace=False,
                p=frequencies_df['adjusted_prob'].values / frequencies_df['adjusted_prob'].sum()
            )
            for num in selected:
                monte_carlo_counts[num] += 1
        
        # Retourner les plus sélectionnés
        return [int(num) for num, _ in monte_carlo_counts.most_common(15)]
    
    def _generate_pairs_optimal_strategy(self, pairs_df: pd.DataFrame) -> List[int]:
        """Stratégie Paires Optimales : basée sur les associations fréquentes."""
        if pairs_df.empty:
            return list(range(1, 16))
        
        # Prendre les meilleures paires et extraire tous les numéros
        top_pairs = pairs_df.head(10)
        pair_numbers = set()
        
        for _, row in top_pairs.iterrows():
            pair_numbers.add(int(row['num1']))
            pair_numbers.add(int(row['num2']))
        
        # Compléter avec d'autres numéros populaires si nécessaire
        pair_list = list(pair_numbers)
        if len(pair_list) < 15:
            # Ajouter des numéros qui apparaissent souvent dans les paires
            all_pair_nums = []
            for _, row in pairs_df.head(20).iterrows():
                all_pair_nums.extend([int(row['num1']), int(row['num2'])])
            
            pair_counter = Counter(all_pair_nums)
            for num, _ in pair_counter.most_common():
                if num not in pair_list and len(pair_list) < 15:
                    pair_list.append(num)
        
        return pair_list[:15]
    
    def _generate_zones_balanced_strategy(self, zones_df: pd.DataFrame, frequencies_df: pd.DataFrame) -> List[int]:
        """Stratégie Zones Équilibrées : répartition optimale par zones."""
        if zones_df.empty:
            return frequencies_df.head(15)['numero'].tolist()
        
        # Analyser la répartition moyenne par zone
        zone1_avg = zones_df['zone1_count'].mean()  # 1-23
        zone2_avg = zones_df['zone2_count'].mean()  # 24-46  
        zone3_avg = zones_df['zone3_count'].mean()  # 47-70
        
        # Calculer le ratio optimal pour une grille de 10 numéros
        total_avg = zone1_avg + zone2_avg + zone3_avg
        if total_avg > 0:
            zone1_target = int((zone1_avg / total_avg) * 10)
            zone2_target = int((zone2_avg / total_avg) * 10)
            zone3_target = 10 - zone1_target - zone2_target
        else:
            zone1_target = zone2_target = zone3_target = 3
        
        # Sélectionner les meilleurs numéros de chaque zone
        zone_picks = []
        
        # Zone 1 (1-23)
        zone1_freq = frequencies_df[frequencies_df['numero'].between(1, 23)]
        zone_picks.extend(zone1_freq.head(max(3, zone1_target))['numero'].tolist())
        
        # Zone 2 (24-46)
        zone2_freq = frequencies_df[frequencies_df['numero'].between(24, 46)]
        zone_picks.extend(zone2_freq.head(max(3, zone2_target))['numero'].tolist())
        
        # Zone 3 (47-70)
        zone3_freq = frequencies_df[frequencies_df['numero'].between(47, 70)]
        zone_picks.extend(zone3_freq.head(max(3, zone3_target))['numero'].tolist())
        
        return zone_picks[:15]
    
    def generate_top_30_balanced_numbers(self, frequencies_df: pd.DataFrame, zones_df: pd.DataFrame, 
                                       delays_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les 30 numéros avec le plus de chances de sortir selon la stratégie équilibrée.
        Combine fréquences, retards, zones et paires pour un score optimal.
        """
        self.logger.info("🎯 Génération des 30 numéros optimaux (stratégie équilibrée)...")
        
        # Créer un DataFrame pour tous les numéros (1-70)
        all_numbers = pd.DataFrame({'numero': range(1, 71)})
        
        # === CALCUL DES SCORES INDIVIDUELS ===
        
        # 1. Score de fréquence (normalisé 0-1)
        freq_scores = frequencies_df.set_index('numero')['frequence']
        max_freq = freq_scores.max()
        freq_scores_norm = freq_scores / max_freq if max_freq > 0 else freq_scores
        
        # 2. Score de retard inversé (plus le retard est important, plus le score est élevé)
        delay_scores = delays_df.set_index('numero')['retard']
        max_delay = delay_scores.max()
        delay_scores_norm = delay_scores / max_delay if max_delay > 0 else delay_scores
        
        # 3. Score des paires (moyenne des fréquences des meilleures paires)
        pair_scores = {}
        if not pairs_df.empty:
            # Pour chaque numéro, calculer son score basé sur ses meilleures paires
            for num in range(1, 71):
                num_pairs = pairs_df[
                    (pairs_df['num1'] == num) | (pairs_df['num2'] == num)
                ].head(10)  # Top 10 paires pour ce numéro
                if not num_pairs.empty:
                    pair_scores[num] = num_pairs['frequence'].mean()
                else:
                    pair_scores[num] = 0
            
            pair_scores_series = pd.Series(pair_scores)
            max_pair = pair_scores_series.max()
            pair_scores_norm = pair_scores_series / max_pair if max_pair > 0 else pair_scores_series
        else:
            pair_scores_norm = pd.Series(0, index=range(1, 71))
        
        # 4. Score d'équilibrage par zones
        zone_scores = {}
        if not zones_df.empty:
            zone1_avg = zones_df['zone1_count'].mean()  # 1-23
            zone2_avg = zones_df['zone2_count'].mean()  # 24-46  
            zone3_avg = zones_df['zone3_count'].mean()  # 47-70
            
            total_avg = zone1_avg + zone2_avg + zone3_avg
            if total_avg > 0:
                # Bonus pour les zones sous-représentées
                zone1_weight = 1.0 - (zone1_avg / total_avg)
                zone2_weight = 1.0 - (zone2_avg / total_avg)
                zone3_weight = 1.0 - (zone3_avg / total_avg)
            else:
                zone1_weight = zone2_weight = zone3_weight = 1.0
            
            for num in range(1, 71):
                if 1 <= num <= 23:
                    zone_scores[num] = zone1_weight
                elif 24 <= num <= 46:
                    zone_scores[num] = zone2_weight
                else:  # 47-70
                    zone_scores[num] = zone3_weight
            
            zone_scores_series = pd.Series(zone_scores)
        else:
            zone_scores_series = pd.Series(1.0, index=range(1, 71))
        
        # === CALCUL DU SCORE COMPOSITE ===
        
        # Pondération équilibrée des différents facteurs
        weights = {
            'frequency': 0.30,    # Fréquence historique
            'delay': 0.25,        # Retard (probabilité de sortie)
            'pairs': 0.25,        # Performance en paires
            'zones': 0.20         # Équilibrage des zones
        }
        
        # Calculer le score final pour chaque numéro
        final_scores = {}
        for num in range(1, 71):
            freq_score = freq_scores_norm.get(num, 0)
            delay_score = delay_scores_norm.get(num, 0)
            pair_score = pair_scores_norm.get(num, 0)
            zone_score = zone_scores_series.get(num, 1.0)
            
            final_score = (
                freq_score * weights['frequency'] +
                delay_score * weights['delay'] +
                pair_score * weights['pairs'] +
                zone_score * weights['zones']
            )
            
            final_scores[num] = final_score
        
        # Créer le DataFrame final avec tous les détails
        results_df = pd.DataFrame({
            'numero': range(1, 71),
            'score_composite': [final_scores[num] for num in range(1, 71)],
            'frequence': [freq_scores_norm.get(num, 0) for num in range(1, 71)],
            'score_retard': [delay_scores_norm.get(num, 0) for num in range(1, 71)],
            'score_paires': [pair_scores_norm.get(num, 0) for num in range(1, 71)],
            'score_zones': [zone_scores_series.get(num, 1.0) for num in range(1, 71)],
            'retard_actuel': [delays_df.set_index('numero')['retard'].get(num, 0) for num in range(1, 71)],
            'freq_absolue': [frequencies_df.set_index('numero')['frequence'].get(num, 0) for num in range(1, 71)]
        })
        
        # Trier par score composite décroissant
        results_df = results_df.sort_values('score_composite', ascending=False).reset_index(drop=True)
        
        # Ajouter le classement
        results_df['rang'] = range(1, len(results_df) + 1)
        
        # Ajouter des informations sur les zones
        results_df['zone'] = results_df['numero'].apply(
            lambda x: 'Zone 1 (1-23)' if 1 <= x <= 23 
                     else 'Zone 2 (24-46)' if 24 <= x <= 46 
                     else 'Zone 3 (47-70)'
        )
        
        self.logger.info("✅ Top 30 numéros équilibrés générés avec succès")
        return results_df.head(30)
    
    def export_top_30_to_csv(self, top_30_df: pd.DataFrame) -> str:
        """Exporte les 30 meilleurs numéros vers un fichier CSV avec horodatage."""
        
        # Créer le nom de fichier avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.STATS_DIR}/top_30_numeros_equilibres_{timestamp}.csv"
        
        # Préparer les données pour l'export
        export_df = top_30_df.copy()
        
        # Arrondir les scores pour la lisibilité
        numeric_cols = ['score_composite', 'frequence', 'score_retard', 'score_paires', 'score_zones']
        for col in numeric_cols:
            export_df[col] = export_df[col].round(4)
        
        # Réorganiser les colonnes pour un meilleur affichage
        column_order = [
            'rang', 'numero', 'score_composite', 'zone',
            'frequence', 'score_retard', 'score_paires', 'score_zones',
            'retard_actuel', 'freq_absolue'
        ]
        export_df = export_df[column_order]
        
        # Exporter vers CSV
        export_df.to_csv(filename, index=False, encoding='utf-8', sep=';')
        
        self.logger.info(f"📊 Top 30 numéros exportés vers: {filename}")
        return filename
    
    def analyze_pairs(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse la fréquence des paires de numéros."""
        self.logger.info("🔗 Analyse des paires de numéros...")
        
        # Générer toutes les combinaisons de paires possibles
        pair_queries = []
        for i in range(len(self.balls_cols)):
            for j in range(i + 1, len(self.balls_cols)):
                col1, col2 = self.balls_cols[i], self.balls_cols[j]
                pair_queries.append(f"""
                    SELECT LEAST({col1}, {col2}) AS num1, 
                           GREATEST({col1}, {col2}) AS num2
                    FROM BaseData 
                    WHERE {col1} IS NOT NULL AND {col2} IS NOT NULL
                """)
        
        union_query = " UNION ALL ".join(pair_queries)
        
        query = f"""
            SELECT num1, num2, COUNT(*) as frequence
            FROM ({union_query})
            WHERE num1 BETWEEN 1 AND {self.max_number} 
            AND num2 BETWEEN 1 AND {self.max_number}
            AND num1 < num2
            GROUP BY num1, num2
            ORDER BY frequence DESC
            LIMIT 100
        """
        
        return db_con.execute(query).fetchdf()

    def analyze_sums(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse les sommes des tirages."""
        self.logger.info("🔢 Analyse des sommes des tirages...")
        
        # Calculer la somme de chaque tirage
        sum_cols = " + ".join(self.balls_cols)
        query = f"""
            SELECT draw_index,
                   ({sum_cols}) as somme_tirage
            FROM BaseData
            ORDER BY draw_index
        """
        
        sums_df = db_con.execute(query).fetchdf()
        
        # Créer les statistiques de sommes
        stats_query = f"""
            SELECT somme_tirage, 
                   COUNT(*) as frequence,
                   COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ({query})) as pourcentage
            FROM ({query})
            GROUP BY somme_tirage
            ORDER BY somme_tirage
        """
        
        return db_con.execute(stats_query).fetchdf()

    def analyze_sectors(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse les statistiques de découpage par secteurs."""
        self.logger.info("🗺️ Analyse des secteurs de découpage...")
        
        # Définir les secteurs de découpage
        # Secteur 1: 1-17, Secteur 2: 18-35, Secteur 3: 36-52, Secteur 4: 53-70
        union_query = " UNION ALL ".join([
            f"SELECT draw_index, {col} as numero FROM BaseData WHERE {col} IS NOT NULL"
            for col in self.balls_cols
        ])
        
        query = f"""
            SELECT draw_index,
                   SUM(CASE WHEN numero BETWEEN 1 AND 17 THEN 1 ELSE 0 END) as secteur1_count,
                   SUM(CASE WHEN numero BETWEEN 18 AND 35 THEN 1 ELSE 0 END) as secteur2_count,
                   SUM(CASE WHEN numero BETWEEN 36 AND 52 THEN 1 ELSE 0 END) as secteur3_count,
                   SUM(CASE WHEN numero BETWEEN 53 AND 70 THEN 1 ELSE 0 END) as secteur4_count
            FROM ({union_query})
            WHERE numero BETWEEN 1 AND {self.max_number}
            GROUP BY draw_index
            ORDER BY draw_index
        """
        
        return db_con.execute(query).fetchdf()

    def analyze_numbers_stats(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse les statistiques des nombres (pairs/impairs, hauts/bas, etc.) sur les 30 derniers tirages."""
        self.logger.info("🔢 Analyse des statistiques des nombres...")
        
        # Créer une requête pour analyser les 30 derniers tirages seulement
        union_query = " UNION ALL ".join([
            f"SELECT draw_index, date_tirage, {col} as numero FROM BaseData WHERE {col} IS NOT NULL"
            for col in self.balls_cols
        ])
        
        query = f"""
            SELECT date_tirage,
                   SUM(CASE WHEN numero % 2 = 0 THEN 1 ELSE 0 END) as pairs_count,
                   SUM(CASE WHEN numero % 2 = 1 THEN 1 ELSE 0 END) as impairs_count,
                   SUM(CASE WHEN numero <= 35 THEN 1 ELSE 0 END) as bas_count,
                   SUM(CASE WHEN numero > 35 THEN 1 ELSE 0 END) as hauts_count,
                   SUM(CASE WHEN numero % 10 = 0 THEN 1 ELSE 0 END) as dizaines_count,
                   SUM(CASE WHEN numero % 5 = 0 THEN 1 ELSE 0 END) as multiples_5_count,
                   SUM(CASE WHEN numero IN (1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67) THEN 1 ELSE 0 END) as premiers_count
            FROM ({union_query})
            WHERE numero BETWEEN 1 AND {self.max_number}
            AND draw_index > (SELECT MAX(draw_index) - 30 FROM BaseData)
            GROUP BY date_tirage, draw_index
            ORDER BY draw_index
        """
        
        return db_con.execute(query).fetchdf()

    def analyze_individual_numbers(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse détaillée de chaque numéro individuel (1 à 70)."""
        self.logger.info("🔢 Analyse détaillée des numéros individuels...")
        
        # Analyser chaque numéro individuellement
        numbers_analysis = []
        
        for numero in range(1, self.max_number + 1):
            # Requête pour trouver toutes les apparitions de ce numéro
            union_query = " UNION ALL ".join([
                f"SELECT draw_index, date_tirage, {col} as numero FROM BaseData WHERE {col} = {numero}"
                for col in self.balls_cols
            ])
            
            # Statistiques complètes pour ce numéro
            stats_query = f"""
            WITH appearances AS (
                SELECT draw_index, date_tirage
                FROM ({union_query})
                ORDER BY draw_index
            )
            SELECT 
                {numero} as numero,
                COUNT(*) as total_apparitions,
                MIN(date_tirage) as premiere_apparition,
                MAX(date_tirage) as derniere_apparition,
                MAX(draw_index) as dernier_tirage_index,
                (SELECT MAX(draw_index) FROM BaseData) - COALESCE(MAX(draw_index), 0) as tirages_depuis_sortie,
                COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT draw_index) FROM BaseData) as pourcentage_presence,
                CASE 
                    WHEN COUNT(*) > 0 THEN (SELECT COUNT(DISTINCT draw_index) FROM BaseData) * 1.0 / COUNT(*)
                    ELSE NULL 
                END as frequence_moyenne_tirages
            FROM appearances
            """
            
            result = db_con.execute(stats_query).fetchone()
            
            if result:
                numbers_analysis.append({
                    'numero': result[0],
                    'total_apparitions': result[1],
                    'premiere_apparition': result[2],
                    'derniere_apparition': result[3],
                    'dernier_tirage_index': result[4] if result[4] else 0,
                    'tirages_depuis_sortie': result[5],
                    'pourcentage_presence': result[6] if result[6] else 0.0,
                    'frequence_moyenne_tirages': result[7] if result[7] else 0.0
                })
        
        return pd.DataFrame(numbers_analysis)

    def analyze_zones(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse la répartition par zones (1-23, 24-46, 47-70)."""
        self.logger.info("🗺️ Analyse des zones de numéros...")
        
        zone1_end = 23
        zone2_end = 46
        
        union_query = " UNION ALL ".join([
            f"SELECT draw_index, {col} as numero FROM BaseData WHERE {col} IS NOT NULL"
            for col in self.balls_cols
        ])
        
        query = f"""
            SELECT draw_index,
                   SUM(CASE WHEN numero BETWEEN 1 AND {zone1_end} THEN 1 ELSE 0 END) as zone1_count,
                   SUM(CASE WHEN numero BETWEEN {zone1_end + 1} AND {zone2_end} THEN 1 ELSE 0 END) as zone2_count,
                   SUM(CASE WHEN numero BETWEEN {zone2_end + 1} AND {self.max_number} THEN 1 ELSE 0 END) as zone3_count
            FROM ({union_query})
            WHERE numero BETWEEN 1 AND {self.max_number}
            GROUP BY draw_index
            ORDER BY draw_index
        """
        
        return db_con.execute(query).fetchdf()

    def _create_intelligent_mix(self, strategies: Dict, target_size: int = 10) -> List[int]:
        """Crée un mix intelligent avec pondération probabiliste."""
        self.logger.info("🧠 Création du mix intelligent avec pondération probabiliste...")
        
        # Calculer le score de chaque numéro selon toutes les stratégies
        number_scores = defaultdict(float)
        number_appearances = defaultdict(int)
        
        for strategy_name, strategy_data in strategies.items():
            numbers = strategy_data['numbers'][:15]  # Limiter pour l'analyse
            weight = strategy_data['weight']
            
            # Attribuer des scores dégressifs selon la position dans la liste
            for i, num in enumerate(numbers):
                position_score = 1.0 - (i / len(numbers))  # Score de 1.0 à ~0.0
                weighted_score = position_score * weight
                number_scores[num] += weighted_score
                number_appearances[num] += 1
        
        # Bonus pour les numéros qui apparaissent dans plusieurs stratégies
        for num in number_scores:
            if number_appearances[num] >= 3:  # Apparaît dans au moins 3 stratégies
                number_scores[num] *= 1.2  # Bonus de 20%
            elif number_appearances[num] >= 2:  # Apparaît dans au moins 2 stratégies
                number_scores[num] *= 1.1  # Bonus de 10%
        
        # Trier par score et sélectionner les meilleurs
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Sélection avec diversification géographique
        selected_numbers = []
        zones_count = {'zone1': 0, 'zone2': 0, 'zone3': 0}  # 1-23, 24-46, 47-70
        max_per_zone = target_size // 3 + 1  # Distribution équilibrée
        
        for num, score in sorted_numbers:
            if len(selected_numbers) >= target_size:
                break
                
            # Déterminer la zone du numéro
            if 1 <= num <= 23:
                zone = 'zone1'
            elif 24 <= num <= 46:
                zone = 'zone2'
            else:
                zone = 'zone3'
            
            # Vérifier si on peut ajouter ce numéro (équilibrage des zones)
            if zones_count[zone] < max_per_zone or len(selected_numbers) >= target_size - 2:
                selected_numbers.append(num)
                zones_count[zone] += 1
        
        # Si pas assez de numéros, compléter avec les meilleurs scores restants
        while len(selected_numbers) < target_size:
            for num, score in sorted_numbers:
                if num not in selected_numbers:
                    selected_numbers.append(num)
                    break
            else:
                break  # Plus de numéros disponibles
        
        return selected_numbers[:target_size]
    
    def generate_recommendations(self, frequencies_df: pd.DataFrame, delays_df: pd.DataFrame, 
                               pairs_df: pd.DataFrame, zones_df: pd.DataFrame, 
                               db_con: duckdb.DuckDBPyConnection) -> Dict:
        """Génère des recommandations de jeu basées sur les analyses probabilistes avancées."""
        self.logger.info("💡 Génération des recommandations probabilistes...")
        
        # === STRATÉGIES CLASSIQUES ===
        # Stratégie HOT : numéros les plus fréquents
        hot_numbers = frequencies_df.head(15)['numero'].tolist()
        
        # Stratégie COLD : numéros en retard
        cold_numbers = delays_df.head(15)['numero'].tolist()
        
        # Stratégie BALANCED : fréquences moyennes
        median_freq = frequencies_df['frequence'].median()
        balanced_numbers = frequencies_df[
            abs(frequencies_df['frequence'] - median_freq) <= median_freq * 0.1
        ].head(15)['numero'].tolist()
        
        # === NOUVELLES STRATÉGIES PROBABILISTES ===
        
        # Stratégie Z-SCORE : écarts statistiques significatifs
        zscore_numbers = self._generate_zscore_strategy(frequencies_df)
        
        # Stratégie FIBONACCI : progression mathématique
        fibonacci_numbers = self._generate_fibonacci_strategy(frequencies_df)
        
        # Stratégie SECTEURS : répartition géographique optimale
        sectors_numbers = self._generate_sectors_strategy(frequencies_df)
        
        # Stratégie TENDANCE : analyse des évolutions récentes
        trend_numbers = self._generate_trend_strategy(db_con)
        
        # Stratégie MONTE CARLO : simulation probabiliste
        montecarlo_numbers = self._generate_montecarlo_strategy(frequencies_df, delays_df)
        
        # Stratégie PAIRES OPTIMALES : basée sur les associations fréquentes
        pairs_optimal_numbers = self._generate_pairs_optimal_strategy(pairs_df)
        
        # Stratégie ZONES ÉQUILIBRÉES : répartition optimale par zones
        zones_balanced_numbers = self._generate_zones_balanced_strategy(zones_df, frequencies_df)
        
        # === STRATÉGIE MIX AMÉLIORÉE AVEC PONDÉRATION PROBABILISTE ===
        mix_numbers = self._create_intelligent_mix({
            'hot': {'numbers': hot_numbers, 'weight': 0.25, 'priority': 1},
            'cold': {'numbers': cold_numbers, 'weight': 0.20, 'priority': 2},
            'balanced': {'numbers': balanced_numbers, 'weight': 0.15, 'priority': 3},
            'zscore': {'numbers': zscore_numbers, 'weight': 0.15, 'priority': 4},
            'montecarlo': {'numbers': montecarlo_numbers, 'weight': 0.15, 'priority': 5},
            'trend': {'numbers': trend_numbers, 'weight': 0.10, 'priority': 6}
        }, target_size=10)
        
        # === GRILLES RECOMMANDÉES COMPLÈTES ===
        recommendations = {
            # Stratégies classiques
            'hot': {
                'numbers': hot_numbers[:10],
                'description': 'Numéros les plus fréquents (stratégie de continuité)',
                'probability_score': 0.85
            },
            'cold': {
                'numbers': cold_numbers[:10],
                'description': 'Numéros en retard (stratégie du retour à l\'équilibre)',
                'probability_score': 0.75
            },
            'balanced': {
                'numbers': balanced_numbers[:10],
                'description': 'Numéros à fréquence équilibrée',
                'probability_score': 0.70
            },
            
            # Nouvelles stratégies probabilistes
            'zscore': {
                'numbers': zscore_numbers[:10],
                'description': 'Numéros avec écarts statistiques significatifs (Z-Score)',
                'probability_score': 0.80
            },
            'fibonacci': {
                'numbers': fibonacci_numbers[:10],
                'description': 'Sélection basée sur la suite de Fibonacci',
                'probability_score': 0.65
            },
            'sectors': {
                'numbers': sectors_numbers[:10],
                'description': 'Répartition géographique optimale (secteurs)',
                'probability_score': 0.70
            },
            'trend': {
                'numbers': trend_numbers[:10],
                'description': 'Tendance récente (20 derniers tirages)',
                'probability_score': 0.75
            },
            'montecarlo': {
                'numbers': montecarlo_numbers[:10],
                'description': 'Simulation probabiliste Monte Carlo (10k itérations)',
                'probability_score': 0.90
            },
            'pairs_optimal': {
                'numbers': pairs_optimal_numbers[:10],
                'description': 'Optimisation basée sur les paires fréquentes',
                'probability_score': 0.80
            },
            'zones_balanced': {
                'numbers': zones_balanced_numbers[:10],
                'description': 'Équilibrage optimal par zones géographiques',
                'probability_score': 0.75
            },
            
            # Mix intelligent amélioré
            'mix_intelligent': {
                'numbers': mix_numbers,
                'description': 'Mix intelligent avec pondération probabiliste multi-stratégies',
                'probability_score': 0.95
            }
        }
        
        # Vérifier qu'il n'y a pas de doublons dans chaque grille
        for strategy, data in recommendations.items():
            numbers = data['numbers']
            if len(numbers) != len(set(numbers)):
                self.logger.warning(f"⚠️ Doublons détectés dans {strategy}: {numbers}")
                # Éliminer les doublons en gardant l'ordre
                unique_numbers = []
                for num in numbers:
                    if num not in unique_numbers:
                        unique_numbers.append(num)
                data['numbers'] = unique_numbers[:10]
                self.logger.info(f"✅ Doublons éliminés dans {strategy}")
        
        return recommendations

    def create_visualizations(self, frequencies_df: pd.DataFrame, delays_df: pd.DataFrame,
                            pairs_df: pd.DataFrame, zones_df: pd.DataFrame, 
                            sums_stats: pd.DataFrame, sectors_df: pd.DataFrame,
                            numbers_stats_df: pd.DataFrame, individual_numbers_df: pd.DataFrame):
        """Crée les visualisations des analyses."""
        self.logger.info("📈 Création des visualisations...")
        
        # Configuration matplotlib
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Graphique des fréquences
        plt.figure(figsize=(15, 8))
        freq_sorted = frequencies_df.sort_values('numero')
        
        bars = plt.bar(freq_sorted['numero'], freq_sorted['frequence'], 
                      color='steelblue', alpha=0.7, width=0.8)
        
        # Colorer les barres selon la fréquence
        colors = ['red' if f > freq_sorted['frequence'].quantile(0.8) else 
                 'orange' if f > freq_sorted['frequence'].quantile(0.6) else 
                 'green' for f in freq_sorted['frequence']]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        plt.title('Fréquence des Numéros au Keno', fontsize=16, fontweight='bold')
        plt.xlabel('Numéro')
        plt.ylabel('Fréquence d\'apparition')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 71, 5))
        
        # Ligne de moyenne
        mean_freq = freq_sorted['frequence'].mean()
        plt.axhline(y=mean_freq, color='red', linestyle='--', alpha=0.7, 
                   label=f'Moyenne: {mean_freq:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.PLOTS_DIR}/frequences_keno.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Graphique des retards amélioré
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        delay_sorted = delays_df.sort_values('numero')
        
        # Graphique 1: Retards par numéro avec code couleur
        bars = ax1.bar(delay_sorted['numero'], delay_sorted['retard'], width=0.8, alpha=0.8)
        
        # Colorer selon l'urgence du retard (ajusté pour 30 tirages)
        for i, (_, row) in enumerate(delay_sorted.iterrows()):
            retard = row['retard']
            if retard >= 25:
                bars[i].set_color('#FF0000')  # Rouge - très en retard
            elif retard >= 15:
                bars[i].set_color('#FF8C00')  # Orange - en retard
            elif retard >= 10:
                bars[i].set_color('#FFD700')  # Jaune - retard modéré
            else:
                bars[i].set_color('#32CD32')  # Vert - récent
        
        ax1.set_xlabel('Numéro', fontweight='bold')
        ax1.set_ylabel('Nombre de tirages depuis dernière apparition', fontweight='bold')
        ax1.set_title('Retard des Numéros au Keno (30 derniers tirages)\n(Rouge: ≥25, Orange: ≥15, Jaune: ≥10, Vert: <10)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, 71, 5))
        
        # Lignes de référence ajustées pour 30 tirages
        ax1.axhline(25, color='red', linestyle='--', alpha=0.6, label='Retard critique (25)')
        ax1.axhline(15, color='orange', linestyle='--', alpha=0.6, label='Retard élevé (15)')
        ax1.axhline(10, color='gold', linestyle='--', alpha=0.6, label='Retard modéré (10)')
        ax1.legend(fontsize=9)
        
        # Graphique 2: Top 15 des numéros les plus en retard
        top_retards = delays_df.nlargest(15, 'retard')
        bars2 = ax2.bar(range(len(top_retards)), top_retards['retard'], 
                       color='crimson', alpha=0.8)
        ax2.set_xticks(range(len(top_retards)))
        ax2.set_xticklabels([f'N°{int(n)}' for n in top_retards['numero']], rotation=45)
        ax2.set_xlabel('Numéros les plus en retard', fontweight='bold')
        ax2.set_ylabel('Nombre de tirages de retard', fontweight='bold')
        ax2.set_title('Top 15 des Numéros les Plus en Retard', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Graphique 3: Distribution des retards (histogramme)
        ax3.hist(delays_df['retard'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Statistiques
        mean_retard = delays_df['retard'].mean()
        median_retard = delays_df['retard'].median()
        ax3.axvline(mean_retard, color='red', linestyle='--', linewidth=2, 
                   label=f'Moyenne: {mean_retard:.1f}')
        ax3.axvline(median_retard, color='orange', linestyle='--', linewidth=2,
                   label=f'Médiane: {median_retard:.1f}')
        
        ax3.set_xlabel('Nombre de tirages de retard', fontweight='bold')
        ax3.set_ylabel('Nombre de numéros', fontweight='bold')
        ax3.set_title('Distribution des Retards', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Graphique 4: Analyse par zones de retard (ajusté pour 30 tirages)
        zones_retard = {
            'Récents (0-9)': len(delays_df[delays_df['retard'] < 10]),
            'Modérés (10-14)': len(delays_df[(delays_df['retard'] >= 10) & (delays_df['retard'] < 15)]),
            'Élevés (15-24)': len(delays_df[(delays_df['retard'] >= 15) & (delays_df['retard'] < 25)]),
            'Critiques (≥25)': len(delays_df[delays_df['retard'] >= 25])
        }
        
        colors4 = ['#32CD32', '#FFD700', '#FF8C00', '#FF0000']
        wedges, texts, autotexts = ax4.pie(zones_retard.values(), labels=zones_retard.keys(), 
                                          colors=colors4, autopct='%1.1f%%', startangle=90)
        
        # Améliorer l'apparence du camembert
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax4.set_title('Répartition des Numéros par Zone de Retard', fontsize=12, fontweight='bold')
        
        plt.suptitle('ANALYSE COMPLÈTE DES RETARDS - KENO', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.PLOTS_DIR}/retards_keno.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap des fréquences (grille 7x10)
        plt.figure(figsize=(12, 8))
        
        # Créer une matrice 7x10 pour les numéros 1-70
        freq_matrix = np.zeros((7, 10))
        labels_matrix = np.zeros((7, 10))
        
        for _, row in frequencies_df.iterrows():
            num = int(row['numero'])
            if 1 <= num <= 70:
                i = (num - 1) // 10
                j = (num - 1) % 10
                freq_matrix[i, j] = row['frequence']
                labels_matrix[i, j] = num
        
        # Créer la heatmap
        sns.heatmap(freq_matrix, annot=labels_matrix.astype(int), fmt='d', 
                   cmap='YlOrRd', cbar_kws={'label': 'Fréquence'},
                   square=True, linewidths=0.5)
        
        plt.title('Heatmap des Fréquences - Keno', fontsize=16, fontweight='bold')
        plt.xlabel('Position (0-9)')
        plt.ylabel('Ligne (1-7)')
        
        plt.tight_layout()
        plt.savefig(f'{self.PLOTS_DIR}/heatmap_keno.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Top 20 paires
        plt.figure(figsize=(12, 8))
        
        top_pairs = pairs_df.head(20)
        pair_labels = [f"{row['num1']}-{row['num2']}" for _, row in top_pairs.iterrows()]
        
        plt.barh(range(len(top_pairs)), top_pairs['frequence'], color='lightcoral', alpha=0.7)
        plt.yticks(range(len(top_pairs)), pair_labels)
        plt.xlabel('Fréquence')
        plt.title('Top 20 des Paires de Numéros les Plus Fréquentes', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{self.PLOTS_DIR}/paires_keno.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Nouveaux graphiques - Sommes
        self.plot_sums_analysis(sums_stats)
        
        # 6. Nouveaux graphiques - Secteurs
        self.plot_sectors_analysis(sectors_df)
        
        # 7. Graphiques - Statistiques des nombres (pairs/impairs)
        self.plot_numbers_stats_analysis(numbers_stats_df)
        
        # 8. Nouveau graphique - Analyse des numéros individuels
        self.plot_individual_numbers_analysis(individual_numbers_df)
        
        self.logger.info("✅ Visualisations sauvegardées (fichiers uniques)")

    def plot_sums_analysis(self, sums_stats: pd.DataFrame):
        """Crée un graphique d'analyse des sommes amélioré."""
        filename = f'{self.PLOTS_DIR}/sommes_keno.png'
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Calculer les statistiques importantes
        mean_sum = (sums_stats['somme_tirage'] * sums_stats['frequence']).sum() / sums_stats['frequence'].sum()
        median_sum = sums_stats.loc[sums_stats['frequence'].idxmax(), 'somme_tirage']
        theoretical_mean = 35.5 * 20  # Moyenne théorique (1+70)/2 * 20 numéros
        
        # Graphique 1: Distribution des sommes avec zones colorées
        bars = ax1.bar(sums_stats['somme_tirage'], sums_stats['frequence'], alpha=0.8, width=3)
        
        # Colorer selon les zones (faibles, moyennes, élevées)
        for i, (_, row) in enumerate(sums_stats.iterrows()):
            somme = row['somme_tirage']
            if somme < 600:
                bars[i].set_color('#FF6B6B')  # Rouge - sommes faibles
            elif somme < 750:
                bars[i].set_color('#4ECDC4')  # Teal - sommes moyennes
            else:
                bars[i].set_color('#45B7D1')  # Bleu - sommes élevées
        
        # Lignes de référence
        ax1.axvline(mean_sum, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Moyenne observée: {mean_sum:.0f}')
        ax1.axvline(theoretical_mean, color='orange', linestyle=':', linewidth=2, alpha=0.8,
                   label=f'Moyenne théorique: {theoretical_mean:.0f}')
        
        ax1.set_xlabel('Somme du tirage (20 numéros)', fontweight='bold')
        ax1.set_ylabel('Fréquence d\'apparition', fontweight='bold')
        ax1.set_title('Distribution des Sommes des Tirages Keno\n(Rouge: <600, Teal: 600-750, Bleu: >750)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Graphique 2: Top 15 des sommes les plus fréquentes
        top_sums = sums_stats.nlargest(15, 'frequence')
        bars2 = ax2.bar(range(len(top_sums)), top_sums['frequence'], 
                       color='lightcoral', alpha=0.8)
        ax2.set_xticks(range(len(top_sums)))
        ax2.set_xticklabels([f'{int(s)}' for s in top_sums['somme_tirage']], rotation=45)
        ax2.set_xlabel('Sommes les plus fréquentes', fontweight='bold')
        ax2.set_ylabel('Nombre d\'occurrences', fontweight='bold')
        ax2.set_title('Top 15 des Sommes les Plus Fréquentes', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Graphique 3: Pourcentages cumulés avec zones
        sums_stats_sorted = sums_stats.sort_values('somme_tirage')
        cumulative_pct = sums_stats_sorted['pourcentage'].cumsum()
        ax3.plot(sums_stats_sorted['somme_tirage'], cumulative_pct, 
                marker='o', markersize=2, color='green', linewidth=2)
        
        # Zones de probabilité
        ax3.axhline(25, color='red', linestyle='--', alpha=0.6, label='25% des tirages')
        ax3.axhline(50, color='orange', linestyle='--', alpha=0.6, label='50% des tirages')
        ax3.axhline(75, color='blue', linestyle='--', alpha=0.6, label='75% des tirages')
        
        ax3.set_xlabel('Somme du tirage', fontweight='bold')
        ax3.set_ylabel('Pourcentage cumulé (%)', fontweight='bold')
        ax3.set_title('Répartition Cumulative des Sommes', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=9)
        
        # Graphique 4: Histogramme par tranches de 50
        tranche_size = 50
        min_sum = int(sums_stats['somme_tirage'].min())
        max_sum = int(sums_stats['somme_tirage'].max())
        
        tranches = []
        labels = []
        for start in range(min_sum, max_sum + tranche_size, tranche_size):
            end = min(start + tranche_size - 1, max_sum)
            mask = (sums_stats['somme_tirage'] >= start) & (sums_stats['somme_tirage'] <= end)
            count = sums_stats[mask]['frequence'].sum()
            tranches.append(count)
            labels.append(f'{start}-{end}')
        
        colors4 = plt.cm.viridis(np.linspace(0, 1, len(tranches)))
        bars4 = ax4.bar(range(len(tranches)), tranches, color=colors4, alpha=0.8)
        ax4.set_xticks(range(len(tranches)))
        ax4.set_xticklabels(labels, rotation=45)
        ax4.set_xlabel('Tranches de sommes', fontweight='bold')
        ax4.set_ylabel('Total des occurrences', fontweight='bold')
        ax4.set_title('Répartition par Tranches de 50 Points', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(tranches)*0.01,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('ANALYSE COMPLÈTE DES SOMMES DES TIRAGES KENO', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"📊 Graphique des sommes amélioré sauvegardé : {filename}")

    def plot_sectors_analysis(self, sectors_df: pd.DataFrame):
        """Crée un graphique d'analyse des secteurs."""
        filename = f'{self.PLOTS_DIR}/secteurs_keno.png'
        
        # Calculer les moyennes par secteur
        sector_means = {
            'Secteur 1 (1-17)': sectors_df['secteur1_count'].mean(),
            'Secteur 2 (18-35)': sectors_df['secteur2_count'].mean(),
            'Secteur 3 (36-52)': sectors_df['secteur3_count'].mean(),
            'Secteur 4 (53-70)': sectors_df['secteur4_count'].mean()
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Graphique 1: Moyennes par secteur
        sectors = list(sector_means.keys())
        means = list(sector_means.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax1.bar(sectors, means, color=colors, alpha=0.8)
        ax1.set_ylabel('Nombre moyen de numéros')
        ax1.set_title('Répartition Moyenne par Secteur')
        ax1.set_ylim(0, max(means) * 1.2)
        
        # Ajouter les valeurs sur les barres
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Graphique 2: Boxplot de la distribution par secteur
        sector_data = [
            sectors_df['secteur1_count'],
            sectors_df['secteur2_count'],
            sectors_df['secteur3_count'],
            sectors_df['secteur4_count']
        ]
        
        box_plot = ax2.boxplot(sector_data, tick_labels=['S1', 'S2', 'S3', 'S4'], patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax2.set_ylabel('Nombre de numéros par tirage')
        ax2.set_title('Distribution des Numéros par Secteur')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"📊 Graphique des secteurs sauvegardé : {filename}")

    def plot_numbers_stats_analysis(self, numbers_stats_df: pd.DataFrame):
        """Crée un graphique d'analyse des statistiques des nombres."""
        filename = f'{self.PLOTS_DIR}/statistiques_nombres_keno.png'
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Calculer les moyennes
        stats_means = {
            'Pairs': numbers_stats_df['pairs_count'].mean(),
            'Impairs': numbers_stats_df['impairs_count'].mean(),
            'Bas (1-35)': numbers_stats_df['bas_count'].mean(),
            'Hauts (36-70)': numbers_stats_df['hauts_count'].mean(),
            'Dizaines': numbers_stats_df['dizaines_count'].mean(),
            'Multiples de 5': numbers_stats_df['multiples_5_count'].mean(),
            'Nombres premiers': numbers_stats_df['premiers_count'].mean()
        }
        
        # Graphique 1: Comparaison Pairs vs Impairs
        pairs_impairs = [stats_means['Pairs'], stats_means['Impairs']]
        colors1 = ['#FF6B6B', '#4ECDC4']
        wedges1, texts1, autotexts1 = ax1.pie(pairs_impairs, labels=['Pairs', 'Impairs'], 
                                             colors=colors1, autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax1.set_title('Répartition Pairs vs Impairs\n(Moyenne par tirage)', fontsize=14, fontweight='bold')
        
        # Graphique 2: Comparaison Bas vs Hauts
        bas_hauts = [stats_means['Bas (1-35)'], stats_means['Hauts (36-70)']]
        colors2 = ['#45B7D1', '#96CEB4']
        wedges2, texts2, autotexts2 = ax2.pie(bas_hauts, labels=['Bas (1-35)', 'Hauts (36-70)'], 
                                             colors=colors2, autopct='%1.1f%%', startangle=90)
        
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax2.set_title('Répartition Bas vs Hauts\n(Moyenne par tirage)', fontsize=14, fontweight='bold')
        
        # Graphique 3: Barres des statistiques spéciales
        special_stats = ['Dizaines', 'Multiples de 5', 'Nombres premiers']
        special_values = [stats_means[stat] for stat in special_stats]
        colors3 = ['#FFD700', '#FF8C00', '#9370DB']
        
        bars3 = ax3.bar(special_stats, special_values, color=colors3, alpha=0.8)
        ax3.set_ylabel('Nombre moyen par tirage', fontweight='bold')
        ax3.set_title('Statistiques Spéciales des Nombres', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars3, special_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotation des labels
        ax3.tick_params(axis='x', rotation=45)
        
        # Graphique 4: Évolution sur les derniers tirages (100 derniers)
        recent_data = numbers_stats_df.tail(100) if len(numbers_stats_df) > 100 else numbers_stats_df
        
        ax4.plot(recent_data.index, recent_data['pairs_count'], 
                label='Pairs', color='#FF6B6B', linewidth=2, alpha=0.8)
        ax4.plot(recent_data.index, recent_data['impairs_count'], 
                label='Impairs', color='#4ECDC4', linewidth=2, alpha=0.8)
        ax4.plot(recent_data.index, recent_data['bas_count'], 
                label='Bas (1-35)', color='#45B7D1', linewidth=2, alpha=0.8)
        ax4.plot(recent_data.index, recent_data['hauts_count'], 
                label='Hauts (36-70)', color='#96CEB4', linewidth=2, alpha=0.8)
        
        # Ligne de référence à 10 (équilibre théorique)
        ax4.axhline(y=10, color='red', linestyle='--', alpha=0.6, label='Équilibre (10)')
        
        ax4.set_xlabel('Index des tirages', fontweight='bold')
        ax4.set_ylabel('Nombre de numéros', fontweight='bold')
        ax4.set_title(f'Évolution sur les {len(recent_data)} Derniers Tirages', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        ax4.set_ylim(0, 20)
        
        plt.suptitle('ANALYSE COMPLÈTE DES STATISTIQUES DES NOMBRES - KENO', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"📊 Graphique des statistiques des nombres sauvegardé : {filename}")

    def plot_individual_numbers_analysis(self, individual_numbers_df: pd.DataFrame):
        """Crée un graphique d'analyse des numéros individuels."""
        filename = f'{self.PLOTS_DIR}/numeros_individuels_keno.png'
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Graphique 1: Nombre total d'apparitions par numéro
        bars1 = ax1.bar(individual_numbers_df['numero'], individual_numbers_df['total_apparitions'], 
                        alpha=0.8, width=0.8)
        
        # Colorer selon le nombre d'apparitions
        mean_apparitions = individual_numbers_df['total_apparitions'].mean()
        for i, (_, row) in enumerate(individual_numbers_df.iterrows()):
            apparitions = row['total_apparitions']
            if apparitions > mean_apparitions * 1.2:
                bars1[i].set_color('#FF6B6B')  # Rouge - très fréquent
            elif apparitions > mean_apparitions * 0.8:
                bars1[i].set_color('#4ECDC4')  # Teal - fréquence normale
            else:
                bars1[i].set_color('#FFD700')  # Jaune - peu fréquent
        
        ax1.set_xlabel('Numéro', fontweight='bold')
        ax1.set_ylabel('Nombre total d\'apparitions', fontweight='bold')
        ax1.set_title('Fréquence d\'Apparition par Numéro\n(Rouge: >120%, Teal: Normal, Jaune: <80%)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, 71, 5))
        
        # Ligne de moyenne
        ax1.axhline(mean_apparitions, color='red', linestyle='--', alpha=0.7, 
                   label=f'Moyenne: {mean_apparitions:.1f}')
        ax1.legend()
        
        # Graphique 2: Tirages depuis la dernière sortie
        bars2 = ax2.bar(individual_numbers_df['numero'], individual_numbers_df['tirages_depuis_sortie'], 
                       alpha=0.8, width=0.8)
        
        # Colorer selon l'urgence du retard (ajusté pour 30 tirages)
        for i, (_, row) in enumerate(individual_numbers_df.iterrows()):
            retard = row['tirages_depuis_sortie']
            if retard >= 25:
                bars2[i].set_color('#FF0000')  # Rouge - très en retard
            elif retard >= 15:
                bars2[i].set_color('#FF8C00')  # Orange - en retard
            elif retard >= 10:
                bars2[i].set_color('#FFD700')  # Jaune - retard modéré
            else:
                bars2[i].set_color('#32CD32')  # Vert - récent
        
        ax2.set_xlabel('Numéro', fontweight='bold')
        ax2.set_ylabel('Tirages depuis dernière sortie', fontweight='bold')
        ax2.set_title('Retard par Numéro\n(Rouge: ≥25, Orange: ≥15, Jaune: ≥10, Vert: <10)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, 71, 5))
        
        # Graphique 3: Top 15 numéros les plus fréquents
        top_frequent = individual_numbers_df.nlargest(15, 'total_apparitions')
        bars3 = ax3.bar(range(len(top_frequent)), top_frequent['total_apparitions'], 
                       color='lightcoral', alpha=0.8)
        ax3.set_xticks(range(len(top_frequent)))
        ax3.set_xticklabels([f'N°{int(n)}' for n in top_frequent['numero']], rotation=45)
        ax3.set_xlabel('Numéros les plus fréquents', fontweight='bold')
        ax3.set_ylabel('Nombre d\'apparitions', fontweight='bold')
        ax3.set_title('Top 15 des Numéros les Plus Fréquents', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Graphique 4: Top 15 numéros les plus en retard
        top_retards = individual_numbers_df.nlargest(15, 'tirages_depuis_sortie')
        bars4 = ax4.bar(range(len(top_retards)), top_retards['tirages_depuis_sortie'], 
                       color='crimson', alpha=0.8)
        ax4.set_xticks(range(len(top_retards)))
        ax4.set_xticklabels([f'N°{int(n)}' for n in top_retards['numero']], rotation=45)
        ax4.set_xlabel('Numéros les plus en retard', fontweight='bold')
        ax4.set_ylabel('Tirages depuis sortie', fontweight='bold')
        ax4.set_title('Top 15 des Numéros les Plus en Retard', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.suptitle('ANALYSE COMPLÈTE DES NUMÉROS INDIVIDUELS - KENO', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"📊 Graphique des numéros individuels sauvegardé : {filename}")

    def export_statistics(self, frequencies_df: pd.DataFrame, delays_df: pd.DataFrame,
                         pairs_df: pd.DataFrame, zones_df: pd.DataFrame, 
                         sums_stats: pd.DataFrame, sectors_df: pd.DataFrame,
                         numbers_stats_df: pd.DataFrame, individual_numbers_df: pd.DataFrame,
                         recommendations: Dict, total_draws: int = 0, top_30_df: pd.DataFrame = None):
        """Exporte toutes les statistiques en CSV."""
        self.logger.info("💾 Export des statistiques...")
        
        # Nettoyer les anciens fichiers avec horodatage
        self._clean_old_timestamped_files()
        
        # Export des fréquences (sans horodatage)
        freq_path = f"{self.STATS_DIR}/frequences_keno.csv"
        frequencies_df.to_csv(freq_path, index=False, encoding='utf-8')
        
        # Export des retards
        delay_path = f"{self.STATS_DIR}/retards_keno.csv"
        delays_df.to_csv(delay_path, index=False, encoding='utf-8')
        
        # Export des paires
        pairs_path = f"{self.STATS_DIR}/paires_keno.csv"
        pairs_df.to_csv(pairs_path, index=False, encoding='utf-8')
        
        # Export des zones
        zones_path = f"{self.STATS_DIR}/zones_keno.csv"
        zones_df.to_csv(zones_path, index=False, encoding='utf-8')
        
        # Export des sommes
        sums_path = f"{self.STATS_DIR}/sommes_keno.csv"
        sums_stats.to_csv(sums_path, index=False, encoding='utf-8')
        
        # Export des secteurs
        sectors_path = f"{self.STATS_DIR}/secteurs_keno.csv"
        sectors_df.to_csv(sectors_path, index=False, encoding='utf-8')
        
        # Export des statistiques de nombres (pairs/impairs, etc.)
        numbers_stats_path = f"{self.STATS_DIR}/statistiques_nombres_keno.csv"
        numbers_stats_df.to_csv(numbers_stats_path, index=False, encoding='utf-8')
        
        # Export des numéros individuels (NOUVEAU - ce que vous cherchiez)
        individual_numbers_path = f"{self.STATS_DIR}/numeros_individuels_keno.csv"
        individual_numbers_df.to_csv(individual_numbers_path, index=False, encoding='utf-8')
        
        # Export des recommandations en Markdown
        rec_path = f"{self.OUTPUT_DIR}/recommandations_keno.md"
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("# 🎯 RECOMMANDATIONS KENO STRATÉGIQUES AVANCÉES\n\n")
            f.write(f"**Généré le :** {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}  \n")
            f.write(f"**Analyse basée sur :** Données historiques complètes  \n")
            f.write(f"**Nombre de grilles traitées :** {total_draws:,} tirages  \n\n")
            
            # Section TOP 30 NUMÉROS ÉQUILIBRÉS
            if top_30_df is not None and not top_30_df.empty:
                f.write("## 🎯 TOP 30 NUMÉROS OPTIMAUX - STRATÉGIE ÉQUILIBRÉE\n\n")
                f.write("**Méthodologie :** Analyse composite basée sur les fréquences, retards, paires et équilibrage des zones  \n")
                f.write("**Pondération :** Fréquence (30%) + Retard (25%) + Paires (25%) + Zones (20%)  \n\n")
                
                f.write("### 📊 TOP 10 NUMÉROS RECOMMANDÉS\n\n")
                f.write("| Rang | Numéro | Score Composite | Zone | Fréquence | Score Retard | Score Paires |\n")
                f.write("|------|--------|-----------------|------|-----------|--------------|-------------|\n")
                
                for idx, row in top_30_df.head(10).iterrows():
                    f.write(f"| {row['rang']:2d} | **{row['numero']:2d}** | {row['score_composite']:.4f} | {row['zone']} | {row['frequence']:.3f} | {row['score_retard']:.3f} | {row['score_paires']:.3f} |\n")
                
                f.write(f"\n### 📋 LISTE COMPLÈTE DES 30 NUMÉROS\n\n")
                f.write("**Numéros classés par score composite :**  \n")
                top_30_numbers = [str(row['numero']) for _, row in top_30_df.iterrows()]
                f.write(f"`{' - '.join(top_30_numbers)}`\n\n")
                
                # Analyse des zones pour les 30 numéros
                zone_counts = {'Zone 1 (1-23)': 0, 'Zone 2 (24-46)': 0, 'Zone 3 (47-70)': 0}
                for _, row in top_30_df.iterrows():
                    zone_counts[row['zone']] += 1
                
                f.write("### 📍 RÉPARTITION PAR ZONES (TOP 30)\n\n")
                f.write(f"- **Zone 1 (1-23) :** {zone_counts['Zone 1 (1-23)']} numéros  \n")
                f.write(f"- **Zone 2 (24-46) :** {zone_counts['Zone 2 (24-46)']} numéros  \n")
                f.write(f"- **Zone 3 (47-70) :** {zone_counts['Zone 3 (47-70)']} numéros  \n\n")
                
                # Suggestions d'utilisation
                f.write("### 💡 SUGGESTIONS D'UTILISATION\n\n")
                f.write("**Pour grilles de 10 numéros :**  \n")
                f.write("- Sélectionnez les 10 premiers numéros du classement  \n")
                f.write("- Ou composez avec 3-4 numéros du TOP 5 + 6-7 numéros du TOP 11-20  \n\n")
                
                f.write("**Pour grilles de 15 numéros :**  \n")
                f.write("- Utilisez les 15 premiers numéros du classement  \n")
                f.write("- Garantit une répartition équilibrée entre toutes les stratégies  \n\n")
                
                f.write("**Pour grilles de 20 numéros :**  \n")
                f.write("- Prenez les 20 premiers numéros pour une couverture optimale  \n")
                f.write("- Idéal pour maximiser les chances avec un investissement mesuré  \n\n")
                
                f.write("---\n\n")
            
            # Trier par score de probabilité
            sorted_strategies = sorted(recommendations.items(), 
                                     key=lambda x: x[1].get('probability_score', 0), 
                                     reverse=True)
            
            f.write("## 📊 CLASSEMENT PAR SCORE DE PROBABILITÉ\n\n")
            
            # Tableau de classement
            f.write("| Rang | Stratégie | Score | Description |\n")
            f.write("|------|-----------|-------|-------------|\n")
            
            for i, (strategy, data) in enumerate(sorted_strategies, 1):
                prob_score = data.get('probability_score', 0)
                f.write(f"| {i:2d} | **{strategy.upper()}** | {prob_score:.2f}/1.00 | {data['description']} |\n")
            
            f.write("\n## 📋 DÉTAIL DES STRATÉGIES\n\n")
            
            for strategy, data in sorted_strategies:
                prob_score = data.get('probability_score', 0)
                f.write(f"### 🔹 {strategy.upper()}\n\n")
                f.write(f"**Score de probabilité :** {prob_score:.2f}/1.00  \n")
                f.write(f"**Description :** {data['description']}  \n")
                f.write(f"**Numéros :** `{data['numbers']}`  \n\n")
                
                # Analyse de la grille
                numbers = data['numbers']
                if numbers:
                    zones = {'Zone 1-23': 0, 'Zone 24-46': 0, 'Zone 47-70': 0}
                    for num in numbers:
                        if 1 <= num <= 23:
                            zones['Zone 1-23'] += 1
                        elif 24 <= num <= 46:
                            zones['Zone 24-46'] += 1
                        else:
                            zones['Zone 47-70'] += 1
                    
                    f.write("**Analyse de la grille :**\n")
                    f.write(f"- **Répartition zones :** {zones}  \n")
                    f.write(f"- **Somme totale :** {sum(numbers)}  \n")
                    f.write(f"- **Moyenne :** {sum(numbers)/len(numbers):.1f}  \n")
                    
                    # Pairs/Impairs
                    pairs = len([n for n in numbers if n % 2 == 0])
                    impairs = len(numbers) - pairs
                    f.write(f"- **Pairs/Impairs :** {pairs}/{impairs}  \n\n")
                
                f.write("---\n\n")
            
            # Section conseils d'utilisation
            f.write("## 💡 CONSEILS D'UTILISATION\n\n")
            f.write("| Stratégie | Description | Recommandation |\n")
            f.write("|-----------|-------------|----------------|\n")
            f.write("| **MIX INTELLIGENT** | Meilleur équilibre probabiliste | ⭐⭐⭐⭐⭐ Optimal |\n")
            f.write("| **MONTE CARLO** | Simulation statistique avancée | ⭐⭐⭐⭐⭐ Excellent |\n")
            f.write("| **HOT** | Continuité des tendances fréquentielles | ⭐⭐⭐⭐ Très bon |\n")
            f.write("| **Z-SCORE** | Écarts statistiques significatifs | ⭐⭐⭐⭐ Très bon |\n")
            f.write("| **COLD** | Stratégie du retour à l'équilibre | ⭐⭐⭐ Bon |\n")
            f.write("| **TREND** | Tendances récentes (20 derniers tirages) | ⭐⭐⭐ Bon |\n")
            f.write("| **PAIRS OPTIMAL** | Optimisation par associations | ⭐⭐⭐ Bon |\n")
            f.write("| **ZONES BALANCED** | Répartition géographique | ⭐⭐⭐ Bon |\n")
            f.write("| **FIBONACCI** | Progression mathématique | ⭐⭐ Moyen |\n")
            f.write("| **SECTORS** | Approche géométrique | ⭐⭐ Moyen |\n")
            f.write("| **BALANCED** | Fréquences moyennes stables | ⭐⭐ Moyen |\n\n")
            
            f.write("## ⚠️ RAPPEL IMPORTANT\n\n")
            f.write("> **Les jeux de hasard comportent des risques.**  \n")
            f.write("> Ces recommandations sont basées sur l'analyse statistique des données historiques et ne garantissent aucun résultat.  \n")
            f.write("> **Jouez avec modération et responsabilité.**\n\n")
            
            f.write("---\n\n")
            f.write("*Rapport généré automatiquement par l'Analyseur Keno Stratégique*\n")
        
        self.logger.info(f"✅ Statistiques exportées dans {self.STATS_DIR}")
        self.logger.info(f"✅ Recommandations exportées dans {rec_path}")

    def _clean_old_timestamped_files(self):
        """Nettoie les anciens fichiers avec horodatage pour éviter l'accumulation."""
        import glob
        import re
        
        # Patterns des fichiers avec horodatage à supprimer
        patterns = [
            f"{self.STATS_DIR}/frequences_keno_*_*.csv",
            f"{self.STATS_DIR}/retards_keno_*_*.csv", 
            f"{self.STATS_DIR}/paires_keno_*_*.csv",
            f"{self.STATS_DIR}/zones_keno_*_*.csv",
            f"{self.STATS_DIR}/sommes_keno_*_*.csv",
            f"{self.STATS_DIR}/secteurs_keno_*_*.csv",
            f"{self.STATS_DIR}/statistiques_nombres_keno_*_*.csv",
            f"{self.STATS_DIR}/numeros_individuels_keno_*_*.csv",
            f"{self.OUTPUT_DIR}/recommandations_keno_*_*.txt"
        ]
        
        files_removed = 0
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    files_removed += 1
                except OSError:
                    pass
        
        if files_removed > 0:
            self.logger.info(f"🧹 {files_removed} anciens fichiers avec horodatage supprimés")

    def run_complete_analysis(self, csv_path: str, create_plots: bool = True, 
                            export_stats: bool = True):
        """Lance l'analyse complète des données Keno."""
        self.logger.info("🚀 Démarrage de l'analyse complète Keno...")
        
        # Initialiser DuckDB
        db_con = duckdb.connect()
        
        try:
            # 1. Charger les données
            table_name = self.load_data_from_csv(csv_path, db_con)
            df_pandas = self._create_base_view(db_con, table_name)
            
            # 2. Analyses statistiques
            frequencies_df = self.analyze_frequencies(db_con)
            delays_df = self.analyze_delays(db_con)
            pairs_df = self.analyze_pairs(db_con)
            zones_df = self.analyze_zones(db_con)
            sums_stats = self.analyze_sums(db_con)
            sectors_df = self.analyze_sectors(db_con)
            numbers_stats_df = self.analyze_numbers_stats(db_con)
            individual_numbers_df = self.analyze_individual_numbers(db_con)
            
            # 3. Générer les recommandations
            recommendations = self.generate_recommendations(frequencies_df, delays_df, pairs_df, zones_df, db_con)
            
            # 4. Générer les 30 numéros optimaux avec la stratégie équilibrée
            top_30_df = self.generate_top_30_balanced_numbers(frequencies_df, zones_df, delays_df, pairs_df)
            
            # 5. Exporter les 30 numéros vers CSV
            csv_file = self.export_top_30_to_csv(top_30_df)
            
            # 6. Créer les visualisations
            if create_plots:
                self.create_visualizations(frequencies_df, delays_df, pairs_df, zones_df, 
                                         sums_stats, sectors_df, numbers_stats_df, individual_numbers_df)
            
            # 7. Exporter les statistiques
            if export_stats:
                self.export_statistics(frequencies_df, delays_df, pairs_df, 
                                     zones_df, sums_stats, sectors_df, numbers_stats_df,
                                     individual_numbers_df, recommendations, len(df_pandas), top_30_df)
            
            # Résumé final
            self.logger.info("✅ Analyse complète terminée avec succès!")
            self.logger.info(f"📊 {len(df_pandas)} tirages analysés")
            self.logger.info(f"📈 {len(frequencies_df)} numéros analysés")
            self.logger.info(f"🔗 {len(pairs_df)} paires analysées")
            self.logger.info(f"🔢 {len(sums_stats)} sommes analysées")
            self.logger.info(f"🗺️ {len(sectors_df)} secteurs analysés")
            self.logger.info(f"🎯 Top 30 numéros équilibrés exportés vers CSV")
            
            print("\n🎯 TOP 30 NUMÉROS ÉQUILIBRÉS - STRATÉGIE OPTIMALE")
            print("=" * 60)
            print(f"📄 Fichier CSV généré: {csv_file}")
            print("\n🏆 TOP 10 NUMÉROS RECOMMANDÉS:")
            for idx, row in top_30_df.head(10).iterrows():
                print(f"   {row['rang']:2d}. Numéro {row['numero']:2d} - Score: {row['score_composite']:.4f} ({row['zone']})")
            
            print(f"\n📋 Les 30 numéros complets sont disponibles dans le fichier CSV")
            print(f"   Localisation: {csv_file}")
            
            print("\n🎯 RECOMMANDATIONS KENO STRATÉGIQUES")
            print("=" * 50)
            
            # Trier les stratégies par score de probabilité
            sorted_strategies = sorted(recommendations.items(), 
                                     key=lambda x: x[1].get('probability_score', 0), 
                                     reverse=True)
            
            for strategy, data in sorted_strategies:
                prob_score = data.get('probability_score', 0)
                print(f"\n🔸 {strategy.upper()} (Score: {prob_score:.2f})")
                print(f"   {data['description']}")
                print(f"   Numéros: {data['numbers']}")
                
                # Analyse de la grille
                numbers = data['numbers']
                if numbers:
                    zones = {'Zone 1-23': 0, 'Zone 24-46': 0, 'Zone 47-70': 0}
                    for num in numbers:
                        if 1 <= num <= 23:
                            zones['Zone 1-23'] += 1
                        elif 24 <= num <= 46:
                            zones['Zone 24-46'] += 1
                        else:
                            zones['Zone 47-70'] += 1
                    
                    print(f"   Répartition: {zones}")
                    print(f"   Somme: {sum(numbers)}, Moyenne: {sum(numbers)/len(numbers):.1f}")
            
            # Affichage spécial pour le mix intelligent
            if 'mix_intelligent' in recommendations:
                print(f"\n{'='*60}")
                print("🧠 ANALYSE DU MIX INTELLIGENT")
                print(f"{'='*60}")
                mix_data = recommendations['mix_intelligent']
                print(f"Score de probabilité: {mix_data['probability_score']:.2f}/1.00")
                print(f"Description: {mix_data['description']}")
                print(f"Numéros sélectionnés: {mix_data['numbers']}")
                
                # Statistiques détaillées du mix
                mix_nums = mix_data['numbers']
                if mix_nums:
                    print(f"\n📊 Statistiques détaillées:")
                    print(f"   • Somme totale: {sum(mix_nums)}")
                    print(f"   • Moyenne: {sum(mix_nums)/len(mix_nums):.1f}")
                    print(f"   • Écart-type: {np.std(mix_nums):.1f}")
                    print(f"   • Paires/Impairs: {len([n for n in mix_nums if n%2==0])}/{len([n for n in mix_nums if n%2==1])}")
                    
                    zones_detail = {'Zone 1-23': 0, 'Zone 24-46': 0, 'Zone 47-70': 0}
                    for num in mix_nums:
                        if 1 <= num <= 23:
                            zones_detail['Zone 1-23'] += 1
                        elif 24 <= num <= 46:
                            zones_detail['Zone 24-46'] += 1
                        else:
                            zones_detail['Zone 47-70'] += 1
                    print(f"   • Répartition zones: {zones_detail}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur durant l'analyse: {e}")
            raise
        finally:
            db_con.close()

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Analyseur Keno avec DuckDB')
    
    # Création d'un groupe pour les sources de données mutuellement exclusives
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--csv', help='Chemin vers le fichier CSV Keno')
    source_group.add_argument('--auto-consolidated', action='store_true', 
                            help='Utilise automatiquement le fichier consolidé s\'il est disponible')
    
    parser.add_argument('--plots', action='store_true', help='Créer les graphiques')
    parser.add_argument('--export-stats', action='store_true', help='Exporter les statistiques')
    
    args = parser.parse_args()
    
    # Déterminer le fichier CSV à utiliser
    csv_path = None
    
    if args.auto_consolidated:
        if load_keno_data and is_consolidated_available:
            if is_consolidated_available():
                from pathlib import Path
                csv_path = Path("keno/keno_data/keno_consolidated.csv")
                print(f"📊 Utilisation du fichier consolidé: {csv_path}")
            else:
                print("❌ Fichier consolidé non disponible")
                if recommend_consolidation:
                    recommend_consolidation()
                return 1
        else:
            print("❌ Module de consolidation non disponible")
            return 1
    else:
        csv_path = args.csv
    
    # Créer l'analyseur
    analyzer = KenoAnalyzer()
    
    # Lancer l'analyse
    analyzer.run_complete_analysis(
        csv_path=str(csv_path),
        create_plots=args.plots,
        export_stats=args.export_stats
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
