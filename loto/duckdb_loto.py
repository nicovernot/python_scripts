#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 GÉNÉRATEUR DE GRILLES LOTO STRATÉGIQUE (VERSION FINALE)
=========================================================

Script complet pour générer des grilles de loto en utilisant des stratégies
de pondération externes définies dans un fichier YAML. Il fournit des analyses
approfondies via des graphiques et des exports CSV.

Utilisation:
    # Générer 3 grilles avec la stratégie par défaut ('equilibre') et créer les analyses
    python loto_strategist.py --csv /path/to/data.csv --grids 3 --plots --export-stats

    # Choisir une stratégie spécifique comme 'focus_retard'
    python loto_strategist.py --csv /path/to/data.csv --strategy focus_retard
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

# --- Dépendances ---
try:
    import pandas as pd
    import numpy as np
    import duckdb
    import matplotlib.pyplot as plt
    import seaborn as sns
    import yaml
    from pathlib import Path
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from scipy import stats
    from scipy.stats import zscore
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les dépendances: pip install pandas numpy duckdb matplotlib seaborn PyYAML scikit-learn scipy")
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning, module='duckdb')

class LotoStrategist:
    """Générateur de grilles de loto utilisant des stratégies de pondération externes et ML."""
    
    PLOTS_DIR = "loto_analyse_plots"
    STATS_DIR = "loto_stats_exports"
    CACHE_DIR = "cache"
    MODELS_DIR = "boost_models"
    LOW_ZONE_END = 17
    MID_ZONE_END = 34

    def __init__(self, max_number: int = 49, numbers_per_draw: int = 5, 
                 config_file: str = './loto/strategies.yml', strategy_name: str = 'equilibre'):
        self.max_number = max_number
        self.numbers_per_draw = numbers_per_draw
        self.BALLS = list(range(1, max_number + 1))
        self.balls_cols = [f'boule_{i}' for i in range(1, numbers_per_draw + 1)]
        
        # Cache pour optimiser les performances
        self._features_cache = {}
        self._models_cache = {}
        
        # Créer les répertoires nécessaires
        for directory in [self.PLOTS_DIR, self.STATS_DIR, self.CACHE_DIR, self.MODELS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.logger = logging.getLogger(__name__)

        self._load_strategies(config_file)
        self.set_strategy(strategy_name)

    def _load_strategies(self, config_file: str):
        """Charge les stratégies depuis un fichier de configuration YAML."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.strategies = yaml.safe_load(f)['strategies']
            self.logger.info(f"✅ Stratégies chargées avec succès depuis '{config_file}'")
        except FileNotFoundError:
            self.logger.error(f"❌ Fichier de configuration '{config_file}' non trouvé. Assurez-vous qu'il existe.")
            raise
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la lecture du fichier de configuration: {e}")
            raise

    def set_strategy(self, strategy_name: str):
        """Sélectionne la stratégie active et ses poids."""
        if strategy_name not in self.strategies:
            raise ValueError(f"La stratégie '{strategy_name}' n'existe pas. Disponibles: {list(self.strategies.keys())}")
        
        self.strategy_name = strategy_name
        self.active_strategy = self.strategies[strategy_name]
        self.weights_config = self.active_strategy['weights']
        self.strategy_description = self.active_strategy.get('description', 'Pas de description.')
        self.logger.info(f"🚀 Stratégie active: '{self.strategy_name}' - {self.strategy_description}")

    def load_data_from_csv(self, csv_path: str, db_con: duckdb.DuckDBPyConnection) -> str:
        """Charge et valide les données depuis un fichier CSV."""
        if not os.path.exists(csv_path): raise FileNotFoundError(f"❌ Fichier non trouvé: {csv_path}")
        self.logger.info(f"📁 Chargement et validation de {csv_path}")
        try:
            sample_df = pd.read_csv(csv_path, nrows=0, sep=';')
            required_cols = self.balls_cols + ['date_de_tirage']
            missing_cols = [col for col in required_cols if col not in sample_df.columns]
            if missing_cols: raise ValueError(f"Colonnes manquantes dans le CSV: {missing_cols}")
            table_name = "loto_historical_data"
            db_con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv('{csv_path}', delim=';', header=true, auto_detect=true)")
            count = db_con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            self.logger.info(f"✅ {count} tirages chargés avec succès")
            return table_name
        except Exception as e: raise Exception(f"❌ Erreur lors du chargement CSV: {e}")

    def _create_base_view(self, db_con, table_name):
        """Crée une vue de données propres et la charge dans un DataFrame."""
        db_con.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW BaseData AS
            SELECT ROW_NUMBER() OVER (ORDER BY date_de_tirage) AS draw_index, 
                   date_de_tirage::DATE as date_de_tirage,
                   {', '.join(self.balls_cols)}
            FROM {table_name}
            WHERE date_de_tirage IS NOT NULL AND {' AND '.join(f'{col} IS NOT NULL' for col in self.balls_cols)}
        """)
        df_pandas = db_con.table('BaseData').fetchdf()
        if df_pandas.empty: raise ValueError("❌ Aucune donnée valide trouvée après filtrage")
        self.logger.info(f"📊 Analyse de {len(df_pandas)} tirages valides")
        return df_pandas
        
    def _analyze_pair_counts(self, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Analyse la fréquence de toutes les paires de numéros."""
        query = " UNION ALL ".join([
            f"SELECT LEAST(t.boule_{i}, t.boule_{j}) AS n1, GREATEST(t.boule_{i}, t.boule_{j}) AS n2 FROM BaseData t"
            for i in range(1, self.numbers_per_draw + 1)
            for j in range(i + 1, self.numbers_per_draw + 1)
        ])
        return db_con.execute(f"SELECT n1, n2, COUNT(*) as count FROM ({query}) GROUP BY n1, n2 ORDER BY count DESC").fetchdf()

    def analyze_complete_criteria(self, db_con: duckdb.DuckDBPyConnection, table_name: str) -> Dict:
        """Orchestre toutes les analyses statistiques."""
        self.logger.info("🔍 Début de l'analyse statistique complète...")
        df_pandas = self._create_base_view(db_con, table_name)
        
        freq = pd.Series(df_pandas[self.balls_cols].values.flatten()).value_counts().reindex(self.BALLS, fill_value=0)
        last_appearance = df_pandas.melt(id_vars=['date_de_tirage'], value_vars=self.balls_cols, value_name='numero').groupby('numero')['date_de_tirage'].max()
        gaps_query = "SELECT numero, AVG(draw_index - lag) as periodicite FROM (SELECT numero, draw_index, LAG(draw_index, 1) OVER (PARTITION BY numero ORDER BY draw_index) as lag FROM (SELECT draw_index, unnest([boule_1, boule_2, boule_3, boule_4, boule_5]) as numero FROM BaseData)) WHERE lag IS NOT NULL GROUP BY numero"
        gaps_df = db_con.execute(gaps_query).fetchdf()
        gaps = pd.Series(gaps_df.set_index('numero')['periodicite']) if not gaps_df.empty else pd.Series(dtype='float64')
        delta = pd.to_datetime('now', utc=True).tz_localize(None) - pd.to_datetime(last_appearance.reindex(self.BALLS))
        
        df_pandas['pairs'] = (df_pandas[self.balls_cols] % 2 == 0).sum(axis=1)
        df_pandas['impairs'] = self.numbers_per_draw - df_pandas['pairs']
        
        all_stats = {
            'freq': freq, 'hot_numbers': freq.nlargest(10).index.tolist(), 'cold_numbers': freq.nsmallest(10).index.tolist(),
            'numbers_analysis': pd.DataFrame({'Numero': self.BALLS, 'Frequence': freq.reindex(self.BALLS, fill_value=0), 'Jours_Depuis_Tirage': delta.dt.days.fillna(999).astype(int), 'Ecart_Moyen_Tirages': gaps.reindex(self.BALLS)}),
            'numbers_to_exclude': set(df_pandas.iloc[-3:][self.balls_cols].values.flatten()),
            'pair_counts': self._analyze_pair_counts(db_con),
            'sums_distribution': df_pandas[self.balls_cols].sum(axis=1),
            'even_odd_distribution': df_pandas.apply(lambda row: f"{row['pairs']}P / {row['impairs']}I", axis=1).value_counts(),
            'df_pandas': df_pandas # Garder le dataframe pour analyses futures
        }
        self.logger.info("✅ Analyse statistique terminée")
        return all_stats

    def calculate_prediction_weights(self, all_stats: Dict) -> Dict[int, float]:
        """Calcule les poids de prédiction en utilisant la stratégie active et analyses avancées."""
        self.logger.info("⚖️  Calcul des poids selon la stratégie avec ML...")
        weights = {num: 0.0 for num in self.BALLS}
        analysis_df = all_stats['numbers_analysis'].set_index('Numero')
        w_conf = self.weights_config

        # Obtenir les probabilités avancées
        probabilities = self._predict_next_draw_probabilities(all_stats)
        
        # Normaliser les probabilités
        max_prob = max(probabilities.values()) if probabilities.values() else 1.0
        
        # Appliquer chaque poids défini dans la stratégie
        if 'frequency' in w_conf:
            max_freq = analysis_df['Frequence'].max()
            if max_freq > 0:
                for num, freq in analysis_df['Frequence'].items(): 
                    weights[num] += (freq / max_freq) * w_conf['frequency']
        
        if 'recency_gap' in w_conf:
            for num in self.BALLS:
                if not pd.isna(analysis_df.loc[num, 'Ecart_Moyen_Tirages']) and analysis_df.loc[num, 'Ecart_Moyen_Tirages'] > 0:
                    retard_ratio = analysis_df.loc[num, 'Jours_Depuis_Tirage'] / (analysis_df.loc[num, 'Ecart_Moyen_Tirages'] * 3)
                    weights[num] += min(retard_ratio, 3) * w_conf['recency_gap']
        
        if 'momentum' in w_conf:
            last_20_draws = all_stats['df_pandas'].tail(20)
            momentum_freq = pd.Series(last_20_draws[self.balls_cols].values.flatten()).value_counts()
            for num, freq in momentum_freq.items(): 
                weights[num] += (freq / 20) * w_conf['momentum']
            
        if 'hot_cold' in w_conf:
            for num in all_stats['hot_numbers']: 
                weights[num] += 0.1 * w_conf['hot_cold']
            for num in all_stats['cold_numbers']: 
                weights[num] += 0.1 * w_conf['hot_cold']
            
        if 'frequent_pairs' in w_conf and not all_stats['pair_counts'].empty:
            top_pairs = all_stats['pair_counts'].head(20)
            for _, row in top_pairs.iterrows():
                weights[row['n1']] += 0.05 * w_conf['frequent_pairs']
                weights[row['n2']] += 0.05 * w_conf['frequent_pairs']

        # Intégrer les probabilités ML avancées
        ml_weight = w_conf.get('ml_boost', 0.3)  # Nouveau paramètre pour les stratégies
        for num, prob in probabilities.items():
            weights[num] += (prob / max_prob) * ml_weight

        # Appliquer la pénalité pour les numéros récents
        penalty_multiplier = 1.0 - w_conf.get('recent_draws_penalty', 0.5)
        for num in all_stats['numbers_to_exclude']:
            if num in weights: 
                weights[num] *= penalty_multiplier
            
        return weights
    
    def generate_grids(self, db_con: duckdb.DuckDBPyConnection, table_name: str, num_grids: int):
        """Génère le nombre de grilles demandé avec algorithmes avancés."""
        self.logger.info(f"🚀 Génération de {num_grids} grille(s) optimisée(s) avec ML")
        all_stats = self.analyze_complete_criteria(db_con, table_name)
        
        # Construire les features historiques pour ML
        historical_features = self._build_historical_features_dataset(all_stats['df_pandas'])
        
        # Entraîner ou charger les modèles ML
        models = self._train_prediction_models(historical_features)
        
        # Calculer les poids de prédiction
        prediction_weights = self.calculate_prediction_weights(all_stats)
        
        # Générer un large pool de combinaisons candidates
        self.logger.info("🎯 Génération de combinaisons candidates...")
        candidate_combinations = self._generate_smart_combinations(prediction_weights, num_combinations=5000)
        
        # Scorer toutes les combinaisons
        self.logger.info("⚖️ Scoring des combinaisons avec ML...")
        scored_combinations = []
        
        for combination in candidate_combinations:
            score = self._score_combination_advanced(combination, all_stats, models)
            validation = self._validate_combination_quality(combination, all_stats)
            
            # Ajuster le score selon la validation
            adjusted_score = score * validation['quality_score']
            
            features = self._calculate_features_for_combination(combination)
            
            scored_combinations.append({
                'grille': [int(x) for x in combination],  # Convertir en int Python
                'score': adjusted_score,
                'validation': validation,
                'somme': features['sum'],
                'pairs': features['even_count'],
                'range': features['range'],
                'consecutifs': features['consecutive_pairs'],
                'diversite_unites': features['last_digit_diversity'],
                'zones': f"{features['zone_low']}-{features['zone_mid']}-{features['zone_high']}"
            })
        
        # Trier par score et appliquer la diversification
        scored_combinations.sort(key=lambda x: x['score'], reverse=True)
        
        # Sélectionner les meilleures grilles avec diversification
        final_grids = []
        used_combinations = set()
        
        for combo in scored_combinations:
            if len(final_grids) >= num_grids:
                break
                
            combo_tuple = tuple(sorted(combo['grille']))
            
            # Vérifier la diversité (pas trop similaire aux grilles déjà sélectionnées)
            is_diverse = True
            for used_combo in used_combinations:
                overlap = len(set(combo_tuple) & set(used_combo))
                if overlap >= 4:  # Trop de similitude
                    is_diverse = False
                    break
            
            if is_diverse and combo['validation']['quality_score'] >= 0.6:
                final_grids.append(combo)
                used_combinations.add(combo_tuple)
        
        # Si pas assez de grilles diverses, compléter avec les meilleures
        while len(final_grids) < num_grids and len(final_grids) < len(scored_combinations):
            for combo in scored_combinations:
                if len(final_grids) >= num_grids:
                    break
                combo_tuple = tuple(sorted(combo['grille']))
                if combo_tuple not in used_combinations:
                    final_grids.append(combo)
                    used_combinations.add(combo_tuple)
                    break
        
        self.logger.info(f"✅ {len(final_grids)} grilles générées et optimisées")
        return final_grids, all_stats

    def _create_visualizations(self, all_stats: Dict):
        """Crée et sauvegarde un ensemble de graphiques d'analyse."""
        self.logger.info(f"📊 Création des graphiques de visualisation...")
        if not os.path.exists(self.PLOTS_DIR): os.makedirs(self.PLOTS_DIR)
        
        sns.set_style("whitegrid")
        # Freq, Retard, Heatmap, Sommes, Pairs/Impairs
        for plot_type in ["freq", "retard", "heatmap", "sommes", "even_odd"]:
            self.logger.info(f"... Création du graphique: {plot_type}")
            plt.figure(figsize=(18, 9))
            if plot_type == "freq":
                all_stats['numbers_analysis'].set_index('Numero')['Frequence'].plot(kind='bar', color='skyblue')
                plt.title("Fréquence d'Apparition de Chaque Numéro", fontsize=16)
            elif plot_type == "retard":
                all_stats['numbers_analysis'].set_index('Numero')['Jours_Depuis_Tirage'].plot(kind='bar', color='salmon')
                plt.title("Nombre de Jours Depuis la Dernière Apparition", fontsize=16)
            elif plot_type == "heatmap":
                plt.figure(figsize=(22, 18))
                pair_matrix = all_stats['pair_counts'].pivot(index='n1', columns='n2', values='count').reindex(index=self.BALLS, columns=self.BALLS, fill_value=0)
                sns.heatmap(pair_matrix + pair_matrix.T, cmap="YlOrRd", linewidths=.5)
                plt.title("Heatmap de Fréquence des Paires de Numéros", fontsize=16)
            elif plot_type == "sommes":
                sns.histplot(all_stats['sums_distribution'], kde=True, bins=30, color='darkcyan')
                plt.title("Distribution de la Somme des Numéros par Tirage", fontsize=16)
            elif plot_type == "even_odd":
                all_stats['even_odd_distribution'].sort_index().plot(kind='bar', color='slateblue')
                plt.title("Fréquence des Répartitions Pairs/Impairs", fontsize=16)
                plt.xticks(rotation=0)

            plt.tight_layout()
            plt.savefig(os.path.join(self.PLOTS_DIR, f"analyse_{plot_type}.png"))
            plt.close()
        self.logger.info(f"📈 Graphiques sauvegardés dans '{self.PLOTS_DIR}'")

    def _export_statistics_to_csv(self, all_stats: Dict):
        """Exporte les statistiques clés dans des fichiers CSV."""
        self.logger.info(f"📄 Export des statistiques au format CSV...")
        if not os.path.exists(self.STATS_DIR): os.makedirs(self.STATS_DIR)

        all_stats['numbers_analysis'].to_csv(os.path.join(self.STATS_DIR, "analyse_numeros.csv"), index=False)
        all_stats['pair_counts'].to_csv(os.path.join(self.STATS_DIR, "frequence_paires.csv"), index=False)
        all_stats['sums_distribution'].to_frame(name='somme_tirage').to_csv(os.path.join(self.STATS_DIR, "distribution_sommes.csv"))
        all_stats['even_odd_distribution'].to_frame(name='occurrences').to_csv(os.path.join(self.STATS_DIR, "distribution_pairs_impairs.csv"))
        self.logger.info(f"💾 Fichiers CSV sauvegardés dans '{self.STATS_DIR}'")
        
    def print_results(self, results: List[Dict]):
        """Affiche les grilles générées avec détails avancés."""
        print("\n" + "="*80 + f"\n🎯 GRILLES GÉNÉRÉES AVEC LA STRATÉGIE '{self.strategy_name.upper()}' (ML-Enhanced)\n" + "="*80)
        
        for i, result in enumerate(results, 1):
            validation = result.get('validation', {})
            quality_indicators = []
            
            if validation.get('sum_ok', False): quality_indicators.append("✓Somme")
            if validation.get('balance_ok', False): quality_indicators.append("✓Équilibre")
            if validation.get('zones_ok', False): quality_indicators.append("✓Zones")
            if validation.get('diversity_ok', False): quality_indicators.append("✓Diversité")
            
            quality_str = " | ".join(quality_indicators) if quality_indicators else "Standard"
            
            print(f"\n🎲 GRILLE #{i} (Score ML: {result['score']:.1f}/100)")
            print(f"   Numéros: {result['grille']}")
            print(f"   Somme: {result['somme']} | Pairs/Impairs: {result['pairs']}/{self.numbers_per_draw - result['pairs']} | Étendue: {result.get('range', 'N/A')}")
            print(f"   Zones: {result.get('zones', 'N/A')} | Consécutifs: {result.get('consecutifs', 0)} | Diversité unités: {result.get('diversite_unites', 'N/A')}")
            print(f"   Qualité: {quality_str}")
        
        # Statistiques globales
        if results:
            avg_score = np.mean([r['score'] for r in results])
            avg_quality = np.mean([r.get('validation', {}).get('quality_score', 0) for r in results])
            print(f"\n📊 STATISTIQUES GLOBALES")
            print(f"   Score ML moyen: {avg_score:.1f}/100")
            print(f"   Qualité moyenne: {avg_quality:.1%}")
            
        print("\n" + "="*80)

    @lru_cache(maxsize=128)
    def _calculate_features_for_combination(self, combination: Tuple[int, ...]) -> Dict:
        """Calcule les features d'une combinaison pour le scoring ML."""
        sorted_combo = sorted(combination)
        
        features = {
            'sum': sum(sorted_combo),
            'range': max(sorted_combo) - min(sorted_combo),
            'even_count': sum(1 for n in sorted_combo if n % 2 == 0),
            'consecutive_pairs': sum(1 for i in range(len(sorted_combo)-1) if sorted_combo[i+1] - sorted_combo[i] == 1),
            'zone_low': sum(1 for n in sorted_combo if n <= self.LOW_ZONE_END),
            'zone_mid': sum(1 for n in sorted_combo if self.LOW_ZONE_END < n <= self.MID_ZONE_END),
            'zone_high': sum(1 for n in sorted_combo if n > self.MID_ZONE_END),
            'gap_variance': np.var([sorted_combo[i+1] - sorted_combo[i] for i in range(len(sorted_combo)-1)]) if len(sorted_combo) > 1 else 0,
            'last_digit_diversity': len(set(n % 10 for n in sorted_combo)),
            'decade_diversity': len(set(n // 10 for n in sorted_combo)),
        }
        
        return features

    def _build_historical_features_dataset(self, df_pandas: pd.DataFrame) -> pd.DataFrame:
        """Construit un dataset de features pour l'entraînement ML."""
        features_list = []
        
        for _, row in df_pandas.iterrows():
            combination = tuple(sorted([row[col] for col in self.balls_cols]))
            features = self._calculate_features_for_combination(combination)
            features['draw_date'] = row['date_de_tirage']
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def _train_prediction_models(self, historical_features: pd.DataFrame) -> Dict:
        """Entraîne des modèles ML pour prédire les probabilités des numéros."""
        self.logger.info("🤖 Entraînement des modèles de machine learning...")
        
        models = {}
        
        # Préparer les features pour chaque numéro
        for ball_num in range(1, self.numbers_per_draw + 1):
            model_file = os.path.join(self.MODELS_DIR, f'xgb_ball_{ball_num}.pkl')
            
            if os.path.exists(model_file):
                # Charger le modèle existant
                with open(model_file, 'rb') as f:
                    models[f'ball_{ball_num}'] = pickle.load(f)
                continue
            
            # Créer les features et targets pour ce numéro
            X = historical_features[['sum', 'range', 'even_count', 'consecutive_pairs', 
                                   'zone_low', 'zone_mid', 'zone_high', 'gap_variance',
                                   'last_digit_diversity', 'decade_diversity']].values
            
            # Target : probabilité qu'un numéro soit tiré à cette position
            y_prob = np.zeros(len(historical_features))
            for i in range(len(y_prob)):
                # Logique simplifiée - à améliorer avec plus de données historiques
                y_prob[i] = np.random.random()  # Placeholder
            
            if len(X) > 10:  # Assez de données pour entraîner
                X_train, X_test, y_train, y_test = train_test_split(X, y_prob, test_size=0.2, random_state=42)
                
                # Modèle Gradient Boosting
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Sauvegarder le modèle
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                
                models[f'ball_{ball_num}'] = model
                
                # Évaluer la performance
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                self.logger.info(f"  Modèle ball_{ball_num}: Train={train_score:.3f}, Test={test_score:.3f}")
        
        return models

    def _score_combination_advanced(self, combination: Tuple[int, ...], all_stats: Dict, models: Dict = None) -> float:
        """Score avancé d'une combinaison utilisant ML et statistiques."""
        if not combination:
            return 0.0
        
        features = self._calculate_features_for_combination(combination)
        
        # Score basé sur les statistiques historiques
        stat_score = 0.0
        
        # 1. Score de fréquence normalisée
        freq_scores = []
        for num in combination:
            freq = all_stats['numbers_analysis'].set_index('Numero').loc[num, 'Frequence']
            max_freq = all_stats['numbers_analysis']['Frequence'].max()
            freq_scores.append(freq / max_freq if max_freq > 0 else 0)
        stat_score += np.mean(freq_scores) * 0.3
        
        # 2. Score de retard (écart depuis dernière apparition)
        retard_scores = []
        for num in combination:
            retard = all_stats['numbers_analysis'].set_index('Numero').loc[num, 'Jours_Depuis_Tirage']
            ecart_moyen = all_stats['numbers_analysis'].set_index('Numero').loc[num, 'Ecart_Moyen_Tirages']
            if pd.notna(ecart_moyen) and ecart_moyen > 0:
                retard_ratio = retard / (ecart_moyen * 2)  # Facteur de retard
                retard_scores.append(min(retard_ratio, 2))
            else:
                retard_scores.append(0.5)
        stat_score += np.mean(retard_scores) * 0.25
        
        # 3. Score de diversité statistique
        diversity_score = 0.0
        diversity_score += features['last_digit_diversity'] / 5 * 0.1  # Diversité des unités
        diversity_score += features['decade_diversity'] / 5 * 0.1      # Diversité des dizaines
        
        # Zones équilibrées (idéal: 1-2 par zone)
        zone_balance = 1 - abs(features['zone_low'] - 1.7) / 5
        zone_balance += 1 - abs(features['zone_mid'] - 1.7) / 5  
        zone_balance += 1 - abs(features['zone_high'] - 1.6) / 5
        diversity_score += zone_balance / 3 * 0.15
        
        stat_score += diversity_score
        
        # 4. Pénalité pour les patterns trop évidents
        penalty = 0.0
        if features['consecutive_pairs'] > 2:  # Trop de suites
            penalty += 0.1
        if abs(features['even_count'] - 2.5) > 2:  # Déséquilibre pair/impair extrême
            penalty += 0.1
        if features['sum'] < 80 or features['sum'] > 180:  # Somme trop extrême
            penalty += 0.1
            
        stat_score = max(0, stat_score - penalty)
        
        # 5. Score ML si disponible
        ml_score = 0.0
        if models:
            try:
                # Extraire les features dans le même ordre que l'entraînement
                feature_order = ['sum', 'range', 'even_count', 'consecutive_pairs', 
                               'zone_low', 'zone_mid', 'zone_high', 'gap_variance',
                               'last_digit_diversity', 'decade_diversity']
                X_features = np.array([[features[f] for f in feature_order]]).reshape(1, -1)
                ml_scores = []
                for model_name, model in models.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_features)[0]
                        ml_scores.append(pred)
                if ml_scores:
                    ml_score = np.mean(ml_scores) * 0.2
            except Exception as e:
                self.logger.warning(f"Erreur ML scoring: {e}")
        
        final_score = (stat_score * 0.8 + ml_score * 0.2) * 100
        return min(100, max(0, final_score))

    def _generate_smart_combinations(self, prediction_weights: Dict[int, float], num_combinations: int = 1000) -> List[Tuple[int, ...]]:
        """Génère des combinaisons intelligentes basées sur les poids et contraintes."""
        combinations_set = set()
        
        # Trier les numéros par poids
        sorted_numbers = sorted(prediction_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Pool de candidats élargi (top 60% des numéros)
        candidate_pool = [num for num, _ in sorted_numbers[:int(len(sorted_numbers) * 0.6)]]
        
        # Génération de combinaisons diversifiées
        attempts = 0
        max_attempts = num_combinations * 10
        
        while len(combinations_set) < num_combinations and attempts < max_attempts:
            attempts += 1
            
            # Stratégie mixte : 70% basé sur les poids, 30% aléatoire pour la diversité
            if np.random.random() < 0.7:
                # Sélection pondérée
                selected = []
                available = candidate_pool.copy()
                
                for _ in range(self.numbers_per_draw):
                    if not available:
                        break
                    
                    # Probabilités basées sur les poids
                    weights = [prediction_weights[num] for num in available]
                    weights = np.array(weights)
                    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
                    
                    # S'assurer que les probabilités sont valides
                    weights = np.clip(weights, 0, 1)
                    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
                    
                    chosen = np.random.choice(available, p=weights)
                    selected.append(chosen)
                    available.remove(chosen)
                
                if len(selected) == self.numbers_per_draw:
                    combination = tuple(sorted(selected))
                    combinations_set.add(combination)
            else:
                # Sélection partiellement aléatoire pour la diversité
                combination = tuple(sorted(np.random.choice(candidate_pool, self.numbers_per_draw, replace=False)))
                combinations_set.add(combination)
        
        return list(combinations_set)

    def _validate_combination_quality(self, combination: Tuple[int, ...], all_stats: Dict) -> Dict:
        """Valide la qualité statistique d'une combinaison."""
        features = self._calculate_features_for_combination(combination)
        
        validation = {
            'sum_ok': 90 <= features['sum'] <= 170,  # Somme dans la plage normale
            'balance_ok': 1 <= features['even_count'] <= 4,  # Équilibre pair/impair
            'zones_ok': all(features[f'zone_{zone}'] >= 1 for zone in ['low', 'mid', 'high']),  # Toutes les zones représentées
            'not_too_consecutive': features['consecutive_pairs'] <= 2,  # Pas trop de consécutifs
            'diversity_ok': features['last_digit_diversity'] >= 3,  # Diversité des unités
            'range_ok': 20 <= features['range'] <= 45,  # Étendue raisonnable
        }
        
        validation['quality_score'] = sum(validation.values()) / len(validation)
        return validation

    def _analyze_advanced_patterns(self, df_pandas: pd.DataFrame) -> Dict:
        """Analyse des patterns avancés dans l'historique."""
        patterns = {
            'hot_cold_cycles': {},
            'seasonal_trends': {},
            'correlation_matrix': None,
            'sequence_patterns': {},
            'gap_analysis': {}
        }
        
        # Analyse des cycles chaud/froid
        for num in self.BALLS:
            appearances = []
            for _, row in df_pandas.iterrows():
                if num in [row[col] for col in self.balls_cols]:
                    appearances.append(row.name)  # Index du tirage
            
            if len(appearances) > 1:
                gaps = np.diff(appearances)
                patterns['hot_cold_cycles'][num] = {
                    'avg_gap': np.mean(gaps),
                    'gap_std': np.std(gaps),
                    'last_gap': len(df_pandas) - appearances[-1] if appearances else len(df_pandas)
                }
        
        # Matrice de corrélation des numéros
        number_matrix = np.zeros((len(df_pandas), self.max_number))
        for i, (_, row) in enumerate(df_pandas.iterrows()):
            for col in self.balls_cols:
                if pd.notna(row[col]):
                    number_matrix[i, int(row[col]) - 1] = 1
        
        if number_matrix.sum() > 0:
            patterns['correlation_matrix'] = np.corrcoef(number_matrix.T)
        
        # Analyse des séquences
        for seq_len in [2, 3]:
            sequences = defaultdict(int)
            for _, row in df_pandas.iterrows():
                nums = sorted([row[col] for col in self.balls_cols if pd.notna(row[col])])
                for i in range(len(nums) - seq_len + 1):
                    seq = tuple(nums[i:i + seq_len])
                    sequences[seq] += 1
            patterns['sequence_patterns'][seq_len] = dict(sequences)
        
        return patterns

    def _calculate_momentum_score(self, num: int, df_pandas: pd.DataFrame, window: int = 10) -> float:
        """Calcule un score de momentum pour un numéro."""
        recent_draws = df_pandas.tail(window)
        recent_appearances = 0
        
        for _, row in recent_draws.iterrows():
            if num in [row[col] for col in self.balls_cols if pd.notna(row[col])]:
                recent_appearances += 1
        
        # Score basé sur la fréquence récente vs fréquence historique
        recent_freq = recent_appearances / window
        
        # Fréquence historique
        total_appearances = 0
        for _, row in df_pandas.iterrows():
            if num in [row[col] for col in self.balls_cols if pd.notna(row[col])]:
                total_appearances += 1
        
        historical_freq = total_appearances / len(df_pandas) if len(df_pandas) > 0 else 0
        
        # Momentum = ratio entre fréquence récente et historique
        if historical_freq > 0:
            momentum = recent_freq / historical_freq
            return min(momentum, 3.0)  # Plafonner à 3
        
        return 1.0

    def _predict_next_draw_probabilities(self, all_stats: Dict, models: Dict = None) -> Dict[int, float]:
        """Prédit les probabilités pour le prochain tirage."""
        probabilities = {}
        df_pandas = all_stats['df_pandas']
        
        # Analyse des patterns avancés
        advanced_patterns = self._analyze_advanced_patterns(df_pandas)
        
        for num in self.BALLS:
            prob = 0.0
            
            # 1. Probabilité basée sur la fréquence (30%)
            freq = all_stats['numbers_analysis'].set_index('Numero').loc[num, 'Frequence']
            max_freq = all_stats['numbers_analysis']['Frequence'].max()
            freq_prob = (freq / max_freq) * 0.3 if max_freq > 0 else 0.0
            
            # 2. Probabilité basée sur le retard (25%)
            retard = all_stats['numbers_analysis'].set_index('Numero').loc[num, 'Jours_Depuis_Tirage']
            ecart_moyen = all_stats['numbers_analysis'].set_index('Numero').loc[num, 'Ecart_Moyen_Tirages']
            
            if pd.notna(ecart_moyen) and ecart_moyen > 0:
                retard_ratio = retard / ecart_moyen
                retard_prob = min(retard_ratio / 2, 1.0) * 0.25
            else:
                retard_prob = 0.125  # Valeur neutre
            
            # 3. Score de momentum (20%)
            momentum_prob = self._calculate_momentum_score(num, df_pandas) * 0.2 / 3.0
            
            # 4. Analyse des cycles (15%)
            cycle_prob = 0.15 / 2  # Valeur neutre par défaut
            if num in advanced_patterns['hot_cold_cycles']:
                cycle_data = advanced_patterns['hot_cold_cycles'][num]
                expected_gap = cycle_data['avg_gap']
                current_gap = cycle_data['last_gap']
                
                if expected_gap > 0:
                    gap_ratio = current_gap / expected_gap
                    cycle_prob = min(gap_ratio / 2, 1.0) * 0.15
            
            # 5. Corrélation avec numéros récents (10%)
            correlation_prob = 0.05  # Valeur neutre par défaut
            if advanced_patterns['correlation_matrix'] is not None and len(df_pandas) > 0:
                recent_numbers = set()
                for _, row in df_pandas.tail(3).iterrows():
                    recent_numbers.update([row[col] for col in self.balls_cols if pd.notna(row[col])])
                
                if recent_numbers:
                    corr_scores = []
                    for recent_num in recent_numbers:
                        if 1 <= recent_num <= self.max_number:
                            corr = advanced_patterns['correlation_matrix'][num-1, int(recent_num)-1]
                            if not np.isnan(corr):
                                corr_scores.append(abs(corr))
                    
                    if corr_scores:
                        correlation_prob = np.mean(corr_scores) * 0.1
            
            # Probabilité finale
            prob = freq_prob + retard_prob + momentum_prob + cycle_prob + correlation_prob
            probabilities[num] = min(prob, 1.0)
        
        return probabilities

def main():
    parser = argparse.ArgumentParser(
        description="🎯 Générateur de Grilles Loto Stratégique",
        formatter_class=argparse.RawTextHelpFormatter
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--csv', type=str, help='Chemin vers fichier CSV des tirages')
    
    parser.add_argument('--grids', type=int, default=1, help='Nombre de grilles à générer (défaut: 1)')
    parser.add_argument('--plots', action='store_true', help='Activer la génération de graphiques d\'analyse')
    parser.add_argument('--export-stats', action='store_true', help='Exporter les statistiques en fichiers CSV')
    
    parser.add_argument('--config-file', type=str, default='strategies.yml', help="Fichier de configuration des stratégies (défaut: strategies.yml)")
    parser.add_argument('--strategy', type=str, default='equilibre', help="Nom de la stratégie à utiliser (défaut: equilibre)")

    args = parser.parse_args()

    try:
        strategist = LotoStrategist(config_file=args.config_file, strategy_name=args.strategy)
        
        db_con = duckdb.connect(':memory:')
        table_name = strategist.load_data_from_csv(args.csv, db_con)

        results, all_stats = strategist.generate_grids(db_con, table_name, args.grids)
        
        strategist.print_results(results)

        res= pd.DataFrame(results)
        res.to_csv(os.path.join(strategist.STATS_DIR,"grilles.csv"))
        print(f"\n💾 Grilles sauvegardées dans 'grilles.csv'")
        if args.plots:
            strategist._create_visualizations(all_stats)
        if args.export_stats:
            strategist._export_statistics_to_csv(all_stats)
        
        db_con.close()

    except (Exception, FileNotFoundError, ValueError) as e:
        logging.getLogger().error(f"❌ Une erreur critique est survenue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()