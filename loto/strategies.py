import duckdb
import yaml
import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, Any, List, Tuple, Set
from scipy.stats import zscore
from datetime import datetime
import logging

# Configuration du logging pour le debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.expanduser("~")
# Utiliser le CSV téléchargé automatiquement
LOTO_CSV_PATH = "/home/nvernot/projets/loto_keno/loto/loto_data/loto_201911.csv"
STATS_DIR = "/home/nvernot/projets/loto_keno/loto_stats_exports" 
STRATEGIES_FILE = "/home/nvernot/projets/loto_keno/loto/strategies.yml"

# Cache pour éviter les recalculs
_cache = {}

class AdvancedLotoAnalyzer:
    """
    Analyseur Loto avancé intégrant des statistiques enrichies,
    un backtesting et une stratégie de sélection de portefeuille.
    Optimisé pour les performances avec cache et vectorisation.
    """
    def __init__(self, strategies: Dict[str, Any]):
        self.strategies = strategies
        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.historical_features_df = None
        self.number_stats_df = None
        self.full_history_df = None  # Cache pour éviter de recharger
        self._features_cache = {}  # Cache pour les features calculées
        
        # Chargement unique des données statistiques
        self._load_number_stats()

    def _load_number_stats(self):
        """Charge les statistiques des numéros une seule fois"""
        try:
            stats_file = os.path.join(STATS_DIR, 'analyse_numeros.csv')
            self._check_file_exists(stats_file, "Analyse des numéros")
            self.number_stats_df = pd.read_csv(stats_file)
            logger.info("Statistiques des numéros chargées avec succès")
        except Exception as e:
            logger.warning(f"Impossible de charger les statistiques: {e}")
            self.number_stats_df = pd.DataFrame()

    def _check_file_exists(self, file_path: str, description: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ERREUR : '{description}' introuvable : {file_path}")

    def load_and_prepare_data(self, until_date: str = None):
        """
        Charge toutes les données et pré-calcule les caractéristiques statistiques
        de l'historique des tirages. `until_date` est utilisé pour le backtesting.
        Optimisé pour éviter les rechargements inutiles.
        """
        # Utiliser le cache si les données sont déjà chargées et pas de filtre de date
        if self.full_history_df is not None and until_date is None:
            self.historical_features_df = self._get_cached_features(self.full_history_df)
            return
            
        # Charger les données depuis CSV seulement si nécessaire
        if self.full_history_df is None:
            logger.info("Chargement initial des données...")
            self.con.execute(f"""
                CREATE OR REPLACE TABLE loto_draws AS
                SELECT 
                    COALESCE(try_strptime(CAST(date_de_tirage AS VARCHAR), '%d/%m/%Y'), try_strptime(CAST(date_de_tirage AS VARCHAR), '%Y-%m-%d')) AS date_de_tirage,
                    CAST(boule_1 AS INTEGER) AS b1, CAST(boule_2 AS INTEGER) AS b2, CAST(boule_3 AS INTEGER) AS b3, CAST(boule_4 AS INTEGER) AS b4, CAST(boule_5 AS INTEGER) AS b5,
                    CAST(numero_chance AS INTEGER) as nc
                FROM read_csv_auto('{LOTO_CSV_PATH}', delim=';', header=true);
            """)
            self.con.execute("DELETE FROM loto_draws WHERE date_de_tirage IS NULL OR b1 IS NULL;")
            self.full_history_df = self.con.execute("SELECT * FROM loto_draws ORDER BY date_de_tirage").fetchdf()
            logger.info(f"Chargé {len(self.full_history_df)} tirages")

        # Filtrer les données selon la date si nécessaire
        if until_date:
            filtered_df = self.full_history_df[self.full_history_df['date_de_tirage'] <= until_date]
        else:
            filtered_df = self.full_history_df
            
        if filtered_df.empty:
            self.historical_features_df = pd.DataFrame(columns=['sum', 'even_count', 'zone_B', 'zone_M', 'zone_H', 'finales_counts', 'dizaines_counts', 'has_suite'])
        else:
            # Calculer les features avec cache
            self.historical_features_df = self._get_cached_features(filtered_df)

    def _calculate_features_for_draws(self, draws: List[Tuple[int, ...]]) -> pd.DataFrame:
        """Méthode conservée pour compatibilité, redirige vers la version vectorisée"""
        return self._calculate_features_vectorized(draws)

    def _score_combination(self, combination: Tuple[int, ...]) -> float:
        """Calcul optimisé du score d'une combinaison"""
        if not isinstance(self.historical_features_df, pd.DataFrame) or self.historical_features_df.empty:
            return 50.0 # Retourne un score neutre si pas d'historique pour comparer

        # Utilisation de la version vectorisée pour calculer les features
        combi_features_df = self._calculate_features_vectorized([combination])
        combi_features = combi_features_df.iloc[0]
        
        weights = {'sum': 0.30, 'even_count': 0.20, 'zone_B': 0.10, 'zone_M': 0.10, 'zone_H': 0.10, 
                   'finales_counts': 0.05, 'dizaines_counts': 0.05, 'has_suite': 0.10}

        # Calculs vectorisés pour tous les features
        total_score = 0
        for feature, weight in weights.items():
            all_values = self.historical_features_df[feature]
            combi_value = combi_features[feature]
            
            mean = all_values.mean()
            std = all_values.std()
            z = 0 if std == 0 else (combi_value - mean) / std
            
            normality_score = np.exp(-0.5 * z**2)
            total_score += normality_score * weight

        return total_score * 100 / sum(weights.values())

    def _analyze_chance_numbers(self, context: pd.DataFrame) -> pd.DataFrame:
        if context.empty or 'nc' not in context.columns:
            return pd.DataFrame(columns=['number', 'frequency'])
        chance_counts = context['nc'].value_counts().reset_index()
        chance_counts.columns = ['number', 'frequency']
        return chance_counts.sort_values(by='frequency', ascending=False)
        
    def generate_combinations_for_strategy(self, strategy_name: str, context: Dict) -> List[Dict]:
        """Génération optimisée des combinaisons avec réduction du pool"""
        weights = self.strategies[strategy_name]['weights']
        scored_numbers_df = self._calculate_scores(weights, context)
        if scored_numbers_df.empty:
            return []
            
        # Réduire le pool de candidats pour les performances
        candidate_pool = scored_numbers_df.head(15)['number'].tolist()  # Réduit de 20 à 15
        
        all_combis = list(combinations(candidate_pool, 5))
        # Limiter encore plus pour le backtest
        max_combis = 10000 if len(all_combis) > 10000 else len(all_combis)
        
        if len(all_combis) > max_combis:
            indices = np.random.choice(len(all_combis), max_combis, replace=False)
            all_combis = [all_combis[i] for i in indices]

        # Utilisation de list comprehension pour de meilleures performances
        scored_combis = [{'combination': sorted(combi), 'score': self._score_combination(combi)} for combi in all_combis]
            
        return sorted(scored_combis, key=lambda x: x['score'], reverse=True)

    def run_analysis(self):
        logger.info("Lancement en mode PREDICTION...")
        self.load_and_prepare_data()

        # Utiliser les données en cache plutôt que de refaire une requête
        draws_df = self.full_history_df
        context = {
            'recent_draws': draws_df.tail(20),
            'recent_numbers': set(pd.concat([draws_df.tail(3)[f'b{i}'] for i in range(1,6)]).dropna().astype(int))
        }

        best_chance_numbers = self._analyze_chance_numbers(draws_df).head(3)['number'].tolist()

        ensemble_results = {}
        all_strategy_results = {}

        for name in self.strategies:
            logger.info(f"Évaluation de la stratégie : '{name}'...")
            top_combinations = self.generate_combinations_for_strategy(name, context)
            if top_combinations:
                best_combi = top_combinations[0]
                ensemble_results[name] = {**best_combi, 'chance': best_chance_numbers[0] if best_chance_numbers else np.random.randint(1,11)}
                all_strategy_results[name] = top_combinations

        self._present_prediction_results(ensemble_results, all_strategy_results, best_chance_numbers)
    
    def run_backtest(self, num_draws_to_test: int):
        logger.info(f"Lancement en mode BACKTEST sur les {num_draws_to_test} derniers tirages...")
        
        # Charger toutes les données une seule fois
        self.load_and_prepare_data()
        
        # S'assurer de ne pas tester plus de tirages qu'il n'y en a
        num_draws_to_test = min(num_draws_to_test, len(self.full_history_df))
        if num_draws_to_test == 0:
            logger.warning("Pas assez de données pour le backtest.")
            return
            
        draws_to_test = self.full_history_df.tail(num_draws_to_test)
        strategy_scores = {name: {'main_matches': 0, 'chance_matches': 0} for name in self.strategies}

        for index, actual_draw in draws_to_test.iterrows():
            test_date = actual_draw['date_de_tirage']
            logger.info(f"Backtest pour le tirage du {test_date.strftime('%Y-%m-%d')}...")

            # Filtrer les données avec pandas plutôt qu'avec SQL
            context_df = self.full_history_df[self.full_history_df['date_de_tirage'] < test_date]
            
            if context_df.empty:
                logger.warning("Pas de données antérieures pour ce tirage, test ignoré.")
                continue

            # Calculer les features pour ce contexte
            self.historical_features_df = self._get_cached_features(context_df)

            context = {
                'recent_draws': context_df.tail(20),
                'recent_numbers': set(pd.concat([context_df.tail(3)[f'b{i}'] for i in range(1,6)]).dropna().astype(int))
            }

            actual_main_numbers = set(actual_draw[[f'b{i}' for i in range(1, 6)]])
            actual_chance = actual_draw['nc']
            best_chance_numbers = self._analyze_chance_numbers(context_df).head(1)['number'].tolist()

            for name in self.strategies:
                top_combinations = self.generate_combinations_for_strategy(name, context)
                if top_combinations:
                    predicted_grid = set(top_combinations[0]['combination'])
                    matches = len(predicted_grid.intersection(actual_main_numbers))
                    strategy_scores[name]['main_matches'] += matches
                    if best_chance_numbers and best_chance_numbers[0] == actual_chance:
                        strategy_scores[name]['chance_matches'] += 1

        self._present_backtest_results(strategy_scores, num_draws_to_test)

    def _present_prediction_results(self, ensemble, all_results, chance):
        print("\n" + "="*80)
        print(" RÉSULTATS DE L'ANALYSE ".center(80, "="))
        print("="*80)
        
        if not ensemble:
            print("Aucune stratégie n'a pu générer de résultat.")
            return

        best_strategy_name = max(ensemble, key=lambda k: ensemble[k]['score'])
        
        print("\n🏆 APPROCHE ENSEMBLE (Meilleure grille de chaque stratégie) 🏆")
        for name, result in ensemble.items():
            is_best = "⭐" if name == best_strategy_name else ""
            print(f"  - {name.capitalize():<15} {is_best} | {str(result['combination']):<25} | Chance: {result['chance']:<3} (Score: {result['score']:.2f})")

        print("\n📈 PORTEFEUILLE DE GRILLES (Top 5 de la meilleure stratégie) 📈")
        print(f"Stratégie la plus performante : {best_strategy_name.capitalize()}")
        top_5_grids = all_results.get(best_strategy_name, [])[:5]
        for i, grid_info in enumerate(top_5_grids):
            print(f"  {i+1}. {str(grid_info['combination']):<25} (Score: {grid_info['score']:.2f})")
        print(f"\nNuméros Chance recommandés (par ordre de pertinence) : {chance}")
        print("\n" + "="*80)

    def _present_backtest_results(self, scores, num_draws):
        print("\n" + "="*80)
        print(" RÉSULTATS DU BACKTESTING ".center(80, "="))
        print("="*80)
        print(f"Évaluation sur {num_draws} tirages.")
        
        sorted_scores = sorted(scores.items(), key=lambda item: item[1]['main_matches'], reverse=True)
        
        print("\n{:<20} | {:<25} | {:<20}".format("Stratégie", "Bons numéros trouvés", "N° Chance trouvés"))
        print("-"*(20+3+25+3+20))
        for name, result in sorted_scores:
            print("{:<20} | {:<25} | {:<20}".format(name.capitalize(), result['main_matches'], result['chance_matches']))
        
        if not sorted_scores:
             print("Aucun score à afficher.")
             return

        best_strategy = sorted_scores[0][0]
        print(f"\n🏆 Stratégie la plus performante en backtest : {best_strategy.capitalize()}")
        print("\n" + "="*80)

    def _calculate_scores(self, weights, context):
        """Version optimisée du calcul de scores avec vectorisation pandas"""
        if self.number_stats_df.empty:
            return pd.DataFrame(columns=['number', 'score'])
            
        momentum_counts = Counter(pd.concat([context['recent_draws'][f'b{i}'] for i in range(1,6)]).dropna())
        
        # Vectorisation avec pandas pour de meilleures performances
        df = self.number_stats_df.copy()
        max_freq = df['Frequence'].max()
        
        # Calculs vectorisés
        df['freq_score'] = df['Frequence'] / max_freq if max_freq > 0 else 0
        df['recency_gap_score'] = df['Jours_Depuis_Tirage'] / (df['Ecart_Moyen_Tirages'] + 1)
        df['momentum_score'] = df['Numero'].map(lambda x: momentum_counts.get(x, 0) / 20)
        
        # Calcul du score final
        df['score'] = (
            weights.get('frequency', 0) * df['freq_score'] +
            weights.get('recency_gap', 0) * df['recency_gap_score'] +
            weights.get('momentum', 0) * df['momentum_score']
        )
        
        # Appliquer la pénalité pour les numéros récents
        recent_penalty = weights.get('recent_draws_penalty', 1.0)
        recent_mask = df['Numero'].isin(context['recent_numbers'])
        df.loc[recent_mask, 'score'] *= recent_penalty
        
        return df[['Numero', 'score']].rename(columns={'Numero': 'number'}).sort_values(by='score', ascending=False)

    def _get_cached_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Récupère ou calcule les features avec cache"""
        cache_key = f"features_{len(df)}_{df.iloc[-1]['date_de_tirage'] if not df.empty else 'empty'}"
        
        if cache_key in self._features_cache:
            return self._features_cache[cache_key]
            
        if df.empty:
            features_df = pd.DataFrame(columns=['sum', 'even_count', 'zone_B', 'zone_M', 'zone_H', 'finales_counts', 'dizaines_counts', 'has_suite'])
        else:
            draws = [tuple(sorted([row[f'b{i}'] for i in range(1, 6)])) for _, row in df.iterrows()]
            features_df = self._calculate_features_vectorized(draws)
            
        self._features_cache[cache_key] = features_df
        return features_df

    def _calculate_features_vectorized(self, draws: List[Tuple[int, ...]]) -> pd.DataFrame:
        """Version vectorisée du calcul de features pour de meilleures performances"""
        if not draws:
            return pd.DataFrame(columns=['sum', 'even_count', 'zone_B', 'zone_M', 'zone_H', 'finales_counts', 'dizaines_counts', 'has_suite'])
            
        # Convertir en numpy array pour la vectorisation
        draws_array = np.array(draws)
        
        features = {
            'sum': np.sum(draws_array, axis=1),
            'even_count': np.sum(draws_array % 2 == 0, axis=1),
            'zone_B': np.sum((draws_array >= 1) & (draws_array <= 16), axis=1),
            'zone_M': np.sum((draws_array >= 17) & (draws_array <= 33), axis=1),
            'zone_H': np.sum((draws_array >= 34) & (draws_array <= 49), axis=1),
        }
        
        # Calculs plus complexes nécessitant des boucles optimisées
        finales_counts = []
        dizaines_counts = []
        has_suite = []
        
        for draw in draws:
            # Finales
            finales = [n % 10 for n in draw]
            finales_counts.append(max(Counter(finales).values()) if finales else 0)
            
            # Dizaines
            dizaines = [n // 10 for n in draw]
            dizaines_counts.append(max(Counter(dizaines).values()) if dizaines else 0)
            
            # Suites
            sorted_draw = sorted(draw)
            has_suite.append(any(sorted_draw[i+1] - sorted_draw[i] == 1 for i in range(len(sorted_draw)-1)))
        
        features['finales_counts'] = finales_counts
        features['dizaines_counts'] = dizaines_counts
        features['has_suite'] = has_suite
        
        return pd.DataFrame(features)

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['predict', 'backtest']:
        print("Erreur : Veuillez spécifier le mode d'exécution.")
        print("Usage : python votre_script.py predict")
        print("   OU : python votre_script.py backtest <nombre_de_tirages>")
        sys.exit(1)

    try:
        # Vérification des fichiers requis avant l'initialisation
        def check_file_exists(file_path: str, description: str):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"'{description}' introuvable : {file_path}")

        check_file_exists(LOTO_CSV_PATH, "Historique Loto")
        check_file_exists(STRATEGIES_FILE, "Fichier de stratégies")

        # Chargement unique des stratégies
        with open(STRATEGIES_FILE, 'r') as f:
            strategies = yaml.safe_load(f)['strategies']
        
        # Initialisation de l'analyseur avec les stratégies
        analyzer = AdvancedLotoAnalyzer(strategies)
        
        mode = sys.argv[1]
        start_time = datetime.now()
        
        if mode == 'predict':
            analyzer.run_analysis()
        elif mode == 'backtest':
            if len(sys.argv) < 3:
                print("Erreur : Veuillez spécifier le nombre de tirages à backtester.")
                sys.exit(1)
            num_draws = int(sys.argv[2])
            analyzer.run_backtest(num_draws)
        
        end_time = datetime.now()
        logger.info(f"Exécution terminée en {(end_time - start_time).total_seconds():.2f} secondes")

    except FileNotFoundError as e:
        logger.error(f"[ERREUR FATALE] {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"[ERREUR] Paramètre invalide : {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        logger.error(f"[ERREUR INATTENDUE] {e}")
        traceback.print_exc()
        sys.exit(1)