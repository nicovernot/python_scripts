#!/usr/bin/env python3
"""
Générateur de grilles Keno intelligent
Basé sur l'analyse statistique complète des données 2020-2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import csv
from datetime import datetime
import json
from collections import Counter, defaultdict

class KenoIntelligentGenerator:
    """Générateur intelligent de grilles Keno"""
    
    def __init__(self, data_dir="keno/keno_data"):
        self.data_dir = Path(data_dir)
        self.frequencies = {}
        self.delays = {}
        self.pairs = {}
        self.triplets = {}
        
        # Paramètres de stratégie
        self.strategies = {
            'hot': {'weight': 0.4, 'desc': 'Numéros fréquents'},
            'cold': {'weight': 0.2, 'desc': 'Numéros en retard'},
            'balanced': {'weight': 0.3, 'desc': 'Équilibré fréquence/retard'},
            'pairs': {'weight': 0.1, 'desc': 'Paires fréquentes'}
        }
        
    def load_analysis_results(self):
        """Charge les résultats d'analyse depuis le CSV"""
        try:
            # Cherche le fichier de fréquences le plus récent
            freq_files = list(Path("keno_stats_exports").glob("frequences_keno_*.csv"))
            if not freq_files:
                print("❌ Aucun fichier d'analyse trouvé. Lancez d'abord duckdb_keno.py pour générer les statistiques")
                return False
            
            latest_file = max(freq_files, key=lambda f: f.stat().st_mtime)
            
            df = pd.read_csv(latest_file)
            self.frequencies = dict(zip(df['numero'], df['frequence']))
            
            print(f"✅ Fréquences chargées depuis {latest_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return False
    
    def analyze_current_delays(self):
        """Analyse les retards actuels sur les 50 derniers tirages"""
        try:
            # Charge les fichiers les plus récents
            csv_files = sorted(list(self.data_dir.glob("keno_*.csv")))[-3:]  # 3 derniers mois
            
            all_recent = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    tirage = {
                        'date': row['date'],
                        'boules': [int(row[f'b{i}']) for i in range(1, 21)]
                    }
                    all_recent.append(tirage)
            
            # Garde les 50 derniers
            recent_tirages = sorted(all_recent, key=lambda x: x['date'])[-50:]
            
            # Calcule les retards
            delays = {i: 50 for i in range(1, 71)}
            
            for idx, tirage in enumerate(reversed(recent_tirages)):
                for boule in tirage['boules']:
                    if delays[boule] == 50:
                        delays[boule] = idx
            
            self.delays = delays
            print(f"✅ Retards calculés sur {len(recent_tirages)} tirages récents")
            return True
            
        except Exception as e:
            print(f"❌ Erreur calcul retards : {e}")
            return False
    
    def get_number_score(self, numero):
        """Calcule le score d'un numéro selon plusieurs critères"""
        freq_score = self.frequencies.get(numero, 0) / 1100  # Normalise sur freq max
        delay_score = min(self.delays.get(numero, 0) / 30, 1)  # Bonus retard
        
        # Score composite
        score = (freq_score * 0.6) + (delay_score * 0.4)
        return score
    
    def generate_smart_grid(self, strategy='balanced', num_count=10):
        """Génère une grille intelligente selon la stratégie"""
        all_numbers = list(range(1, 71))
        
        if strategy == 'hot':
            # Privilégie les numéros fréquents
            scores = [(num, self.frequencies.get(num, 0)) for num in all_numbers]
            scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [num for num, _ in scores[:20]]
            
        elif strategy == 'cold':
            # Privilégie les numéros en retard
            scores = [(num, self.delays.get(num, 0)) for num in all_numbers]
            scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [num for num, _ in scores[:20]]
            
        elif strategy == 'balanced':
            # Équilibre fréquence et retard
            scores = [(num, self.get_number_score(num)) for num in all_numbers]
            scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [num for num, _ in scores[:25]]
            
        else:  # random
            candidates = all_numbers
        
        # Sélection finale avec un peu d'aléatoire
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        
        if len(candidates) >= num_count:
            selected = np.random.choice(candidates, num_count, replace=False)
        else:
            selected = candidates + list(np.random.choice(
                [n for n in all_numbers if n not in candidates], 
                num_count - len(candidates), 
                replace=False
            ))
        
        # Conversion en int Python pour éviter les types numpy
        return sorted([int(x) for x in selected])
    
    def generate_multiple_grids(self, count=5, num_per_grid=10):
        """Génère plusieurs grilles avec différentes stratégies"""
        grids = []
        
        strategies = ['hot', 'cold', 'balanced', 'balanced', 'balanced']  # Plus de balanced
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            grid = self.generate_smart_grid(strategy, num_per_grid)
            
            # Calcule le score de la grille
            grid_score = sum(self.get_number_score(num) for num in grid) / len(grid)
            
            grids.append({
                'grille': grid,
                'strategie': strategy,
                'score': grid_score,
                'description': self.strategies.get(strategy, {}).get('desc', strategy)
            })
        
        # Trie par score décroissant
        grids.sort(key=lambda x: x['score'], reverse=True)
        
        return grids
    
    def analyze_grid_stats(self, grid):
        """Analyse les statistiques d'une grille"""
        freq_total = sum(self.frequencies.get(num, 0) for num in grid)
        freq_avg = freq_total / len(grid)
        
        delay_total = sum(self.delays.get(num, 0) for num in grid)
        delay_avg = delay_total / len(grid)
        
        # Répartition par zones
        zones = {
            'faible': len([n for n in grid if 1 <= n <= 23]),
            'moyenne': len([n for n in grid if 24 <= n <= 46]),
            'forte': len([n for n in grid if 47 <= n <= 70])
        }
        
        # Parité
        pairs = len([n for n in grid if n % 2 == 0])
        impairs = len(grid) - pairs
        
        return {
            'freq_moyenne': freq_avg,
            'retard_moyen': delay_avg,
            'zones': zones,
            'parite': {'pairs': pairs, 'impairs': impairs}
        }
    
    def display_recommendations(self, grids):
        """Affiche les recommandations de grilles"""
        print(f"\n🎯 RECOMMANDATIONS DE GRILLES KENO")
        print("=" * 70)
        
        for i, grid_info in enumerate(grids, 1):
            grid = grid_info['grille']
            strategy = grid_info['strategie']
            score = grid_info['score']
            desc = grid_info['description']
            
            print(f"\n🎲 GRILLE #{i} - {desc.upper()}")
            print(f"   Numéros : {grid}")
            print(f"   Score   : {score:.3f}")
            
            # Statistiques détaillées
            stats = self.analyze_grid_stats(grid)
            print(f"   Fréq moy: {stats['freq_moyenne']:.1f}")
            print(f"   Retard  : {stats['retard_moyen']:.1f} tirages")
            print(f"   Zones   : {stats['zones']['faible']}-{stats['zones']['moyenne']}-{stats['zones']['forte']}")
            
            # ...avant d'utiliser total_pairs...
            total_pairs = stats['parite']['pairs']
            total_impairs = stats['parite']['impairs']
            # ...utilisation...
            print(f"   Parité  : {total_pairs}P/{total_impairs}I")
        
        print(f"\n💡 CONSEILS D'UTILISATION :")
        print(f"   • Les grilles sont classées par score de performance")
        print(f"   • Variez les stratégies pour maximiser vos chances")
        print(f"   • Le Keno reste un jeu de hasard - jouez responsablement")
    
    def save_grids(self, grids):
        """Sauvegarde les grilles générées"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV pour les grilles
        output_dir = Path("keno_output")
        output_dir.mkdir(exist_ok=True)
        
        grid_file = output_dir / f"grilles_keno_{timestamp}.csv"
        
        with open(grid_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['grille_id', 'strategie', 'score'] + [f'n{i}' for i in range(1, 11)])
            
            for i, grid_info in enumerate(grids, 1):
                row = [i, grid_info['strategie'], f"{grid_info['score']:.3f}"] + grid_info['grille']
                writer.writerow(row)
        
        # Rapport détaillé
        report_file = output_dir / f"rapport_keno_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("RAPPORT DE GÉNÉRATION KENO\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}\n")
            f.write(f"Nombre de grilles : {len(grids)}\n\n")
            
            for i, grid_info in enumerate(grids, 1):
                f.write(f"GRILLE #{i} - {grid_info['description']}\n")
                f.write(f"Numéros : {grid_info['grille']}\n")
                f.write(f"Score : {grid_info['score']:.3f}\n")
                
                stats = self.analyze_grid_stats(grid_info['grille'])
                total_pairs = stats['parite']['pairs']
                total_impairs = stats['parite']['impairs']
                f.write(f"Fréquence moyenne : {stats['freq_moyenne']:.1f}\n")
                f.write(f"Retard moyen : {stats['retard_moyen']:.1f}\n")
                f.write(f"Répartition zones : {stats['zones']}\n")
                f.write(f"Parité : {total_pairs}P/{total_impairs}I\n\n")
        
        print(f"\n💾 SAUVEGARDE :")
        print(f"   📊 Grilles : {grid_file}")
        print(f"   📄 Rapport : {report_file}")
    
    def run_generation(self, num_grids=5):
        """Lance la génération complète"""
        print("🎯 GÉNÉRATEUR INTELLIGENT KENO")
        print("=" * 50)
        
        # 1. Charge les analyses
        if not self.load_analysis_results():
            return False
        
        # 2. Analyse les retards actuels  
        if not self.analyze_current_delays():
            return False
        
        # 3. Génère les grilles
        print(f"\n🎲 Génération de {num_grids} grilles...")
        grids = self.generate_multiple_grids(num_grids)
        
        # 4. Affiche les recommandations
        self.display_recommendations(grids)
        
        # 5. Sauvegarde
        self.save_grids(grids)
        
        print(f"\n✅ Génération terminée avec succès !")
        return True

def main():
    generator = KenoIntelligentGenerator()
    success = generator.run_generation(num_grids=5)
    
    if not success:
        print("❌ Échec de la génération")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
