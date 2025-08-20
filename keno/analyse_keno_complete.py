#!/usr/bin/env python3
"""
Analyse complète des données Keno
Traite tous les fichiers CSV disponibles de 2020 à 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime
import csv
from collections import Counter, defaultdict

class KenoAnalysisComplete:
    """Analyseur complet des données Keno"""
    
    def __init__(self, data_dir="keno/keno_data"):
        self.data_dir = Path(data_dir)
        self.all_tirages = []
        self.frequencies = Counter()
        self.last_seen = {}
        
    def load_all_data(self):
        """Charge tous les fichiers CSV disponibles"""
        csv_files = sorted(glob.glob(str(self.data_dir / "keno_*.csv")))
        
        if not csv_files:
            print("❌ Aucun fichier CSV trouvé")
            return False
        
        print(f"📁 Fichiers trouvés : {len(csv_files)}")
        
        total_tirages = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_tirages = len(df)
                total_tirages += file_tirages
                
                # Parse chaque tirage
                for idx, row in df.iterrows():
                    tirage_data = {
                        'date': row['date'],
                        'numero': row['numero_tirage'],
                        'boules': [int(row[f'b{i}']) for i in range(1, 21)]
                    }
                    self.all_tirages.append(tirage_data)
                
                filename = Path(csv_file).name
                print(f"   ✓ {filename}: {file_tirages} tirages")
                
            except Exception as e:
                print(f"   ❌ Erreur avec {csv_file}: {e}")
        
        print(f"\n📊 Total : {total_tirages} tirages chargés")
        
        # Tri par date
        self.all_tirages.sort(key=lambda x: x['date'])
        return True
    
    def analyze_frequencies(self):
        """Analyse les fréquences de tous les numéros"""
        print("\n🔥 ANALYSE DES FRÉQUENCES")
        print("=" * 50)
        
        # Compte toutes les occurrences
        for tirage in self.all_tirages:
            for boule in tirage['boules']:
                self.frequencies[boule] += 1
        
        total_boules = sum(self.frequencies.values())
        avg_frequency = total_boules / 70  # 70 numéros possibles
        
        print(f"📈 Total boules tirées : {total_boules}")
        print(f"📊 Fréquence moyenne : {avg_frequency:.1f}")
        print(f"📅 Période analysée : {self.all_tirages[0]['date']} → {self.all_tirages[-1]['date']}")
        
        # Top 10 plus fréquents
        most_frequent = self.frequencies.most_common(10)
        print(f"\n🔥 TOP 10 - Numéros les plus fréquents :")
        for i, (num, freq) in enumerate(most_frequent, 1):
            percentage = (freq / total_boules) * 100
            print(f"   {i:2d}. Numéro {num:2d} : {freq:4d} fois ({percentage:.2f}%)")
        
        # Top 10 moins fréquents
        least_frequent = self.frequencies.most_common()[-10:]
        least_frequent.reverse()
        print(f"\n❄️  TOP 10 - Numéros les moins fréquents :")
        for i, (num, freq) in enumerate(least_frequent, 1):
            percentage = (freq / total_boules) * 100
            print(f"   {i:2d}. Numéro {num:2d} : {freq:4d} fois ({percentage:.2f}%)")
    
    def analyze_delays(self, max_tirages=100):
        """Analyse les retards sur les derniers tirages"""
        print(f"\n⏰ ANALYSE DES RETARDS ({max_tirages} derniers tirages)")
        print("=" * 50)
        
        if len(self.all_tirages) < max_tirages:
            max_tirages = len(self.all_tirages)
            print(f"⚠️  Seulement {max_tirages} tirages disponibles")
        
        # Initialise les retards
        retards = {i: max_tirages for i in range(1, 71)}
        
        # Analyse des derniers tirages
        recent_tirages = self.all_tirages[-max_tirages:]
        
        for idx, tirage in enumerate(reversed(recent_tirages)):
            for boule in tirage['boules']:
                if retards[boule] == max_tirages:  # Pas encore vu
                    retards[boule] = idx
        
        # Tri par retard décroissant
        sorted_delays = sorted(retards.items(), key=lambda x: x[1], reverse=True)
        
        print(f"🚨 TOP 15 - Numéros en retard :")
        for i, (num, retard) in enumerate(sorted_delays[:15], 1):
            if retard >= 50:
                status = "🔴 TRÈS EN RETARD"
            elif retard >= 25:
                status = "🟡 EN RETARD"
            else:
                status = "🟢 Normal"
            print(f"   {i:2d}. Numéro {num:2d} : {retard:2d} tirages ({status})")
        
        return sorted_delays
    
    def analyze_pairs(self):
        """Analyse les paires de numéros qui sortent ensemble"""
        print(f"\n💑 ANALYSE DES PAIRES")
        print("=" * 50)
        
        pair_count = defaultdict(int)
        
        for tirage in self.all_tirages:
            boules = sorted(tirage['boules'])
            # Génère toutes les paires possibles
            for i in range(len(boules)):
                for j in range(i + 1, len(boules)):
                    pair = (boules[i], boules[j])
                    pair_count[pair] += 1
        
        # Top 10 paires
        top_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"🔥 TOP 10 - Paires les plus fréquentes :")
        for i, ((n1, n2), count) in enumerate(top_pairs, 1):
            total_possible = len(self.all_tirages)
            percentage = (count / total_possible) * 100
            print(f"   {i:2d}. {n1:2d}-{n2:2d} : {count:3d} fois ({percentage:.1f}%)")
    
    def generate_recommendations(self, delays):
        """Génère des recommandations de jeu"""
        print(f"\n💡 RECOMMANDATIONS DE JEU")
        print("=" * 50)
        
        # Stratégies
        hot_numbers = [num for num, _ in self.frequencies.most_common(20)]
        cold_numbers = [num for num, delay in delays[:20]]
        
        print(f"🎯 GRILLES RECOMMANDÉES (10 numéros par grille) :")
        
        # Stratégie HOT
        hot_selection = hot_numbers[:10]
        print(f"\n   🔥 STRATÉGIE HOT (fréquents) :")
        print(f"      {hot_selection}")
        
        # Stratégie COLD  
        cold_selection = cold_numbers[:10]
        print(f"\n   ❄️ STRATÉGIE COLD (retard) :")
        print(f"      {cold_selection}")
        
        # Stratégie BALANCED
        balanced = hot_numbers[:5] + cold_numbers[:5]
        print(f"\n   ⚖️ STRATÉGIE BALANCED (équilibrée) :")
        print(f"      {sorted(balanced)}")
        
        # Stratégie RANDOM parmi les meilleurs
        candidates = list(set(hot_numbers[:15] + cold_numbers[:15]))
        np.random.seed(42)
        random_selection = sorted(np.random.choice(candidates, 10, replace=False))
        print(f"\n   🎲 STRATÉGIE RANDOM (aléatoire optimisée) :")
        print(f"      {random_selection}")
        
        return {
            'hot': hot_selection,
            'cold': cold_selection,
            'balanced': sorted(balanced),
            'random': random_selection
        }
    
    def save_results(self):
        """Sauvegarde les résultats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("keno_stats_exports")
        output_dir.mkdir(exist_ok=True)
        
        # Fréquences
        freq_file = output_dir / f"frequences_keno_{timestamp}.csv"
        with open(freq_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['numero', 'frequence', 'pourcentage'])
            
            total = sum(self.frequencies.values())
            for num in range(1, 71):
                freq = self.frequencies[num]
                pct = (freq / total) * 100 if total > 0 else 0
                writer.writerow([num, freq, f"{pct:.2f}"])
        
        print(f"\n💾 Résultats sauvegardés :")
        print(f"   📄 {freq_file}")
        
        return freq_file
    
    def run_complete_analysis(self):
        """Lance l'analyse complète"""
        print("🎯 ANALYSE COMPLÈTE DES DONNÉES KENO")
        print("=" * 70)
        
        # 1. Chargement des données
        if not self.load_all_data():
            return False
        
        # 2. Analyse des fréquences
        self.analyze_frequencies()
        
        # 3. Analyse des retards
        delays = self.analyze_delays()
        
        # 4. Analyse des paires
        self.analyze_pairs()
        
        # 5. Recommandations
        recommendations = self.generate_recommendations(delays)
        
        # 6. Sauvegarde
        self.save_results()
        
        print(f"\n✅ ANALYSE TERMINÉE AVEC SUCCÈS !")
        print(f"📊 {len(self.all_tirages)} tirages analysés")
        
        return True

def main():
    analyzer = KenoAnalysisComplete()
    success = analyzer.run_complete_analysis()
    
    if not success:
        print("❌ Échec de l'analyse")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
