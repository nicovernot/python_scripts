#!/usr/bin/env python3
"""
GÉNÉRATEUR KENO INTELLIGENT AVEC SYSTÈME RÉDUCTEUR
==================================================

Ce script implémente un générateur avancé basé sur :
- TOP 30 numéros avec profil intelligent
- Système réducteur utilisant PuLP
- Optimisation pair/impair, zones, sommes
- Génération basée sur les paires fréquentes
- 3 profils : Bas (50), Moyen (80), Haut (100 grilles)
"""

import sys
import os
sys.path.append('.')

from keno.keno_generator_advanced import KenoGeneratorAdvanced
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Set
import random
import itertools
import json

# Import conditionnel de PuLP
try:
    import pulp
    HAS_PULP = True
    print("✅ PuLP disponible pour optimisation")
except ImportError:
    HAS_PULP = False
    print("⚠️ PuLP non disponible, utilisation d'algorithme alternatif")

class KenoIntelligentGenerator:
    """Générateur Keno intelligent avec système réducteur"""
    
    def __init__(self):
        self.generator = KenoGeneratorAdvanced()
        self.stats = None
        self.top30_numbers = []
        self.top_pairs = []
        self.optimal_params = {}
        
    def load_and_analyze(self):
        """Charge les données et analyse les patterns"""
        print("📊 Chargement et analyse des données...")
        
        if not self.generator.load_data():
            print("❌ Erreur de chargement des données")
            return False
            
        self.stats = self.generator.analyze_patterns()
        if not self.stats:
            print("❌ Erreur d'analyse des patterns")
            return False
            
        print(f"✅ {len(self.generator.data)} tirages analysés")
        return True
    
    def calculate_intelligent_top30(self) -> List[int]:
        """Calcule le TOP 30 avec profil intelligent optimisé"""
        print("\n🧠 Calcul du TOP 30 avec profil intelligent...")
        
        scores = {}
        
        for num in range(1, 71):
            score = 0
            
            # 1. Fréquence pondérée par période (35%)
            freq_global = self.stats.frequences.get(num, 0)
            freq_recent = self.stats.frequences_recentes.get(num, 0)
            freq_50 = self.stats.frequences_50.get(num, 0)
            
            max_freq_global = max(self.stats.frequences.values()) if self.stats.frequences else 1
            max_freq_recent = max(self.stats.frequences_recentes.values()) if self.stats.frequences_recentes else 1
            max_freq_50 = max(self.stats.frequences_50.values()) if self.stats.frequences_50 else 1
            
            # Pondération intelligente : récent > moyen terme > global
            score += (freq_global / max_freq_global) * 0.15  # 15% global
            score += (freq_50 / max_freq_50) * 0.10         # 10% moyen terme  
            score += (freq_recent / max_freq_recent) * 0.10  # 10% récent
            
            # 2. Retard optimisé (25%)
            retard = self.stats.retards.get(num, 0)
            max_retard = max(self.stats.retards.values()) if self.stats.retards else 1
            
            # Bonus pour retards moyens (ni trop faible, ni trop élevé)
            retard_normalized = retard / max_retard
            if 0.2 <= retard_normalized <= 0.6:  # Retard optimal
                retard_bonus = 1.2
            elif retard_normalized > 0.8:  # Très en retard
                retard_bonus = 1.1
            else:  # Peu en retard
                retard_bonus = 0.9
                
            score += (1 - retard_normalized) * 0.25 * retard_bonus
            
            # 3. Tendances multiples (20%)
            tendance_50 = self.stats.tendances_50.get(num, 1.0)
            tendance_100 = self.stats.tendances_100.get(num, 1.0)
            
            # Moyenne pondérée des tendances
            tendance_score = (tendance_50 * 0.7 + tendance_100 * 0.3)
            score += max(0, (tendance_score - 1.0)) * 0.20
            
            # 4. Popularité dans les paires (15%)
            pair_count = sum(1 for pair, freq in self.stats.paires_freq.items() 
                           if num in pair and freq > 200)  # Paires très fréquentes
            score += min(pair_count / 10, 1.0) * 0.15
            
            # 5. Équilibrage des zones (5%)
            zone = (num - 1) // 18 + 1
            zone_key = f"zone_{zone}_" + ["17", "35", "52", "70"][zone-1] if zone <= 4 else "zone_4_70"
            zone_freq = self.stats.patterns_zones.get(zone_key, 0)
            max_zone_freq = max(self.stats.patterns_zones.values()) if self.stats.patterns_zones else 1
            score += (zone_freq / max_zone_freq) * 0.05
            
            scores[num] = score
        
        # TOP 30 avec profil intelligent
        top30 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:30]
        self.top30_numbers = [num for num, _ in top30]
        
        print(f"🎯 TOP 30 intelligent calculé")
        print(f"   Top 10: {', '.join(map(str, self.top30_numbers[:10]))}")
        
        return self.top30_numbers
    
    def analyze_optimal_parameters(self):
        """Analyse les paramètres optimaux des tirages historiques"""
        print("\n📈 Analyse des paramètres optimaux...")
        
        parite_stats = {'pairs': [], 'impairs': [], 'ratio': []}
        zone_stats = {'zone1': [], 'zone2': [], 'zone3': [], 'zone4': []}
        somme_stats = []
        
    def analyze_optimal_parameters(self):
        """Analyse les paramètres optimaux des tirages historiques"""
        print("\n📈 Analyse des paramètres optimaux...")
        
        parite_stats = {'pairs': [], 'impairs': [], 'ratio': []}
        zone_stats = {'zone1': [], 'zone2': [], 'zone3': [], 'zone4': []}
        somme_stats = []
        
        for _, row in self.generator.data.iterrows():
            # Récupération des 20 numéros tirés (colonnes boule1 à boule20)
            numbers = [row[f'boule{i}'] for i in range(1, 21)]
            # Ne prendre que les 10 premiers pour l'analyse (simulation d'une grille)
            numbers = numbers[:10]
            
            # Analyse parité
            pairs = sum(1 for n in numbers if n % 2 == 0)
            impairs = len(numbers) - pairs
            parite_stats['pairs'].append(pairs)
            parite_stats['impairs'].append(impairs)
            parite_stats['ratio'].append(pairs / len(numbers))
            
            # Analyse zones
            zones = [0, 0, 0, 0]
            for num in numbers:
                zone_idx = min((num - 1) // 18, 3)
                zones[zone_idx] += 1
            
            zone_stats['zone1'].append(zones[0])
            zone_stats['zone2'].append(zones[1]) 
            zone_stats['zone3'].append(zones[2])
            zone_stats['zone4'].append(zones[3])
            
            # Analyse sommes
            somme_stats.append(sum(numbers))
        
        # Calcul des paramètres optimaux
        self.optimal_params = {
            'parite_ratio_optimal': sum(parite_stats['ratio']) / len(parite_stats['ratio']),
            'parite_pairs_moy': sum(parite_stats['pairs']) / len(parite_stats['pairs']),
            'parite_impairs_moy': sum(parite_stats['impairs']) / len(parite_stats['impairs']),
            'zone1_moy': sum(zone_stats['zone1']) / len(zone_stats['zone1']),
            'zone2_moy': sum(zone_stats['zone2']) / len(zone_stats['zone2']),
            'zone3_moy': sum(zone_stats['zone3']) / len(zone_stats['zone3']),
            'zone4_moy': sum(zone_stats['zone4']) / len(zone_stats['zone4']),
            'somme_moyenne': sum(somme_stats) / len(somme_stats),
            'somme_ecart_type': (sum((s - sum(somme_stats)/len(somme_stats))**2 for s in somme_stats) / len(somme_stats))**0.5
        }
        
        print(f"✅ Paramètres optimaux calculés:")
        print(f"   • Ratio pairs optimal: {self.optimal_params['parite_ratio_optimal']:.1%}")
        print(f"   • Répartition zones: {self.optimal_params['zone1_moy']:.1f}-{self.optimal_params['zone2_moy']:.1f}-{self.optimal_params['zone3_moy']:.1f}-{self.optimal_params['zone4_moy']:.1f}")
        print(f"   • Somme moyenne: {self.optimal_params['somme_moyenne']:.0f} ± {self.optimal_params['somme_ecart_type']:.0f}")
    
    def extract_top_pairs(self, top_count: int = 100) -> List[Tuple[int, int]]:
        """Extrait les paires les plus fréquentes du TOP 30"""
        print(f"\n🔗 Extraction des {top_count} meilleures paires du TOP 30...")
        
        # Filtrer les paires qui contiennent uniquement des numéros du TOP 30
        top30_set = set(self.top30_numbers)
        top_pairs_in_top30 = []
        
        for pair, freq in self.stats.paires_freq.items():
            num1, num2 = pair
            if num1 in top30_set and num2 in top30_set:
                top_pairs_in_top30.append((pair, freq))
        
        # Trier et prendre les meilleures
        top_pairs_in_top30.sort(key=lambda x: x[1], reverse=True)
        self.top_pairs = [pair for pair, freq in top_pairs_in_top30[:top_count]]
        
        print(f"✅ {len(self.top_pairs)} paires extraites du TOP 30")
        print(f"   Top 5 paires: {', '.join([f'{p[0]}-{p[1]}' for p in self.top_pairs[:5]])}")
        
        return self.top_pairs
    
    def generate_grids_with_pulp(self, num_grids: int) -> List[List[int]]:
        """Génère des grilles optimisées avec PuLP"""
        print(f"\n🎯 Génération de {num_grids} grilles avec PuLP...")
        
        grids = []
        
        for grid_idx in range(num_grids):
            # Création du problème d'optimisation
            prob = pulp.LpProblem(f"Keno_Grid_{grid_idx}", pulp.LpMaximize)
            
            # Variables : chaque numéro du TOP 30 peut être sélectionné (0 ou 1)
            vars_numbers = {}
            for num in self.top30_numbers:
                vars_numbers[num] = pulp.LpVariable(f"num_{num}", cat='Binary')
            
            # Variables pour les paires
            vars_pairs = {}
            for i, (num1, num2) in enumerate(self.top_pairs[:50]):  # Top 50 paires
                vars_pairs[i] = pulp.LpVariable(f"pair_{i}", cat='Binary')
            
            # Fonction objectif : maximiser les paires fréquentes sélectionnées
            pair_weights = []
            for i, (num1, num2) in enumerate(self.top_pairs[:50]):
                if i < len(self.top_pairs):
                    freq = self.stats.paires_freq.get((num1, num2), 0)
                    pair_weights.append(freq / 1000)  # Normalisation
                else:
                    pair_weights.append(0)
            
            prob += pulp.lpSum([vars_pairs[i] * pair_weights[i] for i in range(len(pair_weights))])
            
            # Contrainte : exactement 10 numéros
            prob += pulp.lpSum([vars_numbers[num] for num in self.top30_numbers]) == 10
            
            # Contraintes de paires
            for i, (num1, num2) in enumerate(self.top_pairs[:50]):
                if i < len(vars_pairs):
                    prob += vars_pairs[i] <= vars_numbers[num1]
                    prob += vars_pairs[i] <= vars_numbers[num2]
                    prob += vars_pairs[i] >= vars_numbers[num1] + vars_numbers[num2] - 1
            
            # Contraintes d'équilibrage pair/impair
            pairs_in_top30 = [n for n in self.top30_numbers if n % 2 == 0]
            impairs_in_top30 = [n for n in self.top30_numbers if n % 2 == 1]
            
            target_pairs = round(self.optimal_params['parite_pairs_moy'])
            prob += pulp.lpSum([vars_numbers[num] for num in pairs_in_top30]) >= target_pairs - 1
            prob += pulp.lpSum([vars_numbers[num] for num in pairs_in_top30]) <= target_pairs + 1
            
            # Contraintes d'équilibrage zones
            zones_in_top30 = [[], [], [], []]
            for num in self.top30_numbers:
                zone_idx = min((num - 1) // 18, 3)
                zones_in_top30[zone_idx].append(num)
            
            for zone_idx in range(4):
                if zones_in_top30[zone_idx]:
                    zone_target = round(self.optimal_params[f'zone{zone_idx+1}_moy'])
                    prob += pulp.lpSum([vars_numbers[num] for num in zones_in_top30[zone_idx]]) >= max(1, zone_target - 1)
                    prob += pulp.lpSum([vars_numbers[num] for num in zones_in_top30[zone_idx]]) <= zone_target + 2
            
            # Résolution
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                # Extraction de la solution
                selected_numbers = []
                for num in self.top30_numbers:
                    if vars_numbers[num].value() == 1:
                        selected_numbers.append(num)
                
                if len(selected_numbers) == 10:
                    grids.append(sorted(selected_numbers))
                else:
                    # Fallback si la solution n'est pas complète
                    grids.append(self.generate_fallback_grid())
            else:
                # Fallback si pas de solution optimale
                grids.append(self.generate_fallback_grid())
        
        print(f"✅ {len(grids)} grilles optimisées générées avec PuLP")
        return grids
    
    def generate_grids_alternative(self, num_grids: int) -> List[List[int]]:
        """Génère des grilles avec algorithme alternatif (sans PuLP)"""
        print(f"\n🎯 Génération de {num_grids} grilles avec algorithme alternatif...")
        
        grids = []
        
        for _ in range(num_grids):
            grid = self.generate_smart_grid()
            grids.append(grid)
        
        print(f"✅ {len(grids)} grilles générées avec algorithme alternatif")
        return grids
    
    def generate_smart_grid(self) -> List[int]:
        """Génère une grille intelligente basée sur les paires et paramètres"""
        grid = []
        used_numbers = set()
        
        # 1. Commencer par les meilleures paires
        pairs_used = 0
        for num1, num2 in self.top_pairs:
            if len(grid) >= 8:  # Laisser de la place pour ajustements
                break
            if num1 not in used_numbers and num2 not in used_numbers:
                grid.extend([num1, num2])
                used_numbers.update([num1, num2])
                pairs_used += 1
                if pairs_used >= 3:  # Maximum 3 paires complètes
                    break
        
        # 2. Compléter avec des numéros du TOP 30 pour équilibrer
        remaining_top30 = [n for n in self.top30_numbers if n not in used_numbers]
        
        # Équilibrage pair/impair
        current_pairs = sum(1 for n in grid if n % 2 == 0)
        target_pairs = round(self.optimal_params['parite_pairs_moy'])
        
        while len(grid) < 10 and remaining_top30:
            need_pair = current_pairs < target_pairs and (target_pairs - current_pairs) > (10 - len(grid)) / 2
            need_impair = not need_pair
            
            candidates = []
            for num in remaining_top30:
                if (need_pair and num % 2 == 0) or (need_impair and num % 2 == 1):
                    candidates.append(num)
            
            if not candidates:
                candidates = remaining_top30
            
            # Sélection pondérée par score
            if candidates:
                selected = random.choice(candidates)
                grid.append(selected)
                remaining_top30.remove(selected)
                if selected % 2 == 0:
                    current_pairs += 1
        
        # 3. Compléter si nécessaire avec le TOP 30 restant
        while len(grid) < 10 and remaining_top30:
            selected = random.choice(remaining_top30)
            grid.append(selected)
            remaining_top30.remove(selected)
        
        return sorted(grid)
    
    def generate_fallback_grid(self) -> List[int]:
        """Génère une grille de fallback simple"""
        return sorted(random.sample(self.top30_numbers, 10))
    
    def validate_grid_quality(self, grid: List[int]) -> Dict[str, float]:
        """Valide la qualité d'une grille selon les paramètres optimaux"""
        
        # Parité
        pairs = sum(1 for n in grid if n % 2 == 0)
        parite_score = 1 - abs(pairs - self.optimal_params['parite_pairs_moy']) / 5
        
        # Zones
        zones = [0, 0, 0, 0]
        for num in grid:
            zone_idx = min((num - 1) // 18, 3)
            zones[zone_idx] += 1
        
        zone_score = 0
        for i in range(4):
            target = self.optimal_params[f'zone{i+1}_moy']
            zone_score += 1 - abs(zones[i] - target) / 3
        zone_score /= 4
        
        # Somme
        grid_sum = sum(grid)
        target_sum = self.optimal_params['somme_moyenne']
        tolerance = self.optimal_params['somme_ecart_type']
        somme_score = max(0, 1 - abs(grid_sum - target_sum) / (2 * tolerance))
        
        # Paires fréquentes
        pairs_count = 0
        for i in range(len(grid)):
            for j in range(i+1, len(grid)):
                pair = tuple(sorted([grid[i], grid[j]]))
                if pair in [tuple(sorted(p)) for p in self.top_pairs[:20]]:
                    pairs_count += 1
        
        pairs_score = min(pairs_count / 3, 1.0)  # 3 paires fréquentes = score max
        
        return {
            'parite': parite_score,
            'zones': zone_score,
            'somme': somme_score,
            'pairs': pairs_score,
            'global': (parite_score + zone_score + somme_score + pairs_score) / 4
        }
    
    def export_top30_intelligent(self, filename: str = None):
        """Exporte le TOP 30 intelligent avec détails"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keno_output/top30_intelligent_{timestamp}.csv"
        
        os.makedirs("keno_output", exist_ok=True)
        
        data = []
        for i, num in enumerate(self.top30_numbers, 1):
            # Calcul du score intelligent
            freq_global = self.stats.frequences.get(num, 0)
            freq_recent = self.stats.frequences_recentes.get(num, 0)
            retard = self.stats.retards.get(num, 0)
            tendance = self.stats.tendances_50.get(num, 1.0)
            
            # Comptage des paires
            pair_count = sum(1 for pair, freq in self.stats.paires_freq.items() 
                           if num in pair and freq > 200)
            
            data.append({
                'rang': i,
                'numero': num,
                'freq_globale': freq_global,
                'freq_recente': freq_recent,
                'retard': retard,
                'tendance_50': round(tendance, 3),
                'nb_paires_frequentes': pair_count,
                'zone': (num - 1) // 18 + 1,
                'parite': 'Pair' if num % 2 == 0 else 'Impair',
                'score_intelligent': round(self.calculate_intelligent_score(num), 4)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"✅ TOP 30 intelligent exporté: {filename}")
        return filename
    
    def calculate_intelligent_score(self, num: int) -> float:
        """Calcule le score intelligent pour un numéro"""
        score = 0
        
        # Réutilise la logique de calculate_intelligent_top30
        freq_global = self.stats.frequences.get(num, 0)
        freq_recent = self.stats.frequences_recentes.get(num, 0)
        freq_50 = self.stats.frequences_50.get(num, 0)
        
        max_freq_global = max(self.stats.frequences.values()) if self.stats.frequences else 1
        max_freq_recent = max(self.stats.frequences_recentes.values()) if self.stats.frequences_recentes else 1
        max_freq_50 = max(self.stats.frequences_50.values()) if self.stats.frequences_50 else 1
        
        score += (freq_global / max_freq_global) * 0.15
        score += (freq_50 / max_freq_50) * 0.10
        score += (freq_recent / max_freq_recent) * 0.10
        
        retard = self.stats.retards.get(num, 0)
        max_retard = max(self.stats.retards.values()) if self.stats.retards else 1
        retard_normalized = retard / max_retard
        
        if 0.2 <= retard_normalized <= 0.6:
            retard_bonus = 1.2
        elif retard_normalized > 0.8:
            retard_bonus = 1.1
        else:
            retard_bonus = 0.9
            
        score += (1 - retard_normalized) * 0.25 * retard_bonus
        
        tendance_50 = self.stats.tendances_50.get(num, 1.0)
        tendance_100 = self.stats.tendances_100.get(num, 1.0)
        tendance_score = (tendance_50 * 0.7 + tendance_100 * 0.3)
        score += max(0, (tendance_score - 1.0)) * 0.20
        
        pair_count = sum(1 for pair, freq in self.stats.paires_freq.items() 
                       if num in pair and freq > 200)
        score += min(pair_count / 10, 1.0) * 0.15
        
        zone = (num - 1) // 18 + 1
        zone_key = f"zone_{zone}_" + ["17", "35", "52", "70"][zone-1] if zone <= 4 else "zone_4_70"
        zone_freq = self.stats.patterns_zones.get(zone_key, 0)
        max_zone_freq = max(self.stats.patterns_zones.values()) if self.stats.patterns_zones else 1
        score += (zone_freq / max_zone_freq) * 0.05
        
        return score
    
    def generate_system_grids(self, profile: str = "moyen") -> Tuple[List[List[int]], Dict]:
        """Génère un système de grilles selon le profil"""
        
        # Configuration des profils
        profiles = {
            "bas": {"grids": 50, "description": "Système Bas - 50 grilles"},
            "moyen": {"grids": 80, "description": "Système Moyen - 80 grilles"},
            "haut": {"grids": 100, "description": "Système Haut - 100 grilles"}
        }
        
        if profile not in profiles:
            profile = "moyen"
        
        config = profiles[profile]
        num_grids = config["grids"]
        
        print(f"\n🎯 {config['description']}")
        print("="*50)
        
        # Génération des grilles
        if HAS_PULP:
            grids = self.generate_grids_with_pulp(num_grids)
        else:
            grids = self.generate_grids_alternative(num_grids)
        
        # Validation de la qualité
        print(f"\n📊 Validation de la qualité des {len(grids)} grilles...")
        
        quality_scores = []
        for grid in grids:
            quality = self.validate_grid_quality(grid)
            quality_scores.append(quality)
        
        # Statistiques globales
        avg_quality = {
            'parite': sum(q['parite'] for q in quality_scores) / len(quality_scores),
            'zones': sum(q['zones'] for q in quality_scores) / len(quality_scores),
            'somme': sum(q['somme'] for q in quality_scores) / len(quality_scores),
            'pairs': sum(q['pairs'] for q in quality_scores) / len(quality_scores),
            'global': sum(q['global'] for q in quality_scores) / len(quality_scores)
        }
        
        print(f"✅ Qualité moyenne du système:")
        print(f"   • Parité: {avg_quality['parite']:.1%}")
        print(f"   • Zones: {avg_quality['zones']:.1%}")
        print(f"   • Sommes: {avg_quality['somme']:.1%}")
        print(f"   • Paires: {avg_quality['pairs']:.1%}")
        print(f"   • Global: {avg_quality['global']:.1%}")
        
        # Métadonnées du système
        metadata = {
            'profile': profile,
            'num_grids': num_grids,
            'top30_numbers': self.top30_numbers,
            'optimal_params': self.optimal_params,
            'quality_avg': avg_quality,
            'generation_method': 'PuLP' if HAS_PULP else 'Alternative',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return grids, metadata
    
    def export_system_grids(self, grids: List[List[int]], metadata: Dict, filename: str = None):
        """Exporte le système de grilles avec métadonnées"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile = metadata['profile']
            filename_base = f"keno_output/system_{profile}_{timestamp}"
        else:
            filename_base = filename.replace('.csv', '')
        
        os.makedirs("keno_output", exist_ok=True)
        
        # 1. Export des grilles
        grids_data = []
        for i, grid in enumerate(grids, 1):
            quality = self.validate_grid_quality(grid)
            
            grids_data.append({
                'grille': i,
                'numeros': ', '.join(map(str, grid)),
                'somme': sum(grid),
                'pairs': sum(1 for n in grid if n % 2 == 0),
                'impairs': sum(1 for n in grid if n % 2 == 1),
                'zone1': sum(1 for n in grid if 1 <= n <= 17),
                'zone2': sum(1 for n in grid if 18 <= n <= 35),
                'zone3': sum(1 for n in grid if 36 <= n <= 52),
                'zone4': sum(1 for n in grid if 53 <= n <= 70),
                'qualite_globale': round(quality['global'], 3),
                'qualite_parite': round(quality['parite'], 3),
                'qualite_zones': round(quality['zones'], 3),
                'qualite_somme': round(quality['somme'], 3),
                'qualite_pairs': round(quality['pairs'], 3)
            })
        
        df_grids = pd.DataFrame(grids_data)
        grids_file = f"{filename_base}_grilles.csv"
        df_grids.to_csv(grids_file, index=False, encoding='utf-8')
        
        # 2. Export des métadonnées
        metadata_file = f"{filename_base}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 3. Export du TOP 30
        top30_file = f"{filename_base}_top30.csv"
        self.export_top30_intelligent(top30_file)
        
        print(f"\n📁 Système exporté:")
        print(f"   • Grilles: {grids_file}")
        print(f"   • Métadonnées: {metadata_file}")
        print(f"   • TOP 30: {top30_file}")
        
        return {
            'grids': grids_file,
            'metadata': metadata_file,
            'top30': top30_file
        }

def main():
    """Fonction principale"""
    print("🎲 GÉNÉRATEUR KENO INTELLIGENT AVEC SYSTÈME RÉDUCTEUR")
    print("="*60)
    
    # Sélection du profil
    print("\n📋 Profils disponibles:")
    print("1️⃣  Bas - 50 grilles (couverture minimale)")
    print("2️⃣  Moyen - 80 grilles (équilibre optimal)")
    print("3️⃣  Haut - 100 grilles (couverture maximale)")
    
    try:
        choice = input("\n🎯 Choisissez un profil (1-3, défaut: 2): ").strip()
        if choice == "1":
            profile = "bas"
        elif choice == "3":
            profile = "haut"
        else:
            profile = "moyen"
    except:
        profile = "moyen"
    
    # Initialisation
    generator = KenoIntelligentGenerator()
    
    # Chargement et analyse
    if not generator.load_and_analyze():
        print("❌ Échec de l'initialisation")
        return
    
    # Calculs intelligents
    generator.calculate_intelligent_top30()
    generator.analyze_optimal_parameters()
    generator.extract_top_pairs()
    
    # Génération du système
    grids, metadata = generator.generate_system_grids(profile)
    
    # Export
    files = generator.export_system_grids(grids, metadata)
    
    print(f"\n🏆 SYSTÈME {profile.upper()} GÉNÉRÉ AVEC SUCCÈS!")
    print(f"📊 {len(grids)} grilles optimisées basées sur le TOP 30")
    print(f"🎯 Qualité globale: {metadata['quality_avg']['global']:.1%}")
    
    # Affichage d'échantillons
    print(f"\n📋 Échantillon de grilles (5 premières):")
    for i, grid in enumerate(grids[:5], 1):
        quality = generator.validate_grid_quality(grid)
        print(f"   {i:2d}. {grid} (Q: {quality['global']:.1%})")

if __name__ == "__main__":
    main()
