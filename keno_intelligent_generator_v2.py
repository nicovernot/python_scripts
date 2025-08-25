#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Générateur Keno Intelligent v2 - Système réducteur avec optimisation PuLP
Basé sur TOP 30 intelligents avec analyse de pairs et paramètres optimaux
Version améliorée avec diversité garantie
"""

import sys
import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent))

# Vérification et import de PuLP
PULP_AVAILABLE = False
try:
    import pulp
    PULP_AVAILABLE = True
    print("✅ PuLP disponible pour optimisation")
except ImportError:
    print("⚠️ PuLP non disponible - utilisation de l'algorithme alternatif")

from keno.keno_generator_advanced import KenoGeneratorAdvanced

class KenoIntelligentGeneratorV2:
    """Générateur intelligent Keno avec système réducteur v2"""
    
    def __init__(self, top30_csv_path: Optional[str] = None):
        """
        Initialisation du générateur intelligent
        
        Args:
            top30_csv_path: Chemin vers un fichier CSV TOP 30 externe (optionnel)
        """
        self.generator = KenoGeneratorAdvanced()
        self.stats = None
        self.top30_numbers = []
        self.top30_scores = {}
        self.top_pairs = []
        self.optimal_params = {}
        self.top30_csv_path = top30_csv_path
        self.external_top30 = False
        self.profiles = {
            'bas': {'grids': 50, 'description': 'couverture minimale'},
            'moyen': {'grids': 80, 'description': 'équilibre optimal'},
            'haut': {'grids': 100, 'description': 'couverture maximale'}
        }
    
    def load_and_analyze_data(self):
        """Charge et analyse les données"""
        print("📊 Chargement et analyse des données...")
        self.generator.load_data()
        self.generator.analyze_patterns()
        self.stats = self.generator.stats
        print(f"✅ {len(self.generator.data)} tirages analysés")
    
    def calculate_intelligent_top30(self) -> List[int]:
        """Calcule le TOP 30 avec scoring intelligent multi-critères ou charge depuis CSV"""
        
        # Vérifier s'il faut charger depuis un CSV externe
        if self.top30_csv_path and Path(self.top30_csv_path).exists():
            return self.load_top30_from_csv()
        
        # Sinon générer le TOP 30 à partir des données
        print("\n🧠 Calcul du TOP 30 avec profil intelligent...")
        
        # Générer et exporter le TOP 30 via le générateur avancé
        self.top30_numbers = self.generator.calculate_and_export_top30()
        
        # Récupérer les scores pour compatibilité
        scores = {}
        max_freq = max(self.stats.frequences.values()) if self.stats.frequences else 1
        max_retard = max(self.stats.retards.values()) if self.stats.retards else 1
        
        for numero in self.top30_numbers:
            # Score fréquence (35%)
            freq_score = (self.stats.frequences.get(numero, 0) / max_freq) * 35
            
            # Score retard inversé (25%) - moins de retard = meilleur score
            retard_score = (1 - (self.stats.retards.get(numero, max_retard) / max_retard)) * 25
            
            # Score tendance (20%)
            trend_score = 0
            if hasattr(self.stats, 'tendances') and numero in self.stats.tendances:
                if self.stats.tendances[numero] > 0:
                    trend_score = 20
                elif self.stats.tendances[numero] < -5:
                    trend_score = 5
                else:
                    trend_score = 10
            else:
                trend_score = 10
            
            # Score pairs (15%)
            pair_score = 0
            if hasattr(self.stats, 'paires_freq'):
                for (n1, n2), freq in self.stats.paires_freq.items():
                    if n1 == numero or n2 == numero:
                        pair_score += freq
                pair_score = min(pair_score / 100, 1) * 15
            
            # Score zones (5%)
            zone_score = 5  # Score de base pour toutes les zones
            
            # Score total
            total_score = freq_score + retard_score + trend_score + pair_score + zone_score
            scores[numero] = total_score
        
        self.top30_scores = scores
        
        print(f"🎯 TOP 30 intelligent calculé")
        print(f"   Top 10: {', '.join(map(str, self.top30_numbers[:10]))}")
        
        return self.top30_numbers
    
    def load_top30_from_csv(self) -> List[int]:
        """Charge le TOP 30 depuis un fichier CSV externe"""
        try:
            print(f"\n📂 Chargement TOP 30 depuis CSV: {self.top30_csv_path}")
            
            top30_df = pd.read_csv(self.top30_csv_path)
            
            # Vérifier les colonnes requises (support ancien et nouveau format)
            score_col = None
            if 'Score_Total' in top30_df.columns:
                score_col = 'Score_Total'  # Nouveau format optimisé
            elif 'Score' in top30_df.columns:
                score_col = 'Score'        # Ancien format
            
            if 'Numero' not in top30_df.columns or score_col is None:
                print(f"⚠️ Colonnes requises manquantes dans le CSV")
                print(f"   Colonnes trouvées: {list(top30_df.columns)}")
                print(f"   Colonnes requises: ['Numero', 'Score' ou 'Score_Total']")
                return self.calculate_intelligent_top30_fallback()
            
            # Trier par score décroissant et prendre les 30 premiers
            top30_df = top30_df.sort_values(score_col, ascending=False).head(30)
            
            self.top30_numbers = top30_df['Numero'].tolist()
            self.top30_scores = dict(zip(top30_df['Numero'], top30_df[score_col]))
            self.external_top30 = True
            
            format_type = "optimisé" if score_col == 'Score_Total' else "standard"
            print(f"✅ TOP 30 chargé depuis CSV ({len(self.top30_numbers)} numéros, format {format_type})")
            print(f"   Top 10: {', '.join(map(str, self.top30_numbers[:10]))}")
            
            return self.top30_numbers
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du CSV: {e}")
            print("🔄 Basculement vers le calcul automatique...")
            return self.calculate_intelligent_top30_fallback()
    
    def calculate_intelligent_top30_fallback(self) -> List[int]:
        """Méthode de fallback pour calculer le TOP 30 en cas d'erreur"""
        print("\n🧠 Calcul du TOP 30 avec profil intelligent (fallback)...")
        
        scores = {}
        max_freq = max(self.stats.frequences.values()) if self.stats.frequences else 1
        max_retard = max(self.stats.retards.values()) if self.stats.retards else 1
        
        for numero in range(1, 71):
            # Score fréquence (35%)
            freq_score = (self.stats.frequences.get(numero, 0) / max_freq) * 35
            
            # Score retard inversé (25%) - moins de retard = meilleur score
            retard_score = (1 - (self.stats.retards.get(numero, max_retard) / max_retard)) * 25
            
            # Score tendance (20%)
            trend_score = 0
            if hasattr(self.stats, 'tendances') and numero in self.stats.tendances:
                if self.stats.tendances[numero] > 0:
                    trend_score = 20
                elif self.stats.tendances[numero] < -5:
                    trend_score = 5
                else:
                    trend_score = 10
            else:
                trend_score = 10
            
            # Score pairs (15%)
            pair_score = 0
            if hasattr(self.stats, 'paires_freq'):
                for (n1, n2), freq in self.stats.paires_freq.items():
                    if n1 == numero or n2 == numero:
                        pair_score += freq
                pair_score = min(pair_score / 100, 1) * 15
            
            # Score zones (5%)
            zone_score = 5  # Score de base pour toutes les zones
            
            # Score total
            total_score = freq_score + retard_score + trend_score + pair_score + zone_score
            scores[numero] = total_score
        
        # Tri et sélection du TOP 30
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.top30_numbers = [num for num, score in sorted_numbers[:30]]
        self.top30_scores = {num: score for num, score in sorted_numbers[:30]}
        
        print(f"🎯 TOP 30 intelligent calculé (fallback)")
        print(f"   Top 10: {', '.join(map(str, self.top30_numbers[:10]))}")
        
        return self.top30_numbers
    
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
    
    def generate_diverse_grids(self, num_grids: int) -> List[List[int]]:
        """Génère des grilles diversifiées avec PuLP et algorithmes hybrides"""
        print(f"\n🎯 Génération de {num_grids} grilles diversifiées...")
        
        grids = []
        used_combinations = set()
        
        # Stratégies de génération pour assurer la diversité
        strategies = [
            'pulp_optimized',     # 40% des grilles
            'pair_focused',       # 30% des grilles  
            'balanced_zones',     # 20% des grilles
            'high_frequency'      # 10% des grilles
        ]
        
        strategy_counts = {
            'pulp_optimized': int(num_grids * 0.4),
            'pair_focused': int(num_grids * 0.3),
            'balanced_zones': int(num_grids * 0.2),
            'high_frequency': num_grids - int(num_grids * 0.9)  # Le reste
        }
        
        for strategy, count in strategy_counts.items():
            print(f"   🔄 Génération {strategy}: {count} grilles")
            
            for i in range(count):
                grid = None
                attempts = 0
                max_attempts = 50
                
                while grid is None and attempts < max_attempts:
                    if strategy == 'pulp_optimized' and PULP_AVAILABLE:
                        grid = self.generate_pulp_grid(grids, attempts)
                    elif strategy == 'pair_focused':
                        grid = self.generate_pair_focused_grid(grids, attempts)
                    elif strategy == 'balanced_zones':
                        grid = self.generate_balanced_zones_grid(grids, attempts)
                    else:  # high_frequency
                        grid = self.generate_frequency_grid(grids, attempts)
                    
                    if grid:
                        grid_tuple = tuple(sorted(grid))
                        if grid_tuple not in used_combinations:
                            grids.append(grid)
                            used_combinations.add(grid_tuple)
                            break
                        else:
                            grid = None
                    
                    attempts += 1
                
                if grid is None:
                    # Dernier recours: grille aléatoire du TOP 30
                    grid = self.generate_random_top30_grid(grids)
                    if grid:
                        grids.append(grid)
        
        print(f"✅ {len(grids)} grilles diversifiées générées")
        return grids
    
    def generate_pulp_grid(self, existing_grids: List[List[int]], attempt: int) -> Optional[List[int]]:
        """Génère une grille avec PuLP"""
        if not PULP_AVAILABLE:
            return None
        
        prob = pulp.LpProblem(f"KenoGrid_PuLP_{attempt}", pulp.LpMaximize)
        
        # Variables de décision
        x = {}
        for num in self.top30_numbers:
            x[num] = pulp.LpVariable(f"x_{num}", cat='Binary')
        
        # Contrainte: exactement 10 numéros
        prob += pulp.lpSum([x[num] for num in self.top30_numbers]) == 10
        
        # Éviter les grilles trop similaires
        for existing_grid in existing_grids[-10:]:  # Seulement les 10 dernières
            common_vars = [x[num] for num in existing_grid if num in self.top30_numbers]
            if len(common_vars) >= 6:
                prob += pulp.lpSum(common_vars) <= 6  # Max 6 numéros en commun
        
        # Fonction objectif avec randomisation
        objective = pulp.lpSum([
            x[num] * (
                self.top30_scores[num] + 
                random.uniform(-10, 10) * (attempt + 1)  # Plus de randomisation après échecs
            ) for num in self.top30_numbers
        ])
        prob += objective
        
        # Contraintes d'équilibrage
        # Parité
        pairs_vars = [x[num] for num in self.top30_numbers if num % 2 == 0]
        prob += pulp.lpSum(pairs_vars) >= 3
        prob += pulp.lpSum(pairs_vars) <= 7
        
        # Zones
        for zone_idx in range(4):
            zone_start = zone_idx * 18 + 1
            zone_end = (zone_idx + 1) * 18
            zone_vars = [x[num] for num in self.top30_numbers 
                        if zone_start <= num <= zone_end]
            if zone_vars:
                prob += pulp.lpSum(zone_vars) >= 0
                prob += pulp.lpSum(zone_vars) <= 6
        
        # Résolution
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            grid = [num for num in self.top30_numbers if x[num].value() == 1]
            if len(grid) == 10:
                return sorted(grid)
        
        return None
    
    def generate_pair_focused_grid(self, existing_grids: List[List[int]], attempt: int) -> Optional[List[int]]:
        """Génère une grille focalisée sur les paires fréquentes"""
        grid = []
        used_numbers = set()
        
        # Sélectionner des paires aléatoirement parmi les meilleures
        available_pairs = self.top_pairs[:20 + attempt * 5]  # Plus de choix après échecs
        random.shuffle(available_pairs)
        
        for pair in available_pairs:
            if len(grid) >= 8:  # Arrêter à 8 pour laisser place à 2 numéros libres
                break
            
            num1, num2 = pair
            if num1 not in used_numbers and num2 not in used_numbers:
                grid.extend([num1, num2])
                used_numbers.update([num1, num2])
        
        # Compléter avec des numéros du TOP 30
        remaining_numbers = [n for n in self.top30_numbers if n not in used_numbers]
        random.shuffle(remaining_numbers)
        
        while len(grid) < 10 and remaining_numbers:
            grid.append(remaining_numbers.pop(0))
        
        if len(grid) == 10:
            return sorted(grid)
        return None
    
    def generate_balanced_zones_grid(self, existing_grids: List[List[int]], attempt: int) -> Optional[List[int]]:
        """Génère une grille avec équilibrage des zones"""
        grid = []
        
        # Répartir par zones
        zones = [[], [], [], []]
        for num in self.top30_numbers:
            zone_idx = min((num - 1) // 18, 3)
            zones[zone_idx].append(num)
        
        # Objectif: 2-3 numéros par zone active
        target_per_zone = [2, 3, 3, 2]  # Répartition 2-3-3-2
        
        for zone_idx, target in enumerate(target_per_zone):
            if zones[zone_idx]:
                available = zones[zone_idx].copy()
                random.shuffle(available)
                selected = available[:min(target, len(available))]
                grid.extend(selected)
        
        # Ajuster à 10 numéros
        if len(grid) > 10:
            random.shuffle(grid)
            grid = grid[:10]
        elif len(grid) < 10:
            remaining = [n for n in self.top30_numbers if n not in grid]
            random.shuffle(remaining)
            grid.extend(remaining[:10 - len(grid)])
        
        if len(grid) == 10:
            return sorted(grid)
        return None
    
    def generate_frequency_grid(self, existing_grids: List[List[int]], attempt: int) -> Optional[List[int]]:
        """Génère une grille basée sur les fréquences"""
        # Prendre les 15 meilleurs en fréquence du TOP 30
        freq_sorted = sorted(
            self.top30_numbers, 
            key=lambda x: self.stats.frequences.get(x, 0), 
            reverse=True
        )
        
        # Sélection avec biais vers les plus fréquents mais avec randomisation
        grid = []
        for i in range(10):
            # Plus on avance, plus on peut prendre des numéros moins fréquents
            max_idx = min(15 + i + attempt, len(freq_sorted))
            selected_idx = random.randint(0, max_idx - 1)
            
            if freq_sorted[selected_idx] not in grid:
                grid.append(freq_sorted[selected_idx])
        
        if len(grid) == 10:
            return sorted(grid)
        return None
    
    def generate_random_top30_grid(self, existing_grids: List[List[int]]) -> Optional[List[int]]:
        """Génère une grille aléatoire du TOP 30"""
        available = self.top30_numbers.copy()
        random.shuffle(available)
        return sorted(available[:10])
    
    def calculate_grid_quality(self, grid: List[int]) -> Dict[str, float]:
        """Calcule la qualité d'une grille selon plusieurs critères"""
        if not grid or len(grid) != 10:
            return {'global': 0.0}
        
        scores = {}
        
        # Score parité (30%)
        pairs = sum(1 for n in grid if n % 2 == 0)
        target_pairs = self.optimal_params['parite_pairs_moy']
        parite_score = max(0, 100 - abs(pairs - target_pairs) * 20)
        scores['parite'] = parite_score
        
        # Score zones (25%)
        zones = [0, 0, 0, 0]
        for num in grid:
            zone_idx = min((num - 1) // 18, 3)
            zones[zone_idx] += 1
        
        zone_score = 0
        for i, count in enumerate(zones):
            target = self.optimal_params[f'zone{i+1}_moy']
            zone_score += max(0, 100 - abs(count - target) * 25)
        zone_score /= 4
        scores['zones'] = zone_score
        
        # Score sommes (20%)
        grid_sum = sum(grid)
        target_sum = self.optimal_params['somme_moyenne']
        ecart_type = self.optimal_params['somme_ecart_type']
        somme_score = max(0, 100 - abs(grid_sum - target_sum) / ecart_type * 30)
        scores['sommes'] = somme_score
        
        # Score paires fréquentes (25%)
        pair_score = 0
        grid_pairs = [(grid[i], grid[j]) for i in range(len(grid)) for j in range(i+1, len(grid))]
        for pair in grid_pairs:
            if pair in [p for p in self.top_pairs[:50]]:
                pair_score += 2
        pair_score = min(100, pair_score)
        scores['paires'] = pair_score
        
        # Score global pondéré
        scores['global'] = (
            scores['parite'] * 0.30 +
            scores['zones'] * 0.25 +
            scores['sommes'] * 0.20 +
            scores['paires'] * 0.25
        )
        
        return scores
    
    def validate_grids_quality(self, grids: List[List[int]]):
        """Valide la qualité de toutes les grilles"""
        print(f"\n📊 Validation de la qualité des {len(grids)} grilles...")
        
        quality_scores = []
        for grid in grids:
            scores = self.calculate_grid_quality(grid)
            quality_scores.append(scores)
        
        # Statistiques globales
        avg_scores = {}
        for key in ['parite', 'zones', 'sommes', 'paires', 'global']:
            avg_scores[key] = sum(s.get(key, 0) for s in quality_scores) / len(quality_scores)
        
        print(f"✅ Qualité moyenne du système:")
        print(f"   • Parité: {avg_scores['parite']:.1f}%")
        print(f"   • Zones: {avg_scores['zones']:.1f}%")
        print(f"   • Sommes: {avg_scores['sommes']:.1f}%")
        print(f"   • Paires: {avg_scores['paires']:.1f}%")
        print(f"   • Global: {avg_scores['global']:.1f}%")
        
        return quality_scores
    
    def export_system(self, grids: List[List[int]], profile: str, quality_scores: List[Dict]):
        """Exporte le système complet"""
        base_filename = f"system_{profile}"
        
        # Créer le dossier de sortie
        output_dir = Path("keno_output")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Export des grilles
        grilles_df = pd.DataFrame(grids, columns=[f'N{i+1}' for i in range(10)])
        grilles_df['Quality'] = [s.get('global', 0) for s in quality_scores]
        grilles_path = output_dir / f"{base_filename}_grilles.csv"
        grilles_df.to_csv(grilles_path, index=False)
        
        # 2. Export des métadonnées
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'profile': profile,
            'num_grids': len(grids),
            'top30_numbers': self.top30_numbers,
            'optimal_parameters': self.optimal_params,
            'top_pairs': [(p[0], p[1]) for p in self.top_pairs[:20]],
            'quality_statistics': {
                'average_quality': sum(s.get('global', 0) for s in quality_scores) / len(quality_scores),
                'min_quality': min(s.get('global', 0) for s in quality_scores),
                'max_quality': max(s.get('global', 0) for s in quality_scores)
            }
        }
        
        metadata_path = output_dir / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 3. Export du TOP 30 intelligent
        top30_df = pd.DataFrame([
            {
                'Numero': num, 
                'Score': self.top30_scores[num],
                'Frequence': self.stats.frequences.get(num, 0),
                'Retard': self.stats.retards.get(num, 0)
            }
            for num in self.top30_numbers
        ])
        top30_path = output_dir / f"{base_filename}_top30.csv"
        top30_df.to_csv(top30_path, index=False)
        print(f"✅ TOP 30 intelligent exporté: {top30_path}")
        
        print(f"\n📁 Système exporté:")
        print(f"   • Grilles: {grilles_path}")
        print(f"   • Métadonnées: {metadata_path}")
        print(f"   • TOP 30: {top30_path}")
        
        return grilles_path, metadata_path, top30_path
    
    def generate_system(self, profile: str):
        """Génère un système complet pour un profil donné"""
        print(f"\n🎯 Système {profile.title()} - {self.profiles[profile]['grids']} grilles")
        print("=" * 50)
        
        # Chargement et analyse
        self.load_and_analyze_data()
        
        # Calcul TOP 30 intelligent
        self.calculate_intelligent_top30()
        
        # Analyse des paramètres optimaux
        self.analyze_optimal_parameters()
        
        # Extraction des paires
        self.extract_top_pairs()
        
        # Génération des grilles
        num_grids = self.profiles[profile]['grids']
        grids = self.generate_diverse_grids(num_grids)
        
        if not grids:
            print("❌ Aucune grille générée")
            return False
        
        # Validation qualité
        quality_scores = self.validate_grids_quality(grids)
        
        # Export
        self.export_system(grids, profile, quality_scores)
        
        # Résumé
        avg_quality = sum(s.get('global', 0) for s in quality_scores) / len(quality_scores)
        print(f"\n🏆 SYSTÈME {profile.upper()} GÉNÉRÉ AVEC SUCCÈS!")
        print(f"📊 {len(grids)} grilles optimisées basées sur le TOP 30")
        print(f"🎯 Qualité globale: {avg_quality:.1f}%")
        
        # Échantillon de grilles
        print(f"\n📋 Échantillon de grilles (5 premières):")
        for i, grid in enumerate(grids[:5], 1):
            quality = quality_scores[i-1].get('global', 0)
            print(f"    {i}. {grid} (Q: {quality:.1f}%)")
        
        return True

def main():
    """Fonction principale"""
    print("🎲 GÉNÉRATEUR KENO INTELLIGENT AVEC SYSTÈME RÉDUCTEUR V2")
    print("=" * 60)
    
    # Vérifier s'il y a un CSV TOP 30 existant
    top30_csv_path = None
    output_dir = Path("keno_output")
    
    if output_dir.exists():
        # Chercher le fichier TOP 30 standard
        standard_top30 = output_dir / "keno_top30.csv"
        
        if standard_top30.exists():
            age_hours = (datetime.now().timestamp() - standard_top30.stat().st_mtime) / 3600
            print(f"\n📂 CSV TOP 30 trouvé: {standard_top30.name}")
            print(f"   📅 Âge: {age_hours:.1f}h")
            use_csv = input("   🤔 Utiliser ce TOP 30 ? (o/N): ").strip().lower()
            
            if use_csv in ['o', 'oui', 'y', 'yes']:
                top30_csv_path = str(standard_top30)
                print(f"   ✅ TOP 30 externe sélectionné")
            else:
                print(f"   🔄 Génération nouveau TOP 30")
    
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
            profile = "moyen"  # Défaut
        
        # Génération du système avec CSV TOP 30 optionnel
        generator = KenoIntelligentGeneratorV2(top30_csv_path=top30_csv_path)
        success = generator.generate_system(profile)
        
        if success:
            print(f"\n✅ Système {profile} généré avec succès!")
        else:
            print(f"\n❌ Échec de génération du système {profile}")
            
    except KeyboardInterrupt:
        print("\n👋 Génération interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

if __name__ == "__main__":
    main()
