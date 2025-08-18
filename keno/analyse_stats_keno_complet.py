#!/usr/bin/env python3
"""
Générateur de Statistiques et Visualisations Keno Complet
=========================================================

Ce script génère toutes les statistiques utiles pour l'analyse du Keno :
- Fréquences des numéros
- Analyse pair/impair
- Zones (1-35, 36-70)
- Sommes des tirages
- Tableau des retards
- Dates des derniers tirages
- Écarts entre tirages
- Combinaisons les plus fréquentes
- Analyse des séquences
- Heatmaps et visualisations

Author: Système Loto/Keno
Date: 18 août 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

class KenoStatsAnalyzer:
    def __init__(self, csv_path):
        """Initialise l'analyseur avec le fichier de données Keno"""
        self.csv_path = Path(csv_path)
        self.output_dir = Path("keno_stats_exports")
        self.plots_dir = Path("keno_analyse_plots")
        
        # Créer les dossiers de sortie
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Charger les données
        print("🔄 Chargement des données Keno...")
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"✅ {len(self.df)} tirages chargés de {self.df['date'].min().strftime('%d/%m/%Y')} à {self.df['date'].max().strftime('%d/%m/%Y')}")
        
        # Colonnes des boules
        self.ball_cols = [f'b{i}' for i in range(1, 21)]
        
        # Timestamp pour les fichiers
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def analyse_frequences(self):
        """Analyse les fréquences de sortie de chaque numéro"""
        print("📊 Analyse des fréquences...")
        
        # Créer un DataFrame avec tous les numéros tirés
        all_numbers = []
        for _, row in self.df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            all_numbers.extend(numbers)
        
        # Compter les fréquences
        freq_series = pd.Series(all_numbers)
        frequencies = freq_series.value_counts().sort_index()
        
        # Compléter avec les numéros manquants (1-70)
        full_frequencies = pd.Series(0, index=range(1, 71))
        full_frequencies.update(frequencies)
        
        # Calculer les statistiques
        total_tirages = len(self.df)
        expected_freq = total_tirages * 20 / 70  # Fréquence théorique
        
        freq_stats = pd.DataFrame({
            'numero': range(1, 71),
            'frequence': full_frequencies.values,
            'frequence_theorique': expected_freq,
            'ecart_theorique': full_frequencies.values - expected_freq,
            'pourcentage': (full_frequencies.values / (total_tirages * 20)) * 100,
            'derniere_sortie': '',
            'retard_tirages': 0
        })
        
        # Calculer les retards et dernières sorties
        for numero in range(1, 71):
            derniere_sortie = self.get_derniere_sortie(numero)
            if derniere_sortie:
                freq_stats.loc[freq_stats['numero'] == numero, 'derniere_sortie'] = derniere_sortie.strftime('%d/%m/%Y')
                retard = self.get_retard_tirages(numero, derniere_sortie)
                freq_stats.loc[freq_stats['numero'] == numero, 'retard_tirages'] = retard
            else:
                freq_stats.loc[freq_stats['numero'] == numero, 'derniere_sortie'] = 'Jamais'
                freq_stats.loc[freq_stats['numero'] == numero, 'retard_tirages'] = total_tirages
        
        # Trier par fréquence décroissante
        freq_stats = freq_stats.sort_values('frequence', ascending=False)
        
        # Sauvegarder
        output_file = self.output_dir / f"frequences_keno_{self.timestamp}.csv"
        freq_stats.to_csv(output_file, index=False)
        print(f"💾 Fréquences sauvegardées: {output_file}")
        
        return freq_stats
    
    def get_derniere_sortie(self, numero):
        """Trouve la date de dernière sortie d'un numéro"""
        for _, row in self.df.sort_values('date', ascending=False).iterrows():
            numbers = [row[col] for col in self.ball_cols]
            if numero in numbers:
                return row['date']
        return None
    
    def get_retard_tirages(self, numero, derniere_date):
        """Calcule le retard en nombre de tirages"""
        tirages_apres = self.df[self.df['date'] > derniere_date]
        return len(tirages_apres)
    
    def analyse_pair_impair(self):
        """Analyse la répartition pair/impair"""
        print("🔢 Analyse pair/impair...")
        
        pair_impair_stats = []
        
        for _, row in self.df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            pairs = sum(1 for n in numbers if n % 2 == 0)
            impairs = 20 - pairs
            
            pair_impair_stats.append({
                'date': row['date'],
                'numero_tirage': row['numero_tirage'],
                'pairs': pairs,
                'impairs': impairs,
                'ratio_pairs': pairs / 20,
                'equilibre': abs(pairs - impairs)
            })
        
        pair_impair_df = pd.DataFrame(pair_impair_stats)
        
        # Statistiques globales
        avg_pairs = pair_impair_df['pairs'].mean()
        avg_impairs = pair_impair_df['impairs'].mean()
        
        summary = {
            'total_tirages': len(pair_impair_df),
            'moyenne_pairs': avg_pairs,
            'moyenne_impairs': avg_impairs,
            'pairs_min': pair_impair_df['pairs'].min(),
            'pairs_max': pair_impair_df['pairs'].max(),
            'impairs_min': pair_impair_df['impairs'].min(),
            'impairs_max': pair_impair_df['impairs'].max(),
            'equilibre_parfait_freq': len(pair_impair_df[pair_impair_df['pairs'] == 10]) / len(pair_impair_df) * 100
        }
        
        # Sauvegarder
        output_file = self.output_dir / f"pair_impair_keno_{self.timestamp}.csv"
        pair_impair_df.to_csv(output_file, index=False)
        
        summary_file = self.output_dir / f"pair_impair_resume_{self.timestamp}.csv"
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        
        print(f"💾 Analyse pair/impair sauvegardée: {output_file}")
        
        return pair_impair_df, summary
    
    def analyse_zones(self):
        """Analyse la répartition par zones (1-35 et 36-70)"""
        print("🗺️ Analyse des zones...")
        
        zones_stats = []
        
        for _, row in self.df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            zone1 = sum(1 for n in numbers if n <= 35)  # Zone 1-35
            zone2 = 20 - zone1  # Zone 36-70
            
            zones_stats.append({
                'date': row['date'],
                'numero_tirage': row['numero_tirage'],
                'zone_1_35': zone1,
                'zone_36_70': zone2,
                'ratio_zone1': zone1 / 20,
                'equilibre_zones': abs(zone1 - zone2)
            })
        
        zones_df = pd.DataFrame(zones_stats)
        
        # Statistiques globales
        summary = {
            'total_tirages': len(zones_df),
            'moyenne_zone1': zones_df['zone_1_35'].mean(),
            'moyenne_zone2': zones_df['zone_36_70'].mean(),
            'zone1_min': zones_df['zone_1_35'].min(),
            'zone1_max': zones_df['zone_1_35'].max(),
            'zone2_min': zones_df['zone_36_70'].min(),
            'zone2_max': zones_df['zone_36_70'].max(),
            'equilibre_parfait_freq': len(zones_df[zones_df['zone_1_35'] == 10]) / len(zones_df) * 100
        }
        
        # Sauvegarder
        output_file = self.output_dir / f"zones_keno_{self.timestamp}.csv"
        zones_df.to_csv(output_file, index=False)
        
        summary_file = self.output_dir / f"zones_resume_{self.timestamp}.csv"
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        
        print(f"💾 Analyse des zones sauvegardée: {output_file}")
        
        return zones_df, summary
    
    def analyse_sommes(self):
        """Analyse les sommes des tirages"""
        print("➕ Analyse des sommes...")
        
        sommes_stats = []
        
        for _, row in self.df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            somme = sum(numbers)
            
            sommes_stats.append({
                'date': row['date'],
                'numero_tirage': row['numero_tirage'],
                'somme': somme,
                'moyenne_numero': somme / 20,
                'ecart_moyenne_theorique': somme - (71 * 10)  # Moyenne théorique = 35.5 * 20 = 710
            })
        
        sommes_df = pd.DataFrame(sommes_stats)
        
        # Statistiques globales
        summary = {
            'total_tirages': len(sommes_df),
            'somme_min': sommes_df['somme'].min(),
            'somme_max': sommes_df['somme'].max(),
            'somme_moyenne': sommes_df['somme'].mean(),
            'somme_mediane': sommes_df['somme'].median(),
            'somme_std': sommes_df['somme'].std(),
            'somme_theorique': 710
        }
        
        # Sauvegarder
        output_file = self.output_dir / f"sommes_keno_{self.timestamp}.csv"
        sommes_df.to_csv(output_file, index=False)
        
        summary_file = self.output_dir / f"sommes_resume_{self.timestamp}.csv"
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        
        print(f"💾 Analyse des sommes sauvegardée: {output_file}")
        
        return sommes_df, summary
    
    def tableau_retards_complet(self):
        """Génère un tableau complet des retards avec toutes les informations"""
        print("📅 Génération du tableau des retards complet...")
        
        tableau = []
        
        for numero in range(1, 71):
            # Statistiques de base
            freq = self.get_frequence_numero(numero)
            derniere_sortie = self.get_derniere_sortie(numero)
            
            if derniere_sortie:
                retard = self.get_retard_tirages(numero, derniere_sortie)
                derniere_str = derniere_sortie.strftime('%d/%m/%Y')
                jours_retard = (datetime.now().date() - derniere_sortie.date()).days
            else:
                retard = len(self.df)
                derniere_str = 'Jamais'
                jours_retard = -1
            
            # Calculer retard théorique
            freq_theorique = len(self.df) * 20 / 70
            retard_theorique = 70 / 20  # En moyenne, un numéro sort tous les 3.5 tirages
            
            # Analyser la régularité
            sorties = self.get_dates_sorties(numero)
            regularite = self.calculer_regularite(sorties)
            
            tableau.append({
                'numero': numero,
                'frequence': freq,
                'frequence_theorique': freq_theorique,
                'ecart_frequence': freq - freq_theorique,
                'derniere_sortie': derniere_str,
                'retard_tirages': retard,
                'retard_jours': jours_retard,
                'retard_theorique': retard_theorique,
                'ratio_retard': retard / retard_theorique if retard_theorique > 0 else 0,
                'regularite': regularite,
                'statut': self.get_statut_numero(numero, freq, retard),
                'priorite': self.calculer_priorite(freq, retard, regularite)
            })
        
        tableau_df = pd.DataFrame(tableau)
        tableau_df = tableau_df.sort_values('priorite', ascending=False)
        
        # Sauvegarder
        output_file = self.output_dir / f"tableau_retards_complet_{self.timestamp}.csv"
        tableau_df.to_csv(output_file, index=False)
        
        print(f"💾 Tableau des retards complet sauvegardé: {output_file}")
        
        return tableau_df
    
    def get_frequence_numero(self, numero):
        """Compte le nombre de sorties d'un numéro"""
        count = 0
        for _, row in self.df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            if numero in numbers:
                count += 1
        return count
    
    def get_dates_sorties(self, numero):
        """Récupère toutes les dates de sortie d'un numéro"""
        dates = []
        for _, row in self.df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            if numero in numbers:
                dates.append(row['date'])
        return sorted(dates)
    
    def calculer_regularite(self, dates_sorties):
        """Calcule un indicateur de régularité basé sur l'écart-type des intervalles"""
        if len(dates_sorties) < 2:
            return 0
        
        intervalles = []
        for i in range(1, len(dates_sorties)):
            delta = (dates_sorties[i] - dates_sorties[i-1]).days
            intervalles.append(delta)
        
        if not intervalles:
            return 0
        
        # Plus l'écart-type est faible, plus c'est régulier
        std_intervalles = np.std(intervalles)
        moyenne_intervalles = np.mean(intervalles)
        
        # Score de régularité (0-100, 100 = très régulier)
        if moyenne_intervalles > 0:
            regularite = max(0, 100 - (std_intervalles / moyenne_intervalles) * 50)
        else:
            regularite = 0
        
        return round(regularite, 1)
    
    def get_statut_numero(self, numero, freq, retard):
        """Détermine le statut d'un numéro"""
        freq_theorique = len(self.df) * 20 / 70
        retard_theorique = 3.5
        
        if retard > retard_theorique * 3:
            return "TRÈS EN RETARD"
        elif retard > retard_theorique * 2:
            return "EN RETARD"
        elif freq > freq_theorique * 1.2:
            return "CHAUD"
        elif freq < freq_theorique * 0.8:
            return "FROID"
        else:
            return "NORMAL"
    
    def calculer_priorite(self, freq, retard, regularite):
        """Calcule un score de priorité pour le prochain tirage"""
        freq_theorique = len(self.df) * 20 / 70
        retard_theorique = 3.5
        
        # Facteurs de priorité
        facteur_retard = min(retard / retard_theorique, 5) * 30  # Max 150 points
        facteur_freq = max(0, (freq_theorique - freq) / freq_theorique) * 50  # Max 50 points
        facteur_regularite = regularite * 0.2  # Max 20 points
        
        priorite = facteur_retard + facteur_freq + facteur_regularite
        return round(priorite, 1)
    
    def generer_visualisations(self, freq_stats, pair_impair_df, zones_df, sommes_df):
        """Génère toutes les visualisations"""
        print("📈 Génération des visualisations...")
        
        # 1. Graphique des fréquences
        plt.figure(figsize=(15, 8))
        plt.bar(freq_stats['numero'], freq_stats['frequence'], alpha=0.7, color='skyblue')
        plt.axhline(y=freq_stats['frequence_theorique'].iloc[0], color='red', linestyle='--', 
                   label=f'Fréquence théorique: {freq_stats["frequence_theorique"].iloc[0]:.1f}')
        plt.xlabel('Numéros')
        plt.ylabel('Fréquence de sortie')
        plt.title('Fréquences de sortie des numéros Keno')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'frequences_keno_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap des retards
        plt.figure(figsize=(14, 10))
        retards_matrix = np.array(freq_stats['retard_tirages'].values).reshape(10, 7)
        sns.heatmap(retards_matrix, annot=True, fmt='d', cmap='RdYlBu_r', 
                   xticklabels=range(1, 8), yticklabels=range(1, 11))
        plt.title('Heatmap des retards (en nombre de tirages)')
        plt.xlabel('Colonnes')
        plt.ylabel('Lignes')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'heatmap_retards_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribution pair/impair
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(pair_impair_df['pairs'], bins=range(0, 22), alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(x=10, color='red', linestyle='--', label='Équilibre parfait (10)')
        plt.xlabel('Nombre de numéros pairs')
        plt.ylabel('Fréquence')
        plt.title('Distribution des numéros pairs par tirage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(pair_impair_df.index[-100:], pair_impair_df['pairs'].tail(100), marker='o', markersize=3)
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Derniers tirages')
        plt.ylabel('Nombre de pairs')
        plt.title('Évolution récente pair/impair')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'pair_impair_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Distribution des zones
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(zones_df['zone_1_35'], bins=range(0, 22), alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(x=10, color='red', linestyle='--', label='Équilibre parfait (10)')
        plt.xlabel('Nombre de numéros zone 1-35')
        plt.ylabel('Fréquence')
        plt.title('Distribution zone 1-35 par tirage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(zones_df.index[-100:], zones_df['zone_1_35'].tail(100), marker='o', markersize=3, color='green')
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Derniers tirages')
        plt.ylabel('Numéros zone 1-35')
        plt.title('Évolution récente des zones')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'zones_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Distribution des sommes
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(sommes_df['somme'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(x=710, color='red', linestyle='--', label='Somme théorique (710)')
        plt.xlabel('Somme des 20 numéros')
        plt.ylabel('Fréquence')
        plt.title('Distribution des sommes par tirage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(sommes_df.index[-100:], sommes_df['somme'].tail(100), marker='o', markersize=3, color='darkorange')
        plt.axhline(y=710, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Derniers tirages')
        plt.ylabel('Somme')
        plt.title('Évolution récente des sommes')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'sommes_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Top 20 des numéros les plus en retard
        plt.figure(figsize=(12, 8))
        top_retards = freq_stats.nlargest(20, 'retard_tirages')
        plt.barh(range(len(top_retards)), top_retards['retard_tirages'], color='red', alpha=0.7)
        plt.yticks(range(len(top_retards)), [f"N°{n}" for n in top_retards['numero']])
        plt.xlabel('Retard (nombre de tirages)')
        plt.title('Top 20 des numéros les plus en retard')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'top_retards_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualisations sauvegardées dans: {self.plots_dir}")
    
    def generer_rapport_complet(self):
        """Génère un rapport complet avec recommandations"""
        print("📋 Génération du rapport complet...")
        
        # Analyser toutes les statistiques
        freq_stats = self.analyse_frequences()
        pair_impair_df, pair_impair_summary = self.analyse_pair_impair()
        zones_df, zones_summary = self.analyse_zones()
        sommes_df, sommes_summary = self.analyse_sommes()
        tableau_retards = self.tableau_retards_complet()
        
        # Générer les visualisations
        self.generer_visualisations(freq_stats, pair_impair_df, zones_df, sommes_df)
        
        # Recommandations
        top_priorites = tableau_retards.head(10)
        top_retards = freq_stats.nlargest(10, 'retard_tirages')
        top_froids = freq_stats.nsmallest(10, 'frequence')
        
        # Créer le rapport
        rapport = {
            'date_generation': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'periode_analyse': f"{self.df['date'].min().strftime('%d/%m/%Y')} - {self.df['date'].max().strftime('%d/%m/%Y')}",
            'total_tirages': len(self.df),
            'derniere_mise_a_jour': self.df['date'].max().strftime('%d/%m/%Y'),
            
            # Top recommandations
            'top_10_priorites': top_priorites[['numero', 'priorite', 'statut', 'retard_tirages']].to_dict('records'),
            'top_10_retards': top_retards[['numero', 'retard_tirages', 'derniere_sortie']].to_dict('records'),
            'top_10_froids': top_froids[['numero', 'frequence', 'derniere_sortie']].to_dict('records'),
            
            # Résumés statistiques
            'resume_pair_impair': pair_impair_summary,
            'resume_zones': zones_summary,
            'resume_sommes': sommes_summary,
            
            # Tendances récentes (30 derniers tirages)
            'tendances_recentes': self.analyser_tendances_recentes()
        }
        
        # Sauvegarder le rapport
        import json
        rapport_file = self.output_dir / f"rapport_complet_{self.timestamp}.json"
        with open(rapport_file, 'w', encoding='utf-8') as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False, default=str)
        
        # Créer aussi un résumé texte
        self.creer_resume_texte(rapport)
        
        print(f"📋 Rapport complet sauvegardé: {rapport_file}")
        
        return rapport
    
    def analyser_tendances_recentes(self, nb_tirages=30):
        """Analyse les tendances sur les derniers tirages"""
        recent_df = self.df.tail(nb_tirages)
        
        # Numéros les plus sortis récemment
        recent_numbers = []
        for _, row in recent_df.iterrows():
            numbers = [row[col] for col in self.ball_cols]
            recent_numbers.extend(numbers)
        
        recent_freq = pd.Series(recent_numbers).value_counts()
        
        return {
            'periode': f"{nb_tirages} derniers tirages",
            'top_10_recents': recent_freq.head(10).to_dict(),
            'moyenne_pairs_recente': recent_df.apply(lambda row: sum(1 for col in self.ball_cols if row[col] % 2 == 0), axis=1).mean(),
            'moyenne_zone1_recente': recent_df.apply(lambda row: sum(1 for col in self.ball_cols if row[col] <= 35), axis=1).mean(),
            'moyenne_somme_recente': recent_df.apply(lambda row: sum(row[col] for col in self.ball_cols), axis=1).mean()
        }
    
    def creer_resume_texte(self, rapport):
        """Crée un résumé textuel du rapport"""
        resume_file = self.output_dir / f"resume_analyse_{self.timestamp}.txt"
        
        with open(resume_file, 'w', encoding='utf-8') as f:
            f.write("🎰 RAPPORT D'ANALYSE KENO COMPLET 🎰\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📅 Génération: {rapport['date_generation']}\n")
            f.write(f"📊 Période analysée: {rapport['periode_analyse']}\n")
            f.write(f"🎲 Total tirages: {rapport['total_tirages']}\n")
            f.write(f"🔄 Dernière MAJ: {rapport['derniere_mise_a_jour']}\n\n")
            
            f.write("🏆 TOP 10 NUMÉROS PRIORITAIRES:\n")
            f.write("-" * 40 + "\n")
            for i, num in enumerate(rapport['top_10_priorites'], 1):
                f.write(f"{i:2d}. N°{num['numero']:2d} - Priorité: {num['priorite']:5.1f} - {num['statut']} (Retard: {num['retard_tirages']} tirages)\n")
            
            f.write(f"\n🔥 TOP 10 PLUS EN RETARD:\n")
            f.write("-" * 40 + "\n")
            for i, num in enumerate(rapport['top_10_retards'], 1):
                f.write(f"{i:2d}. N°{num['numero']:2d} - Retard: {num['retard_tirages']:3d} tirages - Dernière sortie: {num['derniere_sortie']}\n")
            
            f.write(f"\n❄️ TOP 10 NUMÉROS FROIDS:\n")
            f.write("-" * 40 + "\n")
            for i, num in enumerate(rapport['top_10_froids'], 1):
                f.write(f"{i:2d}. N°{num['numero']:2d} - Fréquence: {num['frequence']:3d} - Dernière sortie: {num['derniere_sortie']}\n")
            
            f.write(f"\n📈 TENDANCES RÉCENTES (30 derniers tirages):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Moyenne pairs: {rapport['tendances_recentes']['moyenne_pairs_recente']:.1f}/20\n")
            f.write(f"Moyenne zone 1-35: {rapport['tendances_recentes']['moyenne_zone1_recente']:.1f}/20\n")
            f.write(f"Moyenne somme: {rapport['tendances_recentes']['moyenne_somme_recente']:.0f}\n")
            
            f.write(f"\n🎯 RECOMMANDATIONS POUR LE PROCHAIN TIRAGE:\n")
            f.write("-" * 40 + "\n")
            f.write("Privilégier les numéros avec forte priorité et long retard\n")
            f.write("Équilibrer pair/impair (objectif: 8-12 pairs)\n")
            f.write("Équilibrer zones 1-35 et 36-70 (objectif: 8-12 zone 1)\n")
            f.write("Viser une somme proche de 710\n")
        
        print(f"📋 Résumé textuel sauvegardé: {resume_file}")

def main():
    """Fonction principale"""
    csv_path = "keno/keno_data/keno_consolidated.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("Vérifiez le chemin ou lancez d'abord le téléchargement des données.")
        return
    
    # Créer l'analyseur
    analyzer = KenoStatsAnalyzer(csv_path)
    
    # Générer le rapport complet
    print("\n🚀 Lancement de l'analyse complète...")
    rapport = analyzer.generer_rapport_complet()
    
    print(f"\n✅ Analyse terminée !")
    print(f"📁 Fichiers CSV: {analyzer.output_dir}")
    print(f"📈 Graphiques: {analyzer.plots_dir}")
    print(f"🎯 Consultez le résumé: {analyzer.output_dir}/resume_analyse_{analyzer.timestamp}.txt")

if __name__ == "__main__":
    main()
