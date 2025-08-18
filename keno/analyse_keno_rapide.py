#!/usr/bin/env python3
"""
Générateur de Statistiques Keno Rapide
=======================================

Script simplifié pour générer rapidement les statistiques essentielles du Keno.
Idéal pour des consultations rapides avant un tirage.

Usage:
    python analyse_keno_rapide.py [--top N] [--graphiques]

Author: Système Loto/Keno
Date: 18 août 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import argparse

def analyser_keno_rapide(csv_path, top_n=15, generer_graphiques=False):
    """Analyse rapide du Keno avec les statistiques essentielles"""
    
    print("🔄 Chargement des données...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    ball_cols = [f'b{i}' for i in range(1, 21)]
    total_tirages = len(df)
    
    print(f"✅ {total_tirages} tirages analysés ({df['date'].min().strftime('%d/%m/%Y')} → {df['date'].max().strftime('%d/%m/%Y')})")
    
    # 1. Fréquences et retards
    print("\n📊 ANALYSE DES FRÉQUENCES ET RETARDS")
    print("=" * 50)
    
    stats_numeros = []
    
    for numero in range(1, 71):
        # Compter les sorties
        sorties = 0
        derniere_date = None
        
        for _, row in df.iterrows():
            numbers = [row[col] for col in ball_cols]
            if numero in numbers:
                sorties += 1
                derniere_date = row['date']
        
        # Calculer le retard
        if derniere_date:
            tirages_apres = df[df['date'] > derniere_date]
            retard = len(tirages_apres)
            derniere_str = derniere_date.strftime('%d/%m/%Y')
        else:
            retard = total_tirages
            derniere_str = 'Jamais'
        
        # Calculs
        freq_theorique = total_tirages * 20 / 70
        ecart_freq = sorties - freq_theorique
        
        stats_numeros.append({
            'numero': numero,
            'sorties': sorties,
            'retard': retard,
            'derniere': derniere_str,
            'ecart': ecart_freq,
            'score': retard * 2 + max(0, -ecart_freq)  # Score de priorité simple
        })
    
    # Trier par score de priorité
    stats_df = pd.DataFrame(stats_numeros)
    stats_df = stats_df.sort_values('score', ascending=False)
    
    # Afficher le top
    print(f"\n🏆 TOP {top_n} NUMÉROS PRIORITAIRES:")
    print("-" * 70)
    print("Rang | N° | Sorties | Retard | Dernière sortie | Écart | Score")
    print("-" * 70)
    
    for i, row in stats_df.head(top_n).iterrows():
        print(f"{stats_df.index.get_loc(i)+1:4d} | {row['numero']:2d} | {row['sorties']:7d} | {row['retard']:6d} | {row['derniere']:11s} | {row['ecart']:5.1f} | {row['score']:5.1f}")
    
    # 2. Analyse pair/impair récente
    print(f"\n🔢 ANALYSE PAIR/IMPAIR (10 derniers tirages)")
    print("-" * 50)
    
    recent_df = df.tail(10)
    for _, row in recent_df.iterrows():
        numbers = [row[col] for col in ball_cols]
        pairs = sum(1 for n in numbers if n % 2 == 0)
        date_str = row['date'].strftime('%d/%m')
        print(f"{date_str} | Pairs: {pairs:2d} | Impairs: {20-pairs:2d} | {'⚖️' if pairs == 10 else '📊'}")
    
    # 3. Analyse zones récente
    print(f"\n🗺️ ANALYSE ZONES (10 derniers tirages)")
    print("-" * 50)
    
    for _, row in recent_df.iterrows():
        numbers = [row[col] for col in ball_cols]
        zone1 = sum(1 for n in numbers if n <= 35)
        date_str = row['date'].strftime('%d/%m')
        print(f"{date_str} | Zone 1-35: {zone1:2d} | Zone 36-70: {20-zone1:2d} | {'⚖️' if zone1 == 10 else '📊'}")
    
    # 4. Sommes récentes
    print(f"\n➕ ANALYSE SOMMES (10 derniers tirages)")
    print("-" * 50)
    
    for _, row in recent_df.iterrows():
        numbers = [row[col] for col in ball_cols]
        somme = sum(numbers)
        date_str = row['date'].strftime('%d/%m')
        ecart_710 = somme - 710
        print(f"{date_str} | Somme: {somme:3d} | Écart théorique: {ecart_710:+4d} | {'🎯' if abs(ecart_710) < 50 else '📊'}")
    
    # 5. Recommandations express
    print(f"\n🎯 RECOMMANDATIONS EXPRESS")
    print("-" * 50)
    
    top_5 = stats_df.head(5)['numero'].tolist()
    retards_extremes = stats_df[stats_df['retard'] > 50].head(3)['numero'].tolist()
    
    print(f"🔥 Top 5 priorités: {', '.join(map(str, top_5))}")
    print(f"⏰ Retards extrêmes: {', '.join(map(str, retards_extremes))}")
    
    # Moyennes récentes
    recent_pairs = recent_df.apply(lambda row: sum(1 for col in ball_cols if row[col] % 2 == 0), axis=1).mean()
    recent_zone1 = recent_df.apply(lambda row: sum(1 for col in ball_cols if row[col] <= 35), axis=1).mean()
    recent_somme = recent_df.apply(lambda row: sum(row[col] for col in ball_cols), axis=1).mean()
    
    print(f"📈 Tendance pairs: {recent_pairs:.1f}/20 ({'↗️' if recent_pairs > 10 else '↘️' if recent_pairs < 10 else '➡️'})")
    print(f"📈 Tendance zone 1: {recent_zone1:.1f}/20 ({'↗️' if recent_zone1 > 10 else '↘️' if recent_zone1 < 10 else '➡️'})")
    print(f"📈 Tendance somme: {recent_somme:.0f} ({'↗️' if recent_somme > 710 else '↘️' if recent_somme < 710 else '➡️'})")
    
    # Graphiques optionnels
    if generer_graphiques:
        print(f"\n📈 Génération des graphiques...")
        
        # Graphique simple des fréquences
        plt.figure(figsize=(15, 6))
        plt.bar(range(1, 71), [stats_df[stats_df['numero']==i]['sorties'].iloc[0] for i in range(1, 71)], 
                alpha=0.7, color='skyblue')
        plt.axhline(y=total_tirages * 20 / 70, color='red', linestyle='--', 
                   label=f'Fréquence théorique: {total_tirages * 20 / 70:.1f}')
        plt.xlabel('Numéros')
        plt.ylabel('Nombre de sorties')
        plt.title('Fréquences de sortie - Analyse rapide Keno')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plots_dir = Path("keno_analyse_plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / 'analyse_rapide.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Graphique sauvegardé: {plots_dir}/analyse_rapide.png")
    
    # Sauvegarder les résultats
    output_dir = Path("keno_stats_exports")
    output_dir.mkdir(exist_ok=True)
    
    stats_df.to_csv(output_dir / 'analyse_rapide.csv', index=False)
    print(f"\n💾 Résultats sauvegardés: {output_dir}/analyse_rapide.csv")
    
    return stats_df.head(top_n)

def main():
    """Fonction principale avec arguments"""
    parser = argparse.ArgumentParser(description='Analyse rapide des statistiques Keno')
    parser.add_argument('--top', type=int, default=15, help='Nombre de numéros prioritaires à afficher (défaut: 15)')
    parser.add_argument('--graphiques', action='store_true', help='Générer les graphiques')
    parser.add_argument('--csv', default='keno/keno_data/keno_consolidated.csv', help='Chemin vers le fichier CSV')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("Vérifiez le chemin ou lancez d'abord le téléchargement des données.")
        return
    
    print("🎰 ANALYSE RAPIDE KENO 🎰")
    print("=" * 30)
    
    # Lancer l'analyse
    top_numeros = analyser_keno_rapide(csv_path, args.top, args.graphiques)
    
    print(f"\n✅ Analyse terminée ! Top {args.top} numéros identifiés.")

if __name__ == "__main__":
    main()
