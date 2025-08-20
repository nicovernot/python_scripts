#!/usr/bin/env python3
"""
Script de Génération Rapide des Statistiques Keno
==================================================

Ce script génère toutes les statistiques essentielles du Keno en une seule commande.
Il créé les CSV et affiche un résumé des recommandations.

Usage:
    python stats_keno_express.py

Author: Système Loto/Keno
Date: 18 août 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def main():
    """Fonction principale pour l'analyse express"""
    print("🎰 GÉNÉRATEUR DE STATISTIQUES KENO EXPRESS 🎰")
    print("=" * 55)
    
    # Vérifier le fichier de données
    csv_path = Path("keno/keno_data/keno_consolidated.csv")
    if not csv_path.exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("Vérifiez le chemin ou téléchargez d'abord les données.")
        return
    
    # Charger les données
    print("🔄 Chargement des données...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    ball_cols = [f'b{i}' for i in range(1, 21)]
    total_tirages = len(df)
    
    print(f"✅ {total_tirages} tirages analysés ({df['date'].min().strftime('%d/%m/%Y')} → {df['date'].max().strftime('%d/%m/%Y')})")
    
    # Créer les dossiers de sortie
    output_dir = Path("keno_stats_exports")
    output_dir.mkdir(exist_ok=True)
    
    print("\n📊 GÉNÉRATION DES STATISTIQUES...")
    print("-" * 40)
    
    # 1. FRÉQUENCES ET RETARDS
    print("🔢 Calcul des fréquences et retards...")
    
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
        
        # Calculs avancés
        freq_theorique = total_tirages * 20 / 70
        ecart_freq = sorties - freq_theorique
        priorite = retard * 2 + max(0, -ecart_freq)
        
        # Statut
        if retard > 10:
            statut = "TRÈS EN RETARD"
        elif retard > 7:
            statut = "EN RETARD"
        elif sorties > freq_theorique * 1.2:
            statut = "CHAUD"
        elif sorties < freq_theorique * 0.8:
            statut = "FROID"
        else:
            statut = "NORMAL"
        
        stats_numeros.append({
            'numero': numero,
            'sorties': sorties,
            'retard_tirages': retard,
            'derniere_sortie': derniere_str,
            'ecart_theorique': round(ecart_freq, 1),
            'priorite': round(priorite, 1),
            'statut': statut,
            'pourcentage': round((sorties / (total_tirages * 20)) * 100, 2)
        })
    
    # Créer le DataFrame et trier par priorité
    stats_df = pd.DataFrame(stats_numeros)
    stats_df = stats_df.sort_values('priorite', ascending=False)
    
    # Sauvegarder les fréquences
    freq_file = output_dir / "frequences_express.csv"
    stats_df.to_csv(freq_file, index=False)
    print(f"   💾 Sauvegardé: {freq_file}")
    
    # 2. ANALYSE PAIR/IMPAIR
    print("⚖️ Analyse pair/impair...")
    
    pair_impair_stats = []
    for _, row in df.iterrows():
        numbers = [row[col] for col in ball_cols]
        pairs = sum(1 for n in numbers if n % 2 == 0)
        pair_impair_stats.append({
            'date': row['date'],
            'pairs': pairs,
            'impairs': 20 - pairs,
            'equilibre': abs(pairs - 10)
        })
    
    pair_df = pd.DataFrame(pair_impair_stats)
    pair_file = output_dir / "pair_impair_express.csv"
    pair_df.to_csv(pair_file, index=False)
    print(f"   💾 Sauvegardé: {pair_file}")
    
    # 3. ANALYSE DES ZONES
    print("🗺️ Analyse des zones...")
    
    zones_stats = []
    for _, row in df.iterrows():
        numbers = [row[col] for col in ball_cols]
        zone1 = sum(1 for n in numbers if n <= 35)
        zones_stats.append({
            'date': row['date'],
            'zone_1_35': zone1,
            'zone_36_70': 20 - zone1,
            'equilibre': abs(zone1 - 10)
        })
    
    zones_df = pd.DataFrame(zones_stats)
    zones_file = output_dir / "zones_express.csv"
    zones_df.to_csv(zones_file, index=False)
    print(f"   💾 Sauvegardé: {zones_file}")
    
    # 4. ANALYSE DES SOMMES
    print("➕ Analyse des sommes...")
    
    sommes_stats = []
    for _, row in df.iterrows():
        numbers = [row[col] for col in ball_cols]
        somme = sum(numbers)
        sommes_stats.append({
            'date': row['date'],
            'somme': somme,
            'ecart_710': somme - 710
        })
    
    sommes_df = pd.DataFrame(sommes_stats)
    sommes_file = output_dir / "sommes_express.csv"
    sommes_df.to_csv(sommes_file, index=False)
    print(f"   💾 Sauvegardé: {sommes_file}")
    
    # AFFICHAGE DES RÉSULTATS
    print(f"\n🎯 RÉSULTATS DE L'ANALYSE EXPRESS")
    print("=" * 55)
    
    # Top 15 prioritaires
    print(f"\n🏆 TOP 15 NUMÉROS PRIORITAIRES:")
    print("-" * 55)
    print("Rang | N° | Sorties | Retard | Statut          | Priorité")
    print("-" * 55)
    
    for i, row in stats_df.head(15).iterrows():
        rang = stats_df.index.get_loc(i) + 1
        print(f"{rang:4d} | {row['numero']:2d} | {row['sorties']:7d} | {row['retard_tirages']:6d} | {row['statut']:15s} | {row['priorite']:8.1f}")
    
    # Statistiques récentes (10 derniers tirages)
    recent_df = df.tail(10)
    
    print(f"\n📈 TENDANCES RÉCENTES (10 derniers tirages):")
    print("-" * 55)
    
    # Moyennes récentes
    recent_pairs = recent_df.apply(lambda row: sum(1 for col in ball_cols if row[col] % 2 == 0), axis=1).mean()
    recent_zone1 = recent_df.apply(lambda row: sum(1 for col in ball_cols if row[col] <= 35), axis=1).mean()
    recent_somme = recent_df.apply(lambda row: sum(row[col] for col in ball_cols), axis=1).mean()
    
    print(f"⚖️ Moyenne pairs/impairs: {recent_pairs:.1f}/20 ({'↗️' if recent_pairs > 10 else '↘️' if recent_pairs < 10 else '➡️'})")
    print(f"🗺️ Moyenne zones: {recent_zone1:.1f}/20 ({'↗️' if recent_zone1 > 10 else '↘️' if recent_zone1 < 10 else '➡️'})")
    print(f"➕ Moyenne somme: {recent_somme:.0f} ({'↗️' if recent_somme > 710 else '↘️' if recent_somme < 710 else '➡️'})")
    
    # Recommandations express
    top_5 = stats_df.head(5)['numero'].tolist()
    retards_extremes = stats_df[stats_df['retard_tirages'] > 10]['numero'].tolist()[:3]
    froids = stats_df.nsmallest(5, 'sorties')['numero'].tolist()
    
    print(f"\n🎯 RECOMMANDATIONS EXPRESS:")
    print("-" * 55)
    print(f"🔥 Top 5 priorités absolues: {', '.join(map(str, top_5))}")
    if retards_extremes:
        print(f"⏰ Retards extrêmes (>10 tirages): {', '.join(map(str, retards_extremes))}")
    print(f"❄️ Numéros les plus froids: {', '.join(map(str, froids))}")
    
    print(f"\n💡 CONSEILS STRATÉGIQUES:")
    print("-" * 55)
    if recent_pairs < 8:
        print("⚖️ Privilégier plus de numéros pairs (tendance récente vers les impairs)")
    elif recent_pairs > 12:
        print("⚖️ Privilégier plus de numéros impairs (tendance récente vers les pairs)")
    else:
        print("⚖️ Équilibre pair/impair correct (maintenir 8-12 pairs)")
    
    if recent_zone1 < 8:
        print("🗺️ Privilégier la zone 1-35 (tendance récente vers 36-70)")
    elif recent_zone1 > 12:
        print("🗺️ Privilégier la zone 36-70 (tendance récente vers 1-35)")
    else:
        print("🗺️ Équilibre des zones correct (maintenir 8-12 en zone 1)")
    
    if recent_somme < 660:
        print("➕ Privilégier des numéros plus élevés (sommes récentes basses)")
    elif recent_somme > 760:
        print("➕ Privilégier des numéros plus bas (sommes récentes élevées)")
    else:
        print("➕ Niveau de somme correct (viser 660-760)")
    
    # Tableau des retards par groupes
    print(f"\n📊 RÉPARTITION DES RETARDS:")
    print("-" * 55)
    retard_0_3 = len(stats_df[stats_df['retard_tirages'] <= 3])
    retard_4_7 = len(stats_df[(stats_df['retard_tirages'] >= 4) & (stats_df['retard_tirages'] <= 7)])
    retard_8_plus = len(stats_df[stats_df['retard_tirages'] >= 8])
    
    print(f"🟢 Retard 0-3 tirages: {retard_0_3} numéros")
    print(f"🟡 Retard 4-7 tirages: {retard_4_7} numéros")
    print(f"🔴 Retard 8+ tirages: {retard_8_plus} numéros")
    
    # Sauvegarde du résumé
    resume_file = output_dir / "resume_express.txt"
    with open(resume_file, 'w', encoding='utf-8') as f:
        f.write("🎰 RÉSUMÉ EXPRESS KENO 🎰\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"📅 Génération: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"📊 Période: {df['date'].min().strftime('%d/%m/%Y')} → {df['date'].max().strftime('%d/%m/%Y')}\n")
        f.write(f"🎲 Total tirages: {total_tirages}\n\n")
        
        f.write("🏆 TOP 5 PRIORITÉS:\n")
        for i, num in enumerate(top_5, 1):
            row = stats_df[stats_df['numero'] == num].iloc[0]
            f.write(f"{i}. N°{num:2d} - Priorité: {row['priorite']:5.1f} - {row['statut']}\n")
        
        f.write(f"\n📈 TENDANCES RÉCENTES:\n")
        f.write(f"Pairs: {recent_pairs:.1f}/20, Zone 1: {recent_zone1:.1f}/20, Somme: {recent_somme:.0f}\n")
    
    print(f"\n✅ ANALYSE TERMINÉE !")
    print("=" * 55)
    print(f"📁 Tous les fichiers sauvegardés dans: {output_dir}")
    print(f"📋 Résumé textuel: {resume_file}")
    print("🎯 Utilisez ces données pour optimiser vos prochaines grilles !")

if __name__ == "__main__":
    main()
