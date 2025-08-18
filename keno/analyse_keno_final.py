#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ANALYSE COMPLÈTE DES DONNÉES KENO
====================================

Analyse des fréquences, retards et recommandations pour le Keno.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

def analyze_keno_data():
    """Analyse complète des données Keno"""
    print("🎯 ANALYSE DES DONNÉES KENO")
    print("=" * 50)
    
    # Répertoires
    KENO_DATA_DIR = Path(__file__).parent / "keno_data"
    OUTPUT_DIR = Path(__file__).parent / "keno_analyse"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Charger TOUS les fichiers CSV avec le nouveau format
    csv_files = sorted(list(KENO_DATA_DIR.glob("keno_*.csv")))
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé")
        print("💡 Lancez d'abord: python keno_cli.py extract")
        return
    
    print(f"📂 Chargement de {len(csv_files)} fichiers...")
    
    all_data = []
    total_tirages = 0
    
    try:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            all_data.append(df)
            total_tirages += len(df)
            print(f"   ✓ {csv_file.name}: {len(df)} tirages")
        
        # Concaténer toutes les données
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('date').reset_index(drop=True)
        else:
            print("❌ Aucune donnée chargée")
            return
            
        print(f"📊 {total_tirages} tirages chargés au total")
        print(f"📅 Période: {df['date'].min()} → {df['date'].max()}")
        
        # 1. ANALYSE DES FRÉQUENCES
        print(f"\n📈 1. ANALYSE DES FRÉQUENCES")
        print("-" * 30)
        
        # Nouvelles colonnes format unifié: b1, b2, ..., b20
        ball_columns = [f'b{i}' for i in range(1, 21)]
        all_numbers = Counter()
        
        # Compter toutes les occurrences
        for col in ball_columns:
            if col in df.columns:
                all_numbers.update(df[col].dropna().astype(int))
        
        # Statistiques des fréquences
        total_occurrences = sum(all_numbers.values())
        avg_frequency = total_occurrences / 70  # 70 numéros possibles
        
        print(f"Total occurrences : {total_occurrences:,}")
        print(f"Fréquence moyenne : {avg_frequency:.1f}")
        
        # Top et bottom fréquences
        most_common = all_numbers.most_common(10)
        least_common = all_numbers.most_common()[-10:]
        
        print(f"\n🔥 TOP 10 - Numéros les plus fréquents :")
        for i, (num, count) in enumerate(most_common, 1):
            pct = (count / total_occurrences) * 100
            print(f"   {i:2d}. Numéro {num:2d} : {count:4d} fois ({pct:.1f}%)")
        
        print(f"\n❄️  TOP 10 - Numéros les moins fréquents :")
        for i, (num, count) in enumerate(reversed(least_common), 1):
            pct = (count / total_occurrences) * 100
            print(f"   {i:2d}. Numéro {num:2d} : {count:4d} fois ({pct:.1f}%)")
        
        # 2. ANALYSE DES RETARDS
        print(f"\n⏰ 2. ANALYSE DES RETARDS (100 derniers tirages)")
        print("-" * 40)
        
        recent_draws = df.head(100)  # 100 tirages les plus récents
        last_seen = {}
        
        # Pour chaque numéro, trouver sa dernière apparition
        for num in range(1, 71):
            found = False
            for idx in range(len(recent_draws)):
                row = recent_draws.iloc[idx]
                for col in ball_columns:
                    if col in row and pd.notna(row[col]) and int(row[col]) == num:
                        last_seen[num] = idx
                        found = True
                        break
                if found:
                    break
            if not found:
                last_seen[num] = 100  # Pas vu dans les 100 derniers
        
        # Trier par retard
        delayed_numbers = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
        
        print(f"🚨 TOP 10 - Numéros en retard :")
        for i, (num, delay) in enumerate(delayed_numbers[:10], 1):
            status = "🔴 TRÈS EN RETARD" if delay >= 100 else "🟡 EN RETARD" if delay >= 50 else "🟢 RÉCENT"
            print(f"   {i:2d}. Numéro {num:2d} : {delay:3d} tirages ({status})")
        
        # 3. GÉNÉRATION DE RECOMMANDATIONS
        print(f"\n💡 3. RECOMMANDATIONS DE JEU")
        print("-" * 30)
        
        # Stratégie HOT (chauds)
        hot_numbers = [num for num, count in most_common[:15]]
        
        # Stratégie COLD (retard)
        cold_numbers = [num for num, delay in delayed_numbers[:15]]
        
        # Stratégie BALANCED (équilibrée)
        # Prendre les numéros avec fréquence proche de la moyenne
        balanced_numbers = []
        for num, count in all_numbers.most_common():
            if abs(count - avg_frequency) <= avg_frequency * 0.1:  # ±10% de la moyenne
                balanced_numbers.append(num)
                if len(balanced_numbers) >= 15:
                    break
        
        # Grilles recommandées
        print(f"🎯 GRILLES RECOMMANDÉES (10 numéros par grille) :")
        print(f"\n   🔥 STRATÉGIE HOT (fréquents) :")
        print(f"      {hot_numbers[:10]}")
        
        print(f"\n   ❄️ STRATÉGIE COLD (retard) :")
        print(f"      {cold_numbers[:10]}")
        
        print(f"\n   ⚖️ STRATÉGIE BALANCED (équilibrée) :")
        print(f"      {balanced_numbers[:10]}")
        
        # Mix des stratégies
        mix_grid = hot_numbers[:4] + cold_numbers[:3] + balanced_numbers[:3]
        print(f"\n   🎭 STRATÉGIE MIX (combinée) :")
        print(f"      {mix_grid}")
        
        # 4. SAUVEGARDE DES RÉSULTATS
        print(f"\n💾 4. SAUVEGARDE DES RÉSULTATS")
        print("-" * 30)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les fréquences
        freq_data = []
        for num, count in all_numbers.most_common():
            pct = (count / total_occurrences) * 100
            freq_data.append({
                'numero': num,
                'frequence': count,
                'pourcentage': round(pct, 2),
                'ecart_moyenne': round(count - avg_frequency, 1)
            })
        
        freq_df = pd.DataFrame(freq_data)
        freq_path = OUTPUT_DIR / f"frequences_keno_{timestamp}.csv"
        freq_df.to_csv(freq_path, index=False, encoding='utf-8')
        
        # Sauvegarder les retards
        delay_data = []
        for num, delay in delayed_numbers:
            delay_data.append({
                'numero': num,
                'retard_tirages': delay,
                'statut': 'TRÈS EN RETARD' if delay >= 100 else 'EN RETARD' if delay >= 50 else 'RÉCENT'
            })
        
        delay_df = pd.DataFrame(delay_data)
        delay_path = OUTPUT_DIR / f"retards_keno_{timestamp}.csv"
        delay_df.to_csv(delay_path, index=False, encoding='utf-8')
        
        # Sauvegarder les recommandations
        rec_path = OUTPUT_DIR / f"recommandations_keno_{timestamp}.txt"
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("🎯 RECOMMANDATIONS KENO\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}\n")
            f.write(f"Basé sur {len(df)} tirages\n\n")
            
            f.write("🔥 STRATÉGIE HOT (numéros fréquents) :\n")
            f.write(f"   {hot_numbers[:10]}\n\n")
            
            f.write("❄️ STRATÉGIE COLD (numéros en retard) :\n")
            f.write(f"   {cold_numbers[:10]}\n\n")
            
            f.write("⚖️ STRATÉGIE BALANCED (équilibrée) :\n")
            f.write(f"   {balanced_numbers[:10]}\n\n")
            
            f.write("🎭 STRATÉGIE MIX (combinée) :\n")
            f.write(f"   {mix_grid}\n\n")
            
            f.write("📝 CONSEILS :\n")
            f.write("- HOT : Mise sur la continuité des tendances\n")
            f.write("- COLD : Mise sur le retour à l'équilibre\n")
            f.write("- BALANCED : Mise sur les moyennes statistiques\n")
            f.write("- MIX : Combinaison des trois approches\n")
        
        print(f"📄 Fichiers générés :")
        print(f"   - {freq_path.name}")
        print(f"   - {delay_path.name}")
        print(f"   - {rec_path.name}")
        
        print(f"\n✅ ANALYSE TERMINÉE AVEC SUCCÈS !")
        print(f"📁 Résultats dans : {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_keno_data()
