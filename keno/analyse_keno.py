#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ANALYSE DES DONNÉES KENO
===========================

Script d'analyse des résultats du Keno pour générer des statistiques
et des recommandations de numéros à jouer.

Usage:
    python keno/analyse_keno.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
KENO_DATA_DIR = Path(__file__).parent / "keno_data"
OUTPUT_DIR = Path(__file__).parent / "keno_analyse"

def setup_directories():
    """Créer les répertoires de sortie"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def load_keno_data():
    """Charger les données Keno depuis les fichiers CSV"""
    csv_files = list(KENO_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé dans keno_data/")
        return None
    
    print(f"📂 Fichiers trouvés : {len(csv_files)}")
    
    # Charger le fichier le plus récent ou le plus volumineux
    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
    
    print(f"📊 Chargement de : {largest_file.name}")
    
    try:
        df = pd.read_csv(largest_file, delimiter=';')
        print(f"✅ {len(df)} tirages chargés")
        return df
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return None

def analyze_number_frequencies(df):
    """Analyser les fréquences des numéros"""
    print("\n📊 Analyse des fréquences des numéros...")
    
    # Extraire toutes les boules
    ball_columns = [f'boule{i}' for i in range(1, 21)]
    
    # Créer un dictionnaire pour compter les fréquences
    frequencies = {}
    
    for col in ball_columns:
        if col in df.columns:
            for num in df[col].dropna():
                if pd.notna(num):
                    frequencies[int(num)] = frequencies.get(int(num), 0) + 1
    
    # Convertir en DataFrame
    freq_df = pd.DataFrame(list(frequencies.items()), columns=['numero', 'frequence'])
    freq_df = freq_df.sort_values('frequence', ascending=False)
    freq_df['pourcentage'] = (freq_df['frequence'] / freq_df['frequence'].sum() * 100).round(2)
    
    print(f"📈 Top 10 des numéros les plus fréquents :")
    print(freq_df.head(10).to_string(index=False))
    
    print(f"\n📉 Top 10 des numéros les moins fréquents :")
    print(freq_df.tail(10).to_string(index=False))
    
    return freq_df

def analyze_delay_patterns(df):
    """Analyser les retards (numéros non sortis récemment)"""
    print("\n⏰ Analyse des retards...")
    
    ball_columns = [f'boule{i}' for i in range(1, 21)]
    
    # Obtenir les derniers tirages (100 plus récents)
    recent_draws = df.head(100)
    
    # Compter quand chaque numéro est apparu pour la dernière fois
    last_appearance = {}
    
    for i in range(1, 71):  # Numéros de 1 à 70 au Keno
        last_seen = None
        for idx, row in recent_draws.iterrows():
            for col in ball_columns:
                if col in row and pd.notna(row[col]) and int(row[col]) == i:
                    last_seen = idx
                    break
            if last_seen is not None:
                break
        
        if last_seen is not None:
            last_appearance[i] = last_seen
        else:
            last_appearance[i] = len(recent_draws)  # Pas vu dans les 100 derniers
    
    # Convertir en DataFrame
    delay_df = pd.DataFrame(list(last_appearance.items()), columns=['numero', 'retard'])
    delay_df = delay_df.sort_values('retard', ascending=False)
    
    print(f"⏳ Top 10 des numéros en retard :")
    print(delay_df.head(10).to_string(index=False))
    
    return delay_df

def analyze_number_pairs(df):
    """Analyser les associations de numéros"""
    print("\n🔗 Analyse des paires de numéros...")
    
    ball_columns = [f'boule{i}' for i in range(1, 21)]
    
    # Créer un dictionnaire pour compter les paires
    pairs = {}
    
    for _, row in df.iterrows():
        numbers = []
        for col in ball_columns:
            if col in row and pd.notna(row[col]):
                numbers.append(int(row[col]))
        
        # Générer toutes les paires possibles pour ce tirage
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                pair = tuple(sorted([numbers[i], numbers[j]]))
                pairs[pair] = pairs.get(pair, 0) + 1
    
    # Convertir en DataFrame et trier
    pairs_df = pd.DataFrame(list(pairs.items()), columns=['paire', 'frequence'])
    pairs_df = pairs_df.sort_values('frequence', ascending=False)
    
    print(f"🔗 Top 10 des paires les plus fréquentes :")
    for i in range(min(10, len(pairs_df))):
        pair, freq = pairs_df.iloc[i]
        print(f"   {pair[0]:2d} - {pair[1]:2d} : {freq} fois")
    
    return pairs_df

def generate_recommendations(freq_df, delay_df):
    """Générer des recommandations de jeu"""
    print("\n💡 Génération des recommandations...")
    
    # Stratégie 1 : Numéros les plus fréquents
    hot_numbers = freq_df.head(10)['numero'].tolist()
    
    # Stratégie 2 : Numéros en retard
    delayed_numbers = delay_df.head(10)['numero'].tolist()
    
    # Stratégie 3 : Mix équilibré (moyennes fréquences)
    mid_start = len(freq_df) // 4
    mid_end = 3 * len(freq_df) // 4
    balanced_numbers = freq_df.iloc[mid_start:mid_end].head(10)['numero'].tolist()
    
    recommendations = {
        'chauds': hot_numbers,
        'retard': delayed_numbers,
        'equilibre': balanced_numbers
    }
    
    print("🎯 Recommandations de numéros :")
    for strategy, numbers in recommendations.items():
        print(f"   {strategy.capitalize()} : {numbers}")
    
    return recommendations

def create_visualizations(freq_df, delay_df, plots_dir):
    """Créer des visualisations"""
    print("\n📈 Création des graphiques...")
    
    # Configuration matplotlib pour le français
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Graphique des fréquences
    plt.figure(figsize=(15, 8))
    
    # Trier par numéro pour l'affichage
    freq_sorted = freq_df.sort_values('numero')
    
    plt.bar(freq_sorted['numero'], freq_sorted['frequence'], color='steelblue', alpha=0.7)
    plt.title('Fréquence des Numéros au Keno', fontsize=16, fontweight='bold')
    plt.xlabel('Numéro')
    plt.ylabel('Fréquence d\'apparition')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 71, 5))
    
    # Ajouter une ligne de moyenne
    mean_freq = freq_sorted['frequence'].mean()
    plt.axhline(y=mean_freq, color='red', linestyle='--', alpha=0.7, label=f'Moyenne: {mean_freq:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'frequences_keno.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graphique des retards
    plt.figure(figsize=(15, 8))
    
    delay_sorted = delay_df.sort_values('numero')
    colors = ['red' if r > 50 else 'orange' if r > 25 else 'green' for r in delay_sorted['retard']]
    
    plt.bar(delay_sorted['numero'], delay_sorted['retard'], color=colors, alpha=0.7)
    plt.title('Retard des Numéros au Keno (depuis 100 derniers tirages)', fontsize=16, fontweight='bold')
    plt.xlabel('Numéro')
    plt.ylabel('Nombre de tirages depuis dernière apparition')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 71, 5))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'retards_keno.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap des fréquences (grille 7x10)
    plt.figure(figsize=(12, 8))
    
    # Créer une matrice 7x10 pour les numéros 1-70
    freq_matrix = np.zeros((7, 10))
    for _, row in freq_df.iterrows():
        num = int(row['numero'])
        if 1 <= num <= 70:
            i = (num - 1) // 10
            j = (num - 1) % 10
            freq_matrix[i, j] = row['frequence']
    
    # Créer les labels
    labels = np.arange(1, 71).reshape(7, 10)
    
    sns.heatmap(freq_matrix, annot=labels, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Fréquence'})
    plt.title('Heatmap des Fréquences - Keno', fontsize=16, fontweight='bold')
    plt.xlabel('Position')
    plt.ylabel('Ligne')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'heatmap_keno.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphiques sauvegardés dans : {plots_dir}")

def save_analysis_results(freq_df, delay_df, recommendations):
    """Sauvegarder les résultats d'analyse"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder les fréquences
    freq_path = OUTPUT_DIR / f"frequences_keno_{timestamp}.csv"
    freq_df.to_csv(freq_path, index=False, encoding='utf-8')
    
    # Sauvegarder les retards
    delay_path = OUTPUT_DIR / f"retards_keno_{timestamp}.csv"
    delay_df.to_csv(delay_path, index=False, encoding='utf-8')
    
    # Sauvegarder les recommandations
    rec_path = OUTPUT_DIR / f"recommandations_keno_{timestamp}.txt"
    with open(rec_path, 'w', encoding='utf-8') as f:
        f.write("🎯 RECOMMANDATIONS KENO\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}\n\n")
        
        for strategy, numbers in recommendations.items():
            f.write(f"Stratégie {strategy.upper()} :\n")
            f.write(f"Numéros : {', '.join(map(str, numbers))}\n\n")
        
        f.write("CONSEILS D'UTILISATION :\n")
        f.write("- Chauds : Numéros sortis le plus souvent\n")
        f.write("- Retard : Numéros qui n'apparaissent plus depuis longtemps\n")
        f.write("- Équilibre : Mix de numéros à fréquence moyenne\n")
    
    print(f"💾 Résultats sauvegardés :")
    print(f"   - Fréquences : {freq_path.name}")
    print(f"   - Retards : {delay_path.name}")
    print(f"   - Recommandations : {rec_path.name}")

def main():
    """Fonction principale"""
    print("🎯 ANALYSE DES DONNÉES KENO")
    print("=" * 40)
    
    # Créer les répertoires
    plots_dir = setup_directories()
    
    # Charger les données
    df = load_keno_data()
    if df is None:
        return
    
    # Analyser les fréquences
    freq_df = analyze_number_frequencies(df)
    
    # Analyser les retards
    delay_df = analyze_delay_patterns(df)
    
    # Analyser les paires
    pairs_df = analyze_number_pairs(df)
    
    # Générer les recommandations
    recommendations = generate_recommendations(freq_df, delay_df)
    
    # Créer les visualisations
    create_visualizations(freq_df, delay_df, plots_dir)
    
    # Sauvegarder les résultats
    save_analysis_results(freq_df, delay_df, recommendations)
    
    print(f"\n✅ Analyse terminée !")
    print(f"📁 Résultats dans : {OUTPUT_DIR}")
    print("=" * 40)

if __name__ == "__main__":
    main()
