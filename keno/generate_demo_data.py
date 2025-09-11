#!/usr/bin/env python3
"""
Générateur de données de démonstration pour les tirages Keno
Crée des données synthétiques réalistes pour tester le système d'apprentissage
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_keno_demo_data(num_draws=1000, output_file="keno_demo_data.parquet"):
    """
    Génère des données de démonstration Keno avec des patterns réalistes
    
    Args:
        num_draws: Nombre de tirages à générer
        output_file: Fichier de sortie
    """
    print(f"🎲 Génération de {num_draws} tirages Keno de démonstration...")
    
    # Paramètres pour des tirages réalistes
    total_numbers = 70
    numbers_per_draw = 20
    
    # Créer des biais réalistes pour certains numéros
    # Certains numéros sont légèrement plus/moins fréquents
    weights = np.ones(total_numbers)
    
    # Ajouter des biais subtils (±15% max)
    for i in range(total_numbers):
        bias = random.uniform(0.85, 1.15)
        weights[i] = bias
    
    # Normaliser les poids
    weights = weights / weights.sum() * total_numbers
    
    data = []
    start_date = datetime.now() - timedelta(days=num_draws)
    
    for i in range(num_draws):
        # Date de tirage (un par jour)
        draw_date = start_date + timedelta(days=i)
        
        # Sélectionner 20 numéros avec les poids biaisés
        selected_numbers = np.random.choice(
            range(1, total_numbers + 1),
            size=numbers_per_draw,
            replace=False,
            p=weights / weights.sum()
        )
        
        # Trier les numéros
        selected_numbers = sorted(selected_numbers)
        
        # Créer l'enregistrement
        record = {
            'date_de_tirage': draw_date,
            'numero_tirage': i + 1
        }
        
        # Ajouter les boules (colonnes boule1 à boule20)
        for j, num in enumerate(selected_numbers, 1):
            record[f'boule{j}'] = num
        
        data.append(record)
    
    # Créer le DataFrame
    df = pd.DataFrame(data)
    
    # Sauvegarder en Parquet
    output_path = Path(__file__).parent / "keno_data"
    output_path.mkdir(exist_ok=True)
    
    file_path = output_path / output_file
    df.to_parquet(file_path, index=False)
    
    print(f"✅ {len(df)} tirages sauvegardés dans {file_path}")
    print(f"📊 Période: {df['date_de_tirage'].min()} à {df['date_de_tirage'].max()}")
    
    # Afficher quelques statistiques
    print("\n📈 Statistiques des données générées:")
    all_numbers = []
    for _, row in df.iterrows():
        for i in range(1, 21):
            all_numbers.append(row[f'boule{i}'])
    
    unique_numbers, counts = np.unique(all_numbers, return_counts=True)
    most_frequent = unique_numbers[np.argmax(counts)]
    least_frequent = unique_numbers[np.argmin(counts)]
    
    print(f"   • Numéro le plus fréquent: {most_frequent} ({max(counts)} fois)")
    print(f"   • Numéro le moins fréquent: {least_frequent} ({min(counts)} fois)")
    print(f"   • Fréquence moyenne: {np.mean(counts):.1f} ± {np.std(counts):.1f}")
    
    return file_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Générateur de données Keno de démonstration")
    parser.add_argument("--draws", type=int, default=1000, help="Nombre de tirages à générer")
    parser.add_argument("--output", type=str, default="keno_demo_data.parquet", help="Nom du fichier de sortie")
    
    args = parser.parse_args()
    
    generate_keno_demo_data(args.draws, args.output)