#!/usr/bin/env python3
"""
Script de conversion CSV vers Parquet pour les données Keno
"""

import pandas as pd
import os
from pathlib import Path

def convert_keno_csv_to_parquet():
    """Convertit les fichiers CSV Keno en format Parquet pour optimiser les performances"""
    
    # Chemins des fichiers - correction du chemin relatif
    base_dir = Path(__file__).parent  # Répertoire du script
    csv_path = base_dir / "keno_data/keno_202010.csv"
    parquet_path = base_dir / "keno_data/keno_202010.parquet"
    
    print(f"🔄 Conversion du fichier CSV Keno en Parquet...")
    print(f"   Source: {csv_path}")
    print(f"   Destination: {parquet_path}")
    
    # Vérifier si le fichier source existe
    if not csv_path.exists():
        print(f"❌ Fichier source introuvable: {csv_path}")
        # Chercher d'autres fichiers CSV dans le répertoire
        keno_data_dir = base_dir / "keno_data"
        if keno_data_dir.exists():
            csv_files = list(keno_data_dir.glob("*.csv"))
            if csv_files:
                print(f"📁 Fichiers CSV disponibles:")
                for f in csv_files:
                    print(f"   - {f.name}")
        return None
    
    try:
        # Lecture du CSV avec le bon séparateur
        df = pd.read_csv(csv_path, sep=';')
        
        # Nettoyage des données
        print(f"📊 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Suppression des colonnes vides ou inutiles
        df = df.dropna(axis=1, how='all')  # Supprime les colonnes entièrement vides
        
        # Conversion de la date
        df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
        
        # Conversion des colonnes numériques
        boule_columns = [f'boule{i}' for i in range(1, 21)]  # 20 boules pour le Keno
        for col in boule_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Conversion du multiplicateur et numero_jokerplus
        if 'multiplicateur' in df.columns:
            df['multiplicateur'] = pd.to_numeric(df['multiplicateur'], errors='coerce')
        if 'numero_jokerplus' in df.columns:
            df['numero_jokerplus'] = pd.to_numeric(df['numero_jokerplus'], errors='coerce')
        
        # Suppression des lignes avec des valeurs manquantes critiques
        df = df.dropna(subset=['date_de_tirage'])
        
        print(f"✅ Données nettoyées: {len(df)} lignes conservées")
        
        # Sauvegarde en Parquet
        df.to_parquet(parquet_path, engine='auto', compression='snappy')
        
        # Vérification des tailles de fichiers
        csv_size = csv_path.stat().st_size / (1024*1024)  # MB
        parquet_size = parquet_path.stat().st_size / (1024*1024)  # MB
        
        print(f"📁 Tailles des fichiers:")
        print(f"   CSV: {csv_size:.2f} MB")
        print(f"   Parquet: {parquet_size:.2f} MB")
        print(f"   Compression: {((csv_size - parquet_size) / csv_size * 100):.1f}%")
        
        # Test de lecture
        df_test = pd.read_parquet(parquet_path)
        print(f"✅ Validation: {len(df_test)} lignes lues depuis le fichier Parquet")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion: {e}")
        return None

if __name__ == "__main__":
    convert_keno_csv_to_parquet()
