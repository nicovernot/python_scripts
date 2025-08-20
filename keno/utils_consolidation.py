#!/usr/bin/env python3
"""
Utilitaires pour la gestion des données Keno consolidées
"""

import os
from pathlib import Path
import pandas as pd

def get_consolidated_file_path():
    """Retourne le chemin du fichier consolidé"""
    return Path("keno/keno_data/keno_consolidated.csv")

def is_consolidated_available():
    """Vérifie si le fichier consolidé est disponible et récent"""
    consolidated_file = get_consolidated_file_path()
    
    if not consolidated_file.exists():
        return False
    
    # Vérifier si le fichier est plus récent que les fichiers individuels
    consolidated_time = consolidated_file.stat().st_mtime
    data_dir = Path("keno/keno_data")
    
    for csv_file in data_dir.glob("*.csv"):
        if not csv_file.name.startswith("keno_consolidated"):
            if csv_file.stat().st_mtime > consolidated_time:
                return False  # Un fichier individuel est plus récent
    
    return True

def load_keno_data(prefer_consolidated=True):
    """
    Charge les données Keno, en préférant le fichier consolidé si disponible
    
    Args:
        prefer_consolidated (bool): Préférer le fichier consolidé si disponible
        
    Returns:
        pandas.DataFrame: Les données Keno chargées
        str: Source des données ("consolidated" ou "individual")
    """
    
    if prefer_consolidated and is_consolidated_available():
        consolidated_file = get_consolidated_file_path()
        print(f"📊 Chargement depuis le fichier consolidé: {consolidated_file}")
        df = pd.read_csv(consolidated_file)
        return df, "consolidated"
    
    else:
        # Charger depuis les fichiers individuels
        print("📊 Chargement depuis les fichiers individuels...")
        data_dir = Path("keno/keno_data")
        all_files = []
        
        for csv_file in sorted(data_dir.glob("*.csv")):
            if not csv_file.name.startswith(("keno_consolidated", "keno_backup")):
                all_files.append(csv_file)
        
        if not all_files:
            raise FileNotFoundError("Aucun fichier de données trouvé")
        
        # Lecture et consolidation
        dataframes = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                # Vérification de la structure
                expected_columns = ['date', 'numero_tirage', 'b1', 'b2', 'b3', 'b4', 'b5', 
                                  'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
                                  'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20']
                
                if list(df.columns) == expected_columns:
                    dataframes.append(df)
                    
            except Exception as e:
                print(f"⚠️  Erreur lecture {file.name}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("Aucun fichier valide trouvé")
        
        # Consolidation en mémoire
        df = pd.concat(dataframes, ignore_index=True)
        df = df.drop_duplicates(subset=['numero_tirage'], keep='first')
        df = df.sort_values('numero_tirage')
        
        print(f"📊 {len(dataframes)} fichiers chargés, {len(df)} tirages uniques")
        return df, "individual"

def recommend_consolidation():
    """Recommande la consolidation si nécessaire"""
    if not is_consolidated_available():
        print("\n💡 RECOMMANDATION:")
        print("   Pour améliorer les performances, consolidez les données:")
        print("   python keno_cli.py assemble")
        print()

if __name__ == "__main__":
    # Test des fonctions
    print("🧪 Test des utilitaires de consolidation")
    print("=" * 50)
    
    consolidated_path = get_consolidated_file_path()
    print(f"📁 Chemin fichier consolidé: {consolidated_path}")
    print(f"🔍 Fichier consolidé disponible: {is_consolidated_available()}")
    
    try:
        df, source = load_keno_data()
        print(f"✅ Données chargées depuis: {source}")
        print(f"📊 Nombre de tirages: {len(df)}")
        print(f"📅 Période: {df['date'].min()} → {df['date'].max()}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        recommend_consolidation()
