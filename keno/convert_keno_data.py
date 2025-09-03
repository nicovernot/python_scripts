#!/usr/bin/env python3
"""
Script de conversion CSV vers Parquet pour les données Keno
"""

import pandas as pd
from datetime import datetime

source_csv = "/home/nico/projets/python_scripts/keno/keno_data/keno_consolidated.csv"
date_str = datetime.now().strftime("%Y%m%d")
destination_parquet = f"/home/nico/projets/python_scripts/keno/keno_data/keno_consolidated_{date_str}.parquet"

print("🔄 Conversion du fichier CSV Keno en Parquet...")
print(f"   Source: {source_csv}")
print(f"   Destination: {destination_parquet}")

try:
    df = pd.read_csv(source_csv)
    print(f"📊 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    # Si tu veux filtrer ou manipuler la colonne 'date', utilise bien 'date' :
    # Exemple : df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(destination_parquet, index=False)
    print("✅ Conversion terminée !")
except Exception as e:
    print(f"❌ Erreur lors de la conversion: {e}")
