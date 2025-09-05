#!/usr/bin/env python3
"""
Script de conversion CSV vers Parquet pour les données Keno
"""

import pandas as pd
from datetime import datetime
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
source_csv = os.path.join(base_dir, "keno_data", "keno_consolidated.csv")
date_str = datetime.now().strftime("%Y%m%d")
destination_parquet = os.path.join(base_dir, "keno_data", "keno_consolidated.parquet")

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
