import pandas as pd
from pathlib import Path

# Chemins des fichiers
top30ml_path = Path("keno_output/keno_top30_ml.csv")
consolidated_path = Path("keno/keno_data/keno_consolidated.csv")

# Chargement du TOP 30 ML
df_top30 = pd.read_csv(top30ml_path)
if 'Numero' in df_top30.columns:
    top30_nums = set(df_top30['Numero'])
elif 'numero' in df_top30.columns:
    top30_nums = set(df_top30['numero'])
else:
    raise ValueError("Colonne 'Numero' ou 'numero' non trouvée dans le TOP 30 ML.")

# Chargement du fichier consolidé et récupération du dernier tirage (le plus récent)
df_conso = pd.read_csv(consolidated_path)
df_conso['date'] = pd.to_datetime(df_conso['date'], errors='coerce')
df_conso = df_conso.sort_values('date')
last_draw = df_conso.iloc[-1]
boule_cols = [col for col in df_conso.columns if col.startswith('b')]
last_draw_nums = set(last_draw[boule_cols])

# Comparaison
common_nums = sorted(top30_nums & last_draw_nums)
print(f"🎯 Numéros en commun entre le TOP 30 ML et le dernier tirage ({last_draw['date'].date()}):")
if common_nums:
    print(", ".join(str(n) for n in common_nums))
else:
    print("Aucun numéro en commun.")