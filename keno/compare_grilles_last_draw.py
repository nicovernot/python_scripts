import pandas as pd
from pathlib import Path

# Chemins des fichiers
grilles_path = Path("keno_output/grilles_keno.csv")
consolidated_path = Path("keno/keno_data/keno_consolidated.csv")

# Chargement des grilles
df_grilles = pd.read_csv(grilles_path)
# Chaque grille est une liste de nombres (soit dans une colonne 'grille', soit sous forme de colonnes numero_1, numero_2, ...)
if 'grille' in df_grilles.columns:
    grilles = df_grilles['grille'].apply(lambda x: [int(n) for n in str(x).strip("[]").replace("'", "").split(",")])
else:
    # Colonnes numero_1, numero_2, ...
    num_cols = [col for col in df_grilles.columns if col.startswith('numero_')]
    grilles = df_grilles[num_cols].values.tolist()

# Chargement du fichier consolidé et récupération du dernier tirage (le plus récent)
df_conso = pd.read_csv(consolidated_path)
df_conso['date'] = pd.to_datetime(df_conso['date'], errors='coerce')
df_conso = df_conso.sort_values('date')
last_draw = df_conso.iloc[-1]
boule_cols = [col for col in df_conso.columns if col.startswith('b')]
last_draw_nums = set(last_draw[boule_cols])

print(f"🎯 Dernier tirage du {last_draw['date'].date()} : {', '.join(str(n) for n in last_draw_nums)}\n")

# Comparaison grille par grille, triées par nombre de numéros en commun (du plus grand au plus petit)
resultats = []
for idx, grille in enumerate(grilles, 1):
    grille_set = set(grille)
    common = sorted(grille_set & last_draw_nums)
    resultats.append((idx, grille, common, len(common)))

# Tri par nombre de numéros en commun (décroissant)
resultats = sorted(resultats, key=lambda x: x[3], reverse=True)

for idx, grille, common, nb_common in resultats:
    print(f"Grille {idx:02d} : {', '.join(str(n) for n in grille)}")
    if nb_common > 0:
        print(f"   ➡️  Numéros en commun : {nb_common} ({', '.join(str(n) for n in common)})\n")
    else:
        print(f"   ➡️  Numéros en commun : 0 (aucun)\n")