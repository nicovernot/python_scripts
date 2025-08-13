# convert_to_parquet.py
import duckdb
from pathlib import Path

# Chemin vers votre fichier CSV original
csv_path = Path('~/Téléchargements/loto_201911.csv').expanduser()
# Chemin où vous voulez sauvegarder le nouveau fichier Parquet
parquet_path = csv_path.with_suffix('.parquet')

# Utiliser DuckDB pour la conversion (très rapide)
con = duckdb.connect()

print(f"Conversion de '{csv_path}' en '{parquet_path}'...")

# La commande COPY est la plus efficace
con.execute(f"COPY (SELECT * FROM read_csv_auto('{str(csv_path)}')) TO '{str(parquet_path)}' (FORMAT PARQUET);")

con.close()

print("Conversion terminée avec succès !")