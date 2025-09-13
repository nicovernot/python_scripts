# convert_to_parquet.py
import duckdb
from pathlib import Path
import os

# Configuration des chemins
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'loto_data'
DATA_DIR.mkdir(exist_ok=True)

# Chercher le fichier CSV dans le dossier loto/data
csv_files = list(DATA_DIR.glob('*.csv'))
if not csv_files:
    print(f"❌ Aucun fichier CSV trouvé dans {DATA_DIR}")
    print("Veuillez placer un fichier CSV de données Loto dans ce dossier.")
    exit(1)

# Prendre le premier fichier CSV trouvé
csv_path = csv_files[0]
# Sauvegarder le parquet dans le même dossier
parquet_path = DATA_DIR / csv_path.with_suffix('.parquet').name

print(f"📂 Dossier de données : {DATA_DIR}")
print(f"📥 Fichier CSV source : {csv_path.name}")
print(f"📤 Fichier Parquet cible : {parquet_path.name}")

# Utiliser DuckDB pour la conversion (très rapide)
con = duckdb.connect()

print(f"\n🔄 Conversion en cours...")
try:
    # La commande COPY est la plus efficace
    con.execute(f"COPY (SELECT * FROM read_csv_auto('{str(csv_path)}')) TO '{str(parquet_path)}' (FORMAT PARQUET);")
    
    # Vérifier la taille des fichiers
    csv_size = csv_path.stat().st_size / (1024*1024)
    parquet_size = parquet_path.stat().st_size / (1024*1024)
    compression_ratio = (1 - parquet_size/csv_size) * 100
    
    print(f"✅ Conversion terminée avec succès !")
    print(f"📊 Taille CSV : {csv_size:.1f} MB")
    print(f"📦 Taille Parquet : {parquet_size:.1f} MB")
    print(f"🗜️ Compression : {compression_ratio:.1f}%")
    
except Exception as e:
    print(f"❌ Erreur lors de la conversion : {e}")
finally:
    con.close()