#!/usr/bin/env python3
"""
Utilitaire de nettoyage des fichiers horodatés
==============================================

Nettoie automatiquement les anciens fichiers avec timestamps
pour éviter l'accumulation dans les répertoires de sortie.
"""

import glob
import os
from pathlib import Path

def clean_timestamped_files():
    """Nettoie les anciens fichiers avec horodatage."""
    
    base_path = Path(__file__).parent
    loto_output = base_path / "loto_output"
    keno_output = base_path / "keno_output"
    
    # Patterns des fichiers horodatés à supprimer
    patterns = [
        # Fichiers Loto horodatés
        str(loto_output / "grilles_loto_*_*.md"),
        
        # Fichiers Keno horodatés 
        str(keno_output / "analyse_keno_*_*.md"),
        str(keno_output / "erreur_keno_*_*.md"),
        
        # Autres fichiers horodatés possibles
        str(keno_output / "recommandations_keno_*_*.txt"),
    ]
    
    files_removed = 0
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                files_removed += 1
                print(f"🗑️ Supprimé: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"❌ Erreur suppression {file_path}: {e}")
    
    if files_removed > 0:
        print(f"🧹 {files_removed} anciens fichiers horodatés supprimés")
    else:
        print("✨ Aucun fichier horodaté à supprimer")

if __name__ == "__main__":
    clean_timestamped_files()
