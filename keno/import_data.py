#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📥 IMPORTATEUR DE FICHIERS KENO
==============================

Script pour importer facilement des fichiers Keno CSV depuis les Téléchargements
ou depuis une URL de décompression.

Usage:
    python keno/import_data.py                    # Cherche dans ~/Téléchargements
    python keno/import_data.py fichier.csv        # Importe un fichier spécifique
"""

import shutil
import zipfile
import requests
from pathlib import Path
import sys

def import_keno_data():
    """Importe les données Keno depuis différentes sources"""
    print("📥 IMPORTATEUR DE DONNÉES KENO")
    print("=" * 40)
    
    KENO_DATA_DIR = Path(__file__).parent / "keno_data"
    KENO_DATA_DIR.mkdir(exist_ok=True)
    
    downloads_dir = Path.home() / "Téléchargements"
    
    # Si un fichier est spécifié en argument
    if len(sys.argv) > 1:
        source_file = Path(sys.argv[1])
        if not source_file.is_absolute():
            # Chercher dans les téléchargements
            source_file = downloads_dir / source_file
        
        if source_file.exists():
            import_single_file(source_file, KENO_DATA_DIR)
        else:
            print(f"❌ Fichier non trouvé : {source_file}")
        return
    
    # Chercher tous les fichiers Keno dans Téléchargements
    print(f"🔍 Recherche dans : {downloads_dir}")
    
    patterns = ["keno*.csv", "KENO*.csv", "*keno*.csv", "*KENO*.csv"]
    found_files = []
    
    for pattern in patterns:
        found_files.extend(downloads_dir.glob(pattern))
    
    # Chercher aussi les fichiers ZIP
    zip_patterns = ["keno*.zip", "KENO*.zip", "*keno*.zip", "*KENO*.zip"]
    for pattern in zip_patterns:
        found_files.extend(downloads_dir.glob(pattern))
    
    if not found_files:
        print("❌ Aucun fichier Keno trouvé dans les Téléchargements")
        print("💡 Fichiers acceptés : keno*.csv, keno*.zip")
        return
    
    print(f"📂 {len(found_files)} fichiers trouvés :")
    for file_path in found_files:
        print(f"   - {file_path.name}")
    
    # Importer chaque fichier
    for file_path in found_files:
        import_single_file(file_path, KENO_DATA_DIR)

def import_single_file(source_path, dest_dir):
    """Importe un fichier unique"""
    source_path = Path(source_path)
    
    print(f"\n📁 Import de : {source_path.name}")
    
    try:
        if source_path.suffix.lower() == '.zip':
            # Extraire le ZIP
            extract_dir = dest_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(source_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                print(f"📦 ZIP extrait dans : {extract_dir}")
                
                # Lister les fichiers extraits
                extracted_files = list(extract_dir.glob("*.csv"))
                for csv_file in extracted_files:
                    target_path = dest_dir / csv_file.name
                    shutil.copy2(csv_file, target_path)
                    print(f"   ✅ {csv_file.name} → {target_path.name}")
        
        elif source_path.suffix.lower() == '.csv':
            # Copier le CSV directement
            target_path = dest_dir / source_path.name
            
            if target_path.exists():
                # Vérifier si c'est le même fichier
                if source_path.stat().st_size == target_path.stat().st_size:
                    print(f"   ⚠️ Fichier déjà existant (même taille)")
                    return
                else:
                    # Renommer avec un timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
                    target_path = dest_dir / new_name
            
            shutil.copy2(source_path, target_path)
            print(f"   ✅ Copié vers : {target_path.name}")
        
        else:
            print(f"   ❌ Format non supporté : {source_path.suffix}")
    
    except Exception as e:
        print(f"   ❌ Erreur : {e}")

def show_summary():
    """Affiche un résumé des fichiers disponibles"""
    KENO_DATA_DIR = Path(__file__).parent / "keno_data"
    
    csv_files = list(KENO_DATA_DIR.glob("*.csv"))
    
    print(f"\n📊 RÉSUMÉ DES DONNÉES KENO")
    print("=" * 30)
    print(f"📁 Répertoire : {KENO_DATA_DIR}")
    print(f"📄 Fichiers CSV : {len(csv_files)}")
    
    if csv_files:
        total_size = sum(f.stat().st_size for f in csv_files)
        print(f"💾 Taille totale : {total_size / (1024*1024):.1f} MB")
        
        print(f"\n📋 Liste des fichiers :")
        for file_path in sorted(csv_files):
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   - {file_path.name} ({size_mb:.1f} MB)")
    
    print(f"\n💡 Pour analyser les données :")
    print(f"   python keno/analyse_keno_final.py")

if __name__ == "__main__":
    import_keno_data()
    show_summary()
