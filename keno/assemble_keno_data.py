#!/usr/bin/env python3
"""
Assembleur de données Keno - Consolide tous les fichiers CSV en un seul
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def assemble_keno_data(auto_cleanup=True):
    """Assemble tous les fichiers CSV de keno_data en un seul fichier consolidé
    
    Args:
        auto_cleanup (bool): Si True, nettoie automatiquement les fichiers inutiles après assemblage
    """
    
    # Répertoires
    data_dir = Path("keno/keno_data")
    output_file = data_dir / "keno_consolidated.csv"
    backup_file = data_dir / f"keno_consolidated_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print("🔄 ASSEMBLAGE DES DONNÉES KENO")
    print("=" * 50)
    
    # Sauvegarde de l'ancien fichier consolidé s'il existe
    if output_file.exists():
        print(f"📋 Sauvegarde de l'ancien fichier: {backup_file.name}")
        os.rename(output_file, backup_file)
    
    # Recherche de tous les fichiers CSV (sauf les consolidés et backups)
    csv_files = []
    for csv_file in data_dir.glob("*.csv"):
        if not csv_file.name.startswith(("keno_consolidated", "keno_backup")):
            csv_files.append(csv_file)
    
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé dans keno_data/")
        return False
    
    csv_files.sort()  # Tri par nom (chronologique)
    
    print(f"📁 Fichiers trouvés: {len(csv_files)}")
    for f in csv_files:
        print(f"   - {f.name}")
    
    # Lecture et assemblage (avec gestion des doublons)
    all_dataframes = []
    total_rows = 0
    
    for i, csv_file in enumerate(csv_files):
        print(f"📊 Traitement {i+1}/{len(csv_files)}: {csv_file.name}", end="")
        
        try:
            # Lecture du fichier
            df = pd.read_csv(csv_file)
            
            # Vérification de la structure
            expected_columns = ['date', 'numero_tirage', 'b1', 'b2', 'b3', 'b4', 'b5', 
                              'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 
                              'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20']
            
            if list(df.columns) != expected_columns:
                print(f" ⚠️  Structure différente, ignoré")
                continue
            
            all_dataframes.append(df)
            rows = len(df)
            total_rows += rows
            print(f" ✅ {rows} tirages")
            
        except Exception as e:
            print(f" ❌ Erreur: {str(e)}")
            continue
    
    # Consolidation et suppression des doublons
    print(f"\n🔄 Consolidation de {len(all_dataframes)} fichiers...")
    if all_dataframes:
        # Concaténation de tous les DataFrames
        consolidated_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Suppression des doublons basée sur numero_tirage
        print(f"📊 Avant dédoublonnage: {len(consolidated_df)} tirages")
        consolidated_df = consolidated_df.drop_duplicates(subset=['numero_tirage'], keep='first')
        print(f"📊 Après dédoublonnage: {len(consolidated_df)} tirages")
        
        # Tri par numero_tirage
        consolidated_df = consolidated_df.sort_values('numero_tirage')
        
        # Sauvegarde
        consolidated_df.to_csv(output_file, index=False)
        print(f"💾 Fichier sauvegardé: {output_file}")
    else:
        print("❌ Aucune donnée à consolider")
        return False

    print("\n📈 RÉSUMÉ")
    print("=" * 50)
    print(f"📁 Fichiers traités: {len(all_dataframes)}")
    print(f"📊 Total tirages (avant dédoublonnage): {total_rows}")
    print(f"💾 Fichier consolidé: {output_file}")
    print(f"📦 Taille: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Vérification finale
    if output_file.exists():
        final_df = pd.read_csv(output_file)
        print(f"✅ Vérification: {len(final_df)} lignes dans le fichier final")
        
        # Affichage des statistiques
        print(f"📅 Période: {final_df['date'].min()} → {final_df['date'].max()}")
        print(f"🎲 Numéros de tirage: {final_df['numero_tirage'].min()} → {final_df['numero_tirage'].max()}")
        
        # Vérification des doublons
        duplicates = final_df.duplicated(subset=['numero_tirage']).sum()
        if duplicates > 0:
            print(f"⚠️  {duplicates} doublons restants")
        else:
            print("✅ Aucun doublon dans le fichier final")
        
        # Nettoyage automatique si demandé
        if auto_cleanup:
            cleanup_result = cleanup_unnecessary_files(data_dir, csv_files, keep_latest_backup=True)
            if cleanup_result:
                print("\n🧹 NETTOYAGE AUTOMATIQUE TERMINÉ")
        
        return True
    else:
        print("❌ Échec de la création du fichier consolidé")
        return False

def cleanup_unnecessary_files(data_dir, processed_files, keep_latest_backup=True):
    """Nettoie les fichiers CSV inutiles après consolidation
    
    Args:
        data_dir (Path): Répertoire des données
        processed_files (list): Liste des fichiers qui ont été traités
        keep_latest_backup (bool): Garder le backup le plus récent
    """
    print("\n🧹 NETTOYAGE DES FICHIERS INUTILES")
    print("=" * 50)
    
    files_to_delete = []
    files_to_keep = []
    
    # 1. Garder le fichier consolidé principal
    consolidated_main = data_dir / "keno_consolidated.csv"
    if consolidated_main.exists():
        files_to_keep.append(consolidated_main.name)
    
    # 2. Garder uniquement le backup le plus récent
    backup_files = list(data_dir.glob("keno_consolidated_backup_*.csv"))
    if backup_files and keep_latest_backup:
        latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
        files_to_keep.append(latest_backup.name)
        
        # Marquer les autres backups pour suppression
        for backup in backup_files:
            if backup != latest_backup:
                files_to_delete.append(backup)
    else:
        files_to_delete.extend(backup_files)
    
    # 3. Supprimer tous les fichiers individuels traités (ils sont maintenant dans le consolidé)
    for csv_file in processed_files:
        if csv_file.exists() and csv_file.name not in files_to_keep:
            files_to_delete.append(csv_file)
    
    # 4. Nettoyer les fichiers temporaires et anciens
    temp_patterns = [
        "*_20*_*.csv",  # Fichiers avec timestamp
        "keno_backup_*.csv",
        "*_extracted.csv",
        "*_temp.csv"
    ]
    
    for pattern in temp_patterns:
        temp_files = data_dir.glob(pattern)
        for temp_file in temp_files:
            if temp_file.name not in files_to_keep and temp_file not in files_to_delete:
                files_to_delete.append(temp_file)
    
    # Exécution du nettoyage
    deleted_count = 0
    deleted_size = 0
    
    print(f"📋 Fichiers à conserver: {len(files_to_keep)}")
    for file_to_keep in files_to_keep:
        print(f"   ✅ {file_to_keep}")
    
    print(f"\n🗑️  Fichiers à supprimer: {len(files_to_delete)}")
    
    for file_to_delete in files_to_delete:
        try:
            file_size = file_to_delete.stat().st_size
            print(f"   🗑️  {file_to_delete.name} ({file_size/1024:.1f} KB)")
            file_to_delete.unlink()
            deleted_count += 1
            deleted_size += file_size
        except Exception as e:
            print(f"   ❌ Erreur suppression {file_to_delete.name}: {e}")
    
    print(f"\n📊 RÉSUMÉ DU NETTOYAGE")
    print(f"   🗑️  Fichiers supprimés: {deleted_count}")
    print(f"   💾 Espace libéré: {deleted_size/1024:.1f} KB")
    print(f"   📁 Fichiers conservés: {len(files_to_keep)}")
    
    return deleted_count > 0

def get_consolidated_file_path():
    """Retourne le chemin du fichier consolidé"""
    return Path("keno/keno_data/keno_consolidated.csv")

def is_consolidated_file_available():
    """Vérifie si le fichier consolidé existe et est récent"""
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

if __name__ == "__main__":
    # Assemblage avec nettoyage automatique par défaut
    success = assemble_keno_data(auto_cleanup=True)
    sys.exit(0 if success else 1)
