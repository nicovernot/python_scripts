#!/usr/bin/env python3
"""
Utilitaire de conversion des anciens fichiers Keno vers le nouveau format unifié
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import shutil
from datetime import datetime

class KenoFormatConverter:
    """Convertit les anciens formats vers le format unifié"""
    
    def __init__(self, data_dir="keno/keno_data"):
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backup_old_format"
        
    def detect_format(self, csv_path):
        """Détecte le format d'un fichier CSV"""
        try:
            # Test nouveau format (b1, b2, ...)
            df = pd.read_csv(csv_path, nrows=3)
            if all(f'b{i}' in df.columns for i in range(1, 21)):
                return 'new_unified'
            
            # Test ancien format avec ; 
            df = pd.read_csv(csv_path, nrows=3, sep=';')
            if all(f'boule{i}' in df.columns for i in range(1, 21)):
                return 'old_semicolon'
            
            # Test ancien format avec ,
            df = pd.read_csv(csv_path, nrows=3, sep=',')
            if all(f'boule{i}' in df.columns for i in range(1, 21)):
                return 'old_comma'
            
            return 'unknown'
            
        except Exception:
            return 'error'
    
    def convert_old_to_new(self, input_path, output_path=None):
        """Convertit un fichier ancien format vers le nouveau"""
        format_type = self.detect_format(input_path)
        
        if format_type == 'new_unified':
            print(f"✅ {input_path.name} déjà au bon format")
            return True
        
        if format_type == 'unknown' or format_type == 'error':
            print(f"❌ Format non reconnu pour {input_path.name}")
            return False
        
        try:
            # Lecture selon le format détecté
            if format_type == 'old_semicolon':
                df = pd.read_csv(input_path, sep=';')
            else:
                df = pd.read_csv(input_path, sep=',')
            
            # Conversion vers le nouveau format
            new_df = pd.DataFrame()
            
            # Colonnes essentielles
            if 'date_de_tirage' in df.columns:
                new_df['date'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                new_df['date'] = df['date']
            else:
                print(f"⚠️ Pas de colonne date trouvée dans {input_path.name}")
                return False
            
            if 'annee_numero_de_tirage' in df.columns:
                new_df['numero_tirage'] = df['annee_numero_de_tirage']
            elif 'numero_tirage' in df.columns:
                new_df['numero_tirage'] = df['numero_tirage']
            else:
                # Générer des numéros séquentiels
                new_df['numero_tirage'] = range(1, len(df) + 1)
            
            # Conversion des boules
            for i in range(1, 21):
                old_col = f'boule{i}'
                new_col = f'b{i}'
                
                if old_col in df.columns:
                    new_df[new_col] = df[old_col].astype('Int64')
                else:
                    print(f"⚠️ Colonne {old_col} manquante")
                    return False
            
            # Sauvegarde
            if output_path is None:
                output_path = input_path
            
            new_df.to_csv(output_path, index=False)
            print(f"✅ Converti: {input_path.name} → format unifié")
            return True
            
        except Exception as e:
            print(f"❌ Erreur conversion {input_path.name}: {e}")
            return False
    
    def backup_originals(self):
        """Sauvegarde les fichiers originaux"""
        self.backup_dir.mkdir(exist_ok=True)
        
        original_files = list(self.data_dir.glob("*.csv"))
        if not original_files:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in original_files:
            if file_path.parent == self.backup_dir:
                continue
                
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(file_path, backup_path)
        
        print(f"📁 {len(original_files)} fichiers sauvegardés dans {self.backup_dir}")
    
    def convert_all(self, backup=True):
        """Convertit tous les fichiers du répertoire"""
        print("🔄 CONVERSION DES FICHIERS KENO")
        print("=" * 50)
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            print("❌ Aucun fichier CSV trouvé")
            return False
        
        if backup:
            print("📁 Sauvegarde des originaux...")
            self.backup_originals()
        
        success_count = 0
        
        for csv_file in csv_files:
            if csv_file.parent == self.backup_dir:
                continue
                
            format_type = self.detect_format(csv_file)
            print(f"\n📄 {csv_file.name} - Format: {format_type}")
            
            if self.convert_old_to_new(csv_file):
                success_count += 1
        
        print(f"\n✅ Conversion terminée: {success_count}/{len(csv_files)} fichiers convertis")
        return success_count > 0
    
    def verify_conversion(self):
        """Vérifie que tous les fichiers sont au bon format"""
        print("\n🔍 VÉRIFICATION DES FORMATS")
        print("=" * 30)
        
        csv_files = list(self.data_dir.glob("*.csv"))
        correct_format = 0
        
        for csv_file in csv_files:
            if csv_file.parent == self.backup_dir:
                continue
                
            format_type = self.detect_format(csv_file)
            status = "✅" if format_type == 'new_unified' else "❌"
            print(f"{status} {csv_file.name}: {format_type}")
            
            if format_type == 'new_unified':
                correct_format += 1
        
        print(f"\n📊 Résultat: {correct_format}/{len(csv_files)} fichiers au format unifié")
        return correct_format == len(csv_files)

def main():
    """Point d'entrée principal"""
    converter = KenoFormatConverter()
    
    # Conversion
    if converter.convert_all(backup=True):
        # Vérification
        if converter.verify_conversion():
            print("\n🎉 Tous les fichiers sont maintenant au format unifié !")
        else:
            print("\n⚠️ Certains fichiers n'ont pas pu être convertis")
    else:
        print("\n❌ Échec de la conversion")

if __name__ == "__main__":
    main()
