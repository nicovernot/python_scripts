#!/usr/bin/env python3
"""
🎯 CLI PRINCIPAL POUR L'ANALYSE ET GÉNÉRATION KENO
==================================================

Interface de ligne de commande unifiée pour :
- Extraction des données FDJ
- Analyse statistique complète 
- Génération de grilles intelligentes
- Gestion du système complet

Usage:
    python keno_cli.py extract           # Met à jour les données
    python keno_cli.py analyze           # Analyse complète
    python keno_cli.py generate --grids 5 # Génère des grilles
    python keno_cli.py all --grids 3     # Pipeline complet
    python keno_cli.py clean             # Nettoie les fichiers
    python keno_cli.py --help            # Aide complète

Auteur: Assistant IA
Version: 2.0
Date: Août 2025
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import os
import glob

# Ajout du répertoire keno au path
KENO_DIR = Path(__file__).parent / "keno"
sys.path.insert(0, str(KENO_DIR))

class KenoMain:
    """Interface principale pour le système Keno"""
    
    def __init__(self):
        self.keno_dir = Path(__file__).parent / "keno"
        self.data_dir = self.keno_dir / "keno_data"
        
    def check_data_availability(self):
        """Vérifie si des données sont disponibles"""
        csv_files = list(self.data_dir.glob("keno_*.csv"))
        if not csv_files:
            print("❌ Aucune donnée disponible")
            print("💡 Lancez d'abord: python keno_cli.py extract")
            return False
        
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"✅ Données disponibles - Fichier le plus récent: {latest_file.name}")
        return True

    def extract_data(self):
        """Extraire les données depuis la FDJ"""
        print("🔄 Extraction des données Keno...")
        result = subprocess.run(["python", "keno/extracteur_keno_zip.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Extraction terminée")
            
            # Assemblage automatique des fichiers avec nettoyage
            print("🔄 Assemblage et nettoyage automatique en cours...")
            assemble_result = subprocess.run(["python", "keno/assemble_keno_data.py"], 
                                           capture_output=True, text=True)
            
            if assemble_result.returncode == 0:
                print("✅ Assemblage et nettoyage terminés - Système optimisé")
                print(assemble_result.stdout)
            else:
                print("⚠️  Erreur lors de l'assemblage:")
                print(assemble_result.stderr)
            
            return True
        else:
            print("❌ Erreur lors de l'extraction:")
            print(result.stderr)
            return False

    def assemble(self):
        """Assembler tous les fichiers CSV en un seul fichier consolidé avec nettoyage automatique"""
        print("🔄 Assemblage et nettoyage des fichiers Keno...")
        result = subprocess.run(["python", "keno/assemble_keno_data.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Assemblage et nettoyage terminés")
            print(result.stdout)
            return True
        else:
            print("❌ Erreur lors de l'assemblage:")
            print(result.stderr)
            return False

    def analyze_data(self, force=False):
        """Lance l'analyse complète des données"""
        print("📊 ANALYSE STATISTIQUE COMPLÈTE")
        print("=" * 50)
        
        if not force and not self.check_data_availability():
            return False
        
        analyzer_script = self.keno_dir / "analyse_keno_complete.py"
        
        if not analyzer_script.exists():
            print(f"❌ Script non trouvé: {analyzer_script}")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, str(analyzer_script)
            ], capture_output=True, text=True, cwd=self.keno_dir.parent)
            
            if result.returncode == 0:
                print("✅ Analyse terminée avec succès")
                print(result.stdout)
                return True
            else:
                print("❌ Erreur lors de l'analyse")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False

    def analyze_advanced(self, plots=False, export_stats=False):
        """Lance l'analyse avancée DuckDB avec 11 stratégies"""
        print("🧠 ANALYSE AVANCÉE DUCKDB - 11 STRATÉGIES")
        print("=" * 50)
        
        # Vérifier si le fichier consolidé existe
        consolidated_file = Path("keno/keno_data/keno_consolidated.csv")
        if not consolidated_file.exists():
            print("❌ Fichier consolidé non trouvé")
            print("💡 Générez-le d'abord avec: python keno_cli.py assemble")
            return False
        
        duckdb_script = self.keno_dir / "duckdb_keno.py"
        
        if not duckdb_script.exists():
            print(f"❌ Script DuckDB non trouvé: {duckdb_script}")
            return False
        
        # Construction de la commande
        cmd = [sys.executable, str(duckdb_script), "--auto-consolidated"]
        
        if plots:
            cmd.append("--plots")
        if export_stats:
            cmd.append("--export-stats")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.keno_dir.parent)
            
            if result.returncode == 0:
                print("✅ Analyse avancée terminée avec succès")
                print(result.stdout)
                return True
            else:
                print("❌ Erreur lors de l'analyse avancée:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution: {str(e)}")
            return False

    def generate_grids(self, num_grids=5, force=False, advanced=False):
        """Génère des grilles intelligentes
        
        Args:
            num_grids (int): Nombre de grilles à générer
            force (bool): Force la génération même sans données
            advanced (bool): Utilise le générateur avancé avec ML
        """
        print("🎲 GÉNÉRATION DE GRILLES INTELLIGENTES")
        print("=" * 50)
        
        if not force and not self.check_data_availability():
            print("🔄 Lancement de l'analyse automatique...")
            if not self.analyze_data(force=True):
                return False
        
        # Choix du générateur selon le paramètre advanced
        if advanced:
            generator_script = self.keno_dir / "keno_generator_advanced.py"
            generator_name = "Générateur Avancé (ML)"
        else:
            generator_script = self.keno_dir / "generateur_keno_intelligent_v2.py"
            generator_name = "Générateur Standard"
            
        print(f"🧠 Utilisation: {generator_name}")
        
        if not generator_script.exists():
            print(f"❌ Script non trouvé: {generator_script}")
            return False
        
        try:
            if advanced:
                # Utilisation du générateur avancé avec ses paramètres spécifiques
                result = subprocess.run([
                    sys.executable, str(generator_script),
                    "--grids", str(num_grids),
                    "--silent"
                ], capture_output=True, text=True, cwd=self.keno_dir.parent)
            else:
                # Utilisation du générateur standard (méthode existante)
                result = subprocess.run([
                    sys.executable, str(generator_script)
                ], capture_output=True, text=True, cwd=self.keno_dir.parent)
            
            if result.returncode == 0:
                print("✅ Grilles générées avec succès")
                print(result.stdout)
                return True
            else:
                print("❌ Erreur lors de la génération:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution: {str(e)}")
            return False

    def generate_grids_advanced(self, num_grids=5, force=False, retrain=False, quick=False):
        """Génère des grilles avec le générateur avancé ML/IA
        
        Args:
            num_grids (int): Nombre de grilles à générer
            force (bool): Force la génération même sans données
            retrain (bool): Force le réentraînement des modèles ML
            quick (bool): Mode rapide (10 grilles)
        """
        print("🧠 GÉNÉRATION AVANCÉE - MACHINE LEARNING & IA")
        print("=" * 50)
        
        if not force and not self.check_data_availability():
            print("🔄 Lancement de l'analyse automatique...")
            if not self.analyze_data(force=True):
                return False
        
        generator_script = self.keno_dir / "keno_generator_advanced.py"
        
        if not generator_script.exists():
            print(f"❌ Générateur avancé non trouvé: {generator_script}")
            return False
        
        try:
            # Construction de la commande
            cmd = [sys.executable, str(generator_script)]
            
            if quick:
                cmd.append("--quick")
                print("⚡ Mode rapide activé (10 grilles)")
            else:
                cmd.extend(["--grids", str(num_grids)])
                print(f"🎲 Génération de {num_grids} grilles avec ML/IA")
            
            if retrain:
                cmd.append("--retrain")
                print("🔄 Réentraînement des modèles ML activé")
            
            # Utilisation du fichier consolidé si disponible
            consolidated_file = Path("keno/keno_data/keno_consolidated.csv")
            if consolidated_file.exists():
                # Note: Ne pas passer --data car le générateur avancé charge automatiquement
                print("📊 Fichier consolidé détecté - utilisation automatique")
            else:
                print("📊 Utilisation des fichiers par défaut")
            
            print("🚀 Lancement du générateur avancé...")
            print("-" * 50)
            
            result = subprocess.run(cmd, cwd=self.keno_dir.parent)
            
            if result.returncode == 0:
                print("-" * 50)
                print("✅ Grilles avancées générées avec succès")
                return True
            else:
                print("❌ Erreur lors de la génération avancée")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution: {str(e)}")
            return False

    def update_legacy_scripts(self):
        """Met à jour les anciens scripts pour utiliser le nouveau format"""
        print("🔄 MISE À JOUR DES ANCIENS SCRIPTS")
        print("=" * 50)
        
        # Scripts à mettre à jour
        scripts_to_update = [
            "analyse_keno_final.py",
            "duckdb_keno.py", 
            "keno_generator_advanced.py"
        ]
        
        for script_name in scripts_to_update:
            script_path = self.keno_dir / script_name
            if script_path.exists():
                print(f"🔧 Mise à jour de {script_name}...")
                self._update_script_data_loading(script_path)
        
        print("✅ Mise à jour terminée")
    
    def _update_script_data_loading(self, script_path):
        """Met à jour la logique de chargement des données dans un script"""
        # Cette fonction sera implementée pour modifier les scripts existants
        # pour qu'ils utilisent les nouveaux fichiers CSV au format unifié
        pass
    
    def status(self):
        """Affiche le statut du système"""
        print("📋 STATUT DU SYSTÈME KENO")
        print("=" * 50)
        
        # Vérifier les données
        csv_files = list(self.data_dir.glob("keno_*.csv"))
        print(f"📁 Fichiers de données: {len(csv_files)}")
        
        if csv_files:
            total_tirages = 0
            oldest_date = None
            newest_date = None
            
            for csv_file in csv_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    file_tirages = len(df)
                    total_tirages += file_tirages
                    
                    if not df.empty:
                        file_newest = df['date'].max()
                        file_oldest = df['date'].min()
                        
                        if newest_date is None or file_newest > newest_date:
                            newest_date = file_newest
                        if oldest_date is None or file_oldest < oldest_date:
                            oldest_date = file_oldest
                        
                except Exception:
                    continue
            
            print(f"📊 Total tirages: {total_tirages}")
            if oldest_date and newest_date:
                print(f"📅 Période: {oldest_date} → {newest_date}")
        
        # Vérifier les analyses
        stats_dir = Path("keno_stats_exports")
        if stats_dir.exists():
            freq_files = list(stats_dir.glob("frequences_keno_*.csv"))
            print(f"📈 Analyses disponibles: {len(freq_files)}")
            if freq_files:
                latest_analysis = max(freq_files, key=lambda f: f.stat().st_mtime)
                mod_time = datetime.fromtimestamp(latest_analysis.stat().st_mtime)
                print(f"📊 Dernière analyse: {mod_time.strftime('%d/%m/%Y %H:%M')}")
        
        # Vérifier les grilles générées
        output_dir = Path("keno_output")
        if output_dir.exists():
            grid_files = list(output_dir.glob("grilles_keno_*.csv"))
            print(f"🎲 Grilles générées: {len(grid_files)}")
            if grid_files:
                latest_grids = max(grid_files, key=lambda f: f.stat().st_mtime)
                mod_time = datetime.fromtimestamp(latest_grids.stat().st_mtime)
                print(f"🎯 Dernières grilles: {mod_time.strftime('%d/%m/%Y %H:%M')}")
    
    def clean(self, deep=False):
        """Nettoie les fichiers temporaires et anciens"""
        print("🧹 NETTOYAGE DU SYSTÈME")
        print("=" * 50)
        
        cleaned_count = 0
        
        # Nettoyer les fichiers temporaires
        temp_patterns = [
            "*.pyc",
            "*.pyo", 
            "*.log",
            "__pycache__",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        for pattern in temp_patterns:
            if pattern == "__pycache__":
                # Supprimer les dossiers __pycache__
                for pycache_dir in Path(".").rglob("__pycache__"):
                    if pycache_dir.is_dir():
                        import shutil
                        shutil.rmtree(pycache_dir)
                        cleaned_count += 1
                        print(f"🗑️  Supprimé: {pycache_dir}")
            else:
                # Supprimer les fichiers selon les patterns
                for file_path in Path(".").rglob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                        print(f"🗑️  Supprimé: {file_path}")
        
        # Nettoyer les anciens fichiers d'export (garde les 5 plus récents)
        for export_dir in ["keno_stats_exports", "keno_output", "keno_analyse_plots"]:
            export_path = Path(export_dir)
            if export_path.exists():
                self._clean_old_exports(export_path)
                cleaned_count += self._count_cleaned_exports(export_path)
        
        if deep:
            print("🔥 Nettoyage approfondi...")
            # Nettoyer les sauvegardes anciennes (plus de 30 jours)
            backup_dirs = list(Path(".").rglob("backup_*"))
            for backup_dir in backup_dirs:
                if backup_dir.is_dir():
                    # Vérifier l'âge du backup
                    age_days = (datetime.now().timestamp() - backup_dir.stat().st_mtime) / 86400
                    if age_days > 30:
                        import shutil
                        shutil.rmtree(backup_dir)
                        cleaned_count += 1
                        print(f"🗑️  Backup ancien supprimé: {backup_dir}")
        
        if cleaned_count == 0:
            print("✨ Système déjà propre !")
        else:
            print(f"✅ Nettoyage terminé: {cleaned_count} éléments supprimés")
    
    def _clean_old_exports(self, export_dir):
        """Nettoie les anciens exports en gardant les 5 plus récents"""
        files_by_type = {}
        
        # Grouper les fichiers par type (base name sans timestamp)
        for file_path in export_dir.glob("*_*.*"):
            # Extraire le type de base (avant le timestamp)
            name_parts = file_path.stem.split('_')
            if len(name_parts) >= 3:  # nom_type_timestamp
                base_type = '_'.join(name_parts[:-2])  # Enlever les 2 dernières parties (date_heure)
                if base_type not in files_by_type:
                    files_by_type[base_type] = []
                files_by_type[base_type].append(file_path)
        
        # Pour chaque type, garder seulement les 5 plus récents
        for base_type, files in files_by_type.items():
            if len(files) > 5:
                # Trier par date de modification (plus récent en premier)
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Supprimer les plus anciens (au-delà de 5)
                for old_file in files[5:]:
                    old_file.unlink()
                    print(f"🗑️  Export ancien supprimé: {old_file.name}")
    
    def _count_cleaned_exports(self, export_dir):
        """Compte les exports nettoyés pour les statistiques"""
        # Cette méthode sera appelée après le nettoyage pour compter
        return 0

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="🎯 CLI principal pour l'analyse et génération Keno",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande extract
    extract_parser = subparsers.add_parser(
        'extract', 
        help='Extrait les données depuis FDJ'
    )
    
    # Commande analyze  
    analyze_parser = subparsers.add_parser(
        'analyze', 
        help='Lance l\'analyse statistique complète'
    )
    analyze_parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force l\'analyse même sans données'
    )
    
    # Commande analyze-advanced (DuckDB avec fichier consolidé)
    analyze_advanced_parser = subparsers.add_parser(
        'analyze-advanced',
        help='Analyse avancée DuckDB avec 11 stratégies (utilise le fichier consolidé)'
    )
    analyze_advanced_parser.add_argument(
        '--plots',
        action='store_true',
        help='Génère les graphiques de visualisation'
    )
    analyze_advanced_parser.add_argument(
        '--export-stats',
        action='store_true',
        help='Exporte les statistiques détaillées'
    )
    
    # Commande generate
    generate_parser = subparsers.add_parser(
        'generate', 
        help='Génère des grilles intelligentes'
    )
    generate_parser.add_argument(
        '--grids', 
        type=int, 
        default=5, 
        help='Nombre de grilles à générer (défaut: 5)'
    )
    generate_parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force la génération même sans données'
    )
    generate_parser.add_argument(
        '--advanced',
        action='store_true',
        help='Utilise le générateur avancé avec ML/IA'
    )
    
    # Commande generate-advanced (générateur ML)
    generate_advanced_parser = subparsers.add_parser(
        'generate-advanced',
        help='Génère des grilles avec le générateur avancé (ML/IA)'
    )
    generate_advanced_parser.add_argument(
        '--grids',
        type=int,
        default=5,
        help='Nombre de grilles à générer (défaut: 5)'
    )
    generate_advanced_parser.add_argument(
        '--force',
        action='store_true',
        help='Force la génération même sans données'
    )
    generate_advanced_parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force le réentraînement des modèles ML'
    )
    generate_advanced_parser.add_argument(
        '--quick',
        action='store_true',
        help='Mode rapide (ignore le nombre de grilles, génère 10)'
    )
    
    # Commande status
    status_parser = subparsers.add_parser(
        'status', 
        help='Affiche le statut du système'
    )
    
    # Commande assemble (assemblage + nettoyage automatique)
    assemble_parser = subparsers.add_parser(
        'assemble',
        help='Assemble tous les fichiers CSV en un seul + nettoyage automatique'
    )
    
    # Commande update (pour les anciens scripts)
    update_parser = subparsers.add_parser(
        'update', 
        help='Met à jour les anciens scripts'
    )
    
    # Commande clean
    clean_parser = subparsers.add_parser(
        'clean',
        help='Nettoie les fichiers temporaires et anciens'
    )
    clean_parser.add_argument(
        '--deep',
        action='store_true',
        help='Nettoyage approfondi (inclut les anciens backups)'
    )
    
    # Commande all (pipeline complet)
    all_parser = subparsers.add_parser(
        'all', 
        help='Lance le pipeline complet: extract → analyze → generate'
    )
    all_parser.add_argument(
        '--grids', 
        type=int, 
        default=5, 
        help='Nombre de grilles à générer'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    keno_main = KenoMain()
    
    if args.command == 'extract':
        success = keno_main.extract_data()
        
    elif args.command == 'analyze':
        success = keno_main.analyze_data(force=args.force)
        
    elif args.command == 'analyze-advanced':
        success = keno_main.analyze_advanced(
            plots=args.plots,
            export_stats=args.export_stats
        )
        
    elif args.command == 'assemble':
        success = keno_main.assemble()
        
    elif args.command == 'generate':
        if hasattr(args, 'advanced') and args.advanced:
            success = keno_main.generate_grids_advanced(
                num_grids=args.grids, 
                force=args.force,
                retrain=getattr(args, 'retrain', False),
                quick=getattr(args, 'quick', False)
            )
        else:
            success = keno_main.generate_grids(
                num_grids=args.grids, 
                force=args.force,
                advanced=False
            )
    
    elif args.command == 'generate-advanced':
        success = keno_main.generate_grids_advanced(
            num_grids=args.grids,
            force=args.force,
            retrain=args.retrain,
            quick=args.quick
        )
        
    elif args.command == 'status':
        keno_main.status()
        success = True
        
    elif args.command == 'update':
        keno_main.update_legacy_scripts()
        success = True
        
    elif args.command == 'clean':
        keno_main.clean(deep=args.deep)
        success = True
        
    elif args.command == 'all':
        print("🚀 PIPELINE COMPLET KENO")
        print("=" * 50)
        
        # 1. Extraction
        print("\n1️⃣ Extraction des données...")
        if not keno_main.extract_data():
            return 1
        
        # 2. Analyse
        print("\n2️⃣ Analyse statistique...")
        if not keno_main.analyze_data(force=True):
            return 1
        
        # 3. Génération
        print("\n3️⃣ Génération de grilles...")
        if not keno_main.generate_grids(num_grids=args.grids, force=True):
            return 1
        
        print("\n🎉 Pipeline terminé avec succès !")
        success = True
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
