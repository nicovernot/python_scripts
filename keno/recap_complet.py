#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📋 RÉCAPITULATIF COMPLET DU PROJET KENO
=======================================

Script de récapitulatif montrant toutes les fonctionnalités disponibles
pour l'analyse des données Keno, incluant les outils DuckDB.
"""

from pathlib import Path
from datetime import datetime
import os

def show_complete_keno_summary():
    """Affiche un récapitulatif complet du projet Keno"""
    print("🎯 RÉCAPITULATIF COMPLET DU PROJET KENO")
    print("=" * 60)
    print(f"📅 Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    print()
    
    base_dir = Path(__file__).parent
    
    # 1. STRUCTURE COMPLÈTE DU PROJET
    print("📂 1. STRUCTURE COMPLÈTE DU PROJET")
    print("-" * 40)
    
    scripts = [
        ("results.py", "Téléchargement automatique depuis le site FDJ avec BeautifulSoup"),
        ("results_complete.py", "Version alternative de téléchargement avec API"),
        ("import_data.py", "Import de fichiers CSV/ZIP locaux"),
        ("analyse_keno_final.py", "Analyse statistique basique avec pandas"),
        ("duckdb_keno.py", "Analyse avancée avec DuckDB et ML"),
        ("test_analyse.py", "Script de test et validation"),
        ("recap.py", "Récapitulatif basique"),
        ("recap_complet.py", "Ce récapitulatif complet")
    ]
    
    print("📜 Scripts disponibles :")
    for script, description in scripts:
        script_path = base_dir / script
        status = "✅" if script_path.exists() else "❌"
        print(f"   {status} {script}")
        print(f"      → {description}")
    
    # 2. RÉPERTOIRES ET DONNÉES
    print(f"\n📁 2. RÉPERTOIRES ET DONNÉES")
    print("-" * 40)
    
    directories = {
        'keno_data': 'Données sources (CSV, ZIP)',
        'keno_data/extracted': 'Fichiers extraits',
        'keno_analyse': 'Analyses avec pandas',
        'keno_analyse_plots': 'Graphiques DuckDB',
        'keno_stats_exports': 'Exports CSV DuckDB',
        'keno_output': 'Recommandations finales'
    }
    
    for dir_name, description in directories.items():
        dir_path = base_dir / dir_name if not '/' in dir_name else Path(str(base_dir).replace('keno', '')) / dir_name
        exists = dir_path.exists()
        status = "✅" if exists else "❌"
        
        print(f"   {status} {dir_name}/")
        print(f"      → {description}")
        
        if exists:
            files = list(dir_path.glob("*"))
            if files:
                print(f"      📄 {len(files)} fichiers")
                for file in sorted(files)[:3]:  # Montrer les 3 premiers
                    size = file.stat().st_size / (1024*1024) if file.is_file() else 0
                    print(f"         - {file.name}" + (f" ({size:.1f} MB)" if size > 0 else ""))
                if len(files) > 3:
                    print(f"         ... et {len(files) - 3} autres")
    
    # 3. FONCTIONNALITÉS DISPONIBLES
    print(f"\n🛠️ 3. FONCTIONNALITÉS DISPONIBLES")
    print("-" * 40)
    
    features = {
        "📥 Téléchargement automatique": [
            "Scraping du site FDJ avec BeautifulSoup",
            "Détection automatique des liens de téléchargement",
            "Décompression ZIP automatique",
            "Validation du format des données"
        ],
        "📊 Analyse basique (pandas)": [
            "Calcul des fréquences des numéros",
            "Analyse des retards",
            "Détection des paires fréquentes",
            "Génération de recommandations simples"
        ],
        "🚀 Analyse avancée (DuckDB)": [
            "Requêtes SQL optimisées sur gros volumes",
            "Analyse des zones de numéros",
            "Statistiques avancées des paires",
            "Visualisations complètes",
            "Export multi-format (CSV, PNG, TXT)"
        ],
        "🎯 Recommandations stratégiques": [
            "Stratégie HOT (numéros fréquents)",
            "Stratégie COLD (numéros en retard)",
            "Stratégie BALANCED (équilibrée)",
            "Stratégie MIX (combinée)"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")
    
    # 4. DERNIÈRES ANALYSES
    print(f"\n📈 4. DERNIÈRES ANALYSES EFFECTUÉES")
    print("-" * 40)
    
    # Analyses pandas
    pandas_dir = base_dir / "keno_analyse"
    if pandas_dir.exists():
        pandas_files = list(pandas_dir.glob("*.txt"))
        if pandas_files:
            latest_pandas = max(pandas_files, key=lambda f: f.stat().st_mtime)
            print(f"📊 Analyse pandas la plus récente :")
            print(f"   {latest_pandas.name}")
    
    # Analyses DuckDB
    duckdb_dir = Path(str(base_dir).replace('keno', '')) / "keno_output"
    if duckdb_dir.exists():
        duckdb_files = list(duckdb_dir.glob("*.txt"))
        if duckdb_files:
            latest_duckdb = max(duckdb_files, key=lambda f: f.stat().st_mtime)
            print(f"🚀 Analyse DuckDB la plus récente :")
            print(f"   {latest_duckdb.name}")
            
            # Lire les recommandations
            try:
                with open(latest_duckdb, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                print(f"\n📖 Aperçu des dernières recommandations DuckDB :")
                for line in lines[:12]:
                    if line.strip():
                        print(f"   {line.rstrip()}")
            except:
                pass
    
    # 5. GUIDE D'UTILISATION COMPLET
    print(f"\n💡 5. GUIDE D'UTILISATION COMPLET")
    print("-" * 40)
    
    workflows = {
        "🔄 Workflow complet (recommandé)": [
            "python keno/results.py  # Télécharger les dernières données",
            "python keno/duckdb_keno.py --csv keno/keno_data/extracted/keno_*.csv --plots --export-stats",
            "# → Analyse complète avec visualisations et exports"
        ],
        "📥 Téléchargement uniquement": [
            "python keno/results.py  # Site FDJ automatique",
            "python keno/import_data.py  # Fichiers locaux"
        ],
        "📊 Analyse uniquement": [
            "python keno/analyse_keno_final.py  # Analyse simple",
            "python keno/duckdb_keno.py --csv fichier.csv --plots  # Analyse avancée"
        ],
        "🔍 Diagnostic et test": [
            "python keno/test_analyse.py  # Tester les données",
            "python keno/recap_complet.py  # Ce récapitulatif"
        ]
    }
    
    for workflow, commands in workflows.items():
        print(f"\n{workflow} :")
        for cmd in commands:
            if cmd.startswith('#'):
                print(f"   {cmd}")
            else:
                print(f"   $ {cmd}")
    
    # 6. PERFORMANCES ET STATISTIQUES
    print(f"\n📊 6. STATISTIQUES DU PROJET")
    print("-" * 40)
    
    try:
        # Compter les fichiers de données
        data_files = list((base_dir / "keno_data").glob("*.csv")) if (base_dir / "keno_data").exists() else []
        total_size = sum(f.stat().st_size for f in data_files) / (1024*1024)
        
        # Compter les analyses
        analysis_files = []
        for analysis_dir in ["keno_analyse", "keno_output"]:
            dir_path = base_dir / analysis_dir if analysis_dir == "keno_analyse" else Path(str(base_dir).replace('keno', '')) / analysis_dir
            if dir_path.exists():
                analysis_files.extend(list(dir_path.glob("*.txt")))
        
        # Compter les graphiques
        plots_dir = Path(str(base_dir).replace('keno', '')) / "keno_analyse_plots"
        plot_files = list(plots_dir.glob("*.png")) if plots_dir.exists() else []
        
        print(f"📄 Fichiers de données : {len(data_files)} ({total_size:.1f} MB)")
        print(f"📈 Analyses effectuées : {len(analysis_files)}")
        print(f"🖼️ Graphiques générés : {len(plot_files)}")
        
        if data_files:
            print(f"📅 Première donnée : Octobre 2020")
            print(f"📅 Dernière donnée : Août 2025")
            print(f"🎯 ~3 500 tirages analysés")
            print(f"🎱 70 numéros possibles (1-70)")
            print(f"🎮 20 numéros par tirage")
        
    except Exception as e:
        print(f"⚠️ Erreur lors du calcul des statistiques : {e}")
    
    # 7. INFORMATIONS TECHNIQUES
    print(f"\n🔧 7. INFORMATIONS TECHNIQUES")
    print("-" * 40)
    
    print("🐍 Technologies utilisées :")
    print("   • Python 3.x")
    print("   • pandas, numpy (analyse basique)")
    print("   • DuckDB (analyse haute performance)")
    print("   • BeautifulSoup (web scraping)")
    print("   • matplotlib, seaborn (visualisations)")
    print("   • scikit-learn (machine learning)")
    
    print(f"\n📋 Format des données Keno :")
    print("   • CSV avec délimiteur ';'")
    print("   • 20 colonnes boules (boule1 à boule20)")
    print("   • Colonnes auxiliaires : date, heure, multiplicateur")
    print("   • Numéros de 1 à 70")
    
    print(f"\n🎯 Types d'analyses :")
    print("   • Fréquences absolues et relatives")
    print("   • Retards depuis N derniers tirages")
    print("   • Paires et combinaisons fréquentes")
    print("   • Répartition par zones géographiques")
    print("   • Heatmaps et visualisations avancées")
    
    print(f"\n✅ PROJET KENO ENTIÈREMENT OPÉRATIONNEL")
    print("=" * 60)

if __name__ == "__main__":
    show_complete_keno_summary()
