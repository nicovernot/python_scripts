#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📋 RÉCAPITULATIF DU PROJET KENO
==============================

Script récapitulatif de toutes les fonctionnalités disponibles
pour l'analyse des données Keno.
"""

from pathlib import Path
from datetime import datetime

def show_project_summary():
    """Affiche un récapitulatif complet du projet"""
    print("🎯 RÉCAPITULATIF DU PROJET KENO")
    print("=" * 50)
    print(f"📅 Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    print()
    
    base_dir = Path(__file__).parent
    keno_data_dir = base_dir / "keno_data"
    keno_analyse_dir = base_dir / "keno_analyse"
    
    # 1. FICHIERS DISPONIBLES
    print("📂 1. STRUCTURE DU PROJET")
    print("-" * 30)
    
    scripts = [
        ("results_complete.py", "Téléchargement et traitement des données depuis l'API FDJ"),
        ("import_data.py", "Import de fichiers CSV/ZIP depuis les Téléchargements"),
        ("analyse_keno_final.py", "Analyse complète des données (fréquences, retards, recommandations)"),
        ("test_analyse.py", "Script de test pour vérifier les données"),
    ]
    
    print("📜 Scripts disponibles :")
    for script, description in scripts:
        script_path = base_dir / script
        status = "✅" if script_path.exists() else "❌"
        print(f"   {status} {script}")
        print(f"      → {description}")
    
    # 2. DONNÉES DISPONIBLES
    print(f"\n📊 2. DONNÉES DISPONIBLES")
    print("-" * 30)
    
    csv_files = list(keno_data_dir.glob("*.csv")) if keno_data_dir.exists() else []
    
    if csv_files:
        total_size = sum(f.stat().st_size for f in csv_files)
        print(f"📁 Répertoire données : {keno_data_dir}")
        print(f"📄 Fichiers CSV : {len(csv_files)}")
        print(f"💾 Taille totale : {total_size / (1024*1024):.1f} MB")
        
        print(f"\n📋 Fichiers :")
        for file_path in sorted(csv_files):
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   - {file_path.name} ({size_mb:.1f} MB)")
    else:
        print("❌ Aucune donnée trouvée")
        print("💡 Utilisez 'python keno/import_data.py' pour importer des données")
    
    # 3. RÉSULTATS D'ANALYSE
    print(f"\n📈 3. RÉSULTATS D'ANALYSE")
    print("-" * 30)
    
    if keno_analyse_dir.exists():
        analysis_files = {
            'frequences': list(keno_analyse_dir.glob("frequences_keno_*.csv")),
            'retards': list(keno_analyse_dir.glob("retards_keno_*.csv")),
            'recommandations': list(keno_analyse_dir.glob("recommandations_keno_*.txt"))
        }
        
        total_analyses = sum(len(files) for files in analysis_files.values())
        
        if total_analyses > 0:
            print(f"📁 Répertoire analyses : {keno_analyse_dir}")
            print(f"📊 Analyses effectuées : {total_analyses // 3}")
            
            # Montrer la dernière analyse
            if analysis_files['recommandations']:
                latest_rec = max(analysis_files['recommandations'], key=lambda f: f.stat().st_mtime)
                print(f"📋 Dernière analyse : {latest_rec.name}")
                
                # Lire les premières lignes des recommandations
                try:
                    with open(latest_rec, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:15]
                    
                    print(f"\n📖 Aperçu des dernières recommandations :")
                    for line in lines:
                        if line.strip():
                            print(f"   {line.rstrip()}")
                except:
                    pass
        else:
            print("❌ Aucune analyse trouvée")
    else:
        print("❌ Répertoire d'analyse non trouvé")
        print("💡 Utilisez 'python keno/analyse_keno_final.py' pour générer des analyses")
    
    # 4. GUIDE D'UTILISATION
    print(f"\n💡 4. GUIDE D'UTILISATION RAPIDE")
    print("-" * 30)
    
    steps = [
        ("1️⃣ Importer des données", "python keno/import_data.py"),
        ("2️⃣ Télécharger via API", "python keno/results_complete.py"),
        ("3️⃣ Analyser les données", "python keno/analyse_keno_final.py"),
        ("4️⃣ Voir ce récapitulatif", "python keno/recap.py")
    ]
    
    print("🚀 Étapes recommandées :")
    for step, command in steps:
        print(f"   {step}")
        print(f"      {command}")
    
    # 5. INFORMATIONS TECHNIQUES
    print(f"\n🔧 5. INFORMATIONS TECHNIQUES")
    print("-" * 30)
    
    print("📊 Format des données Keno :")
    print("   - 20 numéros par tirage (boule1 à boule20)")
    print("   - Numéros possibles : 1 à 70")
    print("   - Format CSV avec délimiteur ';'")
    print("   - Colonnes supplémentaires : date, heure, multiplicateur")
    
    print(f"\n📈 Types d'analyses :")
    print("   - Fréquences : Numéros les plus/moins sortis")
    print("   - Retards : Numéros non sortis récemment")
    print("   - Stratégies : HOT, COLD, BALANCED, MIX")
    
    print(f"\n📁 Répertoires :")
    print(f"   - Données : {keno_data_dir}")
    print(f"   - Analyses : {keno_analyse_dir}")
    
    print(f"\n✅ PROJET OPÉRATIONNEL")
    print("=" * 50)

if __name__ == "__main__":
    show_project_summary()
