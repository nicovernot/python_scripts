#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST LOTO - Vérification complète du système Loto
===================================================

Teste que tous les composants Loto fonctionnent correctement avec le CSV téléchargé.
"""

import os
import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_loto_csv_availability():
    """Teste que le fichier CSV Loto est disponible"""
    csv_path = project_root / "loto" / "loto_data" / "loto_201911.csv"
    
    print("🎯 TEST DU SYSTÈME LOTO")
    print("=" * 50)
    
    # Test 1: Vérifier que le CSV existe
    print("\n1. Vérification du fichier CSV téléchargé...")
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024*1024)
        print(f"   ✅ CSV trouvé : {csv_path.name}")
        print(f"   📊 Taille : {size_mb:.1f} MB")
    else:
        print(f"   ❌ CSV manquant : {csv_path}")
        print("   💡 Exécutez d'abord : python loto/result.py")
        return False
    
    # Test 2: Vérifier le contenu du CSV
    print("\n2. Vérification du contenu du CSV...")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, delimiter=';', nrows=5)
        print(f"   ✅ CSV lisible avec {len(df.columns)} colonnes")
        
        # Vérifier les colonnes essentielles
        ball_columns = [col for col in df.columns if 'boule' in col.lower()]
        print(f"   🎱 Colonnes de boules détectées : {len(ball_columns)}")
        
        if len(ball_columns) >= 5:
            print("   ✅ Format Loto valide (5+ boules)")
            return True
        else:
            print("   ⚠️ Format Loto possiblement invalide")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur de lecture CSV : {e}")
        return False

def test_loto_scripts_configuration():
    """Teste que les scripts Loto sont configurés pour utiliser le bon CSV"""
    scripts_to_test = [
        "loto/strategies.py",
        "loto/grilles.py",
        "loto/duckdb_loto.py"
    ]
    
    print("\n3. Vérification de la configuration des scripts...")
    
    all_configured = True
    
    for script_path in scripts_to_test:
        full_path = project_root / script_path
        if full_path.exists():
            print(f"   📄 {script_path}")
            
            # Lire le contenu du script pour vérifier les chemins
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Vérifier les anciens chemins (ne doit plus être présent)
                old_paths_found = []
                if 'Téléchargements' in content:
                    old_paths_found.append('Téléchargements')
                if 'loto_201911.parquet' in content:
                    old_paths_found.append('parquet')
                
                # Vérifier les nouveaux chemins (doit être présent)
                new_paths_found = []
                if 'loto_data/loto_201911.csv' in content:
                    new_paths_found.append('CSV téléchargé')
                
                if old_paths_found:
                    print(f"      ⚠️ Anciens chemins détectés : {old_paths_found}")
                    all_configured = False
                
                if new_paths_found:
                    print(f"      ✅ Nouveaux chemins détectés : {new_paths_found}")
                else:
                    print(f"      ❓ Configuration à vérifier - aucun chemin CSV détecté")
                    
            except Exception as e:
                print(f"      ❌ Erreur de lecture : {e}")
                all_configured = False
        else:
            print(f"   ❌ Script non trouvé : {script_path}")
            all_configured = False
    
    return all_configured

def test_loto_duckdb_execution():
    """Teste l'exécution du script principal duckdb_loto.py"""
    print("\n4. Test d'exécution de duckdb_loto.py...")
    
    csv_path = project_root / "loto" / "loto_data" / "loto_201911.csv"
    
    if not csv_path.exists():
        print("   ❌ CSV non disponible pour le test")
        return False
    
    try:
        # Test d'import du module
        print("   🔧 Test d'import du module...")
        from loto.duckdb_loto import LotoStrategist
        print("   ✅ Import réussi")
        
        # Test d'initialisation
        print("   🔧 Test d'initialisation...")
        config_path = project_root / "loto" / "strategies.yml"
        strategist = LotoStrategist(config_file=str(config_path))
        print("   ✅ Initialisation réussie")
        
        # Test de chargement CSV
        print("   🔧 Test de chargement CSV...")
        import duckdb
        db_con = duckdb.connect(':memory:')
        table_name = strategist.load_data_from_csv(str(csv_path), db_con)
        print(f"   ✅ CSV chargé dans la table : {table_name}")
        
        # Test de génération d'une grille simple
        print("   🔧 Test de génération de grille...")
        grids = strategist.generate_grids(db_con, table_name, 1)
        if grids and len(grids) > 0:
            print(f"   ✅ Grille générée : {grids[0]}")
        else:
            print("   ⚠️ Aucune grille générée")
        
        db_con.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur d'exécution : {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loto_download_script():
    """Teste le script de téléchargement Loto"""
    print("\n5. Test du script de téléchargement...")
    
    try:
        # Test d'import
        from loto.result import main as loto_download_main
        print("   ✅ Import du script de téléchargement réussi")
        
        # Vérifier que le script peut être appelé (sans l'exécuter)
        download_script = project_root / "loto" / "result.py"
        if download_script.exists():
            print("   ✅ Script de téléchargement disponible")
            return True
        else:
            print("   ❌ Script de téléchargement manquant")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur de test du téléchargement : {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🚀 DÉMARRAGE DES TESTS LOTO")
    print("=" * 60)
    
    all_tests_passed = True
    error_summary = []
    
    # Test 1: Disponibilité du CSV
    if not test_loto_csv_availability():
        all_tests_passed = False
        error_summary.append("CSV Loto non disponible ou invalide")
    
    # Test 2: Configuration des scripts
    if not test_loto_scripts_configuration():
        all_tests_passed = False
        error_summary.append("Configuration des scripts incorrecte")
    
    # Test 3: Exécution du script principal
    if not test_loto_duckdb_execution():
        all_tests_passed = False
        error_summary.append("Erreur d'exécution du script principal")
    
    # Test 4: Script de téléchargement
    if not test_loto_download_script():
        all_tests_passed = False
        error_summary.append("Script de téléchargement défaillant")
    
    # Résumé final
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 TOUS LES TESTS LOTO RÉUSSIS")
        print("✅ Le système Loto est opérationnel")
        print("\n🚀 Commandes utiles :")
        print("   python loto/result.py                    # Télécharger les données")
        print("   python loto/duckdb_loto.py --grids 3     # Générer 3 grilles")
        return True
    else:
        print("❌ ÉCHEC DES TESTS LOTO")
        print("\n🔍 Résumé des erreurs:")
        for error in error_summary:
            print(f"   • {error}")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
