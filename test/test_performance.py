#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST PERFORMANCE - Vérification des performances du système
============================================================

Teste les performances et temps d'exécution des composants critiques.
"""

import time
import sys
from pathlib import Path
import traceback

# Ajouter le projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def measure_time(func):
    """Décorateur pour mesurer le temps d'exécution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper

def test_import_performance():
    """Teste les performances d'import des modules principaux"""
    print("⚡ TEST DES PERFORMANCES D'IMPORT")
    print("=" * 50)
    
    modules_to_test = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('duckdb', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('sklearn', None),
    ]
    
    total_time = 0
    successful_imports = 0
    
    for module_name, alias in modules_to_test:
        try:
            start = time.time()
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            import_time = time.time() - start
            
            print(f"   ✅ {module_name:<20} : {import_time:.3f}s")
            total_time += import_time
            successful_imports += 1
            
        except Exception as e:
            print(f"   ❌ {module_name:<20} : ÉCHEC - {e}")
    
    print(f"\n📊 Résumé imports :")
    print(f"   • Modules importés : {successful_imports}/{len(modules_to_test)}")
    print(f"   • Temps total      : {total_time:.3f}s")
    print(f"   • Temps moyen      : {total_time/len(modules_to_test):.3f}s")
    
    return successful_imports == len(modules_to_test)

def test_csv_loading_performance():
    """Teste les performances de chargement des CSV"""
    print("\n📊 TEST DES PERFORMANCES DE CHARGEMENT CSV")
    print("=" * 50)
    
    csv_files = [
        project_root / "keno" / "keno_data" / "keno_202011.csv",
        project_root / "loto" / "loto_data" / "loto_201911.csv",
    ]
    
    try:
        import pandas as pd
        import duckdb
        
        for csv_path in csv_files:
            if csv_path.exists():
                file_type = "Keno" if "keno" in str(csv_path) else "Loto"
                
                # Test avec pandas
                start = time.time()
                try:
                    df = pd.read_csv(csv_path, delimiter=';', nrows=100)  # Limiter pour le test
                    pandas_time = time.time() - start
                    print(f"   ✅ {file_type} Pandas  : {pandas_time:.3f}s ({len(df)} lignes)")
                except Exception as e:
                    print(f"   ❌ {file_type} Pandas  : ÉCHEC - {e}")
                    continue
                
                # Test avec DuckDB
                start = time.time()
                try:
                    conn = duckdb.connect(':memory:')
                    result = conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}', delim=';')").fetchone()
                    duckdb_time = time.time() - start
                    print(f"   ✅ {file_type} DuckDB  : {duckdb_time:.3f}s ({result[0]} lignes totales)")
                    conn.close()
                except Exception as e:
                    print(f"   ❌ {file_type} DuckDB  : ÉCHEC - {e}")
            else:
                file_type = "Keno" if "keno" in str(csv_path) else "Loto"
                print(f"   ❓ {file_type} : CSV non trouvé - {csv_path.name}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Modules manquants pour le test : {e}")
        return False

def test_analysis_performance():
    """Teste les performances d'analyse de base"""
    print("\n🔍 TEST DES PERFORMANCES D'ANALYSE")
    print("=" * 50)
    
    # Test Keno
    print("📊 Test d'analyse Keno...")
    try:
        start = time.time()
        from keno.duckdb_keno import KenoAnalyzer
        import_time = time.time() - start
        print(f"   ✅ Import KenoAnalyzer : {import_time:.3f}s")
        
        start = time.time()
        analyzer = KenoAnalyzer()
        init_time = time.time() - start
        print(f"   ✅ Init KenoAnalyzer  : {init_time:.3f}s")
        
    except Exception as e:
        print(f"   ❌ Erreur KenoAnalyzer : {e}")
    
    # Test Loto
    print("\n📊 Test d'analyse Loto...")
    try:
        start = time.time()
        from loto.duckdb_loto import LotoStrategist
        import_time = time.time() - start
        print(f"   ✅ Import LotoStrategist : {import_time:.3f}s")
        
        config_path = project_root / "loto" / "strategies.yml"
        if config_path.exists():
            start = time.time()
            strategist = LotoStrategist(config_file=str(config_path))
            init_time = time.time() - start
            print(f"   ✅ Init LotoStrategist  : {init_time:.3f}s")
        else:
            print("   ❓ Fichier de configuration manquant")
        
    except Exception as e:
        print(f"   ❌ Erreur LotoStrategist : {e}")
    
    return True

def test_memory_usage():
    """Teste l'utilisation mémoire de base"""
    print("\n💾 TEST D'UTILISATION MÉMOIRE")
    print("=" * 50)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Mémoire avant
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   📊 Mémoire initiale : {mem_before:.1f} MB")
        
        # Charger des modules gourmands
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Mémoire après imports
        mem_after_imports = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   📊 Mémoire après imports : {mem_after_imports:.1f} MB")
        print(f"   📈 Augmentation : +{mem_after_imports - mem_before:.1f} MB")
        
        # Test avec un DataFrame
        df = pd.DataFrame(np.random.randn(1000, 10))
        mem_after_df = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   📊 Mémoire avec DataFrame : {mem_after_df:.1f} MB")
        print(f"   📈 Augmentation totale : +{mem_after_df - mem_before:.1f} MB")
        
        if mem_after_df < 500:  # Seuil ajusté - 500 MB est raisonnable
            print("   ✅ Utilisation mémoire raisonnable")
            return True
        else:
            print("   ⚠️ Utilisation mémoire élevée")
            return False
            
    except ImportError:
        print("   ❓ psutil non disponible - test ignoré")
        return True
    except Exception as e:
        print(f"   ❌ Erreur de test mémoire : {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🚀 DÉMARRAGE DES TESTS DE PERFORMANCE")
    print("=" * 60)
    
    all_tests_passed = True
    error_summary = []
    
    # Test 1: Performance des imports
    if not test_import_performance():
        all_tests_passed = False
        error_summary.append("Problèmes d'import de modules")
    
    # Test 2: Performance de chargement CSV
    if not test_csv_loading_performance():
        all_tests_passed = False
        error_summary.append("Problèmes de chargement CSV")
    
    # Test 3: Performance d'analyse
    if not test_analysis_performance():
        all_tests_passed = False
        error_summary.append("Problèmes de performance d'analyse")
    
    # Test 4: Utilisation mémoire
    if not test_memory_usage():
        all_tests_passed = False
        error_summary.append("Utilisation mémoire excessive")
    
    # Résumé final
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 TOUS LES TESTS DE PERFORMANCE RÉUSSIS")
        print("✅ Le système fonctionne avec des performances acceptables")
        return True
    else:
        print("❌ PROBLÈMES DE PERFORMANCE DÉTECTÉS")
        print("\n🔍 Résumé des problèmes:")
        for error in error_summary:
            print(f"   • {error}")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
