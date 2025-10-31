#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST ESSENTIAL - Vérification des dépendances et imports critiques
================================================================

Ce test vérifie que toutes les dépendances essentielles sont disponibles.
"""

import sys
import traceback
from pathlib import Path

def test_essential_imports():
    """Teste les imports essentiels pour le fonctionnement du système"""
    print("🧪 TEST DES IMPORTS ESSENTIELS")
    print("=" * 50)
    
    essential_modules = [
        ('pandas', 'Manipulation de données'),
        ('numpy', 'Calculs numériques'),
        ('duckdb', 'Base de données'),
        ('matplotlib', 'Visualisations'),
        ('seaborn', 'Graphiques avancés'),
        ('requests', 'Téléchargements web'),
        ('bs4', 'Parsing HTML'),
        ('sklearn', 'Machine Learning'),
        ('xgboost', 'ML avancé'),
        ('yaml', 'Configuration'),
    ]
    
    failed_imports = []
    
    for module, description in essential_modules:
        try:
            __import__(module)
            print(f"✅ {module:<12} - {description}")
        except ImportError as e:
            print(f"❌ {module:<12} - {description} - ERREUR: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"⚠️  {module:<12} - {description} - ATTENTION: {e}")
    
    return len(failed_imports) == 0, failed_imports

def test_project_structure():
    """Vérifie la structure du projet"""
    print("\n🏗️  TEST DE LA STRUCTURE DU PROJET")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    expected_dirs = [
        'keno',
        'loto',
        'test',
    ]
    
    expected_files = [
        'requirements.txt',
        '.gitignore',
    ]
    
    missing_items = []
    
    print("📁 Vérification des dossiers:")
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - MANQUANT")
            missing_items.append(f"dossier {dir_name}")
    
    print("\n📄 Vérification des fichiers:")
    for file_name in expected_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - MANQUANT")
            missing_items.append(f"fichier {file_name}")
    
    return len(missing_items) == 0, missing_items

def test_scripts_compilation():
    """Teste la compilation des scripts principaux"""
    print("\n🔧 TEST DE COMPILATION DES SCRIPTS")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    main_scripts = [
        'keno/duckdb_keno.py',
        'loto/duckdb_loto.py',
        'keno/extracteur_keno_zip.py',
        'loto/result.py',
    ]
    
    import py_compile
    compilation_errors = []
    
    for script_path in main_scripts:
        full_path = project_root / script_path
        if full_path.exists():
            try:
                py_compile.compile(str(full_path), doraise=True)
                print(f"   ✅ {script_path}")
            except py_compile.PyCompileError as e:
                print(f"   ❌ {script_path} - ERREUR: {e}")
                compilation_errors.append(script_path)
        else:
            print(f"   ❓ {script_path} - FICHIER INEXISTANT")
            compilation_errors.append(script_path)
    
    return len(compilation_errors) == 0, compilation_errors

def main():
    """Fonction principale de test"""
    print("🚀 DÉMARRAGE DES TESTS ESSENTIELS")
    print("=" * 60)
    
    all_tests_passed = True
    error_summary = []
    
    # Test 1: Imports essentiels
    imports_ok, failed_imports = test_essential_imports()
    if not imports_ok:
        all_tests_passed = False
        error_summary.append(f"Imports manquants: {', '.join(failed_imports)}")
    
    # Test 2: Structure du projet
    structure_ok, missing_items = test_project_structure()
    if not structure_ok:
        all_tests_passed = False
        error_summary.append(f"Structure incomplète: {', '.join(missing_items)}")
    
    # Test 3: Compilation des scripts
    compilation_ok, compilation_errors = test_scripts_compilation()
    if not compilation_ok:
        all_tests_passed = False
        error_summary.append(f"Erreurs de compilation: {', '.join(compilation_errors)}")
    
    # Résumé final
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 TOUS LES TESTS ESSENTIELS RÉUSSIS")
        print("✅ Le système est prêt à fonctionner")
        return True
    else:
        print("❌ ÉCHEC DES TESTS ESSENTIELS")
        print("\n🔍 Résumé des erreurs:")
        for error in error_summary:
            print(f"   • {error}")
        print("\n💡 Corrigez ces problèmes avant d'utiliser le système")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
