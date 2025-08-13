#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 LANCEUR DE TESTS - Exécution de tous les tests du projet
=========================================================

Ce script lance tous les tests de validation du système Loto/Keno.
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_test_script(script_path, description):
    """Exécute un script de test et retourne le résultat"""
    print(f"\n🔧 Exécution : {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60  # Timeout de 60 secondes
        )
        
        # Afficher la sortie
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0, result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT - Le test a pris trop de temps")
        return False, -1
    except Exception as e:
        print(f"❌ ERREUR D'EXÉCUTION : {e}")
        return False, -2

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Lanceur de tests pour le projet Loto/Keno",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python test/run_all_tests.py                 # Tous les tests
  python test/run_all_tests.py --essential     # Seulement les tests essentiels
  python test/run_all_tests.py --loto          # Seulement les tests Loto
  python test/run_all_tests.py --keno          # Seulement les tests Keno
  python test/run_all_tests.py --performance   # Seulement les tests de performance
  python test/run_all_tests.py --fast          # Tests rapides (exclut performance)
        """
    )
    
    parser.add_argument(
        '--essential', 
        action='store_true',
        help='Exécuter seulement les tests essentiels'
    )
    parser.add_argument(
        '--loto', 
        action='store_true',
        help='Exécuter seulement les tests Loto'
    )
    parser.add_argument(
        '--keno', 
        action='store_true',
        help='Exécuter seulement les tests Keno'
    )
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Exécuter seulement les tests de performance'
    )
    parser.add_argument(
        '--fast', 
        action='store_true',
        help='Exécuter tous les tests sauf les tests de performance'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Affichage détaillé'
    )
    
    args = parser.parse_args()
    
    # Définir les tests disponibles
    test_dir = Path(__file__).parent
    available_tests = [
        (test_dir / "test_essential.py", "Tests essentiels", "essential"),
        (test_dir / "test_loto.py", "Tests Loto", "loto"),
        (test_dir / "test_keno.py", "Tests Keno", "keno"),
        (test_dir / "test_performance.py", "Tests de performance", "performance"),
    ]
    
    # Déterminer quels tests exécuter
    tests_to_run = []
    
    if args.essential:
        tests_to_run = [t for t in available_tests if t[2] == "essential"]
    elif args.loto:
        tests_to_run = [t for t in available_tests if t[2] == "loto"]
    elif args.keno:
        tests_to_run = [t for t in available_tests if t[2] == "keno"]
    elif args.performance:
        tests_to_run = [t for t in available_tests if t[2] == "performance"]
    elif args.fast:
        tests_to_run = [t for t in available_tests if t[2] != "performance"]
    else:
        # Tous les tests par défaut
        tests_to_run = available_tests
    
    # Vérifier que les fichiers de test existent
    tests_to_run = [t for t in tests_to_run if t[0].exists()]
    
    if not tests_to_run:
        print("❌ Aucun test trouvé à exécuter")
        return 1
    
    # Affichage d'en-tête
    print("🧪 LANCEUR DE TESTS - PROJET LOTO/KENO")
    print("=" * 60)
    print(f"📋 Tests à exécuter : {len(tests_to_run)}")
    for script_path, description, category in tests_to_run:
        print(f"   • {description}")
    print()
    
    # Exécution des tests
    results = {}
    total_tests = len(tests_to_run)
    passed_tests = 0
    
    for script_path, description, category in tests_to_run:
        success, return_code = run_test_script(script_path, description)
        results[description] = (success, return_code)
        
        if success:
            passed_tests += 1
        
        if args.verbose:
            status = "✅ RÉUSSI" if success else f"❌ ÉCHEC (code {return_code})"
            print(f"\n{status} - {description}")
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    for description, (success, return_code) in results.items():
        status_icon = "✅" if success else "❌"
        status_text = "RÉUSSI" if success else f"ÉCHEC ({return_code})"
        print(f"{status_icon} {description:<30} : {status_text}")
    
    print(f"\n📈 STATISTIQUES GLOBALES")
    print(f"   • Tests exécutés  : {total_tests}")
    print(f"   • Tests réussis   : {passed_tests}")
    print(f"   • Tests échoués   : {total_tests - passed_tests}")
    print(f"   • Taux de réussite: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 TOUS LES TESTS SONT RÉUSSIS !")
        print("✅ Le système est opérationnel")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TEST(S) ONT ÉCHOUÉ")
        print("💡 Vérifiez les erreurs ci-dessus et corrigez les problèmes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
