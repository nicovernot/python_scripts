#!/usr/bin/env python3
"""
Test de l'API Flask Loto/Keno
============================

Script de test pour vérifier le bon fonctionnement de l'API Flask.
Tests des endpoints principaux et validation des réponses.

Usage:
    python test_api.py
    python test_api.py --verbose
    python test_api.py --endpoint health
    
Author: Système Loto/Keno API Tests
Date: 24 août 2025
"""

import requests
import json
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

class APITester:
    """Testeur de l'API Flask Loto/Keno"""
    
    def __init__(self, base_url: str = "http://localhost:5000", verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'LotoKeno-API-Tester/1.0'
        })
        
    def log(self, message: str, level: str = "INFO"):
        """Log avec timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "SUCCESS"]:
            level_icon = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARNING": "⚠️"
            }.get(level, "📋")
            print(f"[{timestamp}] {level_icon} {message}")
    
    def test_connection(self) -> bool:
        """Test de connexion basique"""
        try:
            self.log("Test de connexion à l'API...")
            response = self.session.get(f"{self.base_url}/", timeout=5)
            
            if response.status_code == 200:
                self.log("✅ Connexion réussie à l'API", "SUCCESS")
                return True
            else:
                self.log(f"❌ Connexion échouée - Status: {response.status_code}", "ERROR")
                return False
                
        except requests.exceptions.ConnectionError:
            self.log("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est démarrée.", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ Erreur de connexion: {e}", "ERROR")
            return False
    
    def test_health(self) -> Dict[str, Any]:
        """Test de l'endpoint de santé"""
        try:
            self.log("Test de l'endpoint de santé...")
            response = self.session.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                self.log("✅ Endpoint de santé opérationnel", "SUCCESS")
                self.log(f"   Status: {health_data.get('status', 'N/A')}")
                self.log(f"   Version: {health_data.get('version', 'N/A')}")
                self.log(f"   Uptime: {health_data.get('uptime', 'N/A')}")
                return health_data
            else:
                self.log(f"❌ Health check échoué - Status: {response.status_code}", "ERROR")
                return {}
                
        except Exception as e:
            self.log(f"❌ Erreur health check: {e}", "ERROR")
            return {}
    
    def test_data_status(self) -> Dict[str, Any]:
        """Test de l'endpoint de statut des données"""
        try:
            self.log("Test du statut des données...")
            response = self.session.get(f"{self.base_url}/api/data/status", timeout=10)
            
            if response.status_code == 200:
                data_status = response.json()
                self.log("✅ Statut des données récupéré", "SUCCESS")
                
                if 'loto' in data_status:
                    loto_status = data_status['loto']
                    self.log(f"   Loto - Fichiers: {loto_status.get('files_count', 0)}")
                    
                if 'keno' in data_status:
                    keno_status = data_status['keno']
                    self.log(f"   Keno - Fichiers: {keno_status.get('files_count', 0)}")
                    
                return data_status
            else:
                self.log(f"❌ Statut des données échoué - Status: {response.status_code}", "ERROR")
                return {}
                
        except Exception as e:
            self.log(f"❌ Erreur statut données: {e}", "ERROR")
            return {}
    
    def test_loto_generation(self) -> Dict[str, Any]:
        """Test de génération de grilles Loto"""
        try:
            self.log("Test de génération Loto...")
            
            payload = {
                "strategy": "equilibre",
                "grids": 3,
                "analysis": True
            }
            
            response = self.session.post(
                f"{self.base_url}/api/loto/generate", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                loto_result = response.json()
                self.log("✅ Génération Loto réussie", "SUCCESS")
                
                if 'grids' in loto_result:
                    grids = loto_result['grids']
                    self.log(f"   Grilles générées: {len(grids)}")
                    
                    if grids and self.verbose:
                        for i, grid in enumerate(grids[:2], 1):
                            self.log(f"   Grille {i}: {grid}")
                            
                return loto_result
            else:
                self.log(f"❌ Génération Loto échouée - Status: {response.status_code}", "ERROR")
                try:
                    error_detail = response.json()
                    self.log(f"   Détail: {error_detail.get('error', 'N/A')}")
                except:
                    pass
                return {}
                
        except Exception as e:
            self.log(f"❌ Erreur génération Loto: {e}", "ERROR")
            return {}
    
    def test_keno_analysis(self) -> Dict[str, Any]:
        """Test d'analyse Keno"""
        try:
            self.log("Test d'analyse Keno...")
            
            payload = {
                "strategy": "equilibre",
                "quick": True
            }
            
            response = self.session.post(
                f"{self.base_url}/api/keno/analyze", 
                json=payload, 
                timeout=45
            )
            
            if response.status_code == 200:
                keno_result = response.json()
                self.log("✅ Analyse Keno réussie", "SUCCESS")
                
                if 'recommendations' in keno_result:
                    recommendations = keno_result['recommendations']
                    self.log(f"   Recommandations: {len(recommendations)}")
                    
                return keno_result
            else:
                self.log(f"❌ Analyse Keno échouée - Status: {response.status_code}", "ERROR")
                try:
                    error_detail = response.json()
                    self.log(f"   Détail: {error_detail.get('error', 'N/A')}")
                except:
                    pass
                return {}
                
        except Exception as e:
            self.log(f"❌ Erreur analyse Keno: {e}", "ERROR")
            return {}
    
    def run_specific_test(self, endpoint: str) -> bool:
        """Exécute un test spécifique"""
        test_methods = {
            'health': self.test_health,
            'data': self.test_data_status,
            'loto': self.test_loto_generation,
            'keno': self.test_keno_analysis,
            'connection': self.test_connection
        }
        
        if endpoint in test_methods:
            result = test_methods[endpoint]()
            return bool(result)
        else:
            self.log(f"❌ Endpoint de test inconnu: {endpoint}", "ERROR")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Exécute tous les tests"""
        self.log("🚀 DÉBUT DES TESTS DE L'API LOTO/KENO", "SUCCESS")
        self.log("=" * 50)
        
        results = {}
        
        # Test 1: Connexion
        results['connection'] = self.test_connection()
        if not results['connection']:
            self.log("❌ Tests interrompus - Pas de connexion", "ERROR")
            return results
        
        # Test 2: Health Check
        results['health'] = bool(self.test_health())
        
        # Test 3: Statut des données
        results['data_status'] = bool(self.test_data_status())
        
        # Test 4: Génération Loto
        results['loto_generation'] = bool(self.test_loto_generation())
        
        # Test 5: Analyse Keno
        results['keno_analysis'] = bool(self.test_keno_analysis())
        
        # Résumé
        self.log("=" * 50)
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        self.log(f"📊 RÉSUMÉ: {passed_tests}/{total_tests} tests réussis", "SUCCESS")
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.log(f"   {test_name}: {status}")
        
        if passed_tests == total_tests:
            self.log("🎉 TOUS LES TESTS SONT PASSÉS !", "SUCCESS")
        else:
            self.log(f"⚠️  {total_tests - passed_tests} test(s) échoué(s)", "WARNING")
        
        return results


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Test de l'API Flask Loto/Keno",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Tests complets avec sortie détaillée
  python test_api.py --verbose
  
  # Test d'un endpoint spécifique
  python test_api.py --endpoint health
  python test_api.py --endpoint loto
  
  # Test avec URL personnalisée
  python test_api.py --url http://192.168.1.100:5000
        """
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:5000',
        help='URL de base de l\'API (défaut: http://localhost:5000)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Affichage détaillé'
    )
    
    parser.add_argument(
        '--endpoint', '-e',
        choices=['health', 'data', 'loto', 'keno', 'connection'],
        help='Test d\'un endpoint spécifique'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout global en secondes (défaut: 60)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialisation du testeur
        tester = APITester(base_url=args.url, verbose=args.verbose)
        
        print(f"🔍 Test de l'API Flask Loto/Keno")
        print(f"📡 URL: {args.url}")
        print(f"⏱️  Timeout: {args.timeout}s")
        print()
        
        # Configuration du timeout global
        requests.adapters.DEFAULT_RETRIES = 3
        
        # Exécution des tests
        if args.endpoint:
            # Test spécifique
            success = tester.run_specific_test(args.endpoint)
            return 0 if success else 1
        else:
            # Tous les tests
            results = tester.run_all_tests()
            
            # Code de sortie basé sur les résultats
            all_passed = all(results.values())
            return 0 if all_passed else 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrompus par l'utilisateur")
        return 130
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
