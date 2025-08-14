#!/usr/bin/env python3
"""
Test de l'API Loto/Keno
=======================

Script pour tester les endpoints de l'API Flask.
"""

import requests
import json
import time
import sys
from threading import Thread
import subprocess

def start_api_server():
    """Démarre le serveur API en arrière-plan"""
    print("🚀 Démarrage du serveur API...")
    process = subprocess.Popen([
        sys.executable, "api/app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Attendre que le serveur soit prêt
    time.sleep(3)
    return process

def test_health_endpoint():
    """Test de l'endpoint de santé"""
    print("\n🔍 Test endpoint /api/health")
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Santé API: {data.get('status', 'unknown')}")
            if 'services' in data:
                for service, status in data['services'].items():
                    status_icon = "✅" if status else "❌"
                    print(f"   {status_icon} Service {service}: {'OK' if status else 'FAIL'}")
        else:
            print(f"   ❌ Erreur: {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")

def test_config_endpoint():
    """Test de l'endpoint de configuration"""
    print("\n🔍 Test endpoint /api/config")
    try:
        response = requests.get("http://localhost:5000/api/config", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Configuration chargée:")
            print(f"   - Version: {data.get('version', 'N/A')}")
            print(f"   - Environnement: {data.get('environment', 'N/A')}")
            print(f"   - Config trouvée: {data.get('config_found', False)}")
        else:
            print(f"   ❌ Erreur: {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")

def test_data_status_endpoint():
    """Test de l'endpoint de statut des données"""
    print("\n🔍 Test endpoint /api/data/status")
    try:
        response = requests.get("http://localhost:5000/api/data/status", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Statut des données:")
            for source, info in data.items():
                if isinstance(info, dict):
                    status = "✅" if info.get('exists', False) else "❌"
                    size = info.get('size_mb', 0)
                    print(f"   {status} {source}: {size:.1f} MB")
        else:
            print(f"   ❌ Erreur: {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")

def test_loto_strategies_endpoint():
    """Test de l'endpoint des stratégies Loto"""
    print("\n🔍 Test endpoint /api/loto/strategies")
    try:
        response = requests.get("http://localhost:5000/api/loto/strategies", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Stratégies disponibles: {len(data.get('strategies', []))}")
            for i, strategy in enumerate(data.get('strategies', [])[:3]):  # Afficher les 3 premières
                print(f"   - {strategy.get('id', f'strategy_{i}')}: {strategy.get('name', 'Sans nom')}")
        else:
            print(f"   ❌ Erreur: {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")

def test_loto_generate_endpoint():
    """Test de l'endpoint de génération Loto"""
    print("\n🔍 Test endpoint /api/loto/generate")
    try:
        payload = {
            "count": 2,
            "strategy": "equilibre",  # Utiliser une stratégie valide
            "export_csv": False
        }
        response = requests.post(
            "http://localhost:5000/api/loto/generate",
            json=payload,
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Grilles générées: {len(data.get('grids', []))}")
            grids = data.get('grids', [])
            if grids:
                print(f"   - Exemple: {grids[0][:5]}...")  # Premiers numéros de la première grille
            print(f"   - Stratégie: {data.get('strategy', 'N/A')}")
            print(f"   - Temps: {data.get('execution_time', 'N/A')}s")
        else:
            error_data = response.json() if response.content else {"error": "Erreur inconnue"}
            print(f"   ❌ Erreur: {error_data.get('error', 'Erreur inconnue')}")
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")

def test_documentation_page():
    """Test de la page de documentation"""
    print("\n🔍 Test page de documentation /")
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            content = response.text
            print(f"   ✅ Page HTML: {len(content)} caractères")
            if "API Loto/Keno" in content:
                print("   ✅ Contenu de documentation détecté")
            else:
                print("   ⚠️  Contenu de documentation non détecté")
        else:
            print(f"   ❌ Erreur: {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")

def main():
    """Fonction principale de test"""
    print("🧪 Test de l'API Loto/Keno")
    print("=" * 30)
    
    # Vérifier si l'API est déjà en cours d'exécution
    api_running = False
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=2)
        api_running = True
        print("✅ API déjà en cours d'exécution")
    except:
        print("⚠️  API non démarrée, démarrage automatique...")
        server_process = start_api_server()
        time.sleep(3)  # Attendre le démarrage
    
    # Exécuter les tests
    try:
        test_documentation_page()
        test_health_endpoint()
        test_config_endpoint()
        test_data_status_endpoint()
        test_loto_strategies_endpoint()
        test_loto_generate_endpoint()
        
        print("\n" + "=" * 50)
        print("🎉 Tests terminés avec succès !")
        print("🌐 API accessible sur http://localhost:5000")
        
    finally:
        # Arrêter le serveur si nous l'avons démarré
        if not api_running and 'server_process' in locals():
            print("\n🔄 Arrêt du serveur de test...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
