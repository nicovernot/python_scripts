#!/usr/bin/env python3
"""
Test Rapide du Menu CLI
======================

Script de test pour vérifier le bon fonctionnement du menu CLI.
"""

import subprocess
import sys
import os
from pathlib import Path


def test_cli_menu():
    """Test basique du menu CLI"""
    print("🧪 Test du Menu CLI...")
    
    # Vérifier que le fichier existe
    cli_path = Path("cli_menu.py")
    if not cli_path.exists():
        print("❌ cli_menu.py non trouvé")
        return False
        
    # Test d'import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli_menu", cli_path)
        cli_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_module)
        print("✅ Module CLI importé avec succès")
        
        # Test d'instanciation
        menu = cli_module.LotoKenoMenu()
        print("✅ Classe LotoKenoMenu instanciée")
        
        # Test des méthodes de base
        menu.clear_screen()
        print("✅ clear_screen() fonctionne")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False


def test_launcher_script():
    """Test du script de lancement"""
    print("\n🧪 Test du Script de Lancement...")
    
    launcher_path = Path("lancer_menu.sh")
    if not launcher_path.exists():
        print("❌ lancer_menu.sh non trouvé")
        return False
        
    # Vérifier les permissions
    if not os.access(launcher_path, os.X_OK):
        print("⚠️  Permissions d'exécution manquantes")
        os.chmod(launcher_path, 0o755)
        print("✅ Permissions corrigées")
        
    print("✅ Script de lancement OK")
    return True


def main():
    """Test principal"""
    print("🎯 Test du Système CLI Loto/Keno")
    print("=" * 40)
    
    success = True
    
    # Test du menu CLI
    if not test_cli_menu():
        success = False
        
    # Test du script de lancement
    if not test_launcher_script():
        success = False
        
    print("\n" + "=" * 40)
    if success:
        print("✅ Tous les tests CLI sont passés !")
        print("\n🚀 Pour utiliser le menu :")
        print("   ./lancer_menu.sh")
        print("   ou")
        print("   python cli_menu.py")
    else:
        print("❌ Certains tests ont échoué")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
