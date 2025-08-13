#!/usr/bin/env python3
"""
Assistant de Configuration - Système Loto/Keno
==============================================

Script interactif pour configurer facilement l'environnement
du système d'analyse Loto/Keno.

Usage:
    python setup_config.py

Author: Système Loto/Keno
Date: 13 août 2025
"""

import os
import sys
from pathlib import Path
import shutil


class Colors:
    """Codes couleurs pour l'affichage"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header():
    """Affiche l'en-tête"""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              🔧 ASSISTANT DE CONFIGURATION 🔧               ║")
    print("║                   Système Loto/Keno v2.0                     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print()


def check_existing_config():
    """Vérifie la configuration existante"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    print(f"{Colors.BOLD}📋 Vérification de la Configuration{Colors.ENDC}")
    print("=" * 50)
    
    if env_file.exists():
        print(f"✅ Fichier .env existant trouvé")
        return True
    elif env_example.exists():
        print(f"⚠️  Fichier .env manquant, mais .env.example disponible")
        return False
    else:
        print(f"❌ Aucun fichier de configuration trouvé")
        return False


def create_config_from_example():
    """Crée le fichier .env depuis l'exemple"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists():
        try:
            shutil.copy2(env_example, env_file)
            print(f"✅ Fichier .env créé depuis .env.example")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la création: {e}")
            return False
    else:
        print(f"❌ Fichier .env.example non trouvé")
        return False


def interactive_config():
    """Configuration interactive"""
    print(f"\n{Colors.BOLD}⚙️  Configuration Interactive{Colors.ENDC}")
    print("=" * 50)
    
    config = {}
    
    # Configuration de base
    print(f"\n{Colors.OKCYAN}📊 Configuration de Base{Colors.ENDC}")
    
    # Nombre de grilles Loto
    while True:
        try:
            grids = input("Nombre de grilles Loto par défaut (1-10) [3]: ").strip()
            if not grids:
                config['DEFAULT_LOTO_GRIDS'] = 3
                break
            grids = int(grids)
            if 1 <= grids <= 10:
                config['DEFAULT_LOTO_GRIDS'] = grids
                break
            else:
                print(f"{Colors.WARNING}Veuillez entrer un nombre entre 1 et 10{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.WARNING}Veuillez entrer un nombre valide{Colors.ENDC}")
    
    # Stratégie par défaut
    strategies = ["equilibre", "agressive", "conservatrice", "ml_focus"]
    print(f"\nStratégies disponibles: {', '.join(strategies)}")
    strategy = input("Stratégie Loto par défaut [equilibre]: ").strip()
    if not strategy or strategy not in strategies:
        strategy = "equilibre"
    config['DEFAULT_LOTO_STRATEGY'] = strategy
    
    # Configuration interface
    print(f"\n{Colors.OKCYAN}🎨 Configuration Interface{Colors.ENDC}")
    
    colors = input("Activer les couleurs dans le terminal ? (O/n) [O]: ").strip().lower()
    config['CLI_COLORS_ENABLED'] = colors != 'n'
    
    clear_screen = input("Nettoyer l'écran automatiquement ? (O/n) [O]: ").strip().lower()
    config['CLI_CLEAR_SCREEN'] = clear_screen != 'n'
    
    # Configuration ML
    print(f"\n{Colors.OKCYAN}🤖 Configuration Machine Learning{Colors.ENDC}")
    
    ml_enabled = input("Activer le machine learning ? (O/n) [O]: ").strip().lower()
    config['ML_ENABLED'] = ml_enabled != 'n'
    
    # Configuration performance
    print(f"\n{Colors.OKCYAN}⚡ Configuration Performance{Colors.ENDC}")
    
    threads = input("Nombre de threads CPU (laisser vide pour auto) [auto]: ").strip()
    if threads.isdigit():
        config['OMP_NUM_THREADS'] = int(threads)
        config['NUMBA_NUM_THREADS'] = int(threads)
        config['DUCKDB_THREADS'] = int(threads)
    
    memory = input("Limite mémoire DuckDB en GB [2]: ").strip()
    if not memory:
        memory = "2"
    config['DUCKDB_MEMORY_LIMIT'] = f"{memory}GB"
    
    # Configuration maintenance
    print(f"\n{Colors.OKCYAN}🧹 Configuration Maintenance{Colors.ENDC}")
    
    cleanup = input("Activer le nettoyage automatique ? (O/n) [O]: ").strip().lower()
    config['AUTO_CLEANUP'] = cleanup != 'n'
    
    if config['AUTO_CLEANUP']:
        days = input("Supprimer les fichiers plus anciens que X jours [30]: ").strip()
        if not days:
            days = "30"
        config['CLEANUP_DAYS_OLD'] = days
    
    return config


def apply_config(config):
    """Applique la configuration au fichier .env"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print(f"❌ Fichier .env non trouvé")
        return False
    
    try:
        # Lire le fichier existant
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Modifier les valeurs
        modified_lines = []
        for line in lines:
            modified = False
            for key, value in config.items():
                if line.startswith(f"{key}="):
                    # Convertir la valeur en string appropriée
                    if isinstance(value, bool):
                        value_str = "true" if value else "false"
                    else:
                        value_str = str(value)
                    
                    modified_lines.append(f"{key}={value_str}\n")
                    modified = True
                    break
            
            if not modified:
                modified_lines.append(line)
        
        # Écrire le fichier modifié
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        
        print(f"✅ Configuration appliquée avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'application: {e}")
        return False


def validate_config():
    """Valide la configuration"""
    print(f"\n{Colors.BOLD}🔍 Validation de la Configuration{Colors.ENDC}")
    print("=" * 50)
    
    try:
        from config_env import load_config, print_config_summary
        
        # Charger et afficher le résumé
        load_config()
        print_config_summary()
        
        return True
        
    except ImportError:
        print(f"{Colors.WARNING}⚠️  Module config_env non disponible{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {e}")
        return False


def main():
    """Fonction principale"""
    print_header()
    
    # Vérifier la configuration existante
    has_config = check_existing_config()
    
    if not has_config:
        print(f"\n{Colors.WARNING}🔧 Configuration manquante{Colors.ENDC}")
        create = input("Créer la configuration depuis l'exemple ? (O/n) [O]: ").strip().lower()
        
        if create != 'n':
            if create_config_from_example():
                has_config = True
            else:
                print(f"\n{Colors.FAIL}❌ Impossible de créer la configuration{Colors.ENDC}")
                return 1
    
    if has_config:
        print(f"\n{Colors.OKGREEN}✅ Configuration trouvée{Colors.ENDC}")
        
        # Proposer la configuration interactive
        configure = input("Voulez-vous configurer interactivement ? (o/N) [N]: ").strip().lower()
        
        if configure in ['o', 'oui', 'y', 'yes']:
            config = interactive_config()
            
            print(f"\n{Colors.BOLD}📝 Résumé de la Configuration{Colors.ENDC}")
            print("=" * 30)
            for key, value in config.items():
                print(f"   {key}: {value}")
            
            apply = input(f"\nAppliquer cette configuration ? (O/n) [O]: ").strip().lower()
            if apply != 'n':
                apply_config(config)
    
    # Validation finale
    print(f"\n{Colors.BOLD}🎯 Test Final{Colors.ENDC}")
    print("=" * 20)
    
    if validate_config():
        print(f"\n{Colors.OKGREEN}🎉 Configuration terminée avec succès !{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Prochaines étapes :{Colors.ENDC}")
        print("  1️⃣  Lancez le menu : ./lancer_menu.sh")
        print("  2️⃣  Ou direct Python : python cli_menu.py")
        print("  3️⃣  Tests du système : python test/run_all_tests.py --essential")
        return 0
    else:
        print(f"\n{Colors.WARNING}⚠️  Configuration incomplète{Colors.ENDC}")
        print("Veuillez vérifier le fichier .env manuellement")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}🛑 Configuration annulée par l'utilisateur{Colors.ENDC}")
        sys.exit(1)
