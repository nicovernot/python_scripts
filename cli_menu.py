#!/usr/bin/env python3
"""
CLI Menu Interactif pour le Système Loto/Keno
============================================

Menu principal pour faciliter l'utilisation des analyses Loto et Keno.
Permet de lancer toutes les commandes principales sans mémoriser les arguments.

Usage:
    python cli_menu.py

Author: Système Loto/Keno
Date: 13 août 2025
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import du gestionnaire de configuration
try:
    from config_env import load_config, get_config, get_config_path, get_config_bool
except ImportError:
    print("⚠️  Gestionnaire de configuration non disponible, utilisation des valeurs par défaut")
    def load_config(): return None
    def get_config(key, default=None): return default
    def get_config_path(key, default=None): return Path(default) if default else None
    def get_config_bool(key, default=False): return default


class Colors:
    """Codes couleurs pour l'affichage terminal"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LotoKenoMenu:
    """Menu interactif principal pour le système Loto/Keno"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        
        # Configuration de l'environnement Python
        venv_python = self.base_path / "venv" / "bin" / "python"
        self.python_path = str(venv_python) if venv_python.exists() else "python"
        
        # Chargement de la configuration
        self.config = load_config()
        
        # Chemins depuis la configuration
        self.loto_csv = get_config_path('LOTO_CSV_PATH') or self.base_path / "loto" / "loto_data" / "loto_201911.csv"
        self.keno_csv = get_config_path('KENO_CSV_PATH') or self.base_path / "keno" / "keno_data" / "keno_consolidated.csv"
        
        # Configuration CLI
        self.colors_enabled = get_config_bool('CLI_COLORS_ENABLED', True)
        self.clear_screen_enabled = get_config_bool('CLI_CLEAR_SCREEN', True)
        self.show_status = get_config_bool('CLI_SHOW_STATUS', True)
        
        # Valeurs par défaut depuis la configuration
        self.default_loto_grids = get_config('DEFAULT_LOTO_GRIDS', 3)
        self.default_loto_strategy = get_config('DEFAULT_LOTO_STRATEGY', 'equilibre')
        self.loto_config_file = get_config('LOTO_CONFIG_FILE', 'loto/strategies.yml')
        
    def clear_screen(self):
        """Nettoie l'écran si activé dans la configuration"""
        if self.clear_screen_enabled:
            os.system('clear' if os.name == 'posix' else 'cls')
    
    def format_date_range(self, date_range):
        """Formate une plage de dates au format MM/YYYY → MM/YYYY"""
        if date_range is None or len(date_range) != 2:
            return None
            
        first_date, last_date = date_range
        
        try:
            # Essayer de parser différents formats de date
            from datetime import datetime
            
            # Format ISO (YYYY-MM-DD) pour Keno
            if '-' in str(first_date) and len(str(first_date)) == 10:
                first_dt = datetime.strptime(str(first_date), '%Y-%m-%d')
                last_dt = datetime.strptime(str(last_date), '%Y-%m-%d')
            # Format français (DD/MM/YYYY) pour Loto
            elif '/' in str(first_date):
                first_dt = datetime.strptime(str(first_date), '%d/%m/%Y')
                last_dt = datetime.strptime(str(last_date), '%d/%m/%Y')
            else:
                return None
                
            first_formatted = first_dt.strftime('%m/%Y')
            last_formatted = last_dt.strftime('%m/%Y')
            
            return f"{first_formatted} → {last_formatted}"
        except:
            return None

    def get_csv_date_range(self, csv_path):
        """Extrait la première et dernière date d'un fichier CSV"""
        try:
            # Essayer d'abord avec délimiteur virgule (pour Keno)
            df = pd.read_csv(csv_path, nrows=1000)
            if 'date' in df.columns:
                first_date = df['date'].iloc[0]
                df_tail = pd.read_csv(csv_path).tail(1)
                last_date = df_tail['date'].iloc[0]
                return first_date, last_date
        except:
            pass
        
        try:
            # Essayer avec délimiteur point-virgule (pour Loto)
            df = pd.read_csv(csv_path, nrows=1000, delimiter=';')
            
            date_column = None
            if 'date' in df.columns:
                date_column = 'date'
            elif 'date_de_tirage' in df.columns:
                date_column = 'date_de_tirage'
            
            if date_column:
                first_date = df[date_column].iloc[0]
                df_tail = pd.read_csv(csv_path, delimiter=';').tail(1)
                last_date = df_tail[date_column].iloc[0]
                return first_date, last_date
        except:
            pass
        
        return None, None
        
    def print_header(self):
        """Affiche l'en-tête du menu"""
        self.clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║                   🎲 SYSTÈME LOTO/KENO 🎰                    ║")
        print("║               Menu Interactif d'Analyse Avancée              ║")
        print("║                     🚀 Version Améliorée                     ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKCYAN}📅 Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}🆕 Nouvelles fonctionnalités: Générateur adaptatif V2 (options 44-46){Colors.ENDC}")
        print()
        
    def print_status(self):
        """Affiche le statut des fichiers de données"""
        print(f"{Colors.BOLD}📊 Statut des Données:{Colors.ENDC}")
        
        # Statut Loto
        if self.loto_csv.exists():
            size = self.loto_csv.stat().st_size / (1024*1024)
            mtime = datetime.fromtimestamp(self.loto_csv.stat().st_mtime)
            date_range_tuple = self.get_csv_date_range(self.loto_csv)
            date_info = ""
            if date_range_tuple and date_range_tuple[0] and date_range_tuple[1]:
                date_range = self.format_date_range(date_range_tuple)
                if date_range:
                    date_info = f", {date_range}"
            print(f"  🎲 Loto:  {Colors.OKGREEN}✓ Disponible{Colors.ENDC} ({size:.1f}MB, MAJ: {mtime.strftime('%d/%m/%Y')}{date_info})")
        else:
            print(f"  🎲 Loto:  {Colors.FAIL}✗ Manquant{Colors.ENDC}")
            
        # Statut Keno
        if self.keno_csv.exists():
            size = self.keno_csv.stat().st_size / (1024*1024)
            mtime = datetime.fromtimestamp(self.keno_csv.stat().st_mtime)
            date_range_tuple = self.get_csv_date_range(self.keno_csv)
            date_info = ""
            if date_range_tuple and date_range_tuple[0] and date_range_tuple[1]:
                date_range = self.format_date_range(date_range_tuple)
                if date_range:
                    date_info = f", {date_range}"
            print(f"  🎰 Keno:  {Colors.OKGREEN}✓ Disponible{Colors.ENDC} ({size:.1f}MB, MAJ: {mtime.strftime('%d/%m/%Y')}{date_info})")
        else:
            print(f"  🎰 Keno:  {Colors.FAIL}✗ Manquant{Colors.ENDC}")
            
        print()
        
    def print_menu(self):
        """Affiche le menu principal"""
        print(f"{Colors.BOLD}📋 Menu Principal:{Colors.ENDC}")
        print()
        
        print(f"{Colors.HEADER}📥 TÉLÉCHARGEMENT DES DONNÉES{Colors.ENDC}")
        print("  1️⃣  Télécharger les données Loto (FDJ)")
        print("  2️⃣  Télécharger les données Keno (FDJ)")
        print("  3️⃣  Mettre à jour toutes les données")
        print("  4️⃣7️⃣ 🔄 Convertir CSV Loto vers Parquet (optimisation)")
        print()
        
        print(f"{Colors.OKBLUE}🎲 ANALYSE LOTO{Colors.ENDC}")
        print("  4️⃣  Générer 3 grilles Loto (rapide)")
        print("  5️⃣  Générer 5 grilles Loto (complet)")
        print("  6️⃣  Générer grilles avec visualisations")
        print("  7️⃣  Analyse Loto personnalisée")
        print("  2️⃣2️⃣ Générateur Loto avancé (ML + IA)")
        print("  4️⃣4️⃣ 🚀 Générateur Loto adaptatif V2 (Auto-amélioration)")
        print("  4️⃣5️⃣ 🧠 Training ML adaptatif (ré-entraînement intelligent)")
        print("  4️⃣6️⃣ ⚡ Générateur sans exclusion (tous les 49 numéros)")
        print()
        
        print(f"{Colors.BOLD}🎯 TOP NUMÉROS ÉQUILIBRÉS{Colors.ENDC}")
        print("  2️⃣8️⃣ 🏆 TOP 25 Loto équilibrés (stratégie optimisée)")
        print("  2️⃣9️⃣ 🏆 TOP 30 Keno équilibrés (stratégie optimisée)")
        print("  3️⃣0️⃣ 📊 Voir TOP 25 Loto (dernière génération)")
        print("  3️⃣1️⃣ 📊 Voir TOP 30 Keno (dernière génération)")
        print()
        
        print(f"{Colors.BOLD}🎲 GÉNÉRATION GRILLES TOP CSV{Colors.ENDC}")
        print("  3️⃣2️⃣ 🎯 Grilles Loto TOP 25 → CSV optimisé")
        print("  3️⃣3️⃣ 🎯 Grilles Keno TOP 30 → CSV optimisé")
        print("  3️⃣4️⃣ 📝 Grilles Loto TOP 25 → Markdown détaillé")
        print("  3️⃣5️⃣ 📝 Grilles Keno TOP 30 → Markdown détaillé")
        print()
        
        print(f"{Colors.OKCYAN}🎰 ANALYSE KENO{Colors.ENDC}")
        print("  8️⃣  Analyse Keno complète (nouveaux algorithmes)")
        print("  9️⃣  Pipeline Keno complet avec visualisations + nettoyage auto")
        print("  1️⃣0️⃣ Analyse Keno personnalisée")
        print("  2️⃣4️⃣ Analyse avancée DuckDB (11 stratégies + optimisé)")
        print("  2️⃣5️⃣ Générateur Keno avancé (ML + IA)")
        print("  2️⃣6️⃣ 📊 Statistiques Keno complètes (CSV + graphiques)")
        print("  2️⃣7️⃣ ⚡ Analyse Keno rapide (recommandations express)")
        print()
        
        print(f"{Colors.BOLD}🔥 KENO PATTERNS AVANCÉS{Colors.ENDC}")
        print("  3️⃣6️⃣ 🧠 Générateur patterns avancés (DuckDB + 15 patterns)")
        print("  3️⃣7️⃣ 📊 TOP 30 numéros patterns → CSV")
        print("  3️⃣8️⃣ 🎯 Générateur personnalisé (paramètres libres)")
        print("  3️⃣9️⃣ 🔍 Test des patterns (vérification complète)")
        print()
        
        print(f"{Colors.BOLD}🎯 SYSTÈME RÉDUCTEUR INTELLIGENT{Colors.ENDC}")
        print("  4️⃣0️⃣ 🧠 Générateur intelligent avec PuLP")
        print("  4️⃣1️⃣ 📊 Systèmes réducteurs (Bas/Moyen/Haut)")
        print("  4️⃣2️⃣ 🎲 Génération basée sur paires fréquentes")
        print("  4️⃣3️⃣ 📊 Générer TOP 30 Keno (KenoGeneratorAdvanced)")
        print()
        
        print(f"{Colors.OKGREEN}🧪 TESTS ET MAINTENANCE{Colors.ENDC}")
        print("  1️⃣1️⃣ Tests complets du système")
        print("  1️⃣2️⃣ Tests essentiels uniquement")
        print("  1️⃣3️⃣ Test de performance")
        print("  1️⃣4️⃣ Nettoyage et optimisation")
        print()
        
        print(f"{Colors.HEADER}🌐 API FLASK{Colors.ENDC}")
        print("  1️⃣5️⃣ Lancer l'API Flask")
        print("  1️⃣6️⃣ Tester l'API Flask")
        print()
        
        print(f"{Colors.OKCYAN}🎯 SYSTÈMES RÉDUITS{Colors.ENDC}")
        print("  1️⃣7️⃣ Générateur de grilles (système réduit)")
        print("  1️⃣8️⃣ Générateur personnalisé")
        print()
        
        print(f"{Colors.WARNING}📊 CONSULTATION DES RÉSULTATS{Colors.ENDC}")
        print("  1️⃣9️⃣ Voir les dernières grilles Loto")
        print("  2️⃣0️⃣ Voir les recommandations Keno")
        print("  2️⃣1️⃣ Ouvrir dossier des graphiques")
        print("  2️⃣3️⃣ Statut détaillé du système")
        print()
        
        print(f"{Colors.FAIL}🚪 QUITTER{Colors.ENDC}")
        print("  0️⃣  Quitter le programme")
        print()
        print("═" * 63)
        
    def execute_command(self, command, description):
        """Exécute une commande et affiche le résultat"""
        print(f"\n{Colors.BOLD}🚀 {description}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Commande: {command}{Colors.ENDC}")
        print("─" * 50)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=self.base_path,
                capture_output=False,
                text=True
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"\n{Colors.OKGREEN}✅ Succès !{Colors.ENDC} (Durée: {execution_time:.1f}s)")
            else:
                print(f"\n{Colors.FAIL}❌ Erreur (Code: {result.returncode}){Colors.ENDC}")
                
        except Exception as e:
            print(f"\n{Colors.FAIL}💥 Exception: {str(e)}{Colors.ENDC}")
            
        print("\n" + "═" * 50)
        input(f"\n{Colors.BOLD}Appuyez sur Entrée pour continuer...{Colors.ENDC}")

    def handle_grilles_loto_top_csv(self):
        """Génération grilles Loto TOP 25 → CSV optimisé"""
        print(f"\n{Colors.BOLD}🎯 GÉNÉRATION GRILLES LOTO TOP 25 → CSV{Colors.ENDC}")
        print("Génération de grilles optimisées basées sur les 25 meilleurs numéros Loto")
        print("  • Utilise le fichier TOP 25 le plus récent")
        print("  • Optimisation PuLP ou algorithme glouton")
        print("  • Export CSV structuré avec analyses détaillées")
        print()
        
        # Configuration de la génération
        try:
            nb_grilles = int(input("Nombre de grilles (5-50) [10]: ").strip() or "10")
            if nb_grilles < 5 or nb_grilles > 50:
                raise ValueError("Nombre invalide")
        except:
            nb_grilles = 10
        
        try:
            top_nombres = int(input("Nombre de numéros TOP à utiliser (10-25) [20]: ").strip() or "20")
            if top_nombres < 10 or top_nombres > 25:
                raise ValueError("Nombre invalide")
        except:
            top_nombres = 20
        
        try:
            taille_grille = int(input("Numéros par grille (5-10) [5]: ").strip() or "5")
            if taille_grille < 5 or taille_grille > 10:
                raise ValueError("Taille invalide")
        except:
            taille_grille = 5
        
        # Choix de l'optimisation
        print(f"\n{Colors.OKBLUE}⚙️ Type d'optimisation:{Colors.ENDC}")
        print("1️⃣  PuLP (optimisation linéaire, recommandé)")
        print("2️⃣  Glouton (algorithme rapide)")
        
        optim_choice = input("Optimisation (1-2) [1]: ").strip() or "1"
        optimisation = "pulp" if optim_choice == "1" else "glouton"
        
        # Construction et exécution de la commande
        command = f"python grilles/generateur_grilles.py --top-csv --jeu loto --top-nombres {top_nombres} --optimisation {optimisation} --taille-grille-loto {taille_grille} --grilles {nb_grilles} --export --format csv --verbose"
        
        print(f"\n{Colors.OKGREEN}✅ Configuration:{Colors.ENDC}")
        print(f"   Jeu: Loto")
        print(f"   Grilles: {nb_grilles}")
        print(f"   TOP numéros: {top_nombres}")
        print(f"   Taille grilles: {taille_grille}")
        print(f"   Optimisation: {optimisation}")
        print(f"   Format: CSV")
        
        confirm = input(f"\n{Colors.BOLD}Lancer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm not in ['n', 'non', 'no']:
            self.execute_command(command, "Génération Grilles Loto TOP 25 → CSV")
        else:
            print("Opération annulée.")
            self.wait_and_continue()

    def handle_grilles_keno_top_csv(self):
        """Génération grilles Keno TOP 30 → CSV optimisé"""
        print(f"\n{Colors.BOLD}🎯 GÉNÉRATION GRILLES KENO TOP 30 → CSV{Colors.ENDC}")
        print("Génération de grilles optimisées basées sur les 30 meilleurs numéros Keno")
        print("  • Utilise le fichier TOP 30 le plus récent")
        print("  • Optimisation PuLP ou algorithme glouton")
        print("  • Export CSV structuré avec analyses détaillées")
        print()
        
        # Configuration de la génération
        try:
            nb_grilles = int(input("Nombre de grilles (5-50) [12]: ").strip() or "12")
            if nb_grilles < 5 or nb_grilles > 50:
                raise ValueError("Nombre invalide")
        except:
            nb_grilles = 12
        
        try:
            top_nombres = int(input("Nombre de numéros TOP à utiliser (15-30) [25]: ").strip() or "25")
            if top_nombres < 15 or top_nombres > 30:
                raise ValueError("Nombre invalide")
        except:
            top_nombres = 25
        
        try:
            taille_grille = int(input("Numéros par grille (6-10) [8]: ").strip() or "8")
            if taille_grille < 6 or taille_grille > 10:
                raise ValueError("Taille invalide")
        except:
            taille_grille = 8
        
        # Choix de l'optimisation
        print(f"\n{Colors.OKBLUE}⚙️ Type d'optimisation:{Colors.ENDC}")
        print("1️⃣  PuLP (optimisation linéaire, recommandé)")
        print("2️⃣  Glouton (algorithme rapide)")
        
        optim_choice = input("Optimisation (1-2) [1]: ").strip() or "1"
        optimisation = "pulp" if optim_choice == "1" else "glouton"
        
        # Construction et exécution de la commande
        command = f"python grilles/generateur_grilles.py --top-csv --jeu keno --top-nombres {top_nombres} --optimisation {optimisation} --taille-grille-keno {taille_grille} --grilles {nb_grilles} --export --format csv --verbose"
        
        print(f"\n{Colors.OKGREEN}✅ Configuration:{Colors.ENDC}")
        print(f"   Jeu: Keno")
        print(f"   Grilles: {nb_grilles}")
        print(f"   TOP numéros: {top_nombres}")
        print(f"   Taille grilles: {taille_grille}")
        print(f"   Optimisation: {optimisation}")
        print(f"   Format: CSV")
        
        confirm = input(f"\n{Colors.BOLD}Lancer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm not in ['n', 'non', 'no']:
            self.execute_command(command, "Génération Grilles Keno TOP 30 → CSV")
        else:
            print("Opération annulée.")
            self.wait_and_continue()

    def handle_grilles_loto_top_markdown(self):
        """Génération grilles Loto TOP 25 → Markdown détaillé"""
        print(f"\n{Colors.BOLD}📝 GÉNÉRATION GRILLES LOTO TOP 25 → MARKDOWN{Colors.ENDC}")
        print("Génération de grilles optimisées avec rapport Markdown détaillé")
        print("  • Utilise le fichier TOP 25 le plus récent")
        print("  • Optimisation PuLP ou algorithme glouton")
        print("  • Rapport Markdown avec tableaux et analyses")
        print("  • Conseils et recommandations d'utilisation")
        print()
        
        # Configuration de la génération
        try:
            nb_grilles = int(input("Nombre de grilles (5-30) [8]: ").strip() or "8")
            if nb_grilles < 5 or nb_grilles > 30:
                raise ValueError("Nombre invalide")
        except:
            nb_grilles = 8
        
        try:
            top_nombres = int(input("Nombre de numéros TOP à utiliser (10-25) [18]: ").strip() or "18")
            if top_nombres < 10 or top_nombres > 25:
                raise ValueError("Nombre invalide")
        except:
            top_nombres = 18
        
        try:
            taille_grille = int(input("Numéros par grille (5-8) [5]: ").strip() or "5")
            if taille_grille < 5 or taille_grille > 8:
                raise ValueError("Taille invalide")
        except:
            taille_grille = 5
        
        # Choix de l'optimisation
        print(f"\n{Colors.OKBLUE}⚙️ Type d'optimisation:{Colors.ENDC}")
        print("1️⃣  PuLP (optimisation linéaire, recommandé)")
        print("2️⃣  Glouton (algorithme rapide)")
        
        optim_choice = input("Optimisation (1-2) [1]: ").strip() or "1"
        optimisation = "pulp" if optim_choice == "1" else "glouton"
        
        # Construction et exécution de la commande
        command = f"python grilles/generateur_grilles.py --top-csv --jeu loto --top-nombres {top_nombres} --optimisation {optimisation} --taille-grille-loto {taille_grille} --grilles {nb_grilles} --export --format markdown --verbose"
        
        print(f"\n{Colors.OKGREEN}✅ Configuration:{Colors.ENDC}")
        print(f"   Jeu: Loto")
        print(f"   Grilles: {nb_grilles}")
        print(f"   TOP numéros: {top_nombres}")
        print(f"   Taille grilles: {taille_grille}")
        print(f"   Optimisation: {optimisation}")
        print(f"   Format: Markdown")
        
        confirm = input(f"\n{Colors.BOLD}Lancer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm not in ['n', 'non', 'no']:
            self.execute_command(command, "Génération Grilles Loto TOP 25 → Markdown")
        else:
            print("Opération annulée.")
            self.wait_and_continue()

    def handle_grilles_keno_top_markdown(self):
        """Génération grilles Keno TOP 30 → Markdown détaillé"""
        print(f"\n{Colors.BOLD}📝 GÉNÉRATION GRILLES KENO TOP 30 → MARKDOWN{Colors.ENDC}")
        print("Génération de grilles optimisées avec rapport Markdown détaillé")
        print("  • Utilise le fichier TOP 30 le plus récent")
        print("  • Optimisation PuLP ou algorithme glouton")
        print("  • Rapport Markdown avec tableaux et analyses")
        print("  • Conseils et recommandations d'utilisation")
        print()
        
        # Configuration de la génération
        try:
            nb_grilles = int(input("Nombre de grilles (5-30) [10]: ").strip() or "10")
            if nb_grilles < 5 or nb_grilles > 30:
                raise ValueError("Nombre invalide")
        except:
            nb_grilles = 10
        
        try:
            top_nombres = int(input("Nombre de numéros TOP à utiliser (15-30) [22]: ").strip() or "22")
            if top_nombres < 15 or top_nombres > 30:
                raise ValueError("Nombre invalide")
        except:
            top_nombres = 22
        
        try:
            taille_grille = int(input("Numéros par grille (6-10) [9]: ").strip() or "9")
            if taille_grille < 6 or taille_grille > 10:
                raise ValueError("Taille invalide")
        except:
            taille_grille = 9
        
        # Choix de l'optimisation
        print(f"\n{Colors.OKBLUE}⚙️ Type d'optimisation:{Colors.ENDC}")
        print("1️⃣  PuLP (optimisation linéaire, recommandé)")
        print("2️⃣  Glouton (algorithme rapide)")
        
        optim_choice = input("Optimisation (1-2) [1]: ").strip() or "1"
        optimisation = "pulp" if optim_choice == "1" else "glouton"
        
        # Construction et exécution de la commande
        command = f"python grilles/generateur_grilles.py --top-csv --jeu keno --top-nombres {top_nombres} --optimisation {optimisation} --taille-grille-keno {taille_grille} --grilles {nb_grilles} --export --format markdown --verbose"
        
        print(f"\n{Colors.OKGREEN}✅ Configuration:{Colors.ENDC}")
        print(f"   Jeu: Keno")
        print(f"   Grilles: {nb_grilles}")
        print(f"   TOP numéros: {top_nombres}")
        print(f"   Taille grilles: {taille_grille}")
        print(f"   Optimisation: {optimisation}")
        print(f"   Format: Markdown")
        
        confirm = input(f"\n{Colors.BOLD}Lancer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm not in ['n', 'non', 'no']:
            self.execute_command(command, "Génération Grilles Keno TOP 30 → Markdown")
        else:
            print("Opération annulée.")
            self.wait_and_continue()
        
    def wait_and_continue(self, message="Appuyez sur Entrée pour continuer..."):
        """Pause avec message"""
        input(f"\n{Colors.BOLD}{message}{Colors.ENDC}")
        
    def show_file_content(self, file_path, max_lines=50):
        """Affiche le contenu d'un fichier"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print(f"\n{Colors.BOLD}📄 Contenu de {file_path.name}:{Colors.ENDC}")
            print("─" * 50)
            
            for i, line in enumerate(lines[:max_lines], 1):
                print(f"{i:3d}: {line.rstrip()}")
                
            if len(lines) > max_lines:
                print(f"\n{Colors.WARNING}... ({len(lines) - max_lines} lignes supplémentaires){Colors.ENDC}")
                
        except FileNotFoundError:
            print(f"\n{Colors.FAIL}❌ Fichier non trouvé: {file_path}{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.FAIL}💥 Erreur: {str(e)}{Colors.ENDC}")
            
    def handle_loto_custom(self):
        """Menu personnalisé pour l'analyse Loto"""
        print(f"\n{Colors.BOLD}🎲 Configuration Loto Personnalisée{Colors.ENDC}")
        print()
        
        # Nombre de grilles
        while True:
            try:
                default_grids = self.default_loto_grids
                nb_grilles = input(f"Nombre de grilles à générer (1-10) [{default_grids}]: ").strip()
                if not nb_grilles:
                    nb_grilles = default_grids
                else:
                    nb_grilles = int(nb_grilles)
                    
                if 1 <= nb_grilles <= 10:
                    break
                else:
                    print(f"{Colors.WARNING}Veuillez entrer un nombre entre 1 et 10{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.WARNING}Veuillez entrer un nombre valide{Colors.ENDC}")
                
        # Stratégie
        strategies = ["equilibre", "agressive", "conservatrice", "ml_focus"]
        print(f"\nStratégies disponibles: {', '.join(strategies)}")
        default_strategy = self.default_loto_strategy
        strategy = input(f"Stratégie [{default_strategy}]: ").strip()
        if not strategy or strategy not in strategies:
            strategy = default_strategy
            
        # Options
        plots = input("Générer les visualisations ? (o/N) [N]: ").strip().lower()
        export_stats = input("Exporter les statistiques ? (o/N) [N]: ").strip().lower()
        
        # Construction de la commande
        command = f"python loto/duckdb_loto.py --csv {self.loto_csv} --grids {nb_grilles} --strategy {strategy} --config {self.loto_config_file}"
        
        if plots in ['o', 'oui', 'y', 'yes']:
            command += " --plots"
            
        if export_stats in ['o', 'oui', 'y', 'yes']:
            command += " --export-stats"
            
        self.execute_command(command, f"Génération Loto Personnalisée ({nb_grilles} grilles, {strategy})")
        
    def handle_keno_custom(self):
        """Menu personnalisé pour l'analyse Keno"""
        print(f"\n{Colors.BOLD}🎰 Configuration Keno Personnalisée{Colors.ENDC}")
        print()
        
        # Options du nouveau système
        print("Choisissez votre analyse :")
        print(f"  {Colors.OKGREEN}1{Colors.ENDC} - Extraction seule")
        print(f"  {Colors.OKBLUE}2{Colors.ENDC} - Analyse statistique seule")
        print(f"  {Colors.OKCYAN}3{Colors.ENDC} - Génération de grilles")
        print(f"  {Colors.WARNING}4{Colors.ENDC} - Pipeline complet personnalisé")
        print(f"  {Colors.HEADER}5{Colors.ENDC} - Analyse DuckDB avancée")
        
        choice = input(f"\n{Colors.BOLD}Votre choix (1-5): {Colors.ENDC}").strip()
        
        if choice == "1":
            self.execute_command("python keno_cli.py extract", "Extraction des Données Keno")
            
        elif choice == "2":
            self.execute_command("python keno_cli.py analyze", "Analyse Statistique Keno")
            
        elif choice == "3":
            nb_grilles = input("Nombre de grilles à générer (1-10) [3]: ").strip() or "3"
            command = f"python keno_cli.py generate --grids {nb_grilles}"
            self.execute_command(command, f"Génération de {nb_grilles} Grilles Keno")
            
        elif choice == "4":
            nb_grilles = input("Nombre de grilles pour le pipeline complet (1-10) [5]: ").strip() or "5"
            command = f"python keno_cli.py all --grids {nb_grilles}"
            self.execute_command(command, f"Pipeline Complet Keno ({nb_grilles} grilles)")
            
        elif choice == "5":
            # Analyse DuckDB avancée avec options
            plots = input("Générer les visualisations ? (o/N) [N]: ").strip().lower()
            export_stats = input("Exporter les statistiques ? (o/N) [N]: ").strip().lower()
            
            # Trouver le fichier de données le plus récent
            data_files = list(Path("keno/keno_data").glob("keno_*.csv"))
            if data_files:
                latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                command = f"python keno/duckdb_keno.py --csv {latest_file}"
                
                if plots in ['o', 'oui', 'y', 'yes']:
                    command += " --plots"
                    
                if export_stats in ['o', 'oui', 'y', 'yes']:
                    command += " --export-stats"
                    
                self.execute_command(command, "Analyse DuckDB Avancée Keno")
            else:
                print(f"{Colors.FAIL}❌ Aucun fichier de données Keno trouvé{Colors.ENDC}")
                print("Lancez d'abord l'extraction des données (option 1)")
                self.wait_and_continue()
        else:
            print(f"{Colors.FAIL}Choix invalide{Colors.ENDC}")
            self.wait_and_continue()
        
    def handle_systeme_reduit_simple(self):
        """Générateur simple de système réduit"""
        print(f"\n{Colors.BOLD}🎯 Générateur de Système Réduit - Simple{Colors.ENDC}")
        print("=" * 50)
        
        # Choix du type de jeu
        print("Type de jeu:")
        print("  loto - Loto (1-49, grilles de 5 numéros)")
        print("  keno - Keno (1-70, grilles de 10 numéros)")
        jeu = input("Jeu (loto/keno): ").strip().lower() or "loto"
        
        # Saisie des numéros favoris
        if jeu == "keno":
            print("Entrez vos numéros favoris Keno (10 à 15 numéros recommandés)")
            print("Format: 5,15,25,35,45,55,65")
        else:
            print("Entrez vos numéros favoris Loto (8 à 15 numéros recommandés)")
            print("Format: 1,7,12,18,23,29,34,39,45,49")
        nombres_input = input("Numéros favoris: ").strip()
        
        if not nombres_input:
            print(f"{Colors.WARNING}⚠️  Aucun numéro saisi.{Colors.ENDC}")
            self.wait_and_continue()
            return
        
        # Nettoyage de l'entrée : enlever les espaces autour des virgules
        nombres_input = ','.join([num.strip() for num in nombres_input.split(',')])
        
        # Nombre de grilles
        try:
            nb_grilles = int(input("Nombre de grilles à générer (5-20): ").strip() or "8")
            if nb_grilles < 1 or nb_grilles > 50:
                raise ValueError("Nombre invalide")
        except:
            nb_grilles = 8
            print(f"Nombre par défaut: {nb_grilles}")
        
        # Format d'export
        print("\nFormat d'export:")
        print("  csv - Tableur (défaut)")
        print("  md  - Markdown")
        format_export = input("Format (csv/md): ").strip() or "csv"
        
        # Nombre de numéros à utiliser
        print(f"\nNombre de numéros à utiliser parmi les favoris:")
        print(f"  Appuyez sur Entrée pour utiliser tous les numéros")
        print(f"  Ou entrez un nombre (minimum 7)")
        nb_nombres_input = input("Nombre de numéros à utiliser: ").strip()
        
        # Construction de la commande avec quotes pour les paramètres
        command = f"python grilles/generateur_grilles.py --jeu {jeu} --nombres \"{nombres_input}\" --grilles {nb_grilles} --export --format {format_export}"
        
        if nb_nombres_input:
            try:
                nb_nombres = int(nb_nombres_input)
                min_requis = 10 if jeu == "keno" else 7
                if nb_nombres >= min_requis:
                    command += f" --nombres-utilises {nb_nombres}"
                else:
                    print(f"⚠️  Minimum {min_requis} numéros requis pour {jeu.upper()}. Utilisation de tous les numéros.")
            except ValueError:
                print("⚠️  Nombre invalide. Utilisation de tous les numéros.")
        self.execute_command(command, "Génération Système Réduit Simple")
        
        # Affichage du dossier de sortie
        print(f"\n{Colors.OKCYAN}📁 Fichiers générés dans: grilles/sorties/{Colors.ENDC}")
        self.wait_and_continue()
    
    def handle_systeme_reduit_personnalise(self):
        """Générateur personnalisé de système réduit"""
        print(f"\n{Colors.BOLD}🎯 Générateur de Système Réduit - Personnalisé{Colors.ENDC}")
        print("=" * 60)
        
        # Choix du type de jeu
        print("Type de jeu:")
        print("  loto - Loto (1-49, grilles de 5 numéros)")
        print("  keno - Keno (1-70, grilles de 10 numéros)")
        jeu = input("Jeu (loto/keno): ").strip().lower() or "loto"
        
        # Méthode de saisie
        print("1. Saisir les numéros directement")
        print("2. Utiliser un fichier de numéros")
        methode_saisie = input("Votre choix (1-2): ").strip()
        
        nombres_param = ""
        fichier_param = ""
        
        if methode_saisie == "2":
            # Fichier
            print(f"\nFichiers disponibles dans grilles/:")
            grilles_dir = self.base_path / "grilles"
            txt_files = list(grilles_dir.glob("*.txt"))
            for i, f in enumerate(txt_files, 1):
                print(f"  {i}. {f.name}")
            
            if txt_files:
                try:
                    file_choice = int(input("Choisir un fichier (numéro): ").strip()) - 1
                    if 0 <= file_choice < len(txt_files):
                        fichier_param = f"--fichier {txt_files[file_choice]}"
                    else:
                        raise ValueError()
                except:
                    fichier_nom = input("Nom du fichier (ex: mes_nombres.txt): ").strip()
                    if fichier_nom:
                        fichier_param = f"--fichier grilles/{fichier_nom}"
            else:
                fichier_nom = input("Nom du fichier (ex: mes_nombres.txt): ").strip()
                if fichier_nom:
                    fichier_param = f"--fichier grilles/{fichier_nom}"
        else:
            # Saisie directe
            print("Entrez vos numéros favoris (8 à 20 numéros)")
            print("Format: 1,7,12,18,23,29,34,39,45,49")
            nombres_input = input("Numéros favoris: ").strip()
            if nombres_input:
                # Nettoyage de l'entrée : enlever les espaces autour des virgules
                nombres_input = ','.join([num.strip() for num in nombres_input.split(',')])
                nombres_param = f"--nombres {nombres_input}"
        
        if not nombres_param and not fichier_param:
            print(f"{Colors.WARNING}⚠️  Aucune source de numéros définie.{Colors.ENDC}")
            self.wait_and_continue()
            return
        
        # Paramètres avancés
        print("\nParamètres avancés:")
        
        # Nombre de grilles
        nb_grilles = input("Nombre de grilles (défaut: 10): ").strip() or "10"
        
        # Niveau de garantie
        print("Niveau de garantie:")
        print("  2 - Garantie faible (plus de grilles)")
        print("  3 - Équilibre optimal (défaut)")
        print("  4 - Garantie élevée")
        print("  5 - Garantie maximale")
        garantie = input("Garantie (2-5, défaut: 3): ").strip() or "3"
        
        # Méthode
        print("Méthode de génération:")
        print("  optimal   - Couverture maximale (défaut)")
        print("  aleatoire - Génération aléatoire intelligente")
        methode = input("Méthode (optimal/aleatoire): ").strip() or "optimal"
        
        # Nombre de numéros à utiliser
        print("Nombre de numéros à utiliser parmi les favoris:")
        print("  Appuyez sur Entrée pour utiliser tous les numéros")
        print("  Ou entrez un nombre (minimum 7)")
        nb_nombres_utilises = input("Nombre de numéros à utiliser: ").strip()
        
        # Format d'export
        print("Format d'export:")
        print("  csv  - Tableur (défaut)")
        print("  json - Données structurées")
        print("  txt  - Fichier texte")
        print("  md   - Markdown")
        format_export = input("Format (csv/json/txt/md): ").strip() or "csv"
        
        # Construction de la commande
        command_parts = ["python grilles/generateur_grilles.py"]
        
        if nombres_param:
            command_parts.append(nombres_param)
        elif fichier_param:
            command_parts.append(fichier_param)
        
        command_parts.extend([
            f"--jeu {jeu}",
            f"--grilles {nb_grilles}",
            f"--garantie {garantie}",
            f"--methode {methode}",
            "--export",
            f"--format {format_export}",
            "--verbose"
        ])
        
        # Ajout du paramètre nombre de numéros si spécifié
        if nb_nombres_utilises:
            try:
                nb_nombres = int(nb_nombres_utilises)
                min_requis = 10 if jeu == "keno" else 7
                if nb_nombres >= min_requis:
                    command_parts.insert(-2, f"--nombres-utilises {nb_nombres}")
            except ValueError:
                pass
        
        command = " ".join(command_parts)
        self.execute_command(command, "Génération Système Réduit Personnalisé")
        
        # Affichage du dossier de sortie
        print(f"\n{Colors.OKCYAN}📁 Fichiers générés dans: grilles/sorties/{Colors.ENDC}")
        print(f"{Colors.OKCYAN}📖 Documentation: grilles/README.md{Colors.ENDC}")
        self.wait_and_continue()
    
    def handle_choice(self, choice):
        """Traite le choix de l'utilisateur"""
        
        if choice == "1":
            self.execute_command("python loto/result.py", "Téléchargement des données Loto")
            
        elif choice == "2":
            self.execute_command("python keno_cli.py extract", "Extraction des données Keno (FDJ)")
            
        elif choice == "3":
            self.execute_command("python loto/result.py && python keno_cli.py extract", "Mise à jour de toutes les données")
            
        elif choice == "4":
            command = f"python loto/duckdb_loto.py --csv {self.loto_csv} --grids {self.default_loto_grids} --config {self.loto_config_file}"
            self.execute_command(command, "Génération 3 grilles Loto (Rapide)")
            
        elif choice == "5":
            command = f"python loto/duckdb_loto.py --csv {self.loto_csv} --grids 5 --config {self.loto_config_file}"
            self.execute_command(command, "Génération 5 grilles Loto (Complet)")
            
        elif choice == "6":
            command = f"python loto/duckdb_loto.py --csv {self.loto_csv} --grids {self.default_loto_grids} --plots --export-stats --config {self.loto_config_file}"
            self.execute_command(command, "Génération Loto avec Visualisations")
            
        elif choice == "7":
            self.handle_loto_custom()
            
        elif choice == "8":
            self.execute_command("python keno_cli.py analyze", "Analyse Keno Complète")
            
        elif choice == "9":
            self.execute_command("python keno_cli.py all --grids 3", "Pipeline Keno Complet avec Visualisations")
            
        elif choice == "10":
            self.handle_keno_custom()
            
        elif choice == "11":
            self.execute_command("python test/run_all_tests.py", "Tests Complets du Système")
            
        elif choice == "12":
            self.execute_command("python test/run_all_tests.py --essential", "Tests Essentiels")
            
        elif choice == "13":
            self.execute_command("python test/test_performance.py", "Test de Performance")
            
        elif choice == "14":
            print(f"\n{Colors.BOLD}🧹 Nettoyage et Optimisation{Colors.ENDC}")
            print("Choisissez le type de nettoyage :")
            print(f"  {Colors.OKGREEN}1{Colors.ENDC} - Nettoyage standard (fichiers temporaires)")
            print(f"  {Colors.WARNING}2{Colors.ENDC} - Nettoyage approfondi (inclut les anciens backups)")
            print(f"  {Colors.OKCYAN}3{Colors.ENDC} - Afficher le statut seulement")
            
            clean_choice = input(f"\n{Colors.BOLD}Votre choix (1-3): {Colors.ENDC}").strip()
            
            if clean_choice == "1":
                self.execute_command("python keno_cli.py clean", "Nettoyage Standard")
            elif clean_choice == "2":
                self.execute_command("python keno_cli.py clean --deep", "Nettoyage Approfondi")
            elif clean_choice == "3":
                self.execute_command("python keno_cli.py status", "Statut du Système")
            else:
                print(f"{Colors.FAIL}Choix invalide{Colors.ENDC}")
                self.wait_and_continue()
            
        elif choice == "15":
            self.execute_command("./lancer_api.sh", "Lancement de l'API Flask")
            
        elif choice == "16":
            self.execute_command("python test/test_api.py", "Test de l'API Flask")
            
        elif choice == "17":
            self.handle_systeme_reduit_simple()
            
        elif choice == "18":
            self.handle_systeme_reduit_personnalise()
            
        elif choice == "19":
            grilles_file = self.base_path / "grilles.csv"
            if grilles_file.exists():
                self.show_file_content(grilles_file, 20)
            else:
                print(f"\n{Colors.WARNING}⚠️  Aucune grille trouvée. Générez d'abord des grilles Loto.{Colors.ENDC}")
            self.wait_and_continue()
            
        elif choice == "20":
            reco_file = self.base_path / "keno_output" / "recommandations_keno.txt"
            if reco_file.exists():
                self.show_file_content(reco_file, 30)
            else:
                print(f"\n{Colors.WARNING}⚠️  Aucune recommandation trouvée. Lancez d'abord une analyse Keno.{Colors.ENDC}")
            self.wait_and_continue()
            
        elif choice == "21":
            plots_dir = self.base_path / "loto_analyse_plots"
            keno_plots_dir = self.base_path / "keno_analyse_plots"
            
            print(f"\n{Colors.BOLD}📊 Dossiers de Graphiques:{Colors.ENDC}")
            
            if plots_dir.exists():
                files = list(plots_dir.glob("*.png"))
                print(f"🎲 Loto: {len(files)} graphiques dans {plots_dir}")
                for f in files:
                    print(f"   - {f.name}")
            else:
                print(f"🎲 Loto: Aucun graphique (dossier inexistant)")
                
            if keno_plots_dir.exists():
                files = list(keno_plots_dir.glob("*.png"))
                print(f"🎰 Keno: {len(files)} graphiques dans {keno_plots_dir}")
                for f in files:
                    print(f"   - {f.name}")
            else:
                print(f"🎰 Keno: Aucun graphique (dossier inexistant)")
                
            print(f"\n{Colors.OKCYAN}💡 Astuce: Utilisez un explorateur de fichiers pour ouvrir les images{Colors.ENDC}")
            self.wait_and_continue()
            
        elif choice == "22":
            print(f"\n{Colors.WARNING}🚧 Générateur Loto Avancé (ML + IA) - Version Classique{Colors.ENDC}")
            print("Ce générateur utilise des techniques avancées:")
            print("  • Machine Learning (XGBoost)")
            print("  • Analyse statistique approfondie") 
            print("  • Optimisation multi-critères")
            print("  • Cache Redis pour les performances")
            print()
            print(f"{Colors.OKBLUE}💡 Pour la version la plus récente avec auto-amélioration,{Colors.ENDC}")
            print(f"{Colors.OKBLUE}   utilisez l'option 44 (Générateur Adaptatif V2){Colors.ENDC}")
            print()
            
            # Configuration des paramètres
            print(f"{Colors.OKBLUE}⚙️  Configuration des paramètres:{Colors.ENDC}")
            print("1️⃣  Mode rapide (1,000 simulations)")
            print("2️⃣  Mode standard (10,000 simulations)")
            print("3️⃣  Mode intensif (50,000 simulations)")
            print("4️⃣  Configuration personnalisée")
            print("0️⃣  Retour au menu principal")
            
            config_choice = input("\n🎯 Votre choix de configuration: ").strip()
            
            if config_choice == "0":
                self.wait_and_continue()
                return
            elif config_choice == "1":
                command = f"{self.python_path} loto/loto_generator_advanced_Version2.py --quick --silent"
                description = "Générateur Loto Avancé (Mode Rapide)"
            elif config_choice == "2":
                command = f"{self.python_path} loto/loto_generator_advanced_Version2.py --silent"
                description = "Générateur Loto Avancé (Mode Standard)"
            elif config_choice == "3":
                command = f"{self.python_path} loto/loto_generator_advanced_Version2.py --intensive"
                description = "Générateur Loto Avancé (Mode Intensif)"
            elif config_choice == "4":
                # Configuration personnalisée
                print(f"\n{Colors.OKBLUE}� Configuration personnalisée:{Colors.ENDC}")
                
                # Nombre de simulations
                while True:
                    try:
                        n_sims = input("📊 Nombre de simulations (100-100000, défaut: 10000): ").strip()
                        if not n_sims:
                            n_sims = 10000
                        else:
                            n_sims = int(n_sims)
                        
                        if n_sims < 100 or n_sims > 100000:
                            print("❌ Le nombre de simulations doit être entre 100 et 100,000")
                            continue
                        break
                    except ValueError:
                        print("❌ Veuillez entrer un nombre valide")
                
                # Nombre de processeurs
                import multiprocessing as mp
                max_cores = mp.cpu_count()
                default_cores = max_cores - 1 if max_cores > 1 else 1
                
                while True:
                    try:
                        n_cores = input(f"🔄 Nombre de processeurs (1-{max_cores}, défaut: {default_cores}): ").strip()
                        if not n_cores:
                            n_cores = default_cores
                        else:
                            n_cores = int(n_cores)
                        
                        if n_cores < 1 or n_cores > max_cores:
                            print(f"❌ Le nombre de processeurs doit être entre 1 et {max_cores}")
                            continue
                        break
                    except ValueError:
                        print("❌ Veuillez entrer un nombre valide")
                
                # Numéros à exclure
                excluded_numbers = None
                while True:
                    exclude_input = input(f"🚫 Numéros à exclure (1-49, séparés par des virgules, ou 'auto' pour les 3 derniers tirages, défaut: auto): ").strip()
                    if not exclude_input or exclude_input.lower() == 'auto':
                        excluded_numbers = None
                        break
                    
                    try:
                        excluded_nums = [int(x.strip()) for x in exclude_input.split(',')]
                        # Vérifier que tous les numéros sont valides (1-49)
                        invalid_nums = [num for num in excluded_nums if num < 1 or num > 49]
                        if invalid_nums:
                            print(f"❌ Numéros invalides détectés: {invalid_nums}. Les numéros doivent être entre 1 et 49.")
                            continue
                        # Vérifier qu'il ne faut pas exclure trop de numéros
                        if len(excluded_nums) > 44:
                            print(f"❌ Trop de numéros exclus ({len(excluded_nums)}). Maximum autorisé: 44.")
                            continue
                        excluded_numbers = excluded_nums
                        break
                    except ValueError:
                        print("❌ Format invalide. Utilisez des numéros séparés par des virgules (ex: 1,5,12)")
                
                # Construction de la commande
                command = f"{self.python_path} loto/loto_generator_advanced_Version2.py -s {n_sims} -c {n_cores} --silent"
                if excluded_numbers:
                    exclude_str = ','.join(map(str, excluded_numbers))
                    command += f" --exclude {exclude_str}"
                    description = f"Générateur Loto Avancé ({n_sims:,} simulations, {n_cores} cœurs, excluant {len(excluded_numbers)} numéros)"
                else:
                    description = f"Générateur Loto Avancé ({n_sims:,} simulations, {n_cores} cœurs, exclusion auto)"
            else:
                print("❌ Choix invalide")
                self.wait_and_continue()
                return
            
            # Confirmation finale
            print(f"\n{Colors.OKGREEN}✅ Configuration choisie:{Colors.ENDC}")
            print(f"   Commande: {command}")
            print()
            confirm = input("Lancer le générateur avancé ? (o/N): ").strip().lower()
            
            if confirm in ['o', 'oui', 'y', 'yes']:
                print(f"\n{Colors.OKBLUE}🚀 Lancement du générateur avancé...{Colors.ENDC}")
                print("⚠️  Note: Ce processus peut prendre plusieurs minutes")
                self.execute_command(command, description)
            else:
                print("Opération annulée.")
                self.wait_and_continue()
        
        elif choice == "23":
            self.execute_command("python keno_cli.py status", "Statut Détaillé du Système")
            
        elif choice == "24":
            print(f"\n{Colors.OKCYAN}🧠 ANALYSE AVANCÉE DUCKDB - 11 STRATÉGIES{Colors.ENDC}")
            print("Cette analyse utilise le fichier consolidé pour une performance optimale")
            print("11 stratégies différentes seront analysées avec scoring avancé")
            
            self.execute_command("python keno_cli.py analyze-advanced --export-stats", 
                               "Analyse Avancée DuckDB")
        
        elif choice == "25":
            print(f"\n{Colors.OKCYAN}🧠 GÉNÉRATEUR KENO AVANCÉ (ML + IA){Colors.ENDC}")
            print("Ce générateur utilise des techniques avancées:")
            print("  • Machine Learning pour la prédiction")
            print("  • Analyse statistique poussée")
            print("  • Optimisation des patterns")
            print("  • Intelligence artificielle")
            print()
            
            # Configuration des paramètres
            print(f"{Colors.OKBLUE}⚙️  Configuration des paramètres:{Colors.ENDC}")
            print("1️⃣  Mode rapide (10 grilles + entraînement)")
            print("2️⃣  Mode standard (5 grilles sans réentraînement)")
            print("3️⃣  Mode intensif (20 grilles + réentraînement complet)")
            print("4️⃣  Configuration personnalisée")
            print("0️⃣  Retour au menu principal")
            
            config_choice = input("\n🎯 Votre choix de configuration: ").strip()
            
            if config_choice == "0":
                self.wait_and_continue()
                return True
            elif config_choice == "1":
                command = "python keno_cli.py generate-advanced --quick"
                description = "Générateur Keno Avancé (Mode Rapide)"
            elif config_choice == "2":
                command = "python keno_cli.py generate-advanced --grids 5"
                description = "Générateur Keno Avancé (Mode Standard)"
            elif config_choice == "3":
                command = "python keno_cli.py generate-advanced --grids 20 --retrain"
                description = "Générateur Keno Avancé (Mode Intensif)"
            elif config_choice == "4":
                # Configuration personnalisée
                print(f"\n{Colors.OKBLUE}🔧 Configuration personnalisée:{Colors.ENDC}")
                
                try:
                    grids = input("Nombre de grilles à générer (5-50): ").strip()
                    grids = int(grids) if grids.isdigit() and 5 <= int(grids) <= 50 else 5
                    
                    retrain = input("Réentraîner les modèles ? (o/N): ").strip().lower()
                    retrain_flag = "--retrain" if retrain in ['o', 'oui', 'y', 'yes'] else ""
                    
                    command = f"python keno_cli.py generate-advanced --grids {grids} {retrain_flag}".strip()
                    description = f"Générateur Keno Avancé ({grids} grilles personnalisées)"
                    
                except ValueError:
                    print(f"{Colors.FAIL}❌ Configuration invalide, utilisation des paramètres par défaut{Colors.ENDC}")
                    command = "python keno_cli.py generate-advanced --grids 5"
                    description = "Générateur Keno Avancé (Mode Standard)"
            else:
                print(f"{Colors.FAIL}Choix invalide{Colors.ENDC}")
                self.wait_and_continue()
                return True
            
            # Confirmation avant exécution
            print(f"\n{Colors.WARNING}⚠️  Paramètres sélectionnés:{Colors.ENDC}")
            print(f"   Commande: {command}")
            print(f"   Description: {description}")
            print(f"\n{Colors.BOLD}💡 Le générateur avancé peut prendre quelques minutes{Colors.ENDC}")
            
            confirm = input(f"\n{Colors.OKGREEN}Confirmer l'exécution ? (O/n): {Colors.ENDC}").strip().lower()
            if confirm in ['', 'o', 'oui', 'y', 'yes']:
                print(f"\n{Colors.OKBLUE}🚀 Lancement du générateur avancé...{Colors.ENDC}")
                print("⚠️  Note: Ce processus peut prendre plusieurs minutes")
                self.execute_command(command, description)
            else:
                print("Opération annulée.")
                self.wait_and_continue()
        
        elif choice == "26":
            print(f"\n{Colors.BOLD}📊 Statistiques Keno Complètes{Colors.ENDC}")
            print("Génération de toutes les statistiques détaillées :")
            print("  • Fréquences et retards de tous les numéros")
            print("  • Analyse pair/impair et zones")
            print("  • Sommes et tableaux de retards")
            print("  • Visualisations et graphiques")
            print("  • Recommandations prioritaires")
            print()
            
            confirm = input(f"{Colors.OKGREEN}Lancer l'analyse complète ? (O/n): {Colors.ENDC}").strip().lower()
            if confirm in ['', 'o', 'oui', 'y', 'yes']:
                self.execute_command("python keno/duckdb_keno.py --auto-consolidated --plots --export-stats", "Statistiques Keno Complètes")
            else:
                print("Opération annulée.")
                self.wait_and_continue()
                
        elif choice == "27":
            print(f"\n{Colors.BOLD}⚡ Analyse Keno Rapide{Colors.ENDC}")
            print("Analyse express avec les informations essentielles :")
            print("  • Top des numéros prioritaires")
            print("  • Retards et tendances récentes")
            print("  • Recommandations immédiates")
            print()
            
            # Demander le nombre de numéros à afficher
            while True:
                try:
                    top_n = input(f"Nombre de numéros prioritaires à afficher (défaut: 15): ").strip()
                    if not top_n:
                        top_n = 15
                    else:
                        top_n = int(top_n)
                    
                    if top_n < 5 or top_n > 50:
                        print("❌ Le nombre doit être entre 5 et 50")
                        continue
                    break
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide")
            
            # Demander si on génère les graphiques
            graphiques = input(f"Générer les graphiques ? (o/N): ").strip().lower()
            graph_option = "--graphiques" if graphiques in ['o', 'oui', 'y', 'yes'] else ""
            
            command = f"python keno/analyse_keno_rapide.py --csv keno/keno_data/keno_consolidated.csv --top {top_n} {graph_option}".strip()
            self.execute_command(command, f"Analyse Keno Rapide (Top {top_n})")
            
        elif choice == "28":
            print(f"\n{Colors.BOLD}🏆 TOP 25 LOTO ÉQUILIBRÉS{Colors.ENDC}")
            print("Génération des 25 numéros Loto avec le plus de chances selon la stratégie équilibrée :")
            print("  • Analyse composite : Fréquence (35%) + Retard (30%) + Paires (20%) + Zones (15%)")
            print("  • Plage 1-49 numéros avec équilibrage 3 zones")
            print("  • Export CSV fixe (remplace le fichier précédent)")
            print("  • Rapport Markdown détaillé avec suggestions")
            print(f"  • {Colors.OKGREEN}Compatible avec le Générateur Adaptatif V2 (option 44){Colors.ENDC}")
            print()
            
            # Sélection du fichier de données Loto
            loto_data_path = Path("loto/loto_data")
            if loto_data_path.exists():
                csv_files = list(loto_data_path.glob("*.csv"))
                if csv_files:
                    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                    print(f"📁 Fichier détecté: {latest_file.name}")
                    date_range = self.format_date_range(self.get_csv_date_range(latest_file))
                    if date_range:
                        print(f"📅 Période: {date_range}")
                    
                    # Options de stratégie
                    print(f"\n{Colors.OKBLUE}⚙️ Stratégies disponibles:{Colors.ENDC}")
                    print("1️⃣  Équilibrée (recommandé)")
                    print("2️⃣  Focus retard")
                    print("3️⃣  Focus fréquence")
                    print("0️⃣  Annuler")
                    
                    strategy_choice = input(f"\n{Colors.BOLD}Stratégie (1-3) [1]: {Colors.ENDC}").strip() or "1"
                    
                    if strategy_choice == "0":
                        self.wait_and_continue()
                        return True
                    
                    strategy_map = {
                        "1": "equilibre",
                        "2": "focus_retard", 
                        "3": "focus_frequence"
                    }
                    
                    strategy = strategy_map.get(strategy_choice, "equilibre")
                    
                    # Options d'export
                    print(f"\n{Colors.OKBLUE}📊 Options d'export:{Colors.ENDC}")
                    export_stats = input("Exporter les statistiques détaillées ? (O/n): ").strip().lower()
                    plots = input("Générer les visualisations ? (o/N): ").strip().lower()
                    
                    # Construction de la commande
                    command = f"python loto/duckdb_loto.py --csv {latest_file} --config-file loto/strategies.yml"
                    
                    if export_stats not in ['n', 'non', 'no']:
                        command += " --export-stats"
                    
                    if plots in ['o', 'oui', 'y', 'yes']:
                        command += " --plots"
                    
                    description = f"TOP 25 Loto Équilibrés (stratégie: {strategy})"
                    
                    print(f"\n{Colors.OKGREEN}✅ Configuration:{Colors.ENDC}")
                    print(f"   Fichier: {latest_file.name}")
                    print(f"   Stratégie: {strategy}")
                    print(f"   Export CSV: loto_stats_exports/top_25_numeros_equilibres_loto.csv")
                    
                    confirm = input(f"\n{Colors.BOLD}Lancer la génération ? (O/n): {Colors.ENDC}").strip().lower()
                    if confirm not in ['n', 'non', 'no']:
                        self.execute_command(command, description)
                    else:
                        print("Opération annulée.")
                        self.wait_and_continue()
                else:
                    print(f"{Colors.FAIL}❌ Aucun fichier CSV Loto trouvé dans {loto_data_path}{Colors.ENDC}")
                    self.wait_and_continue()
            else:
                print(f"{Colors.FAIL}❌ Répertoire loto/loto_data non trouvé{Colors.ENDC}")
                self.wait_and_continue()
                
        elif choice == "29":
            print(f"\n{Colors.BOLD}🏆 TOP 30 KENO ÉQUILIBRÉS{Colors.ENDC}")
            print("Génération des 30 numéros Keno avec le plus de chances selon la stratégie équilibrée :")
            print("  • Analyse composite : Fréquence (30%) + Retard (25%) + Paires (25%) + Zones (20%)")
            print("  • Plage 1-70 numéros avec équilibrage 5 zones")
            print("  • Export CSV fixe (remplace le fichier précédent)")
            print("  • Analyse de 11 stratégies probabilistes")
            print()
            
            # Sélection du fichier de données Keno
            keno_data_path = Path("keno/keno_data")
            if keno_data_path.exists():
                csv_files = list(keno_data_path.glob("*.csv"))
                if csv_files:
                    # Proposer le fichier consolidé par défaut, sinon le plus récent
                    consolidated_file = keno_data_path / "keno_consolidated.csv"
                    if consolidated_file.exists():
                        selected_file = consolidated_file
                        print(f"📁 Fichier recommandé: {selected_file.name} (données consolidées)")
                    else:
                        selected_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                        print(f"📁 Fichier détecté: {selected_file.name}")
                    
                    date_range = self.format_date_range(self.get_csv_date_range(selected_file))
                    if date_range:
                        print(f"📅 Période: {date_range}")
                    
                    # Options d'export
                    print(f"\n{Colors.OKBLUE}📊 Options d'export:{Colors.ENDC}")
                    export_stats = input("Exporter les statistiques détaillées ? (O/n): ").strip().lower()
                    plots = input("Générer les visualisations ? (o/N): ").strip().lower()
                    
                    # Construction de la commande
                    command = f"python keno/duckdb_keno.py --csv {selected_file}"
                    
                    if export_stats not in ['n', 'non', 'no']:
                        command += " --export-stats"
                    
                    if plots in ['o', 'oui', 'y', 'yes']:
                        command += " --plots"
                    
                    description = "TOP 30 Keno Équilibrés (stratégie optimisée)"
                    
                    print(f"\n{Colors.OKGREEN}✅ Configuration:{Colors.ENDC}")
                    print(f"   Fichier: {selected_file.name}")
                    print(f"   Export CSV: keno_stats_exports/top_30_numeros_equilibres_keno.csv")
                    print(f"   Stratégies: 11 algorithmes probabilistes")
                    
                    confirm = input(f"\n{Colors.BOLD}Lancer la génération ? (O/n): {Colors.ENDC}").strip().lower()
                    if confirm not in ['n', 'non', 'no']:
                        self.execute_command(command, description)
                    else:
                        print("Opération annulée.")
                        self.wait_and_continue()
                else:
                    print(f"{Colors.FAIL}❌ Aucun fichier CSV Keno trouvé dans {keno_data_path}{Colors.ENDC}")
                    self.wait_and_continue()
            else:
                print(f"{Colors.FAIL}❌ Répertoire keno/keno_data non trouvé{Colors.ENDC}")
                self.wait_and_continue()
                
        elif choice == "30":
            print(f"\n{Colors.BOLD}📊 AFFICHAGE TOP 25 LOTO{Colors.ENDC}")
            
            # Vérifier l'existence du fichier
            csv_file = Path("loto_stats_exports/top_25_numeros_equilibres_loto.csv")
            if csv_file.exists():
                print(f"📁 Fichier: {csv_file}")
                print(f"📅 Dernière modification: {datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%d/%m/%Y %H:%M:%S')}")
                print()
                
                try:
                    df = pd.read_csv(csv_file, sep=';')
                    
                    print(f"{Colors.OKGREEN}🏆 TOP 25 NUMÉROS LOTO ÉQUILIBRÉS{Colors.ENDC}")
                    print("=" * 80)
                    print(f"{'Rang':<4} {'Numéro':<6} {'Score':<8} {'Zone':<15} {'Retard':<7} {'Fréq':<5}")
                    print("-" * 80)
                    
                    # Afficher les 10 premiers avec couleurs
                    for i, row in df.head(10).iterrows():
                        color = Colors.OKGREEN if i < 3 else Colors.OKCYAN if i < 5 else Colors.ENDC
                        print(f"{color}{row['rang']:<4} {row['numero']:<6} {row['score_composite']:<8.4f} {row['zone']:<15} {row['retard_actuel']:<7} {row['freq_absolue']:<5}{Colors.ENDC}")
                    
                    if len(df) > 10:
                        print(f"\n{Colors.WARNING}... et {len(df) - 10} autres numéros (voir fichier CSV complet){Colors.ENDC}")
                    
                    # Statistiques de zone
                    print(f"\n{Colors.OKBLUE}📍 RÉPARTITION PAR ZONES:{Colors.ENDC}")
                    zone_counts = df['zone'].value_counts()
                    for zone, count in zone_counts.items():
                        percentage = (count / len(df)) * 100
                        print(f"   {zone}: {count} numéros ({percentage:.1f}%)")
                    
                    # Suggestions pratiques
                    top_5 = df.head(5)['numero'].tolist()
                    top_7 = df.head(7)['numero'].tolist()
                    top_10 = df.head(10)['numero'].tolist()
                    
                    print(f"\n{Colors.BOLD}💡 SUGGESTIONS PRATIQUES:{Colors.ENDC}")
                    print(f"   Grille 5 numéros: {top_5}")
                    print(f"   Système 7 numéros: {top_7}")
                    print(f"   Système 10 numéros: {top_10}")
                    
                except Exception as e:
                    print(f"{Colors.FAIL}❌ Erreur lors de la lecture du fichier: {e}{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}❌ Fichier TOP 25 Loto non trouvé{Colors.ENDC}")
                print("💡 Générez d'abord les TOP 25 avec l'option 28")
            
            self.wait_and_continue()
            
        elif choice == "31":
            print(f"\n{Colors.BOLD}📊 AFFICHAGE TOP 30 KENO{Colors.ENDC}")
            
            # Vérifier l'existence du fichier
            csv_file = Path("keno_stats_exports/top_30_numeros_equilibres_keno.csv")
            if csv_file.exists():
                print(f"📁 Fichier: {csv_file}")
                print(f"📅 Dernière modification: {datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%d/%m/%Y %H:%M:%S')}")
                print()
                
                try:
                    df = pd.read_csv(csv_file, sep=';')
                    
                    print(f"{Colors.OKGREEN}🏆 TOP 30 NUMÉROS KENO ÉQUILIBRÉS{Colors.ENDC}")
                    print("=" * 80)
                    print(f"{'Rang':<4} {'Numéro':<6} {'Score':<8} {'Zone':<15} {'Retard':<7} {'Fréq':<5}")
                    print("-" * 80)
                    
                    # Afficher les 10 premiers avec couleurs
                    for i, row in df.head(10).iterrows():
                        color = Colors.OKGREEN if i < 3 else Colors.OKCYAN if i < 5 else Colors.ENDC
                        print(f"{color}{row['rang']:<4} {row['numero']:<6} {row['score_composite']:<8.4f} {row['zone']:<15} {row['retard_actuel']:<7} {row['freq_absolue']:<5}{Colors.ENDC}")
                    
                    if len(df) > 10:
                        print(f"\n{Colors.WARNING}... et {len(df) - 10} autres numéros (voir fichier CSV complet){Colors.ENDC}")
                    
                    # Statistiques de zone
                    print(f"\n{Colors.OKBLUE}📍 RÉPARTITION PAR ZONES:{Colors.ENDC}")
                    zone_counts = df['zone'].value_counts()
                    for zone, count in zone_counts.items():
                        percentage = (count / len(df)) * 100
                        print(f"   {zone}: {count} numéros ({percentage:.1f}%)")
                    
                    # Suggestions pratiques
                    top_10 = df.head(10)['numero'].tolist()
                    top_15 = df.head(15)['numero'].tolist()
                    top_20 = df.head(20)['numero'].tolist()
                    
                    print(f"\n{Colors.BOLD}💡 SUGGESTIONS PRATIQUES:{Colors.ENDC}")
                    print(f"   Grille 10 numéros: {top_10}")
                    print(f"   Système 15 numéros: {top_15}")
                    print(f"   Système 20 numéros: {top_20}")
                    
                except Exception as e:
                    print(f"{Colors.FAIL}❌ Erreur lors de la lecture du fichier: {e}{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}❌ Fichier TOP 30 Keno non trouvé{Colors.ENDC}")
                print("💡 Générez d'abord les TOP 30 avec l'option 29")
            
            self.wait_and_continue()
                
        elif choice == "32":
            self.handle_grilles_loto_top_csv()
            
        elif choice == "33":
            self.handle_grilles_keno_top_csv()
            
        elif choice == "34":
            self.handle_grilles_loto_top_markdown()
            
        elif choice == "35":
            self.handle_grilles_keno_top_markdown()
            
        elif choice == "36":
            self.handle_keno_patterns_advanced()
            
        elif choice == "37":
            self.handle_keno_top30_csv()
            
        elif choice == "38":
            self.handle_keno_generator_custom()
            
        elif choice == "39":
            self.handle_keno_patterns_test()
            
        elif choice == "40":
            self.handle_keno_intelligent_generator()
            
        elif choice == "41":
            self.handle_keno_system_reducer()
            
        elif choice == "42":
            self.handle_keno_pairs_generator()
            
        elif choice == "43":
            self.handle_keno_generate_top30()
            
        elif choice == "44":
            self.handle_loto_adaptive_v2()
            
        elif choice == "45":
            self.handle_loto_adaptive_training()
            
        elif choice == "46":
            self.handle_loto_no_exclusion()
            
        elif choice == "47":
            self.handle_csv_to_parquet()
                
        elif choice == "0":
            print(f"\n{Colors.OKGREEN}👋 Au revoir ! Bonne chance pour vos analyses !{Colors.ENDC}")
            return False
            
        else:
            print(f"\n{Colors.FAIL}❌ Choix invalide. Veuillez réessayer.{Colors.ENDC}")
            self.wait_and_continue()
            
        return True
    
    def handle_keno_patterns_advanced(self):
        """Générateur patterns avancés avec DuckDB et 15 patterns"""
        print(f"\n{Colors.BOLD}🧠 GÉNÉRATEUR KENO PATTERNS AVANCÉS{Colors.ENDC}")
        print("Ce générateur utilise l'analyse complète des patterns :")
        print("  • 🔢 Fréquences multi-période (global, 100, 50, 20 tirages)")
        print("  • ⏰ Retards et overdue numbers avec historique")
        print("  • 🎯 Paires et trios fréquents")
        print("  • 📊 Patterns parité, sommes, zones")
        print("  • 📈 Tendances 10/50/100 tirages")
        print("  • 🚀 DuckDB pour optimisation SQL")
        print()
        
        # Configuration du nombre de grilles
        while True:
            try:
                grids_input = input(f"Nombre de grilles à générer (5-50, défaut: 10): ").strip()
                if not grids_input:
                    grids = 10
                else:
                    grids = int(grids_input)
                    if grids < 5 or grids > 50:
                        print(f"{Colors.FAIL}❌ Nombre de grilles doit être entre 5 et 50{Colors.ENDC}")
                        continue
                break
            except ValueError:
                print(f"{Colors.FAIL}❌ Veuillez entrer un nombre valide{Colors.ENDC}")
        
        command = f"python3 -c \"import sys; sys.path.append('.'); from keno.keno_generator_advanced import KenoGeneratorAdvanced; gen = KenoGeneratorAdvanced(); gen.load_data(); gen.analyze_patterns(); grids = gen.generate_frequency_based_grids({grids}); print('🎯 Grilles générées avec patterns avancés:'); [print(f'  {{i+1:2d}}. {{sorted(grid)}}') for i, grid in enumerate(grids)]\""
        description = f"Générateur Patterns Avancés ({grids} grilles)"
        
        print(f"\n{Colors.WARNING}⚠️  Configuration:{Colors.ENDC}")
        print(f"   • Grilles: {grids}")
        print(f"   • Patterns: 15 types d'analyse")
        print(f"   • Optimisation: DuckDB")
        
        confirm = input(f"\n{Colors.OKGREEN}Confirmer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_keno_top30_csv(self):
        """Génération TOP 30 numéros patterns → CSV"""
        print(f"\n{Colors.BOLD}📊 TOP 30 NUMÉROS PATTERNS → CSV{Colors.ENDC}")
        print("Génération d'un CSV avec les 30 meilleurs numéros basés sur les patterns :")
        print("  • 🔢 Score composite (fréquences + retards + tendances)")
        print("  • 📊 Détails par pattern (global, récent, retard)")
        print("  • 📈 Tendances et recommandations")
        print("  • 💾 Export CSV structuré")
        print()
        
        # Options d'export
        print(f"{Colors.OKBLUE}📋 Options d'export:{Colors.ENDC}")
        print("1️⃣  Export simple (top 30 + scores)")
        print("2️⃣  Export détaillé (tous les patterns)")
        print("3️⃣  Export avec grilles recommandées")
        print("0️⃣  Retour au menu")
        
        export_choice = input("\n🎯 Type d'export: ").strip()
        
        if export_choice == "0":
            self.wait_and_continue()
            return
        elif export_choice == "1":
            detail_level = "simple"
        elif export_choice == "2":
            detail_level = "detaille"
        elif export_choice == "3":
            detail_level = "avec_grilles"
        else:
            print(f"{Colors.FAIL}Choix invalide, export simple par défaut{Colors.ENDC}")
            detail_level = "simple"
        
        command = f"python3 -c \"import sys; sys.path.append('.'); from keno.keno_generator_advanced import KenoGeneratorAdvanced; import pandas as pd; from datetime import datetime; import os; gen = KenoGeneratorAdvanced(); gen.load_data(); stats = gen.analyze_patterns(); scores = {{}}; [scores.update({{num: (stats.frequences.get(num, 0)/max(stats.frequences.values()))*0.4 + (stats.frequences_recentes.get(num, 0)/max(stats.frequences_recentes.values()) if stats.frequences_recentes else 0)*0.3 + (1-stats.retards.get(num, 0)/max(stats.retards.values()) if stats.retards else 0)*0.2 + ((stats.tendances_50.get(num, 1.0)-1.0)*0.1 if stats.tendances_50.get(num, 1.0) > 1.0 else 0)}}) for num in range(1, 71)]; top30 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:30]; df = pd.DataFrame([{{'rang': i+1, 'numero': num, 'score': round(score, 4), 'freq_globale': stats.frequences.get(num, 0), 'freq_recente': stats.frequences_recentes.get(num, 0), 'retard': stats.retards.get(num, 0), 'tendance': round(stats.tendances_50.get(num, 1.0), 3)}} for i, (num, score) in enumerate(top30)]); os.makedirs('keno_output', exist_ok=True); filename = f'keno_top30_{{datetime.now().strftime(\\\"%Y%m%d_%H%M%S\\\")}}.csv'; df.to_csv(f'keno_output/{{filename}}', index=False); print(f'✅ TOP 30 exporté: keno_output/{{filename}}'); print(f'📊 Top 5: {{\\', \\'.join([str(row[\\\'numero\\\']) for _, row in df.head().iterrows()])}}')\""
        description = f"TOP 30 Numéros Patterns → CSV ({detail_level})"
        
        confirm = input(f"\n{Colors.OKGREEN}Confirmer l'export CSV ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_keno_generator_custom(self):
        """Générateur personnalisé avec paramètres libres"""
        print(f"\n{Colors.BOLD}🎯 GÉNÉRATEUR KENO PERSONNALISÉ{Colors.ENDC}")
        print("Configuration complète de votre générateur :")
        print("  • 🔢 Nombre de grilles personnalisé")
        print("  • 🎚️  Stratégies au choix")
        print("  • 📊 Patterns sélectionnables")
        print("  • 💾 Export personnalisé")
        print()
        
        # Configuration du nombre de grilles
        while True:
            try:
                grids_input = input(f"Nombre de grilles (5-100, défaut: 15): ").strip()
                grids = 15 if not grids_input else int(grids_input)
                if grids < 5 or grids > 100:
                    print(f"{Colors.FAIL}❌ Nombre doit être entre 5 et 100{Colors.ENDC}")
                    continue
                break
            except ValueError:
                print(f"{Colors.FAIL}❌ Veuillez entrer un nombre valide{Colors.ENDC}")
        
        # Export des résultats
        export_csv = input("Exporter en CSV ? (O/n): ").strip().lower() in ['', 'o', 'oui', 'y', 'yes']
        
        command = f"python3 -c \"import sys; sys.path.append('.'); from keno.keno_generator_advanced import KenoGeneratorAdvanced; import pandas as pd; from datetime import datetime; gen = KenoGeneratorAdvanced(); gen.load_data(); gen.analyze_patterns(); grids = gen.generate_frequency_based_grids({grids}); print('🎯 {grids} grilles générées:'); [print(f'  {{i+1:2d}}. {{grid}}') for i, grid in enumerate(grids)]"
        
        if export_csv:
            command += f"; import os; os.makedirs('keno_output', exist_ok=True); df = pd.DataFrame([{{'grille': i+1, 'numeros': ', '.join(map(str, grid))}} for i, grid in enumerate(grids)]); filename = f'keno_custom_{{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}}.csv'; df.to_csv(f'keno_output/{{filename}}', index=False); print(f'✅ Export CSV: keno_output/{{filename}}')\""
        else:
            command += '\"'
        
        description = f"Générateur Personnalisé ({grids} grilles)"
        
        print(f"\n{Colors.WARNING}⚠️  Configuration finale:{Colors.ENDC}")
        print(f"   • Grilles: {grids}")
        print(f"   • Export CSV: {'Oui' if export_csv else 'Non'}")
        
        confirm = input(f"\n{Colors.OKGREEN}Confirmer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_keno_patterns_test(self):
        """Test des patterns - vérification complète"""
        print(f"\n{Colors.BOLD}🔍 TEST DES PATTERNS KENO{Colors.ENDC}")
        print("Vérification complète de tous les patterns :")
        print("  • 🔢 15 types de patterns analysés")
        print("  • 🚀 Test DuckDB vs Pandas")
        print("  • 📊 Validation des calculs")
        print("  • 🎯 Test de génération")
        print()
        
        confirm = input(f"{Colors.OKGREEN}Lancer les tests de vérification ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            command = "python test_patterns_verification.py"
            description = "Test Complet des Patterns"
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_keno_intelligent_generator(self):
        """Générateur intelligent v2 avec PuLP et profil intelligent"""
        print(f"\n{Colors.BOLD}🧠 GÉNÉRATEUR KENO INTELLIGENT V2{Colors.ENDC}")
        print("Générateur avancé avec optimisation intelligente :")
        print("  • 🎯 TOP 30 numéros avec profil intelligent")
        print("  • 🔗 Génération basée sur paires fréquentes")
        print("  • ⚖️  Optimisation PuLP avec diversité garantie")
        print("  • 📊 Paramètres optimaux (pair/impair, zones, sommes)")
        print("  • 🎲 Système réducteur multi-stratégies")
        print("  • 🌟 4 algorithmes hybrides pour la diversité")
        print()
        
        # Vérifier s'il y a un CSV TOP 30 standard
        top30_found = False
        top30_file = Path("keno_output/keno_top30.csv")
        
        if top30_file.exists():
            age_hours = (datetime.now().timestamp() - top30_file.stat().st_mtime) / 3600
            print(f"📂 TOP 30 trouvé: {top30_file.name} (âge: {age_hours:.1f}h)")
            top30_found = True
        
        if not top30_found:
            print(f"{Colors.WARNING}⚠️  Aucun TOP 30 récent trouvé{Colors.ENDC}")
            generate_top30 = input("🔄 Générer un nouveau TOP 30 d'abord ? (O/n): ").strip().lower()
            
            if generate_top30 in ['', 'o', 'oui', 'y', 'yes']:
                print(f"\n{Colors.OKBLUE}🚀 Génération TOP 30 avec KenoGeneratorAdvanced...{Colors.ENDC}")
                try:
                    from keno.keno_generator_advanced import KenoGeneratorAdvanced
                    
                    generator = KenoGeneratorAdvanced()
                    if generator.load_data():
                        generator.analyze_patterns()
                        top30_numbers = generator.calculate_and_export_top30()
                        print(f"{Colors.OKGREEN}✅ TOP 30 généré avec succès!{Colors.ENDC}")
                        print(f"   🎯 Top 10: {', '.join(map(str, top30_numbers[:10]))}")
                    else:
                        print(f"{Colors.FAIL}❌ Impossible de charger les données{Colors.ENDC}")
                        
                except Exception as e:
                    print(f"{Colors.FAIL}❌ Erreur génération TOP 30: {e}{Colors.ENDC}")
                    
                print()
        
        # Sélection du profil
        print(f"{Colors.OKBLUE}📋 Profils disponibles:{Colors.ENDC}")
        print("1️⃣  Bas - 50 grilles (couverture minimale)")
        print("2️⃣  Moyen - 80 grilles (équilibre optimal)")
        print("3️⃣  Haut - 100 grilles (couverture maximale)")
        print("0️⃣  Retour au menu")
        
        profile_choice = input("\n🎯 Choisissez un profil (défaut: 2): ").strip()
        
        if profile_choice == "0":
            return
            
        try:
            from keno_intelligent_generator_v2 import KenoIntelligentGeneratorV2
            
            # Mapping des choix
            if profile_choice == "1":
                profile = "bas"
            elif profile_choice == "3":
                profile = "haut"
            else:
                profile = "moyen"  # Défaut
            
            print(f"\n{Colors.OKGREEN}🚀 Lancement du générateur intelligent v2 - Profil {profile.title()}{Colors.ENDC}")
            
            # Génération du système avec détection automatique CSV
            generator = KenoIntelligentGeneratorV2()
            success = generator.generate_system(profile)
            
            if success:
                print(f"\n{Colors.OKGREEN}✅ Système {profile} généré avec succès!{Colors.ENDC}")
                print(f"📁 Fichiers exportés dans le dossier 'keno_output/'")
                input("\n👉 Appuyez sur Entrée pour continuer...")
            else:
                print(f"\n{Colors.FAIL}❌ Échec de génération du système {profile}{Colors.ENDC}")
                input("\n👉 Appuyez sur Entrée pour continuer...")
                
        except ImportError as e:
            print(f"\n{Colors.FAIL}❌ Erreur d'import: {e}{Colors.ENDC}")
            print("Assurez-vous que le module keno_intelligent_generator_v2 est disponible")
            input("\n👉 Appuyez sur Entrée pour continuer...")
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Erreur lors de la génération: {e}{Colors.ENDC}")
            input("\n👉 Appuyez sur Entrée pour continuer...")
    
    def handle_keno_generate_top30(self):
        """Génération du TOP 30 avec KenoGeneratorAdvanced"""
        print(f"\n{Colors.BOLD}📊 GÉNÉRATION TOP 30 KENO AVANCÉ{Colors.ENDC}")
        print("Génération du TOP 30 intelligent avec scoring multi-critères :")
        print("  • 🎯 Analyse de 3520+ tirages historiques")
        print("  • 🧠 Scoring intelligent (fréquence, retard, tendances, pairs)")
        print("  • 📈 Export CSV avec métadonnées complètes")
        print("  • 🚀 Optimisé avec DuckDB")
        print()
        
        confirm = input(f"{Colors.OKGREEN}Générer le TOP 30 ? (O/n): {Colors.ENDC}").strip().lower()
        
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            try:
                from keno.keno_generator_advanced import KenoGeneratorAdvanced
                
                print(f"\n{Colors.OKBLUE}🚀 Lancement KenoGeneratorAdvanced...{Colors.ENDC}")
                
                generator = KenoGeneratorAdvanced()
                
                if generator.load_data():
                    generator.analyze_patterns()
                    top30_numbers = generator.calculate_and_export_top30()
                    
                    print(f"\n{Colors.OKGREEN}✅ TOP 30 généré avec succès!{Colors.ENDC}")
                    print(f"   🎯 Top 10: {', '.join(map(str, top30_numbers[:10]))}")
                    print(f"   📁 Fichier exporté dans 'keno_output/'")
                    
                else:
                    print(f"\n{Colors.FAIL}❌ Impossible de charger les données{Colors.ENDC}")
                    
            except ImportError as e:
                print(f"\n{Colors.FAIL}❌ Erreur d'import: {e}{Colors.ENDC}")
            except Exception as e:
                print(f"\n{Colors.FAIL}❌ Erreur: {e}{Colors.ENDC}")
                
            input("\n👉 Appuyez sur Entrée pour continuer...")
        else:
            print("Opération annulée.")
            self.wait_and_continue()
        """Générateur intelligent v2 avec PuLP et profil intelligent"""
        print(f"\n{Colors.BOLD}🧠 GÉNÉRATEUR KENO INTELLIGENT V2{Colors.ENDC}")
        print("Générateur avancé avec optimisation intelligente :")
        print("  • 🎯 TOP 30 numéros avec profil intelligent")
        print("  • 🔗 Génération basée sur paires fréquentes")
        print("  • ⚖️  Optimisation PuLP avec diversité garantie")
        print("  • 📊 Paramètres optimaux (pair/impair, zones, sommes)")
        print("  • 🎲 Système réducteur multi-stratégies")
        print("  • 🌟 4 algorithmes hybrides pour la diversité")
        print()
        
        # Sélection du profil
        print(f"{Colors.OKBLUE}📋 Profils disponibles:{Colors.ENDC}")
        print("1️⃣  Bas - 50 grilles (couverture minimale)")
        print("2️⃣  Moyen - 80 grilles (équilibre optimal)")
        print("3️⃣  Haut - 100 grilles (couverture maximale)")
        print("0️⃣  Retour au menu")
        
        profile_choice = input("\n🎯 Choisissez un profil (défaut: 2): ").strip()
        
        if profile_choice == "0":
            return
            
        try:
            from keno_intelligent_generator_v2 import KenoIntelligentGeneratorV2
            
            # Mapping des choix
            if profile_choice == "1":
                profile = "bas"
            elif profile_choice == "3":
                profile = "haut"
            else:
                profile = "moyen"  # Défaut
            
            print(f"\n{Colors.OKGREEN}🚀 Lancement du générateur intelligent v2 - Profil {profile.title()}{Colors.ENDC}")
            
            # Génération du système
            generator = KenoIntelligentGeneratorV2()
            success = generator.generate_system(profile)
            
            if success:
                print(f"\n{Colors.OKGREEN}✅ Système {profile} généré avec succès!{Colors.ENDC}")
                print(f"📁 Fichiers exportés dans le dossier 'keno_output/'")
                input("\n👉 Appuyez sur Entrée pour continuer...")
            else:
                print(f"\n{Colors.FAIL}❌ Échec de génération du système {profile}{Colors.ENDC}")
                input("\n👉 Appuyez sur Entrée pour continuer...")
                
        except ImportError as e:
            print(f"\n{Colors.FAIL}❌ Erreur d'import: {e}{Colors.ENDC}")
            print("Assurez-vous que le module keno_intelligent_generator_v2 est disponible")
            input("\n👉 Appuyez sur Entrée pour continuer...")
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Erreur lors de la génération: {e}{Colors.ENDC}")
            input("\n👉 Appuyez sur Entrée pour continuer...")
        print(f"   • Optimisation PuLP/Alternative")
        print(f"   • Export CSV + métadonnées")
        
        confirm = input(f"\n{Colors.OKGREEN}Confirmer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            # Passer le profil via variable d'environnement ou modification du script
            temp_script = f"""
import sys
sys.path.append('.')
from keno_intelligent_generator import KenoIntelligentGenerator

# Simulation du choix automatique
original_input = input
def mock_input(prompt):
    if "profil" in prompt.lower():
        return "{profile_choice or '2'}"
    return original_input(prompt)

__builtins__['input'] = mock_input

# Exécution
generator = KenoIntelligentGenerator()
if generator.load_and_analyze():
    generator.calculate_intelligent_top30()
    generator.analyze_optimal_parameters()
    generator.extract_top_pairs()
    grids, metadata = generator.generate_system_grids("{profile}")
    files = generator.export_system_grids(grids, metadata)
    print("✅ Génération terminée avec succès!")
else:
    print("❌ Échec de la génération")
"""
            
            with open("temp_intelligent_script.py", "w") as f:
                f.write(temp_script)
            
            command = "python temp_intelligent_script.py && rm temp_intelligent_script.py"
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_keno_system_reducer(self):
        """Systèmes réducteurs avec différents profils"""
        print(f"\n{Colors.BOLD}📊 SYSTÈMES RÉDUCTEURS KENO{Colors.ENDC}")
        print("Génération de systèmes optimisés avec réduction intelligente :")
        print("  • 🎯 Basé sur TOP 30 intelligent")
        print("  • 🔗 Utilisation des paires les plus fréquentes")
        print("  • ⚖️  Équilibrage automatique (zones, parité, sommes)")
        print("  • 📊 3 profils de couverture")
        print()
        
        # Choix multiple de profils
        print(f"{Colors.OKBLUE}📋 Générer plusieurs profils ?{Colors.ENDC}")
        print("1️⃣  Un seul profil")
        print("2️⃣  Tous les profils (Bas + Moyen + Haut)")
        print("0️⃣  Retour au menu")
        
        choice = input("\n🎯 Votre choix: ").strip()
        
        if choice == "0":
            self.wait_and_continue()
            return
        elif choice == "2":
            # Génération de tous les profils
            temp_script = """
import sys
sys.path.append('.')
from keno_intelligent_generator import KenoIntelligentGenerator

generator = KenoIntelligentGenerator()
if generator.load_and_analyze():
    generator.calculate_intelligent_top30()
    generator.analyze_optimal_parameters()
    generator.extract_top_pairs()
    
    for profile in ["bas", "moyen", "haut"]:
        print(f"\\n🎯 Génération profil {profile.upper()}...")
        grids, metadata = generator.generate_system_grids(profile)
        files = generator.export_system_grids(grids, metadata)
        print(f"✅ Profil {profile} terminé")
    
    print("\\n🏆 TOUS LES PROFILS GÉNÉRÉS AVEC SUCCÈS!")
else:
    print("❌ Échec de l'initialisation")
"""
            
            with open("temp_all_profiles.py", "w") as f:
                f.write(temp_script)
            
            command = "python temp_all_profiles.py && rm temp_all_profiles.py"
            description = "Génération de tous les profils (Bas + Moyen + Haut)"
            
        else:
            # Un seul profil
            print(f"\n{Colors.OKBLUE}📋 Profil à générer:{Colors.ENDC}")
            print("1️⃣  Bas - 50 grilles")
            print("2️⃣  Moyen - 80 grilles")
            print("3️⃣  Haut - 100 grilles")
            
            profile_choice = input("\n🎯 Profil (défaut: 2): ").strip()
            
            if profile_choice == "1":
                profile = "bas"
            elif profile_choice == "3":
                profile = "haut"
            else:
                profile = "moyen"
            
            temp_script = f"""
import sys
sys.path.append('.')
from keno_intelligent_generator import KenoIntelligentGenerator

generator = KenoIntelligentGenerator()
if generator.load_and_analyze():
    generator.calculate_intelligent_top30()
    generator.analyze_optimal_parameters()
    generator.extract_top_pairs()
    grids, metadata = generator.generate_system_grids("{profile}")
    files = generator.export_system_grids(grids, metadata)
    print("✅ Génération terminée avec succès!")
else:
    print("❌ Échec de la génération")
"""
            
            with open("temp_single_profile.py", "w") as f:
                f.write(temp_script)
            
            command = "python temp_single_profile.py && rm temp_single_profile.py"
            description = f"Système Réducteur - Profil {profile.capitalize()}"
        
        confirm = input(f"\n{Colors.OKGREEN}Confirmer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_keno_pairs_generator(self):
        """Générateur basé sur les paires fréquentes"""
        print(f"\n{Colors.BOLD}🎲 GÉNÉRATEUR BASÉ SUR PAIRES{Colors.ENDC}")
        print("Génération spécialisée utilisant les combinaisons de paires :")
        print("  • 🔗 Paires les plus fréquentes du TOP 30")
        print("  • 🎯 Composition optimale des grilles")
        print("  • ⚖️  Respect des paramètres optimaux")
        print("  • 📊 Export détaillé avec analyse")
        print()
        
        # Configuration
        while True:
            try:
                grids_input = input(f"Nombre de grilles (10-200, défaut: 50): ").strip()
                grids = 50 if not grids_input else int(grids_input)
                if grids < 10 or grids > 200:
                    print(f"{Colors.FAIL}❌ Nombre doit être entre 10 et 200{Colors.ENDC}")
                    continue
                break
            except ValueError:
                print(f"{Colors.FAIL}❌ Veuillez entrer un nombre valide{Colors.ENDC}")
        
        # Stratégie de paires
        print(f"\n{Colors.OKBLUE}🔗 Stratégie de paires:{Colors.ENDC}")
        print("1️⃣  Conservative (3-4 paires par grille max)")
        print("2️⃣  Équilibrée (2-3 paires par grille)")
        print("3️⃣  Agressive (1-2 paires par grille)")
        
        strategy_choice = input("Stratégie (défaut: 2): ").strip() or "2"
        
        strategy_map = {
            "1": "conservative",
            "2": "equilibree", 
            "3": "agressive"
        }
        
        strategy = strategy_map.get(strategy_choice, "equilibree")
        
        temp_script = f"""
import sys
sys.path.append('.')
from keno_intelligent_generator import KenoIntelligentGenerator
import pandas as pd
from datetime import datetime
import os

generator = KenoIntelligentGenerator()
if generator.load_and_analyze():
    generator.calculate_intelligent_top30()
    generator.analyze_optimal_parameters()
    generator.extract_top_pairs()
    
    print("🎲 Génération avec focus sur les paires...")
    
    # Génération spécialisée paires
    grids = []
    strategy = "{strategy}"
    
    for i in range({grids}):
        grid = generator.generate_smart_grid()  # Utilise déjà la logique de paires
        grids.append(grid)
    
    print(f"✅ {{len(grids)}} grilles générées avec stratégie {{strategy}}")
    
    # Export spécialisé
    os.makedirs("keno_output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pairs_data = []
    for i, grid in enumerate(grids, 1):
        # Analyse des paires dans la grille
        pairs_in_grid = []
        for j in range(len(grid)):
            for k in range(j+1, len(grid)):
                pair = tuple(sorted([grid[j], grid[k]]))
                if pair in [tuple(sorted(p)) for p in generator.top_pairs[:50]]:
                    pairs_in_grid.append(pair)
        
        pairs_data.append({{
            'grille': i,
            'numeros': ', '.join(map(str, grid)),
            'nb_paires_frequentes': len(pairs_in_grid),
            'paires_details': '; '.join([f'{{p[0]}}-{{p[1]}}' for p in pairs_in_grid]),
            'somme': sum(grid),
            'pairs': sum(1 for n in grid if n % 2 == 0),
            'qualite': round(generator.validate_grid_quality(grid)['global'], 3)
        }})
    
    df = pd.DataFrame(pairs_data)
    filename = f"keno_output/pairs_generator_{{timestamp}}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Export spécialisé paires: {{filename}}")
    print(f"📊 Paires moyennes par grille: {{df['nb_paires_frequentes'].mean():.1f}}")
    
else:
    print("❌ Échec de l'initialisation")
"""
        
        with open("temp_pairs_generator.py", "w") as f:
            f.write(temp_script)
        
        command = "python temp_pairs_generator.py && rm temp_pairs_generator.py"
        description = f"Générateur Paires ({grids} grilles, stratégie {strategy})"
        
        print(f"\n{Colors.WARNING}⚠️  Configuration:{Colors.ENDC}")
        print(f"   • Grilles: {grids}")
        print(f"   • Stratégie: {strategy}")
        print(f"   • Focus sur paires fréquentes")
        
        confirm = input(f"\n{Colors.OKGREEN}Confirmer la génération ? (O/n): {Colors.ENDC}").strip().lower()
        if confirm in ['', 'o', 'oui', 'y', 'yes']:
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_loto_adaptive_v2(self):
        """Générateur Loto Adaptatif V2 avec Auto-amélioration"""
        print(f"\n{Colors.BOLD}🚀 GÉNÉRATEUR LOTO ADAPTATIF V2{Colors.ENDC}")
        print("Générateur avancé avec auto-amélioration et stratégies adaptatives :")
        print("  • 🧠 Training ML adaptatif automatique")
        print("  • 🎯 Privilégie les numéros du TOP 25 ML")
        print("  • ⚡ Aucune exclusion par défaut (tous les 49 numéros)")
        print("  • 📊 Export automatique du TOP 25 basé sur le ML")
        print("  • 🔄 Auto-amélioration continue des modèles")
        print()
        
        # Configuration des paramètres
        print(f"{Colors.OKBLUE}⚙️  Configuration des paramètres:{Colors.ENDC}")
        print("1️⃣  Mode ultra-rapide (1,000 simulations + training rapide)")
        print("2️⃣  Mode équilibré (10,000 simulations + training standard)")
        print("3️⃣  Mode intensif (50,000 simulations + training complet)")
        print("4️⃣  Mode personnalisé avancé")
        print("0️⃣  Retour au menu principal")
        
        config_choice = input("\n🎯 Votre choix de configuration: ").strip()
        
        if config_choice == "0":
            self.wait_and_continue()
            return
        elif config_choice == "1":
            command = f"python loto/loto_generator_advanced_Version2.py --quick --fast-training --silent"
            description = "Générateur Loto Adaptatif V2 (Mode Ultra-Rapide)"
        elif config_choice == "2":
            command = f"python loto/loto_generator_advanced_Version2.py --silent"
            description = "Générateur Loto Adaptatif V2 (Mode Équilibré)"
        elif config_choice == "3":
            command = f"python loto/loto_generator_advanced_Version2.py --intensive --retrain"
            description = "Générateur Loto Adaptatif V2 (Mode Intensif + Ré-entraînement)"
        elif config_choice == "4":
            # Configuration personnalisée avancée
            print(f"\n{Colors.OKBLUE}🔧 Configuration personnalisée avancée:{Colors.ENDC}")
            
            # Nombre de simulations
            while True:
                try:
                    n_sims = input("📊 Nombre de simulations (100-100000, défaut: 10000): ").strip()
                    if not n_sims:
                        n_sims = 10000
                    else:
                        n_sims = int(n_sims)
                    
                    if n_sims < 100 or n_sims > 100000:
                        print("❌ Le nombre de simulations doit être entre 100 et 100,000")
                        continue
                    break
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide")
            
            # Options avancées
            retrain = input("🔄 Forcer le ré-entraînement des modèles ML ? (o/N): ").strip().lower()
            fast_training = input("⚡ Utiliser le training rapide ? (o/N): ").strip().lower()
            
            # Options d'exclusion
            print(f"\n{Colors.WARNING}🚫 Gestion des exclusions:{Colors.ENDC}")
            print("1️⃣  Aucune exclusion (recommandé - tous les 49 numéros)")
            print("2️⃣  Exclusion automatique (3 derniers tirages)")
            print("3️⃣  Exclusion personnalisée")
            
            exclusion_choice = input("Option d'exclusion (défaut: 1): ").strip() or "1"
            
            # Construction de la commande
            command = f"python loto/loto_generator_advanced_Version2.py -s {n_sims} --silent"
            
            if retrain in ['o', 'oui', 'y', 'yes']:
                command += " --retrain"
            
            if fast_training in ['o', 'oui', 'y', 'yes']:
                command += " --fast-training"
            
            if exclusion_choice == "2":
                command += " --exclude auto"
            elif exclusion_choice == "3":
                excluded_nums = input("Numéros à exclure (1-49, séparés par des virgules): ").strip()
                if excluded_nums:
                    command += f" --exclude {excluded_nums}"
            
            description = f"Générateur Loto Adaptatif V2 Personnalisé ({n_sims:,} simulations)"
        else:
            print("❌ Choix invalide")
            self.wait_and_continue()
            return
        
        # Confirmation finale
        print(f"\n{Colors.OKGREEN}✅ Configuration choisie:{Colors.ENDC}")
        print(f"   Commande: {command}")
        print(f"\n{Colors.WARNING}💡 Nouvelles fonctionnalités V2:{Colors.ENDC}")
        print("   • Auto-amélioration continue des modèles ML")
        print("   • Export automatique du TOP 25 basé sur les prédictions")
        print("   • Privilégie intelligemment les numéros du TOP 25")
        print("   • Aucune exclusion par défaut pour maximiser les possibilités")
        print()
        
        confirm = input("Lancer le générateur adaptatif V2 ? (o/N): ").strip().lower()
        
        if confirm in ['o', 'oui', 'y', 'yes']:
            print(f"\n{Colors.OKBLUE}🚀 Lancement du générateur adaptatif V2...{Colors.ENDC}")
            print("⚠️  Note: Ce processus peut prendre quelques minutes")
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_loto_adaptive_training(self):
        """Training ML adaptatif pour améliorer les modèles"""
        print(f"\n{Colors.BOLD}🧠 TRAINING ML ADAPTATIF LOTO{Colors.ENDC}")
        print("Ré-entraînement intelligent des modèles ML avec adaptation continue :")
        print("  • 🔄 Met à jour les modèles avec les dernières données")
        print("  • 🎯 Optimise les paramètres automatiquement") 
        print("  • ⚡ Mode rapide ou complet disponible")
        print("  • 📊 Validation croisée et métriques de performance")
        print("  • 💾 Sauvegarde automatique des meilleurs modèles")
        print()
        
        # Choix du mode de training
        print(f"{Colors.OKBLUE}⚙️ Mode de training:{Colors.ENDC}")
        print("1️⃣  Training rapide (paramètres optimisés pour la vitesse)")
        print("2️⃣  Training complet (recherche exhaustive des meilleurs paramètres)")
        print("3️⃣  Training personnalisé")
        print("0️⃣  Retour au menu principal")
        
        training_choice = input("\n🎯 Mode de training (défaut: 1): ").strip() or "1"
        
        if training_choice == "0":
            self.wait_and_continue()
            return
        elif training_choice == "1":
            command = "python loto/loto_generator_advanced_Version2.py --retrain --fast-training --silent -s 1000"
            description = "Training ML Adaptatif Rapide"
        elif training_choice == "2":
            command = "python loto/loto_generator_advanced_Version2.py --retrain --silent -s 5000"
            description = "Training ML Adaptatif Complet"
        elif training_choice == "3":
            # Configuration personnalisée du training
            print(f"\n{Colors.OKBLUE}🔧 Configuration du training personnalisé:{Colors.ENDC}")
            
            fast_mode = input("⚡ Utiliser le mode rapide ? (O/n): ").strip().lower()
            
            # Nombre de simulations pour validation
            while True:
                try:
                    n_sims = input("📊 Nombre de simulations pour validation (100-10000, défaut: 2000): ").strip()
                    if not n_sims:
                        n_sims = 2000
                    else:
                        n_sims = int(n_sims)
                    
                    if n_sims < 100 or n_sims > 10000:
                        print("❌ Le nombre doit être entre 100 et 10,000")
                        continue
                    break
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide")
            
            command = f"python loto/loto_generator_advanced_Version2.py --retrain --silent -s {n_sims}"
            
            if fast_mode not in ['n', 'non', 'no']:
                command += " --fast-training"
            
            description = f"Training ML Adaptatif Personnalisé ({n_sims:,} simulations)"
        else:
            print("❌ Choix invalide")
            self.wait_and_continue()
            return
        
        # Confirmation et informations
        print(f"\n{Colors.WARNING}📋 Informations sur le training adaptatif:{Colors.ENDC}")
        print("   • Les modèles existants seront mis à jour")
        print("   • Les performances seront évaluées automatiquement")
        print("   • Seuls les modèles améliorés seront conservés")
        print("   • Le TOP 25 sera recalculé avec les nouveaux modèles")
        
        confirm = input(f"\n{Colors.OKGREEN}Lancer le training adaptatif ? (O/n): {Colors.ENDC}").strip().lower()
        
        if confirm not in ['n', 'non', 'no']:
            print(f"\n{Colors.OKBLUE}🧠 Lancement du training adaptatif...{Colors.ENDC}")
            print("⚠️  Note: Le training peut prendre plusieurs minutes")
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_loto_no_exclusion(self):
        """Générateur Loto sans exclusion - utilise tous les 49 numéros"""
        print(f"\n{Colors.BOLD}⚡ GÉNÉRATEUR LOTO SANS EXCLUSION{Colors.ENDC}")
        print("Générateur utilisant TOUS les 49 numéros sans exclusion :")
        print("  • ✅ Aucun numéro exclu (politique par défaut maintenant)")
        print("  • 🎯 Privilégie les numéros du TOP 25 ML")
        print("  • 🧠 Utilise l'intelligence artificielle pour le choix")
        print("  • 📊 Maximise les possibilités de gains")
        print("  • ⚡ Performance optimisée")
        print()
        
        # Configuration rapide
        print(f"{Colors.OKBLUE}⚙️ Configurations rapides:{Colors.ENDC}")
        print("1️⃣  Génération express (1,000 simulations)")
        print("2️⃣  Génération standard (10,000 simulations)")
        print("3️⃣  Génération intensive (25,000 simulations)")
        print("4️⃣  Configuration personnalisée")
        print("0️⃣  Retour au menu principal")
        
        config_choice = input("\n🎯 Configuration (défaut: 2): ").strip() or "2"
        
        if config_choice == "0":
            self.wait_and_continue()
            return
        elif config_choice == "1":
            command = "python loto/loto_generator_advanced_Version2.py --quick --silent"
            description = "Générateur Sans Exclusion Express"
        elif config_choice == "2":
            command = "python loto/loto_generator_advanced_Version2.py --silent"
            description = "Générateur Sans Exclusion Standard"
        elif config_choice == "3":
            command = "python loto/loto_generator_advanced_Version2.py -s 25000 --silent"
            description = "Générateur Sans Exclusion Intensif"
        elif config_choice == "4":
            # Configuration personnalisée
            print(f"\n{Colors.OKBLUE}🔧 Configuration personnalisée:{Colors.ENDC}")
            
            # Nombre de simulations
            while True:
                try:
                    n_sims = input("📊 Nombre de simulations (100-50000, défaut: 10000): ").strip()
                    if not n_sims:
                        n_sims = 10000
                    else:
                        n_sims = int(n_sims)
                    
                    if n_sims < 100 or n_sims > 50000:
                        print("❌ Le nombre de simulations doit être entre 100 et 50,000")
                        continue
                    break
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide")
            
            # Nombre de processeurs
            import multiprocessing as mp
            max_cores = mp.cpu_count()
            default_cores = max_cores - 1 if max_cores > 1 else 1
            
            while True:
                try:
                    n_cores = input(f"🔄 Nombre de processeurs (1-{max_cores}, défaut: {default_cores}): ").strip()
                    if not n_cores:
                        n_cores = default_cores
                    else:
                        n_cores = int(n_cores)
                    
                    if n_cores < 1 or n_cores > max_cores:
                        print(f"❌ Le nombre de processeurs doit être entre 1 et {max_cores}")
                        continue
                    break
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide")
            
            command = f"python loto/loto_generator_advanced_Version2.py -s {n_sims} -c {n_cores} --silent"
            description = f"Générateur Sans Exclusion Personnalisé ({n_sims:,} simulations, {n_cores} cœurs)"
        else:
            print("❌ Choix invalide")
            self.wait_and_continue()
            return
        
        # Informations importantes
        print(f"\n{Colors.OKGREEN}✅ Avantages du générateur sans exclusion:{Colors.ENDC}")
        print("   • Tous les 49 numéros sont candidats")
        print("   • Pas de biais d'exclusion arbitraire")
        print("   • Le ML décide intelligemment quels numéros privilégier")
        print("   • Meilleure couverture probabiliste")
        
        print(f"\n{Colors.WARNING}💡 Note importante:{Colors.ENDC}")
        print("   Cette approche est maintenant la stratégie par défaut")
        print("   recommandée pour maximiser les chances de gains")
        
        confirm = input(f"\n{Colors.OKGREEN}Lancer la génération sans exclusion ? (O/n): {Colors.ENDC}").strip().lower()
        
        if confirm not in ['n', 'non', 'no']:
            print(f"\n{Colors.OKBLUE}⚡ Lancement du générateur sans exclusion...{Colors.ENDC}")
            print("📊 Utilisation de tous les 49 numéros avec privilège TOP 25 ML")
            self.execute_command(command, description)
        else:
            print("Opération annulée.")
            self.wait_and_continue()
    
    def handle_csv_to_parquet(self):
        """Convertit les fichiers CSV Loto vers le format Parquet pour optimiser les performances"""
        print(f"\n{Colors.BOLD}🔄 CONVERSION CSV → PARQUET{Colors.ENDC}")
        print("Optimisation des données Loto pour de meilleures performances :")
        print("  • 📦 Compression avancée (réduction de 70-80%)")
        print("  • ⚡ Accès ultra-rapide aux données")
        print("  • 🧠 Optimisé pour les analyses ML")
        print("  • 🔧 Compatible avec tous les générateurs")
        print("  • 💾 Économie d'espace disque")
        print()
        
        # Vérifier l'existence du script de conversion
        parquet_script = self.base_path / "loto" / "csvToParquet.py"
        if not parquet_script.exists():
            print(f"{Colors.FAIL}❌ Script de conversion non trouvé: {parquet_script}{Colors.ENDC}")
            self.wait_and_continue()
            return
        
        # Vérifier l'existence de fichiers CSV Loto
        loto_data_dir = self.base_path / "loto" / "loto_data"
        if loto_data_dir.exists():
            csv_files = list(loto_data_dir.glob("*.csv"))
            if csv_files:
                print(f"{Colors.OKGREEN}📂 Fichiers CSV détectés:{Colors.ENDC}")
                for csv_file in csv_files[:3]:  # Afficher max 3 fichiers
                    size = csv_file.stat().st_size / (1024*1024)
                    mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
                    print(f"   📄 {csv_file.name} ({size:.1f}MB, {mtime.strftime('%d/%m/%Y')})")
                
                if len(csv_files) > 3:
                    print(f"   📄 ... et {len(csv_files) - 3} autres fichiers")
            else:
                print(f"{Colors.WARNING}⚠️  Aucun fichier CSV trouvé dans {loto_data_dir}{Colors.ENDC}")
                print("💡 Téléchargez d'abord les données Loto (option 1)")
                self.wait_and_continue()
                return
        else:
            print(f"{Colors.WARNING}⚠️  Dossier loto/loto_data non trouvé{Colors.ENDC}")
            print("💡 Téléchargez d'abord les données Loto (option 1)")
            self.wait_and_continue()
            return
        
        # Vérifier si des fichiers Parquet existent déjà
        parquet_files = list(loto_data_dir.glob("*.parquet"))
        if parquet_files:
            print(f"\n{Colors.OKCYAN}📦 Fichiers Parquet existants:{Colors.ENDC}")
            for parquet_file in parquet_files[:3]:
                size = parquet_file.stat().st_size / (1024*1024)
                mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)
                print(f"   📦 {parquet_file.name} ({size:.1f}MB, {mtime.strftime('%d/%m/%Y')})")
            
            print(f"\n{Colors.WARNING}⚠️  Des fichiers Parquet existent déjà.{Colors.ENDC}")
            print("La conversion remplacera les fichiers existants si le CSV est plus récent.")
        
        # Options de conversion
        print(f"\n{Colors.OKBLUE}🔧 Options de conversion:{Colors.ENDC}")
        print("1️⃣  Conversion automatique (recommandé)")
        print("2️⃣  Conversion forcée (remplace tous les Parquet)")
        print("3️⃣  Vérification et info seulement")
        print("0️⃣  Annuler")
        
        conversion_choice = input(f"\n{Colors.BOLD}Option (défaut: 1): {Colors.ENDC}").strip() or "1"
        
        if conversion_choice == "0":
            print("Opération annulée.")
            self.wait_and_continue()
            return
        elif conversion_choice == "3":
            # Mode information seulement
            print(f"\n{Colors.OKCYAN}📊 Informations sur la conversion:{Colors.ENDC}")
            
            total_csv_size = sum(f.stat().st_size for f in csv_files) / (1024*1024)
            total_parquet_size = sum(f.stat().st_size for f in parquet_files) / (1024*1024) if parquet_files else 0
            
            print(f"   📄 Fichiers CSV: {len(csv_files)} fichiers ({total_csv_size:.1f}MB)")
            print(f"   📦 Fichiers Parquet: {len(parquet_files)} fichiers ({total_parquet_size:.1f}MB)")
            
            if parquet_files:
                compression_ratio = ((total_csv_size - total_parquet_size) / total_csv_size) * 100
                print(f"   🗜️ Compression actuelle: {compression_ratio:.1f}%")
                print(f"   💾 Espace économisé: {total_csv_size - total_parquet_size:.1f}MB")
            
            print(f"\n{Colors.OKGREEN}💡 Avantages de la conversion Parquet:{Colors.ENDC}")
            print("   • Lecture 5-10x plus rapide que CSV")
            print("   • Compression 70-80% (économie d'espace)")
            print("   • Optimisé pour les analyses ML")
            print("   • Type de données préservé")
            
            self.wait_and_continue()
            return
        
        # Exécuter la conversion
        if conversion_choice == "2":
            # Mode forcé - supprimer d'abord les Parquet existants
            print(f"\n{Colors.WARNING}🗑️ Suppression des fichiers Parquet existants...{Colors.ENDC}")
            for parquet_file in parquet_files:
                parquet_file.unlink()
                print(f"   ✓ Supprimé: {parquet_file.name}")
        
        # Confirmation finale
        print(f"\n{Colors.OKGREEN}✅ Configuration de conversion:{Colors.ENDC}")
        print(f"   📂 Dossier: loto/loto_data")
        print(f"   📄 Fichiers CSV: {len(csv_files)}")
        print(f"   🔄 Mode: {'Forcé' if conversion_choice == '2' else 'Automatique'}")
        
        confirm = input(f"\n{Colors.BOLD}Lancer la conversion ? (O/n): {Colors.ENDC}").strip().lower()
        
        if confirm not in ['n', 'non', 'no']:
            print(f"\n{Colors.OKBLUE}🔄 Lancement de la conversion CSV → Parquet...{Colors.ENDC}")
            self.execute_command("python loto/csvToParquet.py", "Conversion CSV vers Parquet")
        else:
            print("Opération annulée.")
            self.wait_and_continue()
        
    def run(self):
        """Lance le menu principal"""
        try:
            while True:
                self.print_header()
                self.print_status()
                self.print_menu()
                
                choice = input(f"{Colors.BOLD}🎯 Votre choix: {Colors.ENDC}").strip()
                
                if not self.handle_choice(choice):
                    break
                    
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}🛑 Interruption par l'utilisateur{Colors.ENDC}")
            print(f"{Colors.OKGREEN}👋 Au revoir !{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.FAIL}💥 Erreur inattendue: {str(e)}{Colors.ENDC}")


def main():
    """Point d'entrée principal"""
    print(f"{Colors.HEADER}🚀 Initialisation du Menu Loto/Keno...{Colors.ENDC}")
    time.sleep(1)
    
    menu = LotoKenoMenu()
    menu.run()


if __name__ == "__main__":
    main()
