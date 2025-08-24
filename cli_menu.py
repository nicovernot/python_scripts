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
        print("╚═══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKCYAN}📅 Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}{Colors.ENDC}")
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
        print()
        
        print(f"{Colors.OKBLUE}🎲 ANALYSE LOTO{Colors.ENDC}")
        print("  4️⃣  Générer 3 grilles Loto (rapide)")
        print("  5️⃣  Générer 5 grilles Loto (complet)")
        print("  6️⃣  Générer grilles avec visualisations")
        print("  7️⃣  Analyse Loto personnalisée")
        print("  2️⃣2️⃣ Générateur Loto avancé (ML + IA)")
        print()
        
        print(f"{Colors.BOLD}🎯 TOP NUMÉROS ÉQUILIBRÉS{Colors.ENDC}")
        print("  2️⃣8️⃣ 🏆 TOP 25 Loto équilibrés (stratégie optimisée)")
        print("  2️⃣9️⃣ 🏆 TOP 30 Keno équilibrés (stratégie optimisée)")
        print("  3️⃣0️⃣ 📊 Voir TOP 25 Loto (dernière génération)")
        print("  3️⃣1️⃣ 📊 Voir TOP 30 Keno (dernière génération)")
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
            self.execute_command("python test_api.py", "Test de l'API Flask")
            
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
            print(f"\n{Colors.WARNING}🚧 Générateur Loto Avancé (ML + IA){Colors.ENDC}")
            print("Ce générateur utilise des techniques avancées:")
            print("  • Machine Learning (XGBoost)")
            print("  • Analyse statistique approfondie") 
            print("  • Optimisation multi-critères")
            print("  • Cache Redis pour les performances")
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
                self.execute_command("python keno/analyse_stats_keno_complet.py", "Statistiques Keno Complètes")
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
                
        elif choice == "0":
            print(f"\n{Colors.OKGREEN}👋 Au revoir ! Bonne chance pour vos analyses !{Colors.ENDC}")
            return False
            
        else:
            print(f"\n{Colors.FAIL}❌ Choix invalide. Veuillez réessayer.{Colors.ENDC}")
            self.wait_and_continue()
            
        return True
        
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
