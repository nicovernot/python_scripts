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
        
        # Chargement de la configuration
        self.config = load_config()
        
        # Chemins depuis la configuration
        self.loto_csv = get_config_path('LOTO_CSV_PATH') or self.base_path / "loto" / "loto_data" / "loto_201911.csv"
        self.keno_csv = get_config_path('KENO_CSV_PATH') or self.base_path / "keno" / "keno_data" / "keno_202010.csv"
        
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
            print(f"  🎲 Loto:  {Colors.OKGREEN}✓ Disponible{Colors.ENDC} ({size:.1f}MB, MAJ: {mtime.strftime('%d/%m/%Y')})")
        else:
            print(f"  🎲 Loto:  {Colors.FAIL}✗ Manquant{Colors.ENDC}")
            
        # Statut Keno
        if self.keno_csv.exists():
            size = self.keno_csv.stat().st_size / (1024*1024)
            mtime = datetime.fromtimestamp(self.keno_csv.stat().st_mtime)
            print(f"  🎰 Keno:  {Colors.OKGREEN}✓ Disponible{Colors.ENDC} ({size:.1f}MB, MAJ: {mtime.strftime('%d/%m/%Y')})")
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
        print()
        
        print(f"{Colors.OKCYAN}🎰 ANALYSE KENO{Colors.ENDC}")
        print("  8️⃣  Analyse Keno (rapide)")
        print("  9️⃣  Analyse Keno avec visualisations")
        print("  1️⃣0️⃣ Analyse Keno personnalisée")
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
        
        # Options
        plots = input("Générer les visualisations ? (o/N) [N]: ").strip().lower()
        export_stats = input("Exporter les statistiques ? (o/N) [N]: ").strip().lower()
        deep_analysis = input("Analyse approfondie (plus lent) ? (o/N) [N]: ").strip().lower()
        
        # Construction de la commande
        command = f"python keno/duckdb_keno.py --csv {self.keno_csv}"
        
        if plots in ['o', 'oui', 'y', 'yes']:
            command += " --plots"
            
        if export_stats in ['o', 'oui', 'y', 'yes']:
            command += " --export-stats"
            
        if deep_analysis in ['o', 'oui', 'y', 'yes']:
            command += " --deep-analysis"
            
        self.execute_command(command, "Analyse Keno Personnalisée")
        
    def handle_systeme_reduit_simple(self):
        """Générateur simple de système réduit"""
        print(f"\n{Colors.BOLD}🎯 Générateur de Système Réduit - Simple{Colors.ENDC}")
        print("=" * 50)
        
        # Saisie des numéros favoris
        print("Entrez vos numéros favoris (8 à 15 numéros recommandés)")
        print("Format: 1,7,12,18,23,29,34,39,45,49")
        nombres_input = input("Numéros favoris: ").strip()
        
        if not nombres_input:
            print(f"{Colors.WARNING}⚠️  Aucun numéro saisi.{Colors.ENDC}")
            self.wait_and_continue()
            return
        
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
        
        # Construction de la commande
        command = f"python grilles/generateur_grilles.py --nombres {nombres_input} --grilles {nb_grilles} --export --format {format_export}"
        
        if nb_nombres_input:
            try:
                nb_nombres = int(nb_nombres_input)
                if nb_nombres >= 7:
                    command += f" --nombres-utilises {nb_nombres}"
                else:
                    print("⚠️  Minimum 7 numéros requis. Utilisation de tous les numéros.")
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
                if nb_nombres >= 7:
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
            self.execute_command("python keno/results_clean.py", "Téléchargement des données Keno")
            
        elif choice == "3":
            self.execute_command("python loto/result.py && python keno/results_clean.py", "Mise à jour de toutes les données")
            
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
            self.execute_command(f"python keno/duckdb_keno.py --csv {self.keno_csv}", "Analyse Keno (Rapide)")
            
        elif choice == "9":
            self.execute_command(f"python keno/duckdb_keno.py --csv {self.keno_csv} --plots --export-stats", "Analyse Keno avec Visualisations")
            
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
            print("Cette fonctionnalité sera bientôt disponible...")
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
