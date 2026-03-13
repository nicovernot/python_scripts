#!/usr/bin/env python3
"""
CLI Menu Interactif — Système Loto
===================================
Menu principal pour toutes les analyses et générations Loto.

Usage:
    python cli_menu.py
"""

import os
import sys
import subprocess
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import pandas as pd

try:
    from config_env import load_config, get_config, get_config_path, get_config_bool
except ImportError:
    def load_config(): return None
    def get_config(key, default=None): return default
    def get_config_path(key, default=None): return Path(default) if default else None
    def get_config_bool(key, default=False): return default


class Colors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKCYAN    = '\033[96m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


class LotoMenu:
    """Menu interactif pour le système Loto."""

    def __init__(self):
        self.base_path   = Path(__file__).parent
        venv_python      = self.base_path / "venv" / "bin" / "python"
        self.python_path = str(venv_python) if venv_python.exists() else "python"

        self.config = load_config()

        self.loto_csv = (
            get_config_path('LOTO_CSV_PATH')
            or self.base_path / "loto" / "loto_data" / "loto_201911.csv"
        )

        self.colors_enabled       = get_config_bool('CLI_COLORS_ENABLED', True)
        self.clear_screen_enabled = get_config_bool('CLI_CLEAR_SCREEN', True)
        self.default_loto_grids   = get_config('DEFAULT_LOTO_GRIDS', 3)
        self.default_loto_strategy= get_config('DEFAULT_LOTO_STRATEGY', 'equilibre')
        self.loto_config_file     = get_config('LOTO_CONFIG_FILE', 'loto/strategies.yml')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clear_screen(self):
        if self.clear_screen_enabled:
            os.system('clear' if os.name == 'posix' else 'cls')

    def wait_and_continue(self, msg="Appuyez sur Entrée pour continuer..."):
        input(f"\n{Colors.BOLD}{msg}{Colors.ENDC}")

    def execute_command(self, command: str, description: str):
        print(f"\n{Colors.BOLD}{description}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}> {command}{Colors.ENDC}")
        print("─" * 55)
        start = time.time()
        try:
            result = subprocess.run(command, shell=True, cwd=self.base_path)
            elapsed = time.time() - start
            if result.returncode == 0:
                print(f"\n{Colors.OKGREEN}Succes  ({elapsed:.1f}s){Colors.ENDC}")
            else:
                print(f"\n{Colors.FAIL}Erreur (code {result.returncode}){Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.FAIL}Exception: {e}{Colors.ENDC}")
        print("=" * 55)
        self.wait_and_continue()

    def show_file_content(self, file_path: Path, max_lines: int = 50):
        try:
            with open(file_path, encoding='utf-8') as f:
                lines = f.readlines()
            print(f"\n{Colors.BOLD}{file_path.name}{Colors.ENDC}")
            print("─" * 50)
            for i, line in enumerate(lines[:max_lines], 1):
                print(f"{i:3d}: {line.rstrip()}")
            if len(lines) > max_lines:
                print(f"{Colors.WARNING}... ({len(lines)-max_lines} lignes supplémentaires){Colors.ENDC}")
        except FileNotFoundError:
            print(f"{Colors.FAIL}Fichier introuvable : {file_path}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Erreur : {e}{Colors.ENDC}")

    def _get_loto_date_range(self) -> str:
        try:
            df = pd.read_csv(self.loto_csv, delimiter=';', nrows=1)
            col = 'date_de_tirage' if 'date_de_tirage' in df.columns else None
            if not col:
                return ""
            df_full = pd.read_csv(self.loto_csv, delimiter=';', usecols=[col])
            dates = pd.to_datetime(df_full[col], dayfirst=True, errors='coerce').dropna()
            if len(dates) == 0:
                return ""
            return f"{dates.min().strftime('%m/%Y')} → {dates.max().strftime('%m/%Y')}"
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def print_header(self):
        self.clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("╔═══════════════════════════════════════════════════════╗")
        print("║              SYSTEME LOTO — Generateur IA             ║")
        print("║          Statistiques Bayesiennes + XGBoost           ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}{Colors.ENDC}")
        print()

    def print_status(self):
        print(f"{Colors.BOLD}Donnees :{Colors.ENDC}")
        if self.loto_csv.exists():
            size  = self.loto_csv.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(self.loto_csv.stat().st_mtime)
            dr    = self._get_loto_date_range()
            dr_str = f", {dr}" if dr else ""
            print(f"  {Colors.OKGREEN}OK{Colors.ENDC}  {self.loto_csv.name}  "
                  f"({size:.1f} MB, MAJ {mtime.strftime('%d/%m/%Y')}{dr_str})")
        else:
            print(f"  {Colors.FAIL}MANQUANT{Colors.ENDC}  {self.loto_csv}")

        bayesian_out = self.base_path / "loto_output" / "bayesian_grilles.json"
        if bayesian_out.exists():
            mtime = datetime.fromtimestamp(bayesian_out.stat().st_mtime)
            print(f"  {Colors.OKGREEN}OK{Colors.ENDC}  bayesian_grilles.json  "
                  f"(genere le {mtime.strftime('%d/%m/%Y %H:%M')})")
        else:
            print(f"  {Colors.WARNING}--{Colors.ENDC}  bayesian_grilles.json  (pas encore genere)")
        print()

    def print_menu(self):
        print(f"{Colors.BOLD}MENU PRINCIPAL{Colors.ENDC}")
        print()

        print(f"{Colors.HEADER}DONNEES{Colors.ENDC}")
        print("   1  Telecharger / mettre a jour les donnees Loto (FDJ)")
        print()

        print(f"{Colors.OKBLUE}GENERATION RAPIDE{Colors.ENDC}")
        print("   2  Generer 3 grilles Loto (rapide)")
        print("   3  Generer 5 grilles Loto (complet)")
        print("   4  Generer grilles avec visualisations")
        print("   5  Generation personnalisee")
        print()

        print(f"{Colors.OKGREEN}GENERATEUR BAYESIEN (RECOMMANDE){Colors.ENDC}")
        print("   6  Generateur Bayesien (DuckDB + XGBoost + PuLP) — 6 grilles optimisees")
        print("   7  Generateur Bayesien avec reentrainement du modele")
        print()

        print(f"{Colors.WARNING}GENERATEUR AVANCE (ML + IA){Colors.ENDC}")
        print("   8  Mode rapide    (1 000 simulations)")
        print("   9  Mode standard  (10 000 simulations)")
        print("  10  Mode intensif  (50 000 simulations)")
        print("  11  Mode personnalise")
        print()

        print(f"{Colors.BOLD}TOP 25 NUMEROS{Colors.ENDC}")
        print("  12  Generer TOP 25 numeros equilibres")
        print("  13  Afficher le dernier TOP 25")
        print("  14  Grilles Loto TOP 25 → CSV optimise")
        print("  15  Grilles Loto TOP 25 → Markdown")
        print()

        print(f"{Colors.BOLD}SYSTEME REDUIT (numeros favoris){Colors.ENDC}")
        print("  16  Systeme reduit simple")
        print("  17  Systeme reduit personnalise")
        print()

        print(f"{Colors.OKCYAN}RESULTATS ET OUTILS{Colors.ENDC}")
        print("  18  Voir les dernieres grilles generees")
        print("  19  Voir les graphiques")
        print("  20  Statut detaille du systeme")
        print()

        print(f"{Colors.OKBLUE}API FLASK{Colors.ENDC}")
        print("  21  Lancer l'API Flask")
        print("  22  Tester l'API Flask")
        print()

        print(f"{Colors.FAIL}QUITTER{Colors.ENDC}")
        print("   0  Quitter")
        print()
        print("=" * 55)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def handle_loto_custom(self):
        print(f"\n{Colors.BOLD}Generation Loto Personnalisee{Colors.ENDC}")

        while True:
            try:
                nb = input(f"Nombre de grilles (1-10) [{self.default_loto_grids}]: ").strip()
                nb = int(nb) if nb else self.default_loto_grids
                if 1 <= nb <= 10:
                    break
                print(f"{Colors.WARNING}Entre 1 et 10{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.WARNING}Nombre invalide{Colors.ENDC}")

        strategies = ["equilibre", "agressive", "conservatrice", "ml_focus",
                      "focus_retard", "momentum_ml", "contrarian_ml", "frequency_ml"]
        print(f"Strategies : {', '.join(strategies)}")
        strat = input(f"Strategie [{self.default_loto_strategy}]: ").strip()
        if strat not in strategies:
            strat = self.default_loto_strategy

        plots  = input("Visualisations ? (o/N): ").strip().lower()
        export = input("Exporter statistiques ? (o/N): ").strip().lower()

        cmd = (f"python loto/duckdb_loto.py --csv {self.loto_csv} "
               f"--grids {nb} --strategy {strat} --config {self.loto_config_file}")
        if plots in ('o', 'oui', 'y', 'yes'):
            cmd += " --plots"
        if export in ('o', 'oui', 'y', 'yes'):
            cmd += " --export-stats"
        self.execute_command(cmd, f"Generation Loto ({nb} grilles, {strat})")

    def handle_advanced_custom(self):
        print(f"\n{Colors.BOLD}Generateur Loto Avance — Configuration Personnalisee{Colors.ENDC}")

        max_cores = mp.cpu_count()
        default_cores = max(1, max_cores - 1)

        while True:
            try:
                n_sims = input("Simulations (100-100000) [10000]: ").strip()
                n_sims = int(n_sims) if n_sims else 10000
                if 100 <= n_sims <= 100000:
                    break
                print("Entre 100 et 100 000")
            except ValueError:
                print("Nombre invalide")

        while True:
            try:
                n_cores = input(f"Processeurs (1-{max_cores}) [{default_cores}]: ").strip()
                n_cores = int(n_cores) if n_cores else default_cores
                if 1 <= n_cores <= max_cores:
                    break
                print(f"Entre 1 et {max_cores}")
            except ValueError:
                print("Nombre invalide")

        exclude_in = input("Numeros a exclure (ex: 1,5,12) ou 'auto' [auto]: ").strip()
        cmd = (f"{self.python_path} loto/loto_generator_advanced_Version2.py "
               f"-s {n_sims} -c {n_cores} --silent")
        if exclude_in and exclude_in.lower() != 'auto':
            cmd += f" --exclude {exclude_in}"
        self.execute_command(cmd, f"Generateur Avance ({n_sims:,} simulations, {n_cores} coeurs)")

    def handle_top25(self):
        print(f"\n{Colors.BOLD}TOP 25 Loto Equilibres{Colors.ENDC}")
        loto_data_path = Path("loto/loto_data")
        csv_files = list(loto_data_path.glob("*.csv")) if loto_data_path.exists() else []
        if not csv_files:
            print(f"{Colors.FAIL}Aucun fichier CSV Loto dans {loto_data_path}{Colors.ENDC}")
            self.wait_and_continue()
            return
        latest = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"Fichier : {latest.name}")
        print("Strategies : 1=Equilibree  2=Focus retard  3=Focus frequence")
        choice = input("Strategie [1]: ").strip() or "1"
        strat  = {"1": "equilibre", "2": "focus_retard", "3": "focus_frequence"}.get(choice, "equilibre")
        plots  = input("Visualisations ? (o/N): ").strip().lower()
        export = input("Exporter statistiques ? (O/n): ").strip().lower()
        cmd = f"python loto/duckdb_loto.py --csv {latest} --config-file loto/strategies.yml"
        if export not in ('n', 'non', 'no'):
            cmd += " --export-stats"
        if plots in ('o', 'oui', 'y', 'yes'):
            cmd += " --plots"
        self.execute_command(cmd, f"TOP 25 Loto (strategie: {strat})")

    def handle_show_top25(self):
        csv_file = Path("loto_stats_exports/top_25_numeros_equilibres_loto.csv")
        if not csv_file.exists():
            print(f"{Colors.FAIL}Fichier TOP 25 introuvable. Lancez d'abord l'option 12.{Colors.ENDC}")
            self.wait_and_continue()
            return
        mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
        print(f"\n{Colors.BOLD}TOP 25 LOTO — {mtime.strftime('%d/%m/%Y %H:%M')}{Colors.ENDC}")
        try:
            df = pd.read_csv(csv_file, sep=';')
            print(f"{'Rang':<4} {'Num':<5} {'Score':<9} {'Zone':<15} {'Retard':<8} {'Freq'}")
            print("─" * 60)
            for i, row in df.head(25).iterrows():
                color = Colors.OKGREEN if i < 3 else (Colors.OKCYAN if i < 7 else Colors.ENDC)
                print(f"{color}{row['rang']:<4} {row['numero']:<5} "
                      f"{row['score_composite']:<9.4f} {row['zone']:<15} "
                      f"{row['retard_actuel']:<8} {row['freq_absolue']}{Colors.ENDC}")
            top5 = df.head(5)['numero'].tolist()
            top10 = df.head(10)['numero'].tolist()
            print(f"\n  Grille 5 numeros : {top5}")
            print(f"  Systeme 10       : {top10}")
        except Exception as e:
            print(f"{Colors.FAIL}Erreur lecture : {e}{Colors.ENDC}")
        self.wait_and_continue()

    def handle_grilles_top_csv(self):
        print(f"\n{Colors.BOLD}Grilles Loto TOP 25 → CSV{Colors.ENDC}")
        try:
            nb = int(input("Nombre de grilles (5-50) [10]: ").strip() or "10")
            nb = max(5, min(50, nb))
        except ValueError:
            nb = 10
        try:
            top = int(input("Numeros TOP a utiliser (10-25) [20]: ").strip() or "20")
            top = max(10, min(25, top))
        except ValueError:
            top = 20
        opt = input("Optimisation : 1=PuLP  2=Glouton [1]: ").strip()
        opt_name = "glouton" if opt == "2" else "pulp"
        cmd = (f"python grilles/generateur_grilles.py --top-csv --jeu loto "
               f"--top-nombres {top} --optimisation {opt_name} "
               f"--taille-grille-loto 5 --grilles {nb} --export --format csv --verbose")
        confirm = input(f"Lancer ? (O/n): ").strip().lower()
        if confirm not in ('n', 'non', 'no'):
            self.execute_command(cmd, f"Grilles Loto TOP {top} → CSV ({nb} grilles)")
        else:
            self.wait_and_continue()

    def handle_grilles_top_markdown(self):
        print(f"\n{Colors.BOLD}Grilles Loto TOP 25 → Markdown{Colors.ENDC}")
        try:
            nb = int(input("Nombre de grilles (5-30) [8]: ").strip() or "8")
            nb = max(5, min(30, nb))
        except ValueError:
            nb = 8
        try:
            top = int(input("Numeros TOP a utiliser (10-25) [18]: ").strip() or "18")
            top = max(10, min(25, top))
        except ValueError:
            top = 18
        opt = input("Optimisation : 1=PuLP  2=Glouton [1]: ").strip()
        opt_name = "glouton" if opt == "2" else "pulp"
        cmd = (f"python grilles/generateur_grilles.py --top-csv --jeu loto "
               f"--top-nombres {top} --optimisation {opt_name} "
               f"--taille-grille-loto 5 --grilles {nb} --export --format markdown --verbose")
        confirm = input("Lancer ? (O/n): ").strip().lower()
        if confirm not in ('n', 'non', 'no'):
            self.execute_command(cmd, f"Grilles Loto TOP {top} → Markdown ({nb} grilles)")
        else:
            self.wait_and_continue()

    def handle_systeme_reduit_simple(self):
        print(f"\n{Colors.BOLD}Systeme Reduit — Numeros Favoris{Colors.ENDC}")
        print("Format : 1,7,12,18,23,29,34,39,45,49")
        nums = input("Vos numeros favoris : ").strip()
        if not nums:
            print(f"{Colors.WARNING}Aucun numero saisi.{Colors.ENDC}")
            self.wait_and_continue()
            return
        nums = ','.join(n.strip() for n in nums.split(','))
        try:
            nb = int(input("Nombre de grilles (5-20) [8]: ").strip() or "8")
            nb = max(1, min(50, nb))
        except ValueError:
            nb = 8
        fmt = input("Format : csv/md [csv]: ").strip() or "csv"
        cmd = (f"python grilles/generateur_grilles.py --jeu loto "
               f"--nombres \"{nums}\" --grilles {nb} --export --format {fmt}")
        self.execute_command(cmd, f"Systeme Reduit ({nb} grilles)")
        print(f"{Colors.OKCYAN}Fichiers dans : grilles/sorties/{Colors.ENDC}")
        self.wait_and_continue()

    def handle_systeme_reduit_personnalise(self):
        print(f"\n{Colors.BOLD}Systeme Reduit Personnalise{Colors.ENDC}")
        print("1 - Saisir les numeros  2 - Utiliser un fichier")
        methode = input("Choix (1/2) [1]: ").strip() or "1"

        nombres_param = ""
        fichier_param = ""

        if methode == "2":
            grilles_dir = self.base_path / "grilles"
            txt_files = list(grilles_dir.glob("*.txt"))
            for i, f in enumerate(txt_files, 1):
                print(f"  {i}. {f.name}")
            if txt_files:
                try:
                    idx = int(input("Numero du fichier : ").strip()) - 1
                    fichier_param = f"--fichier {txt_files[idx]}"
                except Exception:
                    pass
            if not fichier_param:
                nom = input("Nom du fichier (grilles/xxx.txt) : ").strip()
                if nom:
                    fichier_param = f"--fichier grilles/{nom}"
        else:
            nums = input("Numeros favoris (8-20) : ").strip()
            if nums:
                nums = ','.join(n.strip() for n in nums.split(','))
                nombres_param = f"--nombres {nums}"

        if not nombres_param and not fichier_param:
            print(f"{Colors.WARNING}Aucune source definie.{Colors.ENDC}")
            self.wait_and_continue()
            return

        nb      = input("Grilles [10]: ").strip() or "10"
        garantie= input("Garantie 2-5 [3]: ").strip() or "3"
        methode2= input("Methode optimal/aleatoire [optimal]: ").strip() or "optimal"
        fmt     = input("Format csv/json/md [csv]: ").strip() or "csv"

        parts = ["python grilles/generateur_grilles.py"]
        parts.append(nombres_param or fichier_param)
        parts += [f"--jeu loto", f"--grilles {nb}", f"--garantie {garantie}",
                  f"--methode {methode2}", "--export", f"--format {fmt}", "--verbose"]
        self.execute_command(" ".join(parts), "Systeme Reduit Personnalise")
        print(f"{Colors.OKCYAN}Fichiers dans : grilles/sorties/{Colors.ENDC}")
        self.wait_and_continue()

    # ------------------------------------------------------------------
    # Dispatch principal
    # ------------------------------------------------------------------

    def handle_choice(self, choice: str) -> bool:
        c = choice.strip()

        if c == "1":
            self.execute_command("python loto/result.py",
                                 "Telechargement donnees Loto (FDJ)")

        elif c == "2":
            cmd = (f"python loto/duckdb_loto.py --csv {self.loto_csv} "
                   f"--grids {self.default_loto_grids} --config {self.loto_config_file}")
            self.execute_command(cmd, "Generation 3 grilles Loto (rapide)")

        elif c == "3":
            cmd = (f"python loto/duckdb_loto.py --csv {self.loto_csv} "
                   f"--grids 5 --config {self.loto_config_file}")
            self.execute_command(cmd, "Generation 5 grilles Loto (complet)")

        elif c == "4":
            cmd = (f"python loto/duckdb_loto.py --csv {self.loto_csv} "
                   f"--grids {self.default_loto_grids} "
                   f"--plots --export-stats --config {self.loto_config_file}")
            self.execute_command(cmd, "Generation Loto avec visualisations")

        elif c == "5":
            self.handle_loto_custom()

        elif c == "6":
            cmd = f"{self.python_path} loto/generateur_bayesien_loto.py"
            self.execute_command(cmd, "Generateur Bayesien (DuckDB + XGBoost + PuLP)")

        elif c == "7":
            cmd = f"{self.python_path} loto/generateur_bayesien_loto.py --retrain"
            self.execute_command(cmd, "Generateur Bayesien avec reentrainement")

        elif c == "8":
            cmd = f"{self.python_path} loto/loto_generator_advanced_Version2.py --quick --silent"
            self.execute_command(cmd, "Generateur Avance — Mode rapide (1 000 sim.)")

        elif c == "9":
            cmd = f"{self.python_path} loto/loto_generator_advanced_Version2.py --silent"
            self.execute_command(cmd, "Generateur Avance — Mode standard (10 000 sim.)")

        elif c == "10":
            cmd = f"{self.python_path} loto/loto_generator_advanced_Version2.py --intensive"
            self.execute_command(cmd, "Generateur Avance — Mode intensif (50 000 sim.)")

        elif c == "11":
            self.handle_advanced_custom()

        elif c == "12":
            self.handle_top25()

        elif c == "13":
            self.handle_show_top25()

        elif c == "14":
            self.handle_grilles_top_csv()

        elif c == "15":
            self.handle_grilles_top_markdown()

        elif c == "16":
            self.handle_systeme_reduit_simple()

        elif c == "17":
            self.handle_systeme_reduit_personnalise()

        elif c == "18":
            grille_file = self.base_path / "grilles.csv"
            bayesian    = self.base_path / "loto_output" / "bayesian_grilles.json"
            if bayesian.exists():
                self.show_file_content(bayesian, 60)
            elif grille_file.exists():
                self.show_file_content(grille_file, 25)
            else:
                print(f"{Colors.WARNING}Aucune grille generee.{Colors.ENDC}")
            self.wait_and_continue()

        elif c == "19":
            plots_dir = self.base_path / "loto_analyse_plots"
            if plots_dir.exists():
                files = list(plots_dir.glob("*.png"))
                print(f"\n{len(files)} graphique(s) dans {plots_dir}")
                for f in files:
                    print(f"  - {f.name}")
            else:
                print(f"{Colors.WARNING}Aucun graphique.{Colors.ENDC}")
            self.wait_and_continue()

        elif c == "20":
            print(f"\n{Colors.BOLD}Statut du systeme{Colors.ENDC}")
            for label, path in [
                ("Donnees CSV",        self.loto_csv),
                ("Strategies YAML",    self.base_path / "loto" / "strategies.yml"),
                ("Strategies ML YAML", self.base_path / "loto" / "strategies_ml.yml"),
                ("Boost models",       self.base_path / "boost_models"),
                ("Grilles output",     self.base_path / "loto_output"),
                ("Stats exports",      self.base_path / "loto_stats_exports"),
                ("Graphiques",         self.base_path / "loto_analyse_plots"),
                ("Bayesian JSON",      self.base_path / "loto_output" / "bayesian_grilles.json"),
            ]:
                p = Path(path)
                if p.exists():
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                    print(f"  {Colors.OKGREEN}OK{Colors.ENDC}  {label:<22} {mtime.strftime('%d/%m/%Y %H:%M')}")
                else:
                    print(f"  {Colors.FAIL}--{Colors.ENDC}  {label}")
            self.wait_and_continue()

        elif c == "21":
            self.execute_command("./lancer_api.sh", "Lancement API Flask")

        elif c == "22":
            self.execute_command("python test/test_api.py", "Test API Flask")

        elif c == "0":
            print(f"\n{Colors.OKGREEN}Au revoir !{Colors.ENDC}")
            return False

        else:
            print(f"\n{Colors.FAIL}Choix invalide.{Colors.ENDC}")
            self.wait_and_continue()

        return True

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self):
        try:
            while True:
                self.print_header()
                self.print_status()
                self.print_menu()
                choice = input(f"{Colors.BOLD}Votre choix : {Colors.ENDC}").strip()
                if not self.handle_choice(choice):
                    break
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Interruption.{Colors.ENDC}")
            print(f"{Colors.OKGREEN}Au revoir !{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.FAIL}Erreur inattendue : {e}{Colors.ENDC}")


def main():
    print(f"{Colors.HEADER}Initialisation du menu Loto...{Colors.ENDC}")
    time.sleep(0.5)
    LotoMenu().run()


if __name__ == "__main__":
    main()
