#!/usr/bin/env python3
"""
Extracteur de données Keno FDJ - Version ZIP
Télécharge et traite le fichier ZIP FDJ "Depuis octobre 2020" (tous tirages récents)
"""

import requests
import zipfile
import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path
import csv
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keno_extract.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KenoZipExtractor:
    """Extracteur de données Keno depuis le ZIP FDJ (octobre 2020 à aujourd'hui)"""
    
    def __init__(self, data_dir: str = "keno/keno_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.historique_url = "https://www.fdj.fr/jeux-de-tirage/keno/historique"
        self.zip_pattern = r'https://www\.sto\.api\.fdj\.fr/anonymous/service-draw-info/v3/documentations/[a-f0-9-]+6'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def get_zip_download_url(self) -> Optional[str]:
        """Récupère l'URL du ZIP FDJ contenant tous les tirages depuis octobre 2020"""
        try:
            logger.info(f"Récupération de la page historique : {self.historique_url}")
            response = requests.get(self.historique_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                link_text = link.get_text(strip=True).lower()
                # Cherche le lien qui contient "octobre 2020" ou "depuis octobre 2020"
                if 'sto.api.fdj.fr' in href and 'documentations' in href:
                    if 'octobre 2020' in link_text or 'depuis octobre 2020' in link_text or '2020' in link_text:
                        logger.info(f"URL du ZIP trouvée : {href}")
                        return href
            # Fallback regex si aucun lien trouvé
            html_content = response.text
            matches = re.findall(self.zip_pattern, html_content)
            if matches:
                logger.info(f"URL ZIP trouvée par regex : {matches[0]}")
                return matches[0]
            logger.error("Aucune URL de ZIP trouvée sur la page")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'URL ZIP : {e}")
            return None

    def download_and_extract_zip(self, zip_url: str) -> Optional[str]:
        """Télécharge et extrait le fichier ZIP, retourne le chemin du CSV principal"""
        try:
            logger.info(f"Téléchargement du ZIP : {zip_url}")
            response = requests.get(zip_url, headers=self.headers, timeout=60)
            response.raise_for_status()
            zip_name = zip_url.split("/")[-1].split("?")[0]
            zip_path = self.data_dir / zip_name
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"ZIP téléchargé ({len(response.content)} bytes)")
            extract_dir = self.data_dir / f"extract_{zip_name}"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = zip_ref.namelist()
                logger.info(f"Fichiers extraits : {extracted_files}")
            csv_files = list(extract_dir.glob("*.csv"))
            if not csv_files:
                csv_files = list(extract_dir.rglob("*.csv"))
            if csv_files:
                main_csv = csv_files[0]
                logger.info(f"Fichier CSV principal trouvé : {main_csv}")
                return str(main_csv)
            else:
                logger.error("Aucun fichier CSV trouvé dans l'archive")
                return None
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement/extraction : {e}")
            return None

    def parse_csv_data(self, csv_path: str) -> List[Dict]:
        """Parse les données du fichier CSV FDJ"""
        tirages = []
        try:
            logger.info(f"Analyse du fichier CSV : {csv_path}")
            encodings = ['utf-8', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=';')
                    logger.info(f"Fichier lu avec l'encodage : {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                logger.error("Impossible de lire le fichier CSV avec les encodages testés")
                return []
            logger.info(f"Colonnes trouvées : {list(df.columns)}")
            logger.info(f"Nombre de lignes : {len(df)}")
            logger.info("Aperçu des données :")
            logger.info(df.head().to_string())
            date_col = None
            numero_col = None
            boules_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if 'date_de_tirage' in col_lower:
                    date_col = col
                elif date_col is None and ('date' in col_lower or 'jour' in col_lower) and 'forclusion' not in col_lower:
                    date_col = col
                elif 'numero' in col_lower and 'tirage' in col_lower:
                    numero_col = col
                elif 'boule' in col_lower and col_lower != 'numero_jokerplus':
                    boules_cols.append(col)
            if not date_col or not boules_cols:
                logger.error("Impossible d'identifier les colonnes nécessaires")
                return []
            logger.info(f"Colonne date : {date_col}")
            logger.info(f"Colonne numéro : {numero_col}")
            logger.info(f"Colonnes boules : {boules_cols}")
            for index, row in df.iterrows():
                try:
                    date_str = str(row[date_col])
                    date_tirage = None
                    for date_format in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                        try:
                            date_tirage = datetime.strptime(date_str, date_format)
                            break
                        except ValueError:
                            continue
                    if not date_tirage:
                        logger.warning(f"Format de date non reconnu : {date_str}")
                        continue
                    numero_tirage = str(row[numero_col]) if numero_col else f"T{index+1}"
                    boules = []
                    for col in boules_cols:
                        try:
                            boule = int(row[col])
                            if 1 <= boule <= 70:
                                boules.append(boule)
                        except (ValueError, TypeError):
                            continue
                    if len(boules) == 20:
                        tirage = {
                            'date': date_tirage.strftime('%Y-%m-%d'),
                            'numero_tirage': numero_tirage,
                            'boules': sorted(boules)
                        }
                        tirages.append(tirage)
                        if len(tirages) % 1000 == 0:
                            logger.info(f"Traité {len(tirages)} tirages...")
                except Exception as e:
                    logger.warning(f"Erreur ligne {index} : {e}")
                    continue
            logger.info(f"Total des tirages valides extraits : {len(tirages)}")
            return tirages
        except Exception as e:
            logger.error(f"Erreur lors du parsing : {e}")
            return []

    def save_to_monthly_files(self, tirages: List[Dict]) -> None:
        """Sauvegarde les tirages dans des fichiers mensuels"""
        try:
            monthly_data = {}
            for tirage in tirages:
                date_obj = datetime.strptime(tirage['date'], '%Y-%m-%d')
                month_key = date_obj.strftime('%Y%m')
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                monthly_data[month_key].append(tirage)
            total_saved = 0
            for month_key, month_tirages in monthly_data.items():
                filename = f"keno_{month_key}.csv"
                filepath = self.data_dir / filename
                existing_count = 0
                if filepath.exists():
                    existing_df = pd.read_csv(filepath)
                    existing_count = len(existing_df)
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['date', 'numero_tirage', 'b1', 'b2', 'b3', 'b4', 'b5',
                                   'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14',
                                   'b15', 'b16', 'b17', 'b18', 'b19', 'b20'])
                    for tirage in month_tirages:
                        row = [tirage['date'], tirage['numero_tirage']] + tirage['boules']
                        writer.writerow(row)
                new_count = len(month_tirages)
                added_count = new_count - existing_count
                total_saved += added_count
                if added_count > 0:
                    logger.info(f"Fichier {filename} : {new_count} tirages ({added_count} nouveaux)")
                else:
                    logger.info(f"Fichier {filename} : {new_count} tirages (aucun nouveau)")
            logger.info(f"Total de nouveaux tirages sauvegardés : {total_saved}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde : {e}")

    def save_to_single_file(self, tirages: List[Dict], filename: str = "keno_consolidated.csv") -> None:
        """Sauvegarde tous les tirages dans un seul fichier CSV consolidé"""
        try:
            filepath = self.data_dir / filename
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'numero_tirage', 'b1', 'b2', 'b3', 'b4', 'b5',
                                 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14',
                                 'b15', 'b16', 'b17', 'b18', 'b19', 'b20'])
                for tirage in tirages:
                    row = [tirage['date'], tirage['numero_tirage']] + tirage['boules']
                    writer.writerow(row)
            logger.info(f"Fichier unique {filename} sauvegardé avec {len(tirages)} tirages")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier unique : {e}")

    def cleanup_temp_files(self) -> None:
        """Supprime tous les fichiers temporaires et mensuels, ne garde que le fichier consolidé"""
        try:
            # Suppression des fichiers ZIP et dossiers extraits
            for item in self.data_dir.glob("*.zip"):
                if item.exists():
                    item.unlink()
            for extract_dir in self.data_dir.glob("extract_*"):
                if extract_dir.exists():
                    import shutil
                    shutil.rmtree(extract_dir)
            # Suppression des fichiers mensuels
            for month_file in self.data_dir.glob("keno_*.csv"):
                if month_file.name != "keno_consolidated.csv":
                    month_file.unlink()
            logger.info("Fichiers temporaires et mensuels supprimés, seul le fichier consolidé est conservé")
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage : {e}")

    def extract_keno_data(self) -> bool:
        """Méthode principale d'extraction pour le ZIP unique FDJ"""
        try:
            logger.info("=== DÉBUT DE L'EXTRACTION KENO ===")
            zip_url = self.get_zip_download_url()
            if not zip_url:
                logger.error("Impossible de récupérer l'URL du ZIP")
                return False
            csv_path = self.download_and_extract_zip(zip_url)
            if not csv_path:
                logger.error(f"Impossible de télécharger/extraire le ZIP : {zip_url}")
                return False
            tirages = self.parse_csv_data(csv_path)
            if not tirages:
                logger.error("Aucun tirage valide extrait")
                return False
            self.save_to_single_file(tirages, filename="keno_consolidated.csv")  # Sauvegarde unique
            self.cleanup_temp_files()  # Supprime tout sauf le consolidé
            logger.info("=== EXTRACTION TERMINÉE AVEC SUCCÈS ===")
            return True
        except Exception as e:
            logger.error(f"Erreur générale : {e}")
            return False

def main():
    """Point d'entrée principal"""
    extractor = KenoZipExtractor()
    success = extractor.extract_keno_data()
    if success:
        print("✅ Extraction réussie !")
    else:
        print("❌ Échec de l'extraction")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
