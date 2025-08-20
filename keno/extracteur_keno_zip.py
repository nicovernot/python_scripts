#!/usr/bin/env python3
"""
Extracteur de données Keno FDJ - Version ZIP
Télécharge et traite le fichier ZIP historique depuis la FDJ
Période : Octobre 2020 à Août 2025
"""

import requests
import zipfile
import io
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
    """Extracteur de données Keno depuis le fichier ZIP FDJ"""
    
    def __init__(self, data_dir: str = "keno/keno_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URL de la page historique FDJ
        self.historique_url = "https://www.fdj.fr/jeux-de-tirage/keno/historique"
        
        # Pattern pour extraire le lien du ZIP récent
        self.zip_pattern = r'https://www\.sto\.api\.fdj\.fr/anonymous/service-draw-info/v3/documentations/[a-f0-9-]+6'
        
        # Headers pour les requêtes
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def get_zip_download_url(self) -> Optional[str]:
        """Récupère l'URL de téléchargement du ZIP depuis la page historique"""
        try:
            logger.info(f"Récupération de la page historique : {self.historique_url}")
            response = requests.get(self.historique_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse avec BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Cherche tous les liens vers les archives
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                # Cherche le lien qui correspond au pattern (octobre 2020 à août 2025)
                if 'sto.api.fdj.fr' in href and 'documentations' in href:
                    # Vérifie le texte du lien pour identifier le bon fichier
                    link_text = link.get_text(strip=True).lower()
                    if 'octobre 2020' in link_text or '2020' in link_text:
                        logger.info(f"URL du ZIP trouvée : {href}")
                        return href
            
            # Fallback : cherche avec regex dans le HTML brut
            html_content = response.text
            matches = re.findall(self.zip_pattern, html_content)
            if matches:
                logger.info(f"URL du ZIP trouvée via regex : {matches[0]}")
                return matches[0]
            
            logger.error("Aucune URL de ZIP trouvée sur la page")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'URL du ZIP : {e}")
            return None

    def download_and_extract_zip(self, zip_url: str) -> Optional[str]:
        """Télécharge et extrait le fichier ZIP"""
        try:
            logger.info(f"Téléchargement du ZIP : {zip_url}")
            response = requests.get(zip_url, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            # Sauvegarde temporaire du ZIP
            zip_path = self.data_dir / "keno_historique.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"ZIP téléchargé ({len(response.content)} bytes)")
            
            # Extraction du ZIP
            extract_dir = self.data_dir / "temp_extract"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = zip_ref.namelist()
                logger.info(f"Fichiers extraits : {extracted_files}")
            
            # Cherche le fichier CSV principal
            csv_files = list(extract_dir.glob("*.csv"))
            if not csv_files:
                # Cherche dans les sous-dossiers
                csv_files = list(extract_dir.rglob("*.csv"))
            
            if csv_files:
                main_csv = csv_files[0]  # Prend le premier fichier CSV trouvé
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
            
            # Lecture avec différents encodages possibles
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
            
            # Affiche les premières lignes pour comprendre la structure
            logger.info("Aperçu des données :")
            logger.info(df.head().to_string())
            
            # Identifie les colonnes importantes
            date_col = None
            numero_col = None
            boules_cols = []
            
            for col in df.columns:
                col_lower = col.lower()
                # Priorise la vraie date de tirage, pas la date de forclusion
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
            
            # Traitement des données
            for index, row in df.iterrows():
                try:
                    # Date du tirage
                    date_str = str(row[date_col])
                    
                    # Parse de la date (plusieurs formats possibles)
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
                    
                    # Numéro de tirage
                    numero_tirage = str(row[numero_col]) if numero_col else f"T{index+1}"
                    
                    # Extraction des boules
                    boules = []
                    for col in boules_cols:
                        try:
                            boule = int(row[col])
                            if 1 <= boule <= 70:  # Vérification de validité
                                boules.append(boule)
                        except (ValueError, TypeError):
                            continue
                    
                    if len(boules) == 20:  # Keno = 20 boules tirées
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
            # Groupe par mois
            monthly_data = {}
            
            for tirage in tirages:
                date_obj = datetime.strptime(tirage['date'], '%Y-%m-%d')
                month_key = date_obj.strftime('%Y%m')
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                
                monthly_data[month_key].append(tirage)
            
            # Sauvegarde chaque mois
            total_saved = 0
            for month_key, month_tirages in monthly_data.items():
                filename = f"keno_{month_key}.csv"
                filepath = self.data_dir / filename
                
                # Vérifie si le fichier existe déjà
                existing_count = 0
                if filepath.exists():
                    existing_df = pd.read_csv(filepath)
                    existing_count = len(existing_df)
                
                # Écrit les données
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # En-tête
                    writer.writerow(['date', 'numero_tirage', 'b1', 'b2', 'b3', 'b4', 'b5',
                                   'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14',
                                   'b15', 'b16', 'b17', 'b18', 'b19', 'b20'])
                    
                    # Données
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

    def cleanup_temp_files(self) -> None:
        """Nettoie les fichiers temporaires"""
        try:
            # Supprime le ZIP
            zip_path = self.data_dir / "keno_historique.zip"
            if zip_path.exists():
                zip_path.unlink()
            
            # Supprime le dossier d'extraction temporaire
            extract_dir = self.data_dir / "temp_extract"
            if extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir)
            
            logger.info("Fichiers temporaires nettoyés")
            
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage : {e}")

    def extract_keno_data(self) -> bool:
        """Méthode principale d'extraction"""
        try:
            logger.info("=== DÉBUT DE L'EXTRACTION KENO ===")
            
            # 1. Récupère l'URL du ZIP
            zip_url = self.get_zip_download_url()
            if not zip_url:
                logger.error("Impossible de récupérer l'URL du ZIP")
                return False
            
            # 2. Télécharge et extrait le ZIP
            csv_path = self.download_and_extract_zip(zip_url)
            if not csv_path:
                logger.error("Impossible de télécharger/extraire le ZIP")
                return False
            
            # 3. Parse les données CSV
            tirages = self.parse_csv_data(csv_path)
            if not tirages:
                logger.error("Aucun tirage valide extrait")
                return False
            
            # 4. Sauvegarde dans des fichiers mensuels
            self.save_to_monthly_files(tirages)
            
            # 5. Nettoyage
            self.cleanup_temp_files()
            
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
