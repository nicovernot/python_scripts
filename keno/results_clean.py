#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📥 TÉLÉCHARGEUR DE RÉSULTATS KENO
================================

Script pour télécharger les résultats du Keno depuis la page d'historique FDJ
avec extraction automatique du lien de téléchargement et décompression du ZIP.
Version améliorée avec nettoyage automatique des anciens fichiers.

Usage:
    python keno/results_clean.py
"""

import os
import sys
import requests
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import time
from bs4 import BeautifulSoup
import re

# Configuration
KENO_DATA_DIR = Path(__file__).parent / "keno_data"

# URL de la page d'historique FDJ
HISTORIQUE_URL = "https://www.fdj.fr/jeux-de-tirage/keno/historique"

def setup_directories():
    """Créer les répertoires nécessaires"""
    KENO_DATA_DIR.mkdir(exist_ok=True)
    print(f"📁 Répertoire créé/vérifié : {KENO_DATA_DIR}")

def clean_old_downloads():
    """Supprime les anciens téléchargements pour éviter l'accumulation"""
    print("🧹 Nettoyage des anciens téléchargements...")
    
    files_removed = 0
    total_size_freed = 0
    
    # Patterns des fichiers à supprimer
    patterns_to_remove = [
        "keno_*.csv",
        "keno_*.zip", 
        "keno_historique*.zip",
        "keno_historique*.csv"
    ]
    
    for pattern in patterns_to_remove:
        for file_path in KENO_DATA_DIR.glob(pattern):
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                files_removed += 1
                total_size_freed += file_size
                print(f"   ❌ Supprimé : {file_path.name}")
            except Exception as e:
                print(f"   ⚠️ Impossible de supprimer {file_path.name} : {e}")
    
    # Nettoyer aussi le répertoire extracted
    extracted_dir = KENO_DATA_DIR / "extracted"
    if extracted_dir.exists():
        for file_path in extracted_dir.glob("*"):
            try:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    files_removed += 1
                    total_size_freed += file_size
                    print(f"   ❌ Supprimé : extracted/{file_path.name}")
            except Exception as e:
                print(f"   ⚠️ Impossible de supprimer extracted/{file_path.name} : {e}")
    
    if files_removed > 0:
        size_mb = total_size_freed / (1024 * 1024)
        print(f"✅ {files_removed} fichiers supprimés ({size_mb:.1f} MB libérés)")
    else:
        print("✅ Aucun ancien fichier à supprimer")

def get_download_links():
    """Récupère les liens de téléchargement depuis la page d'historique"""
    print("🔍 Recherche des liens de téléchargement...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        response = requests.get(HISTORIQUE_URL, headers=headers, timeout=15)
        response.raise_for_status()
        
        print(f"✅ Page chargée (statut: {response.status_code})")
        
        # Parser le HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Chercher tous les liens de téléchargement
        download_links = []
        
        # Méthode 1: Chercher par pattern dans le code source complet
        content = response.text
        print(f"📄 Taille de la page : {len(content)} caractères")
        
        # Pattern pour les URLs de l'API FDJ - plus permissif
        patterns = [
            r'https://www\.sto\.api\.fdj\.fr/[^"\s]+',
            r'https://[^"\s]*api[^"\s]*fdj[^"\s]*keno[^"\s]*',
            r'https://[^"\s]*fdj[^"\s]*keno[^"\s]*\.zip',
            r'href="([^"]*keno[^"]*)"',
            r'download="([^"]*keno[^"]*)"'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            print(f"   Pattern '{pattern[:50]}...' : {len(matches)} résultats")
            
            for match in matches:
                # Si c'est un tuple (groupe de capture), prendre le premier élément
                if isinstance(match, tuple):
                    match = match[0]
                
                if match and len(match) > 10:  # URL valide
                    download_links.append({
                        'url': match,
                        'title': 'Historique Keno (détecté)',
                        'download': 'keno_historique',
                        'text': 'Lien détecté automatiquement'
                    })
        
        # Méthode 2: Chercher dans les balises <a>
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            download_attr = link.get('download', '')
            title_attr = link.get('title', '')
            
            # Vérifier si c'est un lien Keno
            if (href and 
                (('keno' in href.lower()) or 
                 ('keno' in download_attr.lower()) or 
                 ('keno' in title_attr.lower()))):
                
                download_links.append({
                    'url': href,
                    'title': title_attr or 'Historique Keno',
                    'download': download_attr or 'keno',
                    'text': link.get_text(strip=True)
                })
        
        # Méthode 3: URLs hardcodées comme fallback
        if not download_links:
            print("⚠️ Aucun lien trouvé dynamiquement, utilisation de liens de fallback")
            
            fallback_urls = [
                "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/documentations/1a2b3c4d-9876-4562-b3fc-2c963f66aft6",
                "https://www.fdj.fr/api/game/keno/historique/download",
                "https://media.fdj.fr/generated/game/keno/keno_202010.zip"
            ]
            
            for url in fallback_urls:
                download_links.append({
                    'url': url,
                    'title': 'Historique Keno (fallback)',
                    'download': 'keno_fallback',
                    'text': 'URL de secours'
                })
        
        # Nettoyer et dédoublonner
        unique_links = []
        seen_urls = set()
        
        for link in download_links:
            url = link['url']
            
            # Nettoyer l'URL
            if url.startswith('//'):
                url = 'https:' + url
            elif url.startswith('/'):
                url = 'https://www.fdj.fr' + url
            
            # Éviter les doublons
            if url not in seen_urls and len(url) > 10:
                seen_urls.add(url)
                link['url'] = url
                unique_links.append(link)
        
        print(f"🔗 {len(unique_links)} liens uniques trouvés")
        
        for i, link in enumerate(unique_links, 1):
            print(f"   {i}. {link['title']}")
            print(f"      URL: {link['url'][:80]}...")
            print(f"      Download: {link['download']}")
        
        return unique_links
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de l'accès à la page : {e}")
        
        # Retourner des URLs de fallback
        print("🔄 Utilisation d'URLs de secours...")
        return [
            {
                'url': "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/documentations/1a2b3c4d-9876-4562-b3fc-2c963f66aft6",
                'title': 'Historique Keno (URL de secours)',
                'download': 'keno_backup',
                'text': 'Lien de secours'
            }
        ]
    except Exception as e:
        print(f"❌ Erreur lors du parsing : {e}")
        return []

def download_file(url, filename=None):
    """Télécharge un fichier depuis une URL"""
    if not filename:
        # Extraire le nom du fichier depuis l'URL ou créer un nom par défaut
        filename = url.split('/')[-1]
        if not filename or '.' not in filename:
            filename = "keno_historique"
    
    # Assurer une extension
    if not any(ext in filename.lower() for ext in ['.zip', '.csv']):
        filename += '.zip'
    
    filepath = KENO_DATA_DIR / filename
    
    # Supprimer l'ancien fichier s'il existe
    if filepath.exists():
        filepath.unlink()
        print(f"   🗑️ Ancien téléchargement supprimé : {filename}")
    
    print(f"📥 Téléchargement : {filename}")
    print(f"   URL : {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'fr-FR,fr;q=0.9',
        'Referer': 'https://www.fdj.fr/',
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Vérifier le type de contenu
        content_type = response.headers.get('content-type', '').lower()
        print(f"   Type : {content_type}")
        
        # Télécharger le fichier
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progression : {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Téléchargé : {filepath} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Erreur de téléchargement : {e}")
        return None
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        return None

def extract_zip(zip_path):
    """Extrait un fichier ZIP"""
    if not zip_path or not zip_path.exists():
        return []
    
    print(f"📦 Extraction : {zip_path.name}")
    
    extracted_files = []
    extract_dir = KENO_DATA_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Lister le contenu
            file_list = zip_ref.namelist()
            print(f"   Fichiers dans l'archive : {len(file_list)}")
            
            for file_name in file_list:
                print(f"   - {file_name}")
            
            # Extraire tout
            zip_ref.extractall(extract_dir)
            
            # Copier les fichiers CSV vers le répertoire principal
            for file_name in file_list:
                extracted_path = extract_dir / file_name
                
                if extracted_path.exists() and file_name.lower().endswith('.csv'):
                    # Utiliser un nom simple et fixe (pas de timestamp)
                    target_path = KENO_DATA_DIR / file_name
                    
                    # Supprimer l'ancien fichier s'il existe
                    if target_path.exists():
                        target_path.unlink()
                        print(f"   🗑️ Ancien fichier supprimé : {file_name}")
                    
                    import shutil
                    shutil.copy2(extracted_path, target_path)
                    extracted_files.append(target_path)
                    print(f"   ✅ Extrait : {target_path.name}")
        
        return extracted_files
        
    except zipfile.BadZipFile:
        print(f"   ❌ Fichier ZIP corrompu")
        return []
    except Exception as e:
        print(f"   ❌ Erreur d'extraction : {e}")
        return []

def verify_csv_data(file_path):
    """Vérifie les données CSV téléchargées"""
    if not file_path or not file_path.exists():
        print("❌ Aucun fichier à vérifier")
        return False
    
    try:
        print(f"\n🔍 Vérification : {file_path.name}")
        
        # Essayer différents délimiteurs
        delimiters = [';', ',', '\t']
        df = None
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=5)
                if len(df.columns) > 10:  # Keno a beaucoup de colonnes
                    print(f"   ✅ Délimiteur détecté : '{delimiter}'")
                    break
            except:
                continue
        
        if df is None:
            print("   ❌ Format CSV non reconnu")
            return False
        
        # Lire le fichier complet
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        print(f"   📊 Données : {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Vérifier les colonnes Keno
        ball_columns = [col for col in df.columns if col.lower().startswith('boule')]
        print(f"   🎱 Colonnes boules : {len(ball_columns)}")
        
        if len(ball_columns) >= 20:
            print(f"   ✅ Format Keno valide")
            
            # Analyser la période couverte
            if 'date_de_tirage' in df.columns:
                try:
                    df['date_parsed'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
                    dates_valides = df['date_parsed'].dropna()
                    
                    if len(dates_valides) > 0:
                        date_min = dates_valides.min()
                        date_max = dates_valides.max()
                        
                        print(f"   📅 PÉRIODE COUVERTE :")
                        print(f"       Du {date_min.strftime('%d/%m/%Y')} au {date_max.strftime('%d/%m/%Y')}")
                        
                        # Analyser par année
                        yearly_counts = dates_valides.dt.year.value_counts().sort_index()
                        print(f"   📈 RÉPARTITION PAR ANNÉE :")
                        for year, count in yearly_counts.items():
                            print(f"       {year}: {count:,} tirages")
                        
                        # Calculer la durée
                        duree_jours = (date_max - date_min).days
                        duree_annees = duree_jours / 365.25
                        print(f"   ⏱️  DURÉE TOTALE : {duree_jours:,} jours ({duree_annees:.1f} années)")
                        
                except Exception as e:
                    print(f"   ⚠️ Impossible d'analyser les dates : {e}")
            
            # Afficher un échantillon
            print(f"\n   📋 Aperçu des données :")
            print(f"   Colonnes : {list(df.columns[:5])}...")
            
            if len(df) > 0:
                # Afficher quelques numéros du premier tirage
                first_numbers = []
                for col in ball_columns[:5]:
                    if col in df.iloc[0] and pd.notna(df.iloc[0][col]):
                        first_numbers.append(int(df.iloc[0][col]))
                print(f"   Premier tirage : {first_numbers}")
            
            return True
        else:
            print(f"   ⚠️ Nombre de colonnes boules insuffisant : {len(ball_columns)}")
            return False
        
    except Exception as e:
        print(f"   ❌ Erreur de vérification : {e}")
        return False

def main():
    """Fonction principale"""
    print("🎯 TÉLÉCHARGEUR DE RÉSULTATS KENO - VERSION NETTOYAGE AUTO")
    print("=" * 65)
    
    # 1. Créer les répertoires
    setup_directories()
    
    # 2. Nettoyer les anciens téléchargements
    clean_old_downloads()
    
    # 3. Rechercher les liens de téléchargement
    print(f"\n🔍 Recherche des liens de téléchargement...")
    download_links = get_download_links()
    
    if not download_links:
        print("❌ Aucun lien de téléchargement trouvé")
        return
    
    # 4. Télécharger le premier lien trouvé (le plus récent)
    best_link = download_links[0]
    print(f"\n📡 Téléchargement du fichier le plus récent...")
    print(f"   Titre : {best_link['title']}")
    
    # Déterminer le nom du fichier (nom fixe)
    filename = "keno_historique"
    
    downloaded_file = download_file(best_link['url'], filename)
    
    if not downloaded_file:
        print("❌ Échec du téléchargement")
        return
    
    # 5. Extraire le fichier si c'est un ZIP
    extracted_files = []
    if downloaded_file.suffix.lower() == '.zip':
        extracted_files = extract_zip(downloaded_file)
    elif downloaded_file.suffix.lower() == '.csv':
        extracted_files = [downloaded_file]
    
    # 6. Vérifier les fichiers extraits
    valid_files = []
    for file_path in extracted_files:
        if verify_csv_data(file_path):
            valid_files.append(file_path)
    
    # 7. Résumé
    print(f"\n✅ TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 40)
    print(f"📁 Répertoire : {KENO_DATA_DIR}")
    print(f"📦 Fichier téléchargé : {downloaded_file.name}")
    print(f"📄 Fichiers CSV valides : {len(valid_files)}")
    
    for file_path in valid_files:
        size_mb = file_path.stat().st_size / (1024*1024)
        print(f"   ✅ {file_path.name} ({size_mb:.1f} MB)")
    
    if valid_files:
        print(f"\n💡 Pour analyser les données :")
        print(f"   python keno/duckdb_keno.py --csv {valid_files[0]} --plots --export-stats")
    
    print("=" * 65)

if __name__ == "__main__":
    main()
