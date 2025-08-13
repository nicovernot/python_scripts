#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📥 TÉLÉCHARGEUR ET PROCESSEUR DE RÉSULTATS KENO
===============================================

Script pour télécharger et traiter les résultats du Keno depuis l'API officielle
de la FDJ ou depuis des fichiers CSV locaux.

Usage:
    python keno/results_complete.py                    # Télécharge depuis l'API
    python keno/results_complete.py --local            # Traite les fichiers locaux
    python keno/results_complete.py --file fichier.csv # Traite un fichier spécifique
"""

import os
import sys
import requests
import json
import csv
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import time
import argparse

# Configuration
KENO_DATA_DIR = Path(__file__).parent / "keno_data"
BASE_API_URL = "https://www.fdj.fr/api/game/keno"

def setup_directories():
    """Créer les répertoires nécessaires"""
    KENO_DATA_DIR.mkdir(exist_ok=True)
    print(f"📁 Répertoire créé/vérifié : {KENO_DATA_DIR}")

def process_local_csv(file_path):
    """Traite un fichier CSV local"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Fichier non trouvé : {file_path}")
        return None
    
    print(f"📂 Traitement du fichier local : {file_path.name}")
    
    try:
        # Lire le fichier CSV
        df = pd.read_csv(file_path, delimiter=';')
        print(f"📊 Fichier lu avec succès : {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Afficher les colonnes pour comprendre la structure
        print(f"📋 Colonnes disponibles : {list(df.columns)}")
        
        # Afficher un aperçu
        print(f"\n📋 Aperçu des premières lignes :")
        print(df.head())
        
        # Copier le fichier dans le répertoire keno_data s'il n'y est pas déjà
        if not str(file_path).startswith(str(KENO_DATA_DIR)):
            target_path = KENO_DATA_DIR / file_path.name
            shutil.copy2(file_path, target_path)
            print(f"📋 Fichier copié vers : {target_path}")
            return target_path
        
        return file_path
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement du fichier : {e}")
        return None

def find_local_csv_files():
    """Trouve tous les fichiers CSV dans le répertoire keno_data"""
    csv_files = list(KENO_DATA_DIR.glob("*.csv"))
    return csv_files

def get_keno_data_for_date(target_date):
    """Récupère les données Keno pour une date donnée depuis l'API"""
    date_str = target_date.strftime('%Y-%m-%d')
    url = f"{BASE_API_URL}/draws/{date_str}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'fr-FR,fr;q=0.9',
        'Referer': 'https://www.fdj.fr/',
        'Origin': 'https://www.fdj.fr'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return process_daily_data(data, date_str)
        elif response.status_code == 404:
            return []  # Pas de tirages ce jour-là
        else:
            print(f"⚠️ Erreur {response.status_code} pour {date_str}")
            return []
    except Exception as e:
        print(f"⚠️ Erreur pour {date_str}: {e}")
        return []

def process_daily_data(data, date_str):
    """Traite les données d'une journée"""
    results = []
    
    try:
        if isinstance(data, dict) and 'draws' in data:
            draws = data['draws']
        elif isinstance(data, list):
            draws = data
        else:
            return results
        
        for draw in draws:
            if isinstance(draw, dict):
                processed = extract_keno_draw(draw, date_str)
                if processed:
                    results.append(processed)
                    
    except Exception as e:
        print(f"⚠️ Erreur traitement {date_str}: {e}")
    
    return results

def extract_keno_draw(draw, date_str):
    """Extrait les informations d'un tirage Keno"""
    try:
        result = {
            'date': date_str,
            'heure': draw.get('drawTime', ''),
            'numero_tirage': draw.get('drawNumber', ''),
        }
        
        # Récupérer les numéros gagnants
        if 'winningNumbers' in draw:
            numbers = draw['winningNumbers']
        elif 'numbers' in draw:
            numbers = draw['numbers']
        elif 'results' in draw:
            numbers = draw['results']
        else:
            return None
        
        # Ajouter les 20 numéros du Keno
        if isinstance(numbers, list) and len(numbers) >= 20:
            for i, num in enumerate(numbers[:20]):
                result[f'numero_{i+1}'] = int(num)
        else:
            return None
            
        return result
        
    except Exception as e:
        print(f"⚠️ Erreur extraction tirage: {e}")
        return None

def download_keno_data_recent():
    """Télécharge les résultats Keno récents (7 derniers jours)"""
    print("📥 Téléchargement des résultats Keno récents...")
    
    all_results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Derniers 7 jours
    
    current_date = start_date
    
    print(f"📅 Période : du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")
    
    while current_date <= end_date:
        day_results = get_keno_data_for_date(current_date)
        all_results.extend(day_results)
        
        print(f"   📊 {current_date.strftime('%d/%m/%Y')} - {len(day_results)} tirages")
        
        # Pause pour éviter de surcharger l'API
        time.sleep(0.5)
        
        current_date += timedelta(days=1)
    
    print(f"✅ Téléchargement terminé : {len(all_results)} tirages au total")
    return all_results

def save_api_data(all_results):
    """Sauvegarde les données API en CSV"""
    if not all_results:
        print("❌ Aucune donnée à sauvegarder")
        return None
    
    try:
        df = pd.DataFrame(all_results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'heure'])
        df = df.drop_duplicates(subset=['date', 'numero_tirage'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = KENO_DATA_DIR / f"keno_api_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"💾 {len(df)} tirages sauvegardés dans : {csv_path}")
        return csv_path
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")
        return None

def verify_csv_data(file_path):
    """Vérifie et analyse un fichier CSV"""
    if not file_path or not Path(file_path).exists():
        print("❌ Aucun fichier à vérifier")
        return
    
    try:
        file_path = Path(file_path)
        print(f"\n🔍 Analyse du fichier : {file_path.name}")
        
        # Essayer différents délimiteurs
        delimiters = [';', ',', '\t']
        df = None
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=5)
                if len(df.columns) > 1:  # Si on a plusieurs colonnes, c'est bon
                    print(f"✅ Délimiteur détecté : '{delimiter}'")
                    break
            except:
                continue
        
        if df is None:
            print("❌ Impossible de lire le fichier CSV")
            return
        
        # Lire le fichier complet
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        print(f"📊 Résumé :")
        print(f"   - Lignes : {len(df):,}")
        print(f"   - Colonnes : {len(df.columns)}")
        print(f"   - Taille du fichier : {file_path.stat().st_size / (1024*1024):.1f} MB")
        
        print(f"\n📋 Colonnes :")
        for i, col in enumerate(df.columns):
            print(f"   {i+1:2d}. {col}")
        
        if len(df) > 0:
            print(f"\n📋 Aperçu (5 premières lignes) :")
            print(df.head())
            
            print(f"\n📋 Aperçu (5 dernières lignes) :")
            print(df.tail())
            
            # Analyser les dates si disponible
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'jour' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    print(f"\n📅 Période couverte :")
                    print(f"   - Du : {df[date_col].min().strftime('%d/%m/%Y')}")
                    print(f"   - Au : {df[date_col].max().strftime('%d/%m/%Y')}")
                    print(f"   - Nombre de jours : {(df[date_col].max() - df[date_col].min()).days + 1}")
                except:
                    print(f"⚠️ Impossible d'analyser les dates dans la colonne {date_col}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Téléchargeur et processeur de résultats Keno')
    parser.add_argument('--local', action='store_true', help='Traiter les fichiers CSV locaux')
    parser.add_argument('--file', type=str, help='Traiter un fichier CSV spécifique')
    parser.add_argument('--download', action='store_true', help='Télécharger depuis l\'API')
    
    args = parser.parse_args()
    
    print("🎯 TÉLÉCHARGEUR ET PROCESSEUR DE RÉSULTATS KENO")
    print("=" * 60)
    
    # Créer les répertoires
    setup_directories()
    
    if args.file:
        # Traiter un fichier spécifique
        file_path = Path(args.file)
        if not file_path.is_absolute():
            # Si le chemin n'est pas absolu, chercher dans plusieurs endroits
            possible_paths = [
                Path.cwd() / file_path,
                Path.home() / "Téléchargements" / file_path,
                KENO_DATA_DIR / file_path
            ]
            
            for p in possible_paths:
                if p.exists():
                    file_path = p
                    break
        
        processed_file = process_local_csv(file_path)
        if processed_file:
            verify_csv_data(processed_file)
    
    elif args.local:
        # Traiter tous les fichiers CSV locaux
        csv_files = find_local_csv_files()
        
        if not csv_files:
            print("❌ Aucun fichier CSV trouvé dans le répertoire keno_data")
            print(f"💡 Copiez vos fichiers CSV dans : {KENO_DATA_DIR}")
        else:
            print(f"📂 {len(csv_files)} fichiers CSV trouvés :")
            for file_path in csv_files:
                print(f"   - {file_path.name}")
                verify_csv_data(file_path)
                print("-" * 40)
    
    else:
        # Par défaut : télécharger depuis l'API
        print(f"\n📅 Démarrage du téléchargement : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        all_results = download_keno_data_recent()
        
        if all_results:
            csv_file = save_api_data(all_results)
            if csv_file:
                verify_csv_data(csv_file)
                
            print(f"\n✅ Téléchargement terminé avec succès !")
        else:
            print(f"\n❌ Aucune donnée récupérée depuis l'API")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
