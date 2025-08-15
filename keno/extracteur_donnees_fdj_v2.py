#!/usr/bin/env python3
"""
Extracteur de données Keno depuis le site FDJ - Version améliorée.
"""

import requests
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

class ExtracteurDonneesFDJ:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyser_structure_html(self, html_content: str) -> Dict:
        """
        Analyse la structure du HTML pour comprendre l'organisation des données.
        """
        stats = {
            'taille_total': len(html_content),
            'nb_scripts': len(re.findall(r'<script[^>]*>', html_content)),
            'nb_keno': html_content.lower().count('keno'),
            'nb_numbers': html_content.count('numbers'),
            'nb_json_objects': len(re.findall(r'\{[^{}]*\}', html_content)),
            'patterns_next_js': len(re.findall(r'self\.__next_f\.push', html_content)),
            'patterns_api': len(re.findall(r'"drawGame', html_content)),
        }
        
        self.logger.info(f"Structure HTML: {stats}")
        return stats
    
    def extraire_donnees_next_js(self, html_content: str) -> List[Dict]:
        """
        Extrait les données depuis les scripts Next.js.
        """
        tirages = []
        
        # Chercher les scripts Next.js
        next_pattern = r'self\.__next_f\.push\(\[1,"([^"]+)"\]\)'
        scripts = re.findall(next_pattern, html_content)
        
        self.logger.info(f"Trouvé {len(scripts)} scripts Next.js")
        
        for i, script_content in enumerate(scripts):
            try:
                # Décoder le contenu échappé
                decoded = script_content.encode().decode('unicode_escape')
                
                # Si le script contient "keno", l'analyser plus en détail
                if 'keno' in decoded.lower():
                    self.logger.info(f"Script {i} contient 'keno' (taille: {len(decoded)})")
                    
                    # Chercher différents patterns de données Keno
                    patterns = [
                        r'"id":\s*"(\d+)"[^}]*"gameName":\s*"keno"[^}]*"numbers":\s*(\[[^\]]+\])[^}]*"date":\s*"([^"]+)"',
                        r'"gameName":\s*"keno"[^}]*"numbers":\s*(\[[^\]]+\])[^}]*"date":\s*"([^"]+)"[^}]*"externalId":\s*"([^"]+)"',
                        r'"externalId":\s*"([^"]+)"[^}]*"gameName":\s*"keno"[^}]*"numbers":\s*(\[[^\]]+\])',
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, decoded)
                        self.logger.info(f"Pattern trouvé {len(matches)} correspondances")
                        
                        for match in matches:
                            try:
                                tirage = self.parser_match_keno(match, pattern)
                                if tirage:
                                    tirages.append(tirage)
                                    self.logger.info(f"Tirage extrait: {tirage.get('numero_tirage', 'N/A')}")
                            except Exception as e:
                                self.logger.warning(f"Erreur parsing match: {e}")
                                continue
                    
                    # Sauvegarder le script intéressant pour debug
                    if 'numbers' in decoded and 'keno' in decoded.lower():
                        timestamp = datetime.now().strftime("%H%M%S")
                        with open(f"debug_script_keno_{timestamp}.json", 'w', encoding='utf-8') as f:
                            f.write(decoded)
                        self.logger.info(f"Script Keno sauvegardé dans debug_script_keno_{timestamp}.json")
                        
            except Exception as e:
                self.logger.warning(f"Erreur décodage script {i}: {e}")
                continue
        
        return tirages
    
    def parser_match_keno(self, match: tuple, pattern: str) -> Optional[Dict]:
        """
        Parse un match de pattern Keno et retourne un tirage formaté.
        """
        try:
            if pattern.startswith('"id"'):
                # Pattern: id, numbers, date
                tirage_id, numeros_str, date = match
                external_id = tirage_id
            elif pattern.startswith('"gameName"'):
                # Pattern: numbers, date, externalId
                numeros_str, date, external_id = match
                tirage_id = f"keno_{external_id}"
            else:
                # Pattern: externalId, numbers
                external_id, numeros_str = match
                tirage_id = f"keno_{external_id}"
                date = datetime.now().isoformat()
            
            # Parser les numéros
            numeros = json.loads(numeros_str)
            if isinstance(numeros[0], str):
                numeros = [int(n) for n in numeros]
            
            return {
                'id': tirage_id,
                'date': date,
                'numeros': numeros,
                'numero_tirage': external_id,
                'multiplicateur': '1',
                'joker': '',
                'source': 'FDJ'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur parsing match: {e}")
            return None
    
    def extraire_donnees_alternatives(self, html_content: str) -> List[Dict]:
        """
        Méthodes alternatives d'extraction.
        """
        tirages = []
        
        # Chercher dans les attributs data-*
        data_patterns = [
            r'data-[^=]*=[\'"]\{[^}]*keno[^}]*\}[\'"]',
            r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});',
            r'window\.FDJ_DATA\s*=\s*(\{.*?\});',
        ]
        
        for pattern in data_patterns:
            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            self.logger.info(f"Pattern alternatif: {len(matches)} matches")
            
            for match in matches:
                try:
                    if isinstance(match, str) and match.startswith('{'):
                        data = json.loads(match)
                        # Parcourir récursivement pour trouver des données Keno
                        tirages_found = self.extraire_recursif_keno(data)
                        tirages.extend(tirages_found)
                except json.JSONDecodeError:
                    continue
        
        return tirages
    
    def extraire_recursif_keno(self, data: any, path: str = "") -> List[Dict]:
        """
        Extrait récursivement les données Keno depuis une structure JSON.
        """
        tirages = []
        
        if isinstance(data, dict):
            # Vérifier si c'est un tirage Keno
            if (data.get('gameName') == 'keno' or 
                'keno' in str(data.get('id', '')).lower()) and 'numbers' in data:
                tirage = self.convertir_tirage_fdj(data)
                if tirage:
                    tirages.append(tirage)
            
            # Parcourir récursivement
            for key, value in data.items():
                if key.lower() in ['keno', 'draws', 'results', 'data', 'games']:
                    sub_tirages = self.extraire_recursif_keno(value, f"{path}.{key}")
                    tirages.extend(sub_tirages)
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                sub_tirages = self.extraire_recursif_keno(item, f"{path}[{i}]")
                tirages.extend(sub_tirages)
        
        return tirages
    
    def convertir_tirage_fdj(self, tirage_data: Dict) -> Optional[Dict]:
        """
        Convertit les données FDJ en format standardisé.
        """
        try:
            numeros = tirage_data.get('numbers', [])
            if isinstance(numeros, list) and numeros:
                # Convertir en entiers si ce sont des chaînes
                if isinstance(numeros[0], str):
                    numeros = [int(n) for n in numeros]
                
                return {
                    'id': tirage_data.get('id', ''),
                    'date': tirage_data.get('date', ''),
                    'numeros': numeros,
                    'numero_tirage': tirage_data.get('externalId', tirage_data.get('id', '')),
                    'multiplicateur': str(tirage_data.get('options', {}).get('multiplier', '1')),
                    'joker': tirage_data.get('options', {}).get('joker', ''),
                    'source': 'FDJ'
                }
        except Exception as e:
            self.logger.error(f"Erreur conversion tirage: {e}")
        
        return None
    
    def recuperer_donnees_recentes(self, nb_tirages: int = 10) -> List[Dict]:
        """
        Récupère les données récentes depuis FDJ.
        """
        self.logger.info("Récupération des données FDJ...")
        
        # URLs à essayer
        urls = [
            'https://www.fdj.fr/jeux-de-tirage/keno',
            'https://www.fdj.fr/resultats-et-rapports-officiels/keno',
            'https://www.fdj.fr/resultats-et-rapports-officiels'
        ]
        
        tous_tirages = []
        
        for url in urls:
            try:
                self.logger.info(f"Tentative de récupération depuis {url}")
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    self.logger.info(f"Contenu récupéré ({len(response.content)} bytes)")
                    
                    # Analyser la structure
                    stats = self.analyser_structure_html(response.text)
                    
                    # Sauvegarder pour debug
                    timestamp = datetime.now().strftime("%H%M%S")
                    debug_file = f"debug_fdj_{timestamp}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    self.logger.info(f"Contenu sauvegardé dans {debug_file}")
                    
                    # Essayer différentes méthodes d'extraction
                    methodes = [
                        self.extraire_donnees_next_js,
                        self.extraire_donnees_alternatives
                    ]
                    
                    for methode in methodes:
                        try:
                            tirages_extraits = methode(response.text)
                            if tirages_extraits:
                                self.logger.info(f"{methode.__name__}: {len(tirages_extraits)} tirages extraits")
                                tous_tirages.extend(tirages_extraits)
                        except Exception as e:
                            self.logger.warning(f"Erreur avec {methode.__name__}: {e}")
                    
                    # Si on a trouvé des données, pas besoin d'essayer les autres URLs
                    if tous_tirages:
                        break
                        
                else:
                    self.logger.warning(f"Erreur HTTP {response.status_code} pour {url}")
                    
            except Exception as e:
                self.logger.error(f"Erreur lors de la récupération de {url}: {e}")
        
        # Dédoublonner par numéro de tirage
        tirages_uniques = []
        numeros_vus = set()
        
        for tirage in tous_tirages:
            numero = tirage.get('numero_tirage', '')
            if numero and numero not in numeros_vus:
                numeros_vus.add(numero)
                tirages_uniques.append(tirage)
        
        self.logger.info(f"{len(tirages_uniques)} tirages uniques finaux")
        return tirages_uniques[:nb_tirages]

def main():
    extracteur = ExtracteurDonneesFDJ()
    
    # Essayer d'abord avec le fichier existant pour debug
    try:
        with open('debug_fdj_123331.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("=== Analyse du fichier HTML existant ===")
        stats = extracteur.analyser_structure_html(content)
        
        # Tester les méthodes d'extraction
        tirages_next = extracteur.extraire_donnees_next_js(content)
        tirages_alt = extracteur.extraire_donnees_alternatives(content)
        
        print(f"Next.js: {len(tirages_next)} tirages")
        print(f"Alternatif: {len(tirages_alt)} tirages")
        
        if tirages_next or tirages_alt:
            tous_tirages = tirages_next + tirages_alt
            print(f"\n=== {len(tous_tirages)} tirages trouvés ===")
            for tirage in tous_tirages[:3]:
                print(f"Tirage {tirage.get('numero_tirage', 'N/A')} du {tirage.get('date', 'N/A')}")
                print(f"Numéros: {tirage.get('numeros', [])}")
                print("---")
        else:
            print("Aucun tirage trouvé dans le fichier existant")
            
    except FileNotFoundError:
        print("Aucun fichier debug existant")
    
    # Récupérer de nouvelles données
    print("\n=== Récupération de nouvelles données ===")
    tirages = extracteur.recuperer_donnees_recentes(5)
    
    if tirages:
        print(f"\n=== {len(tirages)} tirages récupérés ===")
        for tirage in tirages:
            print(f"Tirage {tirage.get('numero_tirage', 'N/A')} du {tirage.get('date', 'N/A')}")
            print(f"Numéros: {tirage.get('numeros', [])}")
            print("---")
    else:
        print("Aucun nouveau tirage récupéré")

if __name__ == "__main__":
    main()
