"""
Service Loto
============

Service pour la génération de grilles Loto optimisées via l'API.
"""

import os
import subprocess
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from config_env import get_config, get_config_path
except ImportError:
    def get_config(key, default=None): return default
    def get_config_path(key, default=None): return Path(default) if default else None


class LotoService:
    """Service pour la génération de grilles Loto"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.loto_csv = get_config_path('LOTO_CSV_PATH') or self.base_path / "loto" / "loto_data" / "loto_201911.csv"
        self.config_file = get_config_path('LOTO_CONFIG_FILE') or self.base_path / "loto" / "strategies.yml"
        self.script_path = self.base_path / "loto" / "duckdb_loto.py"
        self.output_file = self.base_path / "grilles.csv"
        
    def generate_grids(self, grids: int = 3, strategy: str = 'equilibre', 
                      plots: bool = False, export_stats: bool = False) -> Dict[str, Any]:
        """Génère des grilles Loto optimisées"""
        
        # Vérifications préalables
        if not self.loto_csv.exists():
            raise Exception("Fichier de données Loto manquant. Veuillez d'abord mettre à jour les données.")
        
        if not self.script_path.exists():
            raise Exception("Script de génération Loto introuvable.")
        
        # Construction de la commande
        command = [
            sys.executable,
            str(self.script_path),
            '--csv', str(self.loto_csv),
            '--grids', str(grids),
            '--strategy', strategy
        ]
        
        if self.config_file.exists():
            command.extend(['--config', str(self.config_file)])
        
        if plots:
            command.append('--plots')
        
        if export_stats:
            command.append('--export-stats')
        
        try:
            # Exécution du script
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=600  # 10 minutes max
            )
            
            if result.returncode != 0:
                raise Exception(f"Erreur lors de la génération: {result.stderr}")
            
            # Lecture des grilles générées
            grids_data = self._read_generated_grids()
            
            # Statistiques de génération
            stats = self._extract_generation_stats(result.stdout)
            
            return {
                'grids': grids_data,
                'stats': stats,
                'execution_time': self._extract_execution_time(result.stdout),
                'strategy_used': strategy,
                'plots_generated': plots,
                'stats_exported': export_stats
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Timeout lors de la génération des grilles")
        except Exception as e:
            raise Exception(f"Erreur lors de la génération: {str(e)}")
    
    def _read_generated_grids(self) -> List[Dict[str, Any]]:
        """Lit les grilles générées depuis le fichier CSV"""
        if not self.output_file.exists():
            return []
        
        grids = []
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parser les numéros
                    numbers = []
                    for i in range(1, 6):  # B1 à B5
                        if f'B{i}' in row:
                            numbers.append(int(row[f'B{i}']))
                    
                    grid = {
                        'id': int(row.get('Grille', 0)),
                        'numbers': numbers,
                        'score_ml': float(row.get('Score_ML', 0)),
                        'sum': int(row.get('Somme', 0)),
                        'balance': row.get('Equilibre', ''),
                        'zones': row.get('Zones', ''),
                        'quality': row.get('Qualite', '')
                    }
                    grids.append(grid)
            
            return grids
            
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture des grilles: {str(e)}")
    
    def _extract_generation_stats(self, output: str) -> Dict[str, Any]:
        """Extrait les statistiques de génération depuis la sortie"""
        stats = {
            'total_grids': 0,
            'average_score': 0.0,
            'average_quality': 0.0,
            'strategy_info': ''
        }
        
        lines = output.split('\n')
        for line in lines:
            if 'grilles générées' in line.lower():
                # Extraire le nombre de grilles
                words = line.split()
                for i, word in enumerate(words):
                    if word.isdigit() and 'grille' in words[i+1:i+3]:
                        stats['total_grids'] = int(word)
                        break
            
            elif 'score ml moyen' in line.lower():
                # Extraire le score ML moyen
                try:
                    score_part = line.split(':')[1].strip()
                    score = float(score_part.split('/')[0])
                    stats['average_score'] = score
                except:
                    pass
            
            elif 'qualité moyenne' in line.lower():
                # Extraire la qualité moyenne
                try:
                    quality_part = line.split(':')[1].strip()
                    quality = float(quality_part.replace('%', ''))
                    stats['average_quality'] = quality
                except:
                    pass
            
            elif 'stratégie' in line.lower() and 'active' in line.lower():
                stats['strategy_info'] = line.strip()
        
        return stats
    
    def _extract_execution_time(self, output: str) -> Optional[float]:
        """Extrait le temps d'exécution depuis la sortie"""
        lines = output.split('\n')
        for line in lines:
            if 'durée' in line.lower() and 's' in line:
                try:
                    # Chercher un nombre suivi de 's'
                    words = line.split()
                    for word in words:
                        if word.endswith('s') and word[:-1].replace('.', '').isdigit():
                            return float(word[:-1])
                except:
                    pass
        return None
    
    def get_available_strategies(self) -> List[Dict[str, str]]:
        """Retourne la liste des stratégies disponibles"""
        strategies = [
            {
                'id': 'equilibre',
                'name': 'Équilibrée',
                'description': 'Approche équilibrée entre fréquence, retard et momentum'
            },
            {
                'id': 'agressive',
                'name': 'Agressive',
                'description': 'Favorise les tendances récentes et le momentum'
            },
            {
                'id': 'conservatrice',
                'name': 'Conservatrice',
                'description': 'Basée sur l\'historique long terme'
            },
            {
                'id': 'ml_focus',
                'name': 'ML Focus',
                'description': 'Priorité maximale au machine learning'
            }
        ]
        
        return strategies
    
    def get_last_generation_info(self) -> Optional[Dict[str, Any]]:
        """Retourne les informations de la dernière génération"""
        if not self.output_file.exists():
            return None
        
        try:
            stat = self.output_file.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Compter les grilles
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                lines = list(reader)
                grid_count = len(lines) - 1 if lines else 0  # -1 pour l'en-tête
            
            return {
                'file_path': str(self.output_file),
                'last_modified': modified.isoformat(),
                'last_modified_readable': modified.strftime('%d/%m/%Y %H:%M:%S'),
                'grid_count': grid_count,
                'age_hours': (datetime.now() - modified).total_seconds() / 3600
            }
            
        except Exception as e:
            return {
                'error': f'Erreur lors de la lecture des informations: {str(e)}'
            }
    
    def is_healthy(self) -> bool:
        """Vérifie si le service est en bonne santé"""
        try:
            return (
                self.loto_csv.exists() and
                self.script_path.exists() and
                self.loto_csv.stat().st_size > 1024  # Au moins 1KB
            )
        except Exception:
            return False
    
    def validate_strategy(self, strategy: str) -> bool:
        """Valide qu'une stratégie est disponible"""
        available = [s['id'] for s in self.get_available_strategies()]
        return strategy in available
    
    def get_strategy_info(self, strategy: str) -> Optional[Dict[str, str]]:
        """Retourne les informations d'une stratégie"""
        strategies = self.get_available_strategies()
        for s in strategies:
            if s['id'] == strategy:
                return s
        return None
