"""
Service Keno
============

Service pour l'analyse Keno et génération de recommandations via l'API.
"""

import os
import subprocess
import sys
import json
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


class KenoService:
    """Service pour l'analyse Keno"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.keno_csv = get_config_path('KENO_CSV_PATH') or self.base_path / "keno" / "keno_data" / "keno_202010.csv"
        self.script_path = self.base_path / "keno" / "duckdb_keno.py"
        self.output_dir = get_config_path('KENO_OUTPUT_PATH') or self.base_path / "keno_output"
        self.output_file = self.output_dir / "recommandations_keno.txt"
        
    def analyze(self, strategies: int = 7, deep_analysis: bool = False,
                plots: bool = False, export_stats: bool = False) -> Dict[str, Any]:
        """Effectue une analyse Keno et génère des recommandations"""
        
        # Vérifications préalables
        if not self.keno_csv.exists():
            raise Exception("Fichier de données Keno manquant. Veuillez d'abord mettre à jour les données.")
        
        if not self.script_path.exists():
            raise Exception("Script d'analyse Keno introuvable.")
        
        # Construction de la commande
        command = [
            sys.executable,
            str(self.script_path),
            '--csv', str(self.keno_csv)
        ]
        
        if deep_analysis:
            command.append('--deep-analysis')
        
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
                raise Exception(f"Erreur lors de l'analyse: {result.stderr}")
            
            # Lecture des recommandations générées
            recommendations = self._read_recommendations()
            
            # Extraction des statistiques
            analysis_stats = self._extract_analysis_stats(result.stdout)
            
            return {
                'recommendations': recommendations,
                'stats': analysis_stats,
                'execution_time': self._extract_execution_time(result.stdout),
                'strategies_count': strategies,
                'deep_analysis': deep_analysis,
                'plots_generated': plots,
                'stats_exported': export_stats
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Timeout lors de l'analyse Keno")
        except Exception as e:
            raise Exception(f"Erreur lors de l'analyse: {str(e)}")
    
    def _read_recommendations(self) -> Dict[str, Any]:
        """Lit les recommandations générées"""
        if not self.output_file.exists():
            return {'error': 'Fichier de recommandations non trouvé'}
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            recommendations = {
                'raw_content': content,
                'strategies': [],
                'summary': {}
            }
            
            # Parser le contenu pour extraire les stratégies
            lines = content.split('\n')
            current_strategy = None
            
            for line in lines:
                line = line.strip()
                
                # Détecter une nouvelle stratégie
                if line.startswith('🎯') and 'Stratégie' in line:
                    if current_strategy:
                        recommendations['strategies'].append(current_strategy)
                    
                    current_strategy = {
                        'name': line,
                        'numbers': [],
                        'confidence': 0,
                        'description': ''
                    }
                
                # Extraire les numéros recommandés
                elif current_strategy and '[' in line and ']' in line:
                    try:
                        # Chercher les listes de numéros [1, 2, 3, ...]
                        start = line.find('[')
                        end = line.find(']', start)
                        if start != -1 and end != -1:
                            numbers_str = line[start+1:end]
                            numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                            current_strategy['numbers'] = numbers
                    except:
                        pass
                
                # Extraire la confiance ou description
                elif current_strategy and line:
                    if 'confiance' in line.lower() or 'score' in line.lower():
                        try:
                            # Extraire un pourcentage ou score
                            words = line.split()
                            for word in words:
                                if '%' in word or word.replace('.', '').isdigit():
                                    current_strategy['confidence'] = word
                                    break
                        except:
                            pass
                    else:
                        current_strategy['description'] += line + ' '
            
            # Ajouter la dernière stratégie
            if current_strategy:
                recommendations['strategies'].append(current_strategy)
            
            # Résumé général
            recommendations['summary'] = {
                'total_strategies': len(recommendations['strategies']),
                'file_size': len(content),
                'generation_time': datetime.now().isoformat()
            }
            
            return recommendations
            
        except Exception as e:
            return {'error': f'Erreur lors de la lecture des recommandations: {str(e)}'}
    
    def _extract_analysis_stats(self, output: str) -> Dict[str, Any]:
        """Extrait les statistiques d'analyse depuis la sortie"""
        stats = {
            'draws_analyzed': 0,
            'strategies_generated': 0,
            'analysis_type': 'standard',
            'data_period': '',
            'success': True
        }
        
        lines = output.split('\n')
        for line in lines:
            # Nombre de tirages analysés
            if 'tirages' in line.lower() and ('analysés' in line.lower() or 'chargés' in line.lower()):
                words = line.split()
                for word in words:
                    if word.isdigit():
                        stats['draws_analyzed'] = int(word)
                        break
            
            # Type d'analyse
            elif 'analyse approfondie' in line.lower():
                stats['analysis_type'] = 'deep'
            elif 'analyse rapide' in line.lower():
                stats['analysis_type'] = 'quick'
            
            # Période des données
            elif 'période' in line.lower() or 'données' in line.lower():
                if '2020' in line or '2021' in line or '2022' in line or '2023' in line or '2024' in line or '2025' in line:
                    stats['data_period'] = line.strip()
        
        return stats
    
    def _extract_execution_time(self, output: str) -> Optional[float]:
        """Extrait le temps d'exécution depuis la sortie"""
        lines = output.split('\n')
        for line in lines:
            if 'durée' in line.lower() or 'temps' in line.lower():
                try:
                    # Chercher un nombre suivi de 's' ou 'sec'
                    words = line.split()
                    for word in words:
                        if word.endswith('s') and word[:-1].replace('.', '').isdigit():
                            return float(word[:-1])
                        elif 'sec' in word and word.replace('sec', '').replace('.', '').isdigit():
                            return float(word.replace('sec', ''))
                except:
                    pass
        return None
    
    def get_available_analysis_types(self) -> List[Dict[str, str]]:
        """Retourne les types d'analyse disponibles"""
        return [
            {
                'id': 'quick',
                'name': 'Analyse Rapide',
                'description': 'Analyse standard avec algorithmes de base'
            },
            {
                'id': 'deep',
                'name': 'Analyse Approfondie',
                'description': 'Analyse complète avec ML avancé (plus lent)'
            }
        ]
    
    def get_last_analysis_info(self) -> Optional[Dict[str, Any]]:
        """Retourne les informations de la dernière analyse"""
        if not self.output_file.exists():
            return None
        
        try:
            stat = self.output_file.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Lire le début du fichier pour des infos
            with open(self.output_file, 'r', encoding='utf-8') as f:
                first_lines = f.readlines()[:10]
            
            summary_info = ''
            for line in first_lines:
                if 'RECOMMANDATIONS KENO' in line:
                    summary_info = line.strip()
                    break
            
            return {
                'file_path': str(self.output_file),
                'last_modified': modified.isoformat(),
                'last_modified_readable': modified.strftime('%d/%m/%Y %H:%M:%S'),
                'file_size': stat.st_size,
                'summary': summary_info,
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
                self.keno_csv.exists() and
                self.script_path.exists() and
                self.keno_csv.stat().st_size > 1024  # Au moins 1KB
            )
        except Exception:
            return False
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des statistiques Keno"""
        if not self.keno_csv.exists():
            return {'error': 'Fichier de données Keno manquant'}
        
        try:
            # Lecture rapide pour obtenir des infos de base
            with open(self.keno_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_draws = len(lines) - 1 if lines else 0  # -1 pour l'en-tête
            
            # Dates approximatives (première et dernière ligne)
            first_draw = ''
            last_draw = ''
            
            if len(lines) > 1:
                first_draw = lines[1].split(',')[0] if ',' in lines[1] else ''
                last_draw = lines[-1].split(',')[0] if ',' in lines[-1] else ''
            
            return {
                'total_draws': total_draws,
                'first_draw': first_draw,
                'last_draw': last_draw,
                'file_size_mb': self.keno_csv.stat().st_size / (1024 * 1024),
                'data_health': 'good' if total_draws > 3000 else 'insufficient'
            }
            
        except Exception as e:
            return {
                'error': f'Erreur lors de l\'analyse des statistiques: {str(e)}'
            }
