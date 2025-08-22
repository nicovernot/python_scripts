"""
Service de Gestion des Données
=============================

Service pour gérer les données Loto et Keno (téléchargement, statut, validation).
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from config_env import get_config_path, get_config_bool
except ImportError:
    def get_config_path(key, default=None): return Path(default) if default else None
    def get_config_bool(key, default=False): return default


class DataService:
    """Service de gestion des données"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.loto_csv = get_config_path('LOTO_CSV_PATH') or self.base_path / "loto" / "loto_data" / "loto_201911.csv"
        self.keno_csv = get_config_path('KENO_CSV_PATH') or self.base_path / "keno" / "keno_data" / "keno_202010.csv"
        
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut des fichiers de données"""
        loto_status = self._get_file_status(self.loto_csv, "Loto")
        keno_status = self._get_file_status(self.keno_csv, "Keno")
        
        return {
            'loto': loto_status,
            'keno': keno_status,
            'overall_status': 'ok' if loto_status['exists'] and keno_status['exists'] else 'missing_data'
        }
    
    def _get_file_status(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Retourne le statut d'un fichier de données"""
        if not file_path.exists():
            return {
                'exists': False,
                'type': file_type,
                'path': str(file_path),
                'message': f'Fichier {file_type} manquant'
            }
        
        stat = file_path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        return {
            'exists': True,
            'type': file_type,
            'path': str(file_path),
            'size_mb': round(size_mb, 2),
            'modified': modified.isoformat(),
            'modified_readable': modified.strftime('%d/%m/%Y %H:%M:%S'),
            'age_hours': (datetime.now() - modified).total_seconds() / 3600
        }
    
    def update_data(self, update_loto: bool = True, update_keno: bool = True) -> Dict[str, Any]:
        """Met à jour les données depuis FDJ"""
        results = {
            'loto': None,
            'keno': None,
            'overall_success': True
        }
        
        if update_loto:
            results['loto'] = self._update_loto_data()
            if not results['loto']['success']:
                results['overall_success'] = False
        
        if update_keno:
            results['keno'] = self._update_keno_data()
            if not results['keno']['success']:
                results['overall_success'] = False
        
        return results
    
    def _update_loto_data(self) -> Dict[str, Any]:
        """Met à jour les données Loto"""
        try:
            script_path = self.base_path / "loto" / "result.py"
            if not script_path.exists():
                return {
                    'success': False,
                    'error': 'Script de téléchargement Loto introuvable'
                }
            
            # Exécuter le script de téléchargement
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=300  # 5 minutes max
            )
            
            if result.returncode == 0:
                # Vérifier que le fichier a été créé/mis à jour
                if self.loto_csv.exists():
                    return {
                        'success': True,
                        'message': 'Données Loto mises à jour avec succès',
                        'file_status': self._get_file_status(self.loto_csv, "Loto")
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Fichier Loto non créé après téléchargement'
                    }
            else:
                return {
                    'success': False,
                    'error': f'Erreur lors du téléchargement Loto: {result.stderr}'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Timeout lors du téléchargement Loto'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur inattendue lors du téléchargement Loto: {str(e)}'
            }
    
    def _update_keno_data(self) -> Dict[str, Any]:
        """Met à jour les données Keno"""
        try:
            script_path = self.base_path / "keno" / "results_clean.py"
            if not script_path.exists():
                return {
                    'success': False,
                    'error': 'Script de téléchargement Keno introuvable'
                }
            
            # Exécuter le script de téléchargement
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                timeout=300  # 5 minutes max
            )
            
            if result.returncode == 0:
                # Vérifier que le fichier a été créé/mis à jour
                if self.keno_csv.exists():
                    return {
                        'success': True,
                        'message': 'Données Keno mises à jour avec succès',
                        'file_status': self._get_file_status(self.keno_csv, "Keno")
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Fichier Keno non créé après téléchargement'
                    }
            else:
                return {
                    'success': False,
                    'error': f'Erreur lors du téléchargement Keno: {result.stderr}'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Timeout lors du téléchargement Keno'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur inattendue lors du téléchargement Keno: {str(e)}'
            }
    
    def is_healthy(self) -> bool:
        """Vérifie si le service est en bonne santé"""
        try:
            status = self.get_status()
            return status['overall_status'] == 'ok'
        except Exception:
            return False
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Valide l'intégrité des données"""
        results = {
            'loto': self._validate_loto_integrity(),
            'keno': self._validate_keno_integrity()
        }
        
        results['overall_valid'] = results['loto']['valid'] and results['keno']['valid']
        return results
    
    def _validate_loto_integrity(self) -> Dict[str, Any]:
        """Valide l'intégrité des données Loto"""
        if not self.loto_csv.exists():
            return {'valid': False, 'error': 'Fichier Loto manquant'}
        
        try:
            with open(self.loto_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 800:  # Minimum attendu
                return {
                    'valid': False, 
                    'error': f'Trop peu de lignes dans le fichier Loto: {len(lines)}'
                }
            
            # Vérifier la structure de la première ligne (en-tête)
            if lines and not lines[0].startswith('date_tirage'):
                return {
                    'valid': False,
                    'error': 'Format d\'en-tête Loto invalide'
                }
            
            return {
                'valid': True,
                'lines': len(lines),
                'message': f'Fichier Loto valide avec {len(lines)} lignes'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Erreur lors de la validation Loto: {str(e)}'
            }
    
    def _validate_keno_integrity(self) -> Dict[str, Any]:
        """Valide l'intégrité des données Keno"""
        if not self.keno_csv.exists():
            return {'valid': False, 'error': 'Fichier Keno manquant'}
        
        try:
            with open(self.keno_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 3000:  # Minimum attendu
                return {
                    'valid': False,
                    'error': f'Trop peu de lignes dans le fichier Keno: {len(lines)}'
                }
            
            # Vérifier la structure de la première ligne (en-tête)
            if lines and 'date_tirage' not in lines[0]:
                return {
                    'valid': False,
                    'error': 'Format d\'en-tête Keno invalide'
                }
            
            return {
                'valid': True,
                'lines': len(lines),
                'message': f'Fichier Keno valide avec {len(lines)} lignes'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Erreur lors de la validation Keno: {str(e)}'
            }
    
    def check_data_status(self) -> Dict[str, Any]:
        """Vérifie le statut global des données pour le dashboard"""
        status = self.get_status()
        
        return {
            'loto_available': status['loto']['exists'],
            'keno_available': status['keno']['exists'], 
            'last_update': datetime.now().isoformat(),
            'overall_health': status['overall_status'] == 'ok',
            'details': {
                'loto': {
                    'file_size': status['loto'].get('size_human', 'N/A'),
                    'last_modified': status['loto'].get('last_modified', 'N/A')
                },
                'keno': {
                    'file_size': status['keno'].get('size_human', 'N/A'), 
                    'last_modified': status['keno'].get('last_modified', 'N/A')
                }
            }
        }
