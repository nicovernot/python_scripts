#!/usr/bin/env python3
"""
Gestionnaire de Configuration Environnement
==========================================

Utilitaire pour charger et gérer les variables d'environnement
du système Loto/Keno.

Usage:
    from config_env import load_config, get_config
    
    # Charger la configuration
    config = load_config()
    
    # Accéder aux valeurs
    loto_csv = get_config('LOTO_CSV_PATH')
    
Author: Système Loto/Keno
Date: 13 août 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Gestionnaire de configuration centralisé"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.config = {}
        self._load_env_file()
        
    def _load_env_file(self):
        """Charge le fichier .env"""
        if not self.env_file.exists():
            logging.warning(f"Fichier {self.env_file} non trouvé")
            return
            
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Ignorer les commentaires et lignes vides
                    if not line or line.startswith('#') or line.startswith('='):
                        continue
                        
                    # Parser les variables
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Supprimer les guillemets si présents
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                            
                        self.config[key] = value
                        
        except Exception as e:
            logging.error(f"Erreur lors du chargement de {self.env_file}: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une variable de configuration"""
        # Priorité: variable d'environnement > fichier .env > défaut
        value = os.environ.get(key) or self.config.get(key, default)
        
        # Conversion automatique des types
        if isinstance(value, str):
            # Boolean
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            # Integer
            elif value.isdigit():
                return int(value)
            # Float
            elif value.replace('.', '').isdigit():
                try:
                    return float(value)
                except ValueError:
                    pass
                    
        return value
        
    def get_path(self, key: str, default: str = None) -> Path:
        """Récupère un chemin et le convertit en Path"""
        path_str = self.get(key, default)
        if path_str:
            return Path(path_str).expanduser()
        return None
        
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Récupère une valeur booléenne"""
        value = self.get(key, default)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
        
    def get_int(self, key: str, default: int = 0) -> int:
        """Récupère une valeur entière"""
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            return default
            
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Récupère une valeur décimale"""
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            return default
            
    def set_env_vars(self):
        """Définit les variables d'environnement système"""
        for key, value in self.config.items():
            if key not in os.environ:
                os.environ[key] = str(value)
                
    def validate_paths(self) -> Dict[str, bool]:
        """Valide l'existence des chemins critiques"""
        path_keys = [
            'LOTO_DATA_PATH', 'KENO_DATA_PATH', 
            'MODEL_SAVE_PATH', 'CACHE_PATH'
        ]
        
        results = {}
        for key in path_keys:
            path = self.get_path(key)
            if path:
                results[key] = path.exists()
            else:
                results[key] = False
                
        return results
        
    def create_missing_dirs(self):
        """Crée les dossiers manquants"""
        dir_keys = [
            'LOTO_DATA_PATH', 'KENO_DATA_PATH', 'LOTO_PLOTS_PATH', 
            'KENO_PLOTS_PATH', 'LOTO_STATS_PATH', 'KENO_STATS_PATH',
            'KENO_OUTPUT_PATH', 'CACHE_PATH', 'LOGS_PATH'
        ]
        
        created = []
        for key in dir_keys:
            path = self.get_path(key)
            if path and not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    created.append(str(path))
                except Exception as e:
                    logging.error(f"Impossible de créer {path}: {e}")
                    
        return created
        
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la configuration"""
        return {
            'env_file_exists': self.env_file.exists(),
            'config_loaded': len(self.config),
            'paths_valid': self.validate_paths(),
            'ml_enabled': self.get_bool('ML_ENABLED'),
            'auto_cleanup': self.get_bool('AUTO_CLEANUP'),
            'cli_colors': self.get_bool('CLI_COLORS_ENABLED')
        }


# Instance globale
_config_manager = None


def load_config(env_file: str = ".env") -> ConfigManager:
    """Charge la configuration depuis le fichier .env"""
    global _config_manager
    _config_manager = ConfigManager(env_file)
    _config_manager.set_env_vars()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Raccourci pour récupérer une valeur de configuration"""
    global _config_manager
    if _config_manager is None:
        load_config()
    return _config_manager.get(key, default)


def get_config_path(key: str, default: str = None) -> Path:
    """Raccourci pour récupérer un chemin"""
    global _config_manager
    if _config_manager is None:
        load_config()
    return _config_manager.get_path(key, default)


def get_config_bool(key: str, default: bool = False) -> bool:
    """Raccourci pour récupérer un booléen"""
    global _config_manager
    if _config_manager is None:
        load_config()
    return _config_manager.get_bool(key, default)


def initialize_environment():
    """Initialise l'environnement complet"""
    config = load_config()
    
    # Créer les dossiers manquants
    created_dirs = config.create_missing_dirs()
    if created_dirs:
        print(f"📁 Dossiers créés: {', '.join(created_dirs)}")
        
    # Configurer le logging
    log_level = config.get('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=config.get('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # Configurer les variables Python
    if config.get_bool('PYTHON_WARNINGS') == 'ignore':
        import warnings
        warnings.filterwarnings('ignore')
        
    return config


def print_config_summary():
    """Affiche un résumé de la configuration"""
    config = load_config()
    summary = config.get_summary()
    
    print("🔧 RÉSUMÉ DE LA CONFIGURATION")
    print("=" * 40)
    print(f"📄 Fichier .env: {'✅' if summary['env_file_exists'] else '❌'}")
    print(f"⚙️  Variables chargées: {summary['config_loaded']}")
    print(f"🤖 ML activé: {'✅' if summary['ml_enabled'] else '❌'}")
    print(f"🧹 Auto-cleanup: {'✅' if summary['auto_cleanup'] else '❌'}")
    print(f"🎨 Couleurs CLI: {'✅' if summary['cli_colors'] else '❌'}")
    
    print("\n📁 VALIDATION DES CHEMINS:")
    for path_key, exists in summary['paths_valid'].items():
        status = '✅' if exists else '❌'
        path = config.get_path(path_key)
        print(f"   {status} {path_key}: {path}")


if __name__ == "__main__":
    """Test du gestionnaire de configuration"""
    print("🧪 Test du Gestionnaire de Configuration")
    print("=" * 50)
    
    # Test de chargement
    config = initialize_environment()
    
    # Affichage du résumé
    print_config_summary()
    
    # Test d'accès aux variables
    print("\n🎯 VARIABLES PRINCIPALES:")
    print(f"   LOTO_CSV_PATH: {get_config_path('LOTO_CSV_PATH')}")
    print(f"   KENO_CSV_PATH: {get_config_path('KENO_CSV_PATH')}")
    print(f"   DEFAULT_LOTO_GRIDS: {get_config('DEFAULT_LOTO_GRIDS')}")
    print(f"   ML_ENABLED: {get_config_bool('ML_ENABLED')}")
    
    print("\n✅ Test terminé avec succès !")
