"""
Validateurs de Requêtes API
==========================

Validation des paramètres pour les endpoints Loto et Keno.
"""

from typing import Dict, List, Any


def validate_loto_request(data: Dict[str, Any]) -> List[str]:
    """Valide une requête de génération Loto"""
    errors = []
    
    # Validation du nombre de grilles
    if 'grids' in data:
        grids = data['grids']
        if not isinstance(grids, int):
            errors.append("'grids' doit être un entier")
        elif grids < 1 or grids > 20:
            errors.append("'grids' doit être entre 1 et 20")
    
    # Validation de la stratégie
    if 'strategy' in data:
        strategy = data['strategy']
        valid_strategies = ['equilibre', 'frequences', 'retards', 'zones', 'paires', 'mixte']
        if not isinstance(strategy, str):
            errors.append("'strategy' doit être une chaîne de caractères")
        elif strategy not in valid_strategies:
            errors.append(f"'strategy' doit être un de: {', '.join(valid_strategies)}")
    
    # Validation des options booléennes
    bool_fields = ['plots', 'export_stats', 'ml_enabled']
    for field in bool_fields:
        if field in data and not isinstance(data[field], bool):
            errors.append(f"'{field}' doit être un booléen")
    
    return errors


def validate_keno_request(data: Dict[str, Any]) -> List[str]:
    """Valide une requête d'analyse Keno"""
    errors = []
    
    # Validation du nombre de stratégies
    if 'strategies' in data:
        strategies = data['strategies']
        if not isinstance(strategies, int):
            errors.append("'strategies' doit être un entier")
        elif strategies < 1 or strategies > 15:
            errors.append("'strategies' doit être entre 1 et 15")
    
    # Validation des options booléennes
    bool_fields = ['deep_analysis', 'plots', 'export_stats', 'ml_enhanced', 
                   'trend_analysis', 'auto_consolidated', 'generate_plots', 
                   'export_csv', 'generate_markdown']
    for field in bool_fields:
        if field in data and not isinstance(data[field], bool):
            errors.append(f"'{field}' doit être un booléen")
    
    return errors


def validate_data_update_request(data: Dict[str, Any]) -> List[str]:
    """Valide une requête de mise à jour des données"""
    errors = []
    
    # Validation des options booléennes
    bool_fields = ['loto', 'keno', 'force', 'verify']
    for field in bool_fields:
        if field in data and not isinstance(data[field], bool):
            errors.append(f"'{field}' doit être un booléen")
    
    return errors


def validate_positive_integer(value: Any, field_name: str, min_val: int = 1, max_val: int = None) -> List[str]:
    """Valide qu'une valeur est un entier positif dans une plage"""
    errors = []
    
    if not isinstance(value, int):
        errors.append(f"'{field_name}' doit être un entier")
        return errors
    
    if value < min_val:
        errors.append(f"'{field_name}' doit être >= {min_val}")
    
    if max_val is not None and value > max_val:
        errors.append(f"'{field_name}' doit être <= {max_val}")
    
    return errors


def validate_string_choice(value: Any, field_name: str, choices: List[str]) -> List[str]:
    """Valide qu'une valeur est une chaîne dans une liste de choix"""
    errors = []
    
    if not isinstance(value, str):
        errors.append(f"'{field_name}' doit être une chaîne de caractères")
        return errors
    
    if value not in choices:
        errors.append(f"'{field_name}' doit être un de: {', '.join(choices)}")
    
    return errors


def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Nettoie et standardise les données de requête"""
    sanitized = {}
    
    for key, value in data.items():
        # Nettoyer les chaînes
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                continue
        
        # Convertir les chaînes booléennes
        if isinstance(value, str) and value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        
        # Convertir les chaînes numériques
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        
        sanitized[key] = value
    
    return sanitized
