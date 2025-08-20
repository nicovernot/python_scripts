"""
Gestionnaire d'Erreurs API
=========================

Gestion centralisée des erreurs pour l'API Flask Loto/Keno.
"""

from flask import jsonify
from datetime import datetime


class APIError(Exception):
    """Exception personnalisée pour l'API"""
    
    def __init__(self, message, status_code=400, payload=None):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code
        self.payload = payload
        
    def to_dict(self):
        """Convertit l'erreur en dictionnaire"""
        rv = {'success': False, 'error': self.message}
        if self.payload:
            rv.update(self.payload)
        return rv


def handle_api_error(error):
    """Gestionnaire d'erreur API standard"""
    response = jsonify({
        'success': False,
        'error': error.message,
        'status_code': error.status_code,
        'timestamp': datetime.now().isoformat()
    })
    response.status_code = error.status_code
    return response


def validate_required_fields(data, required_fields):
    """Valide la présence des champs requis"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise APIError(f"Champs requis manquants: {', '.join(missing_fields)}", 400)


def validate_field_type(data, field, expected_type, required=False):
    """Valide le type d'un champ"""
    if field not in data:
        if required:
            raise APIError(f"Champ requis manquant: {field}", 400)
        return
    
    if not isinstance(data[field], expected_type):
        raise APIError(f"Type invalide pour {field}: attendu {expected_type.__name__}", 400)


def validate_field_range(data, field, min_val=None, max_val=None):
    """Valide qu'un champ numérique est dans une plage"""
    if field not in data:
        return
    
    value = data[field]
    if min_val is not None and value < min_val:
        raise APIError(f"{field} doit être >= {min_val}", 400)
    
    if max_val is not None and value > max_val:
        raise APIError(f"{field} doit être <= {max_val}", 400)


def validate_field_choices(data, field, choices):
    """Valide qu'un champ est dans une liste de choix"""
    if field not in data:
        return
    
    if data[field] not in choices:
        raise APIError(f"{field} doit être un de: {', '.join(map(str, choices))}", 400)
