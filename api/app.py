#!/usr/bin/env python3
"""
API Flask pour le Système Loto/Keno
===================================

API RESTful pour exposer les fonctionnalités d'analyse Loto et Keno
via des endpoints HTTP.

Endpoints principaux:
- POST /api/loto/generate - Génération de grilles Loto
- POST /api/keno/analyze - Analyse Keno
- GET /api/data/status - Statut des données
- GET /api/health - Santé de l'API

Usage:
    python api/app.py
    
Accès:
    http://localhost:5000

Author: Système Loto/Keno API
Date: 13 août 2025
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config_env import load_config, get_config, get_config_path, get_config_bool
except ImportError:
    print("⚠️  Module config_env non disponible, utilisation des valeurs par défaut")
    def load_config(): return None
    def get_config(key, default=None): return default
    def get_config_path(key, default=None): return Path(default) if default else None
    def get_config_bool(key, default=False): return default

from api.services.loto_service import LotoService
from api.services.keno_service import KenoService
from api.services.data_service import DataService
from api.utils.validators import validate_loto_request, validate_keno_request
from api.utils.error_handler import APIError, handle_api_error


def create_app(config_name='default'):
    """Factory pour créer l'application Flask"""
    app = Flask(__name__)
    
    # Configuration
    app.config.update(
        SECRET_KEY=get_config('FLASK_SECRET_KEY', 'loto-keno-api-secret-key'),
        DEBUG=get_config_bool('FLASK_DEBUG', False),
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True
    )
    
    # CORS pour permettre les requêtes cross-origin
    CORS(app)
    
    # Configuration du logging
    if not app.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s'
        )
    
    # Initialisation des services
    loto_service = LotoService()
    keno_service = KenoService()
    data_service = DataService()
    
    # Gestionnaire d'erreurs global
    @app.errorhandler(APIError)
    def handle_api_error_global(error):
        return handle_api_error(error)
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.error(f"Erreur inattendue: {str(error)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Erreur interne du serveur',
            'message': str(error) if app.debug else 'Une erreur inattendue s\'est produite'
        }), 500
    
    # ============================================================================
    # ROUTES PRINCIPALES
    # ============================================================================
    
    @app.route('/')
    def index():
        """Page d'accueil avec documentation de l'API"""
        html_doc = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🎲🎰 API Loto/Keno</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
                .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; margin-right: 10px; }
                .get { background: #27ae60; }
                .post { background: #e74c3c; }
                .status { text-align: center; margin: 20px 0; }
                .success { color: #27ae60; }
                .warning { color: #f39c12; }
                code { background: #2c3e50; color: #ecf0f1; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎲🎰 API Système Loto/Keno</h1>
                
                <div class="status">
                    <h3>📊 Statut de l'API: <span class="success">✅ OPÉRATIONNELLE</span></h3>
                    <p>Dernière mise à jour: {{ timestamp }}</p>
                </div>
                
                <h2>📚 Documentation des Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method post">POST</span><strong>/api/loto/generate</strong>
                    <p>Génère des grilles Loto optimisées</p>
                    <p><strong>Paramètres:</strong> <code>grids</code>, <code>strategy</code>, <code>config</code></p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span><strong>/api/keno/analyze</strong>
                    <p>Effectue une analyse Keno et génère des recommandations</p>
                    <p><strong>Paramètres:</strong> <code>strategies</code>, <code>deep_analysis</code></p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span><strong>/api/data/update</strong>
                    <p>Met à jour les données Loto et/ou Keno depuis FDJ</p>
                    <p><strong>Paramètres:</strong> <code>loto</code>, <code>keno</code></p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span><strong>/api/data/status</strong>
                    <p>Retourne le statut des fichiers de données</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span><strong>/api/health</strong>
                    <p>Vérification de santé de l'API</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span><strong>/api/config</strong>
                    <p>Retourne la configuration active de l'API</p>
                </div>
                
                <h2>🚀 Exemples d'Utilisation</h2>
                
                <h3>Génération Loto</h3>
                <pre><code>curl -X POST http://localhost:5000/api/loto/generate \\
     -H "Content-Type: application/json" \\
     -d '{"grids": 3, "strategy": "equilibre"}'</code></pre>
                
                <h3>Analyse Keno</h3>
                <pre><code>curl -X POST http://localhost:5000/api/keno/analyze \\
     -H "Content-Type: application/json" \\
     -d '{"strategies": 7, "deep_analysis": false}'</code></pre>
                
                <h3>Statut des Données</h3>
                <pre><code>curl http://localhost:5000/api/data/status</code></pre>
                
            </div>
        </body>
        </html>
        """
        return render_template_string(html_doc, timestamp=datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    
    # ============================================================================
    # ENDPOINTS LOTO
    # ============================================================================
    
    @app.route('/api/loto/generate', methods=['POST'])
    def generate_loto():
        """Génère des grilles Loto optimisées"""
        try:
            data = request.get_json() or {}
            
            # Validation des paramètres
            errors = validate_loto_request(data)
            if errors:
                raise APIError(f"Paramètres invalides: {', '.join(errors)}", 400)
            
            # Paramètres avec valeurs par défaut
            grids = data.get('grids', get_config('DEFAULT_LOTO_GRIDS', 3))
            strategy = data.get('strategy', get_config('DEFAULT_LOTO_STRATEGY', 'equilibre'))
            plots = data.get('plots', False)
            export_stats = data.get('export_stats', False)
            
            app.logger.info(f"Génération Loto: {grids} grilles, stratégie {strategy}")
            
            # Génération des grilles
            result = loto_service.generate_grids(
                grids=grids,
                strategy=strategy,
                plots=plots,
                export_stats=export_stats
            )
            
            return jsonify({
                'success': True,
                'data': result,
                'message': f'{grids} grilles générées avec succès',
                'timestamp': datetime.now().isoformat()
            })
            
        except APIError:
            raise
        except Exception as e:
            app.logger.error(f"Erreur génération Loto: {str(e)}")
            raise APIError(f"Erreur lors de la génération: {str(e)}", 500)
    
    @app.route('/api/loto/strategies', methods=['GET'])
    def get_loto_strategies():
        """Retourne la liste des stratégies Loto disponibles"""
        try:
            strategies = loto_service.get_available_strategies()
            return jsonify({
                'success': True,
                'data': {
                    'strategies': strategies,
                    'default': get_config('DEFAULT_LOTO_STRATEGY', 'equilibre')
                }
            })
        except Exception as e:
            app.logger.error(f"Erreur récupération stratégies: {str(e)}")
            raise APIError(f"Erreur lors de la récupération des stratégies: {str(e)}", 500)
    
    # ============================================================================
    # ENDPOINTS KENO
    # ============================================================================
    
    @app.route('/api/keno/analyze', methods=['POST'])
    def analyze_keno():
        """Effectue une analyse Keno et génère des recommandations"""
        try:
            data = request.get_json() or {}
            
            # Validation des paramètres
            errors = validate_keno_request(data)
            if errors:
                raise APIError(f"Paramètres invalides: {', '.join(errors)}", 400)
            
            # Paramètres avec valeurs par défaut
            strategies = data.get('strategies', get_config('DEFAULT_KENO_STRATEGIES', 7))
            deep_analysis = data.get('deep_analysis', False)
            plots = data.get('plots', False)
            export_stats = data.get('export_stats', False)
            
            app.logger.info(f"Analyse Keno: {strategies} stratégies, analyse {'approfondie' if deep_analysis else 'standard'}")
            
            # Analyse Keno
            result = keno_service.analyze(
                strategies=strategies,
                deep_analysis=deep_analysis,
                plots=plots,
                export_stats=export_stats
            )
            
            return jsonify({
                'success': True,
                'data': result,
                'message': f'Analyse Keno terminée avec {strategies} stratégies',
                'timestamp': datetime.now().isoformat()
            })
            
        except APIError:
            raise
        except Exception as e:
            app.logger.error(f"Erreur analyse Keno: {str(e)}")
            raise APIError(f"Erreur lors de l'analyse: {str(e)}", 500)
    
    # ============================================================================
    # ENDPOINTS DONNÉES
    # ============================================================================
    
    @app.route('/api/data/status', methods=['GET'])
    def get_data_status():
        """Retourne le statut des fichiers de données"""
        try:
            status = data_service.get_status()
            return jsonify({
                'success': True,
                'data': status,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            app.logger.error(f"Erreur statut données: {str(e)}")
            raise APIError(f"Erreur lors de la vérification du statut: {str(e)}", 500)
    
    @app.route('/api/data/update', methods=['POST'])
    def update_data():
        """Met à jour les données depuis FDJ"""
        try:
            data = request.get_json() or {}
            
            update_loto = data.get('loto', True)
            update_keno = data.get('keno', True)
            
            app.logger.info(f"Mise à jour données: Loto={update_loto}, Keno={update_keno}")
            
            result = data_service.update_data(
                update_loto=update_loto,
                update_keno=update_keno
            )
            
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Mise à jour des données terminée',
                'timestamp': datetime.now().isoformat()
            })
            
        except APIError:
            raise
        except Exception as e:
            app.logger.error(f"Erreur mise à jour données: {str(e)}")
            raise APIError(f"Erreur lors de la mise à jour: {str(e)}", 500)
    
    # ============================================================================
    # ENDPOINTS UTILITAIRES
    # ============================================================================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Vérification de santé de l'API"""
        try:
            # Vérifications basiques
            checks = {
                'api': True,
                'config': load_config() is not None,
                'data_service': data_service.is_healthy(),
                'loto_service': loto_service.is_healthy(),
                'keno_service': keno_service.is_healthy()
            }
            
            all_healthy = all(checks.values())
            
            return jsonify({
                'success': all_healthy,
                'data': {
                    'status': 'healthy' if all_healthy else 'unhealthy',
                    'checks': checks,
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0'
                }
            }), 200 if all_healthy else 503
            
        except Exception as e:
            app.logger.error(f"Erreur health check: {str(e)}")
            return jsonify({
                'success': False,
                'data': {
                    'status': 'error',
                    'error': str(e)
                }
            }), 500
    
    @app.route('/api/config', methods=['GET'])
    def get_config_info():
        """Retourne la configuration active"""
        try:
            config_info = {
                'default_loto_grids': get_config('DEFAULT_LOTO_GRIDS', 3),
                'default_loto_strategy': get_config('DEFAULT_LOTO_STRATEGY', 'equilibre'),
                'default_keno_strategies': get_config('DEFAULT_KENO_STRATEGIES', 7),
                'ml_enabled': get_config_bool('ML_ENABLED', True),
                'auto_cleanup': get_config_bool('AUTO_CLEANUP', True),
                'api_version': '2.0'
            }
            
            return jsonify({
                'success': True,
                'data': config_info
            })
            
        except Exception as e:
            app.logger.error(f"Erreur config: {str(e)}")
            raise APIError(f"Erreur lors de la récupération de la configuration: {str(e)}", 500)
    
    return app


def main():
    """Point d'entrée principal"""
    print("🚀 Démarrage de l'API Loto/Keno...")
    
    # Chargement de la configuration
    load_config()
    
    # Création de l'application
    app = create_app()
    
    # Configuration du serveur
    host = get_config('FLASK_HOST', '0.0.0.0')
    port = int(get_config('FLASK_PORT', 5000))
    debug = get_config_bool('FLASK_DEBUG', False)
    
    print(f"🌐 API disponible sur: http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}/")
    print(f"🔧 Mode debug: {'Activé' if debug else 'Désactivé'}")
    
    # Démarrage du serveur
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )


# Création de l'instance app pour permettre l'import direct
app = create_app()

if __name__ == "__main__":
    main()
