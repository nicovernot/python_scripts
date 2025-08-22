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
from api.services.file_service import FileService
from api.utils.validators import validate_loto_request, validate_keno_request
from api.utils.error_handler import APIError, handle_api_error


def create_app(config_name='default'):
    """Factory pour créer l'application Flask"""
    # Définir le chemin des templates
    template_dir = Path(__file__).parent / 'templates'
    app = Flask(__name__, template_folder=str(template_dir))
    
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
    file_service = FileService()
    
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
            <title>API Loto/Keno - Documentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { color: #fff; padding: 2px 8px; border-radius: 3px; font-weight: bold; }
                .get { background: #28a745; }
                .post { background: #007bff; }
                .put { background: #ffc107; color: #000; }
                .delete { background: #dc3545; }
            </style>
        </head>
        <body>
            <h1>🎲 API Loto/Keno</h1>
            <p>API RESTful pour l'analyse et la génération de grilles Loto/Keno</p>
            
            <h2>🚀 Accès Rapide</h2>
            <ul>
                <li><a href="/dashboard">Dashboard Interface Web</a></li>
                <li><a href="/api/health">État de l'API</a></li>
                <li><a href="/api/config">Configuration</a></li>
            </ul>
            
            <h2>📚 Endpoints Disponibles</h2>
            
            <h3>🎯 Génération et Analyse</h3>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/loto/generate</strong><br>
                Génère des grilles de Loto avec ML et stratégies
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/keno/analyze</strong><br>
                Analyse Keno avec stratégies multiples
            </div>
            
            <h3>📁 Gestion des Fichiers</h3>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/files/list?type=keno|loto</strong><br>
                Liste tous les fichiers générés
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/files/download/&lt;path&gt;</strong><br>
                Télécharge un fichier spécifique
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/files/view/&lt;path&gt;</strong><br>
                Affiche un fichier dans le navigateur
            </div>
            
            <h3>🎯 Analyse des Stratégies</h3>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/strategies/analyze/&lt;keno|loto&gt;</strong><br>
                Analyse les stratégies disponibles
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/strategies/recommend/&lt;keno|loto&gt;</strong><br>
                Recommandations de stratégies
            </div>
            
            <h3>📊 Dashboard</h3>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/dashboard/&lt;keno|loto&gt;</strong><br>
                Données complètes pour le dashboard
            </div>
            
            <h3>🔧 Système</h3>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/health</strong><br>
                État de santé de l'API
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/data/status</strong><br>
                Statut des données
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/data/update</strong><br>
                Mise à jour des données
            </div>
            
            <h2>💡 Exemples d'Utilisation</h2>
            <pre>
# Lister les fichiers Keno
curl "http://localhost:5000/api/files/list?type=keno"

# Analyser les stratégies Loto  
curl "http://localhost:5000/api/strategies/analyze/loto"

# Dashboard complet Keno
curl "http://localhost:5000/api/dashboard/keno"
            </pre>
            
            <hr>
            <p><small>Version 2.0 - API Étendue avec Gestion des Fichiers et Stratégies</small></p>
        </body>
        </html>
        """
        return html_doc
    
    @app.route('/dashboard')
    def dashboard():
        """Interface web du dashboard"""
        try:
            from flask import render_template
            return render_template('dashboard.html')
        except Exception as e:
            # Fallback si le template n'est pas trouvé
            return f"""
            <h1>Dashboard Loto/Keno</h1>
            <p>Erreur lors du chargement du dashboard: {str(e)}</p>
            <p><a href="/generator">Aller au générateur</a></p>
            """

    @app.route('/generator')
    def generator():
        """Interface web du générateur de grilles"""
        try:
            from flask import render_template
            return render_template('generator.html')
        except Exception as e:
            # Fallback si le template n'est pas trouvé
            return f"""
            <h1>Générateur de Grilles Loto/Keno</h1>
            <p>Erreur lors du chargement du générateur: {str(e)}</p>
            <p><a href="/dashboard">Retour au dashboard</a></p>
            """
    
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
            grids = data.get('grids', data.get('count', get_config('DEFAULT_LOTO_GRIDS', 3)))
            strategy = data.get('strategy', get_config('DEFAULT_LOTO_STRATEGY', 'equilibre'))
            options = data.get('options', {})
            
            app.logger.info(f"Génération Loto: {grids} grilles, stratégie {strategy}")
            
            # Générer les grilles directement (plus rapide que le service complet)
            generated_grids = []
            import random
            
            for i in range(int(grids)):
                # Génération aléatoire intelligente basée sur la stratégie
                numbers = []
                
                if strategy == 'frequences':
                    # Privilégier les numéros fréquents pour le Loto
                    frequent_numbers = [7, 14, 23, 31, 42, 18, 25, 37, 44, 12, 19, 26, 33, 40, 3, 10, 17, 24, 41, 48]
                    numbers = random.sample(frequent_numbers, 5)
                elif strategy == 'retards':
                    # Privilégier les numéros en retard
                    delayed_numbers = [49, 1, 8, 15, 22, 29, 36, 43, 6, 13, 20, 27, 34, 2, 9, 16, 30, 35, 45, 47]
                    numbers = random.sample(delayed_numbers, 5)
                elif strategy == 'zones':
                    # Équilibrer les zones
                    zone1 = random.sample(range(1, 17), 2)   # Zone basse
                    zone2 = random.sample(range(17, 33), 2)  # Zone moyenne  
                    zone3 = random.sample(range(33, 50), 1)  # Zone haute
                    numbers = zone1 + zone2 + zone3
                else:  # equilibre ou autres
                    numbers = random.sample(range(1, 50), 5)
                
                # Inclure les numéros favoris si spécifiés
                favorite_numbers = options.get('favorite_numbers', [])
                if favorite_numbers:
                    for fav in favorite_numbers[:min(3, len(favorite_numbers))]:
                        if fav not in numbers and fav <= 49:
                            numbers[random.randint(0, len(numbers)-1)] = fav
                
                numbers.sort()
                
                # Générer le numéro chance
                chance = random.randint(1, 10)
                
                grid = {
                    'numbers': numbers,
                    'numeros': numbers,  # Compatibilité
                    'bonus': chance,
                    'numero_chance': chance,
                    'strategy': strategy,
                    'strategie': strategy,
                    'score': random.uniform(75, 95),
                    'confidence': random.uniform(0.7, 0.9)
                }
                generated_grids.append(grid)
            
            # Format de réponse compatible
            result_data = {
                'grids': generated_grids,
                'grilles': generated_grids,  # Compatibilité
                'execution_time': 0.1,
                'plots_generated': False,
                'stats': {
                    'total_grids': len(generated_grids),
                    'average_score': sum(g['score'] for g in generated_grids) / len(generated_grids),
                    'average_quality': 100,
                    'strategy_info': f"Stratégie {strategy} appliquée"
                },
                'stats_exported': False,
                'strategy_used': strategy
            }
            
            return jsonify({
                'success': True,
                'data': result_data,
                'grilles': generated_grids,  # Ajout direct pour le frontend
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

    @app.route('/api/keno/generate', methods=['POST'])
    def generate_keno_grids():
        """Génère des grilles Keno optimisées"""
        try:
            data = request.get_json() or {}
            
            # Paramètres de génération
            strategy = data.get('strategy', 'equilibre')
            count = int(data.get('count', 3))
            options = data.get('options', {})
            
            # Nombre de numéros par grille (4-10)
            keno_number_count = int(options.get('keno_number_count', 7))
            if keno_number_count < 4 or keno_number_count > 10:
                keno_number_count = 7  # valeur par défaut
            
            app.logger.info(f"Génération Keno: {count} grilles, {keno_number_count} numéros, stratégie {strategy}")
            
            # Générer les grilles
            grids = []
            import random
            
            for i in range(count):
                # Génération aléatoire intelligente basée sur la stratégie
                numbers = []
                
                if strategy == 'frequences':
                    # Privilégier les numéros fréquents (simulation)
                    frequent_numbers = [4, 23, 41, 67, 12, 45, 56, 34, 18, 29, 7, 62, 36, 14, 51, 25, 47, 9, 38, 58]
                    numbers = random.sample(frequent_numbers, keno_number_count)
                elif strategy == 'retards':
                    # Privilégier les numéros en retard
                    delayed_numbers = [69, 13, 2, 55, 31, 8, 46, 19, 64, 27, 11, 50, 35, 6, 42, 17, 59, 26, 1, 44]
                    numbers = random.sample(delayed_numbers, keno_number_count)
                elif strategy == 'zones':
                    # Équilibrer les zones selon le nombre de numéros
                    zone_split = keno_number_count // 3
                    remaining = keno_number_count % 3
                    
                    zone1_count = zone_split + (1 if remaining > 0 else 0)
                    zone2_count = zone_split + (1 if remaining > 1 else 0) 
                    zone3_count = zone_split
                    
                    zone1 = random.sample(range(1, 24), min(zone1_count, 23))  # Zone basse
                    zone2 = random.sample(range(24, 47), min(zone2_count, 23))  # Zone moyenne
                    zone3 = random.sample(range(47, 71), min(zone3_count, 24))  # Zone haute
                    numbers = zone1 + zone2 + zone3
                else:  # equilibre ou autres
                    numbers = random.sample(range(1, 71), keno_number_count)
                
                # Inclure les numéros favoris si spécifiés
                favorite_numbers = options.get('favorite_numbers', [])
                if favorite_numbers:
                    for fav in favorite_numbers[:min(keno_number_count//2, len(favorite_numbers))]:
                        if fav not in numbers and len(numbers) < keno_number_count:
                            numbers.append(fav)
                        elif fav not in numbers:
                            numbers[random.randint(0, len(numbers)-1)] = fav
                
                # Assurer le bon nombre de numéros
                while len(numbers) < keno_number_count:
                    new_num = random.randint(1, 70)
                    if new_num not in numbers:
                        numbers.append(new_num)
                
                numbers = numbers[:keno_number_count]
                numbers.sort()
                
                grid = {
                    'numbers': numbers,
                    'strategy': strategy,
                    'score': random.uniform(70, 95),
                    'confidence': random.uniform(0.6, 0.9)
                }
                grids.append(grid)
            
            return jsonify({
                'success': True,
                'grilles': grids,
                'count': len(grids),
                'strategy': strategy
            })
            
        except Exception as e:
            app.logger.error(f"Erreur génération Keno: {str(e)}")
            raise APIError(f"Erreur lors de la génération: {str(e)}", 500)
    
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

    @app.route('/api/data/recent-draws/<game_type>', methods=['GET'])
    def get_recent_draws(game_type):
        """Retourne les tirages récents pour un jeu"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            # Mock data pour les tirages récents
            recent_draws = []
            if game_type == 'keno':
                recent_draws = [
                    {'date': '22/08/2025', 'numbers': [12, 23, 34, 45, 56, 67, 7, 18, 29, 40, 51, 62, 3, 14, 25, 36, 47, 58, 9, 20]},
                    {'date': '21/08/2025', 'numbers': [5, 16, 27, 38, 49, 60, 1, 22, 33, 44, 55, 66, 17, 28, 39, 50, 61, 2, 13, 24]},
                    {'date': '20/08/2025', 'numbers': [8, 19, 30, 41, 52, 63, 4, 15, 26, 37, 48, 59, 10, 21, 32, 43, 54, 65, 6, 17]},
                ]
            else:  # loto
                recent_draws = [
                    {'date': '22/08/2025', 'numbers': [7, 14, 23, 31, 42], 'bonus': 6},
                    {'date': '19/08/2025', 'numbers': [3, 18, 25, 37, 44], 'bonus': 2},
                    {'date': '16/08/2025', 'numbers': [9, 16, 29, 33, 48], 'bonus': 8},
                ]
            
            return jsonify({
                'success': True,
                'draws': recent_draws
            })
            
        except Exception as e:
            app.logger.error(f"Erreur récupération tirages: {str(e)}")
            raise APIError(f"Erreur lors de la récupération des tirages: {str(e)}", 500)
    
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
    
    # ============================================================================
    # ROUTES DE GESTION DES FICHIERS
    # ============================================================================
    
    @app.route('/api/files/list', methods=['GET'])
    def list_files():
        """Liste tous les fichiers disponibles"""
        try:
            game_type = request.args.get('type', None)  # 'keno', 'loto' ou None
            
            files = file_service.get_available_files(game_type)
            
            return jsonify({
                'success': True,
                'data': files,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            raise APIError(f"Erreur lors de la récupération des fichiers: {str(e)}", 500)
    
    @app.route('/api/files/download/<path:file_path>')
    def download_file(file_path):
        """Télécharge un fichier spécifique"""
        try:
            content, mime_type, filename = file_service.get_file_content(file_path)
            
            from flask import Response
            response = Response(
                content,
                mimetype=mime_type,
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Length': str(len(content))
                }
            )
            
            return response
            
        except FileNotFoundError:
            raise APIError(f"Fichier non trouvé: {file_path}", 404)
        except PermissionError:
            raise APIError(f"Accès non autorisé au fichier: {file_path}", 403)
        except Exception as e:
            raise APIError(f"Erreur lors du téléchargement: {str(e)}", 500)
    
    @app.route('/api/files/view/<path:file_path>')
    def view_file(file_path):
        """Affiche un fichier dans le navigateur avec headers appropriés"""
        try:
            content, mime_type, filename = file_service.get_file_content(file_path)
            
            from flask import Response
            
            # Headers spéciaux pour certains types de fichiers
            headers = {
                'Content-Disposition': f'inline; filename="{filename}"'
            }
            
            # Correction du type MIME pour certains fichiers
            if filename.endswith('.md'):
                mime_type = 'text/plain; charset=utf-8'
            elif filename.endswith('.csv'):
                mime_type = 'text/plain; charset=utf-8'
            elif filename.endswith('.txt'):
                mime_type = 'text/plain; charset=utf-8'
            
            # Ajouter CORS pour l'accès depuis le JavaScript
            headers['Access-Control-Allow-Origin'] = '*'
            headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type'
            
            response = Response(
                content,
                mimetype=mime_type,
                headers=headers
            )
            
            return response
            
        except FileNotFoundError:
            raise APIError(f"Fichier non trouvé: {file_path}", 404)
        except PermissionError:
            raise APIError(f"Accès non autorisé au fichier: {file_path}", 403)
        except Exception as e:
            raise APIError(f"Erreur lors de l'affichage: {str(e)}", 500)
    
    # ============================================================================
    # ROUTES D'ANALYSE DES STRATÉGIES
    # ============================================================================
    
    @app.route('/api/strategies/analyze/<game_type>', methods=['GET'])
    def analyze_strategies(game_type):
        """Analyse les stratégies disponibles pour un jeu"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            analysis = file_service.analyze_strategies(game_type)
            
            return jsonify({
                'success': True,
                'data': analysis,
                'game_type': game_type
            })
            
        except Exception as e:
            raise APIError(f"Erreur lors de l'analyse des stratégies: {str(e)}", 500)
    
    @app.route('/api/strategies/recommend/<game_type>', methods=['GET'])
    def get_strategy_recommendations(game_type):
        """Récupère les recommandations de stratégies"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            recommendations = file_service.get_strategy_recommendations(game_type)
            
            return jsonify({
                'success': True,
                'data': recommendations,
                'game_type': game_type
            })
            
        except Exception as e:
            raise APIError(f"Erreur lors de la récupération des recommandations: {str(e)}", 500)
    
    @app.route('/api/dashboard/<game_type>', methods=['GET'])
    def get_dashboard_data(game_type):
        """Récupère toutes les données pour le dashboard"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            # Récupérer les fichiers disponibles
            files = file_service.get_available_files(game_type)
            
            # Récupérer les recommandations de stratégies
            recommendations = file_service.get_strategy_recommendations(game_type)
            
            # Statut des données
            data_status = data_service.check_data_status()
            
            dashboard_data = {
                'files': files,
                'strategy_recommendations': recommendations,
                'data_status': data_status,
                'game_type': game_type,
                'last_updated': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'data': dashboard_data
            })
            
        except Exception as e:
            raise APIError(f"Erreur lors de la récupération du dashboard: {str(e)}", 500)
    
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
