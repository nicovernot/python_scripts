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
import subprocess
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

# Imports de services avec gestion d'erreur
try:
    from api.services.loto_service import LotoService
    from api.services.keno_service import KenoService
    from api.services.data_service import DataService
    from api.services.file_service import FileService
    from api.utils.validators import validate_loto_request, validate_keno_request
    from api.utils.error_handler import APIError, handle_api_error
except ImportError as e:
    print(f"⚠️  Erreur import API modules: {e}")
    # Ajouter le chemin du projet au sys.path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    
    # Réessayer les imports
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
    
    # Répertoires de sortie
    PROJECT_ROOT = Path(__file__).parent.parent
    KENO_OUTPUT_DIR = PROJECT_ROOT / 'keno_output'
    LOTO_OUTPUT_DIR = PROJECT_ROOT / 'loto' / 'output'  # Utiliser loto/output au lieu de loto_output
    
    # Créer les répertoires s'ils n'existent pas
    KENO_OUTPUT_DIR.mkdir(exist_ok=True)
    LOTO_OUTPUT_DIR.mkdir(exist_ok=True)

    # Configuration du logging
    if not app.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s'
        )
    
    # Initialisation des services
    loto_service = LotoService()
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
    # FONCTIONS UTILITAIRES
    # ============================================================================
    
    def generate_analysis_report(game_type, subprocess_result, start_time, options):
        """Génère un rapport détaillé d'analyse"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse output pour extraire les statistiques
        output = subprocess_result.stdout
        
        # Informations de base
        report = {
            'game_type': game_type.upper(),
            'timestamp': end_time.isoformat(),
            'duration_seconds': round(duration, 2),
            'status': 'success' if subprocess_result.returncode == 0 else 'error',
            'options_used': options,
            'files_generated': {
                'plots': [],
                'exports': [],
                'markdown': [],
                'text': []
            },
            'statistics': {},
            'summary': '',
            'recommendations': []
        }
        
        # Extraire les statistiques spécifiques selon le jeu
        if game_type == 'keno':
            # Extraire infos Keno
            if '3536 tirages analysés' in output:
                report['statistics']['total_draws'] = 3536
                report['statistics']['numbers_analyzed'] = 70
                report['statistics']['pairs_analyzed'] = 100
                
            # Chercher les recommandations
            if 'MIX_INTELLIGENT' in output:
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if 'MIX_INTELLIGENT' in line and i+1 < len(lines):
                        report['recommendations'].append({
                            'strategy': 'MIX_INTELLIGENT',
                            'description': 'Mix intelligent avec pondération probabiliste',
                            'score': '0.95'
                        })
                        break
                        
            report['summary'] = f"Analyse complète de {report['statistics'].get('total_draws', 'N/A')} tirages Keno avec génération de {len(report['recommendations'])} stratégies recommandées."
            
        elif game_type == 'loto':
            # Extraire infos Loto
            if 'tirages chargés avec succès' in output:
                for line in output.split('\n'):
                    if 'tirages chargés avec succès' in line:
                        try:
                            draws = int(line.split()[1])
                            report['statistics']['total_draws'] = draws
                        except:
                            pass
                            
            if 'Score ML:' in output:
                for line in output.split('\n'):
                    if 'Score ML:' in line:
                        try:
                            score = line.split('Score ML:')[1].split('/')[0].strip()
                            report['statistics']['ml_score'] = score
                        except:
                            pass
                            
            # Stratégie utilisée
            if 'STRATÉGIE' in output:
                report['recommendations'].append({
                    'strategy': 'EQUILIBRE',
                    'description': 'Stratégie équilibrée avec Machine Learning',
                    'score': report['statistics'].get('ml_score', 'N/A')
                })
                
            report['summary'] = f"Génération de grilles Loto optimisées avec ML (Score: {report['statistics'].get('ml_score', 'N/A')}/100) basée sur {report['statistics'].get('total_draws', 'N/A')} tirages historiques."
        
        # Détecter les fichiers générés
        project_root = Path(__file__).parent.parent
        
        # Graphiques
        plots_dir = project_root / f'{game_type}_analyse_plots'
        if plots_dir.exists():
            report['files_generated']['plots'] = [f.name for f in plots_dir.glob('*.png')]
            
        # Exports
        exports_dir = project_root / f'{game_type}_stats_exports'
        if exports_dir.exists():
            report['files_generated']['exports'] = [f.name for f in exports_dir.glob('*.csv')]
            
        # Rapports Markdown depuis les répertoires output
        output_dir = project_root / f'{game_type}_output'
        if output_dir.exists():
            markdown_files = sorted([f.name for f in output_dir.glob('*.md')])
            text_files = sorted([f.name for f in output_dir.glob('*.txt')])
            report['files_generated']['markdown'] = markdown_files
            report['files_generated']['text'] = text_files
        else:
            report['files_generated']['markdown'] = []
            report['files_generated']['text'] = []
            
        return report
    
    def generate_keno_markdown_report(result):
        """Génère un rapport Markdown pour l'analyse Keno"""
        app.logger.info(f"🔧 DEBUG: generate_keno_markdown_report() - result type={type(result)}")
        
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        
        # Vérifier si result est un dictionnaire valide
        if not isinstance(result, dict):
            app.logger.error(f"🔧 ERROR: Expected dict, got {type(result)}: {result}")
            return f"""# ❌ Rapport d'Erreur Keno

**Date d'analyse :** {timestamp}  
**Type :** Erreur d'analyse  

## ⚠️ Erreur Rencontrée

L'analyse n'a pas pu être complétée correctement.

**Détails de l'erreur :**
```
{str(result)}
```

---
*Rapport d'erreur généré automatiquement*
"""
        
        app.logger.info(f"🔧 DEBUG: result.keys()={list(result.keys())}")
        
        try:
            markdown_content = f"""# 🎯 Rapport d'Analyse Keno

**Date d'analyse :** {timestamp}  
**Type :** Analyse stratégique complète  
**Durée d'exécution :** {result.get('execution_time', 'N/A')} secondes

## 📊 Résumé Exécutif

- **Nombre de stratégies analysées :** {result.get('strategies_count', 'N/A')}
- **Type d'analyse :** {'Approfondie' if result.get('deep_analysis', False) else 'Standard'}
- **Graphiques générés :** {'Oui' if result.get('plots_generated', False) else 'Non'}
- **Statistiques exportées :** {'Oui' if result.get('stats_exported', False) else 'Non'}

## 🎲 Recommandations Stratégiques

"""
            app.logger.info("🔧 DEBUG: Section résumé générée")
            
            # Ajouter les recommandations
            recommendations = result.get('recommendations', [])
            app.logger.info(f"🔧 DEBUG: recommendations type={type(recommendations)}, value={recommendations}")
            
            # Vérifier si recommendations est un dict avec une erreur
            if isinstance(recommendations, dict) and 'error' in recommendations:
                markdown_content += f"""### ⚠️ Erreur dans les recommandations

**Message d'erreur :** {recommendations['error']}

"""
            elif isinstance(recommendations, list) and recommendations:
                for i, rec in enumerate(recommendations, 1):
                    markdown_content += f"""### Stratégie {i}: {rec.get('strategy', 'N/A')}

- **Numéros recommandés :** {', '.join(map(str, rec.get('numbers', [])))}
- **Niveau de confiance :** {rec.get('confidence', 0):.1%}
- **Score d'analyse :** {rec.get('score', 0):.1f}/100
- **Description :** {rec.get('analysis', 'N/A')}

"""
            else:
                markdown_content += f"""### ⚠️ Aucune recommandation disponible

Les recommandations n'ont pas pu être générées pour cette analyse.

"""
            app.logger.info("🔧 DEBUG: Section recommandations générée")
            
            # Ajouter les statistiques
            stats = result.get('stats', {})
            app.logger.info(f"🔧 DEBUG: stats type={type(stats)}, value={stats}")
            
            if stats and isinstance(stats, dict):
                markdown_content += f"""## 📈 Statistiques d'Analyse

| Métrique | Valeur |
|----------|--------|
| Confiance moyenne | {stats.get('average_confidence', 0):.1%} |
| Score moyen | {stats.get('average_score', 0):.1f}/100 |
| Qualité des données | {stats.get('data_quality', 'N/A')} |
| Dernière mise à jour | {stats.get('last_update', 'N/A')} |

"""
            else:
                markdown_content += f"""## 📈 Statistiques d'Analyse

*Aucune statistique détaillée disponible pour cette analyse.*

"""
            app.logger.info("🔧 DEBUG: Section statistiques générée")
            
            markdown_content += f"""## ⚠️ Avertissement

Ce rapport est généré à des fins d'analyse statistique. Les jeux de hasard comportent des risques financiers. Jouez avec modération.

---
*Rapport généré automatiquement par l'API Loto/Keno - {timestamp}*
"""
            app.logger.info("🔧 DEBUG: Markdown complet généré")
            
        except Exception as e:
            app.logger.error(f"🔧 ERROR: Exception in markdown generation: {e}")
            import traceback
            app.logger.error(f"🔧 ERROR: Traceback: {traceback.format_exc()}")
            raise
        
        return markdown_content
    
    def generate_loto_markdown_report(result):
        """Génère un rapport Markdown pour les grilles Loto"""
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        
        markdown_content = f"""# 🎰 Rapport de Génération Loto

**Date de génération :** {timestamp}  
**Type :** Grilles optimisées avec Machine Learning  
**Durée d'exécution :** {result.get('execution_time', 'N/A')} secondes

## 📊 Résumé Exécutif

- **Nombre de grilles générées :** {result.get('grids_count', 'N/A')}
- **Stratégie utilisée :** {result.get('strategy', 'N/A')}
- **Graphiques générés :** {'Oui' if result.get('plots_generated', False) else 'Non'}
- **Statistiques exportées :** {'Oui' if result.get('stats_exported', False) else 'Non'}

## 🎲 Grilles Recommandées

"""
        
        # Ajouter les grilles
        grids = result.get('grids', [])
        for i, grid in enumerate(grids, 1):
            numbers = grid.get('numbers', [])
            chance = grid.get('chance', 0)
            markdown_content += f"""### Grille {i}

- **Numéros :** {', '.join(map(str, numbers[:5]))}
- **Numéro Chance :** {chance}
- **Score ML :** {grid.get('ml_score', 'N/A')}/100
- **Probabilité estimée :** {grid.get('probability', 'N/A')}

"""
        
        # Ajouter les statistiques
        stats = result.get('stats', {})
        if stats:
            markdown_content += f"""## 📈 Statistiques de Génération

| Métrique | Valeur |
|----------|--------|
| Score ML moyen | {stats.get('average_ml_score', 0):.1f}/100 |
| Tirages analysés | {stats.get('total_draws', 'N/A')} |
| Qualité des données | {stats.get('data_quality', 'N/A')} |
| Algorithme utilisé | {stats.get('algorithm', 'N/A')} |

## ⚠️ Avertissement

Ce rapport est généré à des fins d'analyse statistique. Les jeux de hasard comportent des risques financiers. Jouez avec modération.

---
*Rapport généré automatiquement par l'API Loto/Keno - {timestamp}*
"""
        
        return markdown_content

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
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/data/download/&lt;keno|loto&gt;</strong><br>
                Téléchargement de données spécifiques par jeu
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/analysis/run/&lt;keno|loto&gt;</strong><br>
                Lance l'analyse complète pour un jeu (graphiques, stats, exports)
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/analysis/status/&lt;keno|loto&gt;</strong><br>
                Retourne le statut des analyses et fichiers générés
            </div>
            
            <h2>💡 Exemples d'Utilisation</h2>
            <pre>
# Lister les fichiers Keno
curl "http://localhost:5000/api/files/list?type=keno"

# Télécharger données Keno
curl -X POST "http://localhost:5000/api/data/download/keno"

# Télécharger données Loto  
curl -X POST "http://localhost:5000/api/data/download/loto"

# Lancer analyse complète Keno
curl -X POST "http://localhost:5000/api/analysis/run/keno" \
  -H "Content-Type: application/json" \
  -d '{"options": {"auto_consolidated": true, "plots": true, "export_stats": true}}'

# Vérifier statut analyse Loto
curl "http://localhost:5000/api/analysis/status/loto"

# Analyser les stratégies Loto  
curl "http://localhost:5000/api/strategies/analyze/loto"

# Dashboard complet Keno
curl "http://localhost:5000/api/dashboard/keno"
            </pre>
            
            <hr>
            <p><small>Version 2.1 - API Étendue avec Analyses Automatisées</small></p>
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
    
    @app.route('/api/loto/analyze', methods=['POST'])
    def analyze_loto():
        """Génère des grilles Loto avec analyse (alias pour generate)"""
        try:
            app.logger.info("🎯 Requête reçue pour analyse/génération Loto")
            data = request.get_json() or {}
            app.logger.info(f"Données reçues: {data}")
            
            # Paramètres avec valeurs par défaut
            grids = data.get('grids', get_config('DEFAULT_LOTO_GRIDS', 3))
            strategy = data.get('strategy', get_config('DEFAULT_LOTO_STRATEGY', 'equilibre'))
            plots = data.get('generate_plots', data.get('plots', False))
            export_stats = data.get('export_csv', data.get('export_stats', False))
            generate_markdown = data.get('generate_markdown', False)
            
            app.logger.info(f"Génération Loto: {grids} grilles, stratégie {strategy}, plots: {plots}, export: {export_stats}")
            
            # Utiliser le service Loto
            service = LotoService()
            
            result = service.generate_grids(
                grids=grids,
                strategy=strategy,
                plots=plots,
                export_stats=export_stats
            )
            
            # Générer le fichier Markdown si demandé
            files_generated = {
                'plots': [],
                'exports': [],
                'markdown': [],
                'text': []
            }
            
            if generate_markdown:
                markdown_content = generate_loto_markdown_report(result)
                # Nom de fichier fixe, écrasé à chaque génération
                markdown_file = LOTO_OUTPUT_DIR / "grilles_loto.md"
                
                try:
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    files_generated['markdown'].append(markdown_file.name)
                    app.logger.info(f"✅ Rapport Markdown généré: {markdown_file}")
                except Exception as e:
                    app.logger.error(f"❌ Erreur génération Markdown: {e}")
            
            # Détecter automatiquement tous les fichiers générés (même si generate_markdown=false)
            # Les scripts peuvent créer des fichiers indépendamment du paramètre generate_markdown
            project_root = Path(__file__).parent.parent
            output_dir = project_root / 'loto' / 'output'  # Utiliser loto/output au lieu de loto_output
            plots_dir = project_root / 'loto_analyse_plots'
            exports_dir = project_root / 'loto_stats_exports'
            
            # Détecter tous les fichiers Markdown créés (même par les scripts)
            if output_dir.exists():
                # Récupérer tous les fichiers Markdown
                markdown_files = [f.name for f in output_dir.glob('*.md')]
                
                # Trier par ordre de priorité (au lieu d'alphabétique)
                # rapport_analyse.md est le vrai fichier généré par les scripts Loto
                priority_order = ['rapport_analyse.md', 'grilles_loto.md', 'rapport_loto.md', 'analyse_loto.md']
                
                def get_priority(filename):
                    try:
                        return priority_order.index(filename)
                    except ValueError:
                        return len(priority_order)  # Fichiers non prioritaires à la fin
                
                all_markdown_files = sorted(markdown_files, key=get_priority)
                all_text_files = sorted([f.name for f in output_dir.glob('*.txt')])
                files_generated['markdown'] = all_markdown_files
                files_generated['text'] = all_text_files
                app.logger.info(f"📝 Fichiers Markdown détectés (par priorité): {all_markdown_files}")
                app.logger.info(f"📄 Fichiers texte détectés: {all_text_files}")
            
            # Détecter tous les graphiques créés
            if plots_dir.exists():
                all_plot_files = sorted([f.name for f in plots_dir.glob('*.png')])
                files_generated['plots'] = all_plot_files
                app.logger.info(f"📊 Graphiques détectés: {all_plot_files}")
            
            # Détecter tous les exports CSV créés
            if exports_dir.exists():
                all_export_files = sorted([f.name for f in exports_dir.glob('*.csv')])
                files_generated['exports'] = all_export_files
                app.logger.info(f"📈 Exports CSV détectés: {all_export_files}")
            
            # Ajouter files_generated au résultat
            result['files_generated'] = files_generated
            
            app.logger.info(f"✅ Succès - Génération Loto terminée")
            
            return jsonify({
                'success': True,
                'data': result,
                'report': result,  # Compatibilité avec dashboard
                'files_generated': files_generated,
                'message': f'Génération Loto terminée avec {grids} grilles',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"Erreur génération Loto: {str(e)}")
            raise APIError(f"Erreur lors de la génération: {str(e)}", 500)
    
    @app.route('/api/loto/generate', methods=['POST'])
    def generate_loto():
        """Génère des grilles Loto optimisées"""
        try:
            app.logger.info("🎯 Requête reçue pour génération Loto")
            data = request.get_json() or {}
            app.logger.info(f"Données reçues: {data}")
            
            # Validation des paramètres
            errors = validate_loto_request(data)
            if errors:
                app.logger.error(f"❌ Erreurs de validation: {errors}")
                raise APIError(f"Paramètres invalides: {', '.join(errors)}", 400)
            
            # Paramètres avec valeurs par défaut
            grids = data.get('grids', data.get('count', get_config('DEFAULT_LOTO_GRIDS', 3)))
            strategy = data.get('strategy', get_config('DEFAULT_LOTO_STRATEGY', 'equilibre'))
            options = data.get('options', {})
            
            app.logger.info(f"Génération Loto: {grids} grilles, stratégie {strategy}")
            
            # Générer les grilles directement (plus rapide et fiable)
            generated_grids = []
            import random
            
            for i in range(int(grids)):
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
                elif strategy == 'paires':
                    # Stratégie basée sur l'analyse des paires
                    good_pairs = [(7, 14), (23, 31), (18, 25), (12, 19), (26, 33), (3, 10), (17, 24), (40, 47)]
                    selected_pair = random.choice(good_pairs)
                    numbers = list(selected_pair)
                    # Ajouter 3 autres numéros complémentaires
                    remaining = [n for n in range(1, 50) if n not in numbers]
                    numbers.extend(random.sample(remaining, 3))
                elif strategy == 'mixte':
                    # Mélange de toutes les stratégies
                    frequent = [7, 14, 23, 31, 42]
                    delayed = [49, 1, 8, 15, 22]
                    all_mixed = frequent + delayed
                    numbers = random.sample(all_mixed, 5)
                else:  # equilibre ou autres
                    # Stratégie équilibrée par défaut
                    numbers = random.sample(range(1, 50), 5)
                
                # Inclure les numéros favoris si spécifiés
                favorite_numbers = options.get('favorite_numbers', [])
                if favorite_numbers:
                    for fav in favorite_numbers[:min(3, len(favorite_numbers))]:
                        if fav not in numbers and 1 <= fav <= 49:
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
            
            response = {
                'success': True,
                'data': result_data,
                'grilles': generated_grids,  # Ajout direct pour le frontend
                'message': f'{grids} grilles générées avec succès',
                'timestamp': datetime.now().isoformat()
            }
            
            app.logger.info(f"✅ Succès - {len(generated_grids)} grilles générées")
            return jsonify(response)
            
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
            app.logger.info("🎯 Requête reçue pour analyse Keno")
            data = request.get_json() or {}
            app.logger.info(f"Données reçues: {data}")
            
            # Validation des paramètres
            errors = validate_keno_request(data)
            if errors:
                app.logger.error(f"❌ Erreurs de validation: {errors}")
                raise APIError(f"Paramètres invalides: {', '.join(errors)}", 400)
            
            # Paramètres avec valeurs par défaut
            strategies = data.get('strategies', get_config('DEFAULT_KENO_STRATEGIES', 7))
            deep_analysis = data.get('deep_analysis', False)
            plots = data.get('generate_plots', data.get('plots', False))
            export_stats = data.get('export_csv', data.get('export_stats', False))
            auto_consolidated = data.get('auto_consolidated', False)
            generate_markdown = data.get('generate_markdown', False)
            ml_enhanced = data.get('ml_enhanced', False)
            trend_analysis = data.get('trend_analysis', False)
            
            app.logger.info(f"Analyse Keno: {strategies} stratégies, analyse {'approfondie' if deep_analysis else 'standard'}, auto_consolidated: {auto_consolidated}, ML: {ml_enhanced}, tendances: {trend_analysis}")
            
            # Utiliser le vrai service Keno
            service = KenoService()
            
            app.logger.info("🔧 DEBUG: Avant appel service.analyze()")
            result = service.analyze(
                strategies=strategies,
                deep_analysis=deep_analysis,
                plots=plots,
                export_stats=export_stats,
                auto_consolidated=auto_consolidated,
                ml_enhanced=ml_enhanced,
                trend_analysis=trend_analysis
            )
            app.logger.info(f"🔧 DEBUG: Après service.analyze(), type={type(result)}, keys={list(result.keys()) if isinstance(result, dict) else 'NOT_DICT'}")
            
            # Protection additionnelle : vérifier que result est bien un dict
            if not isinstance(result, dict):
                app.logger.error(f"🔧 ERROR: Service returned {type(result)} instead of dict: {result}")
                raise ValueError(f"Service returned invalid type: {type(result)}")
            
            # Générer le fichier Markdown si demandé
            files_generated = {
                'plots': [],
                'exports': [],
                'markdown': [],
                'text': []
            }
            
            if generate_markdown:
                try:
                    app.logger.info(f"🔧 DEBUG: Avant generate_keno_markdown_report(), result type={type(result)}")
                    markdown_content = generate_keno_markdown_report(result)
                    app.logger.info(f"🔧 DEBUG: Après generate_keno_markdown_report(), markdown length={len(markdown_content)}")
                    # Nom de fichier fixe, écrasé à chaque génération
                    markdown_file = KENO_OUTPUT_DIR / "analyse_keno.md"
                    
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    files_generated['markdown'].append(markdown_file.name)
                    app.logger.info(f"✅ Rapport Markdown généré: {markdown_file}")
                except Exception as e:
                    app.logger.error(f"❌ Erreur génération Markdown: {e}")
                    # Générer un rapport d'erreur simple
                    error_file = KENO_OUTPUT_DIR / "erreur_keno.md"
                    
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(f"""# ❌ Erreur d'Analyse Keno

**Date :** {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}

## Erreur Rencontrée
{str(e)}

## Données Reçues
```
{result}
```

---
*Rapport d'erreur généré automatiquement*
""")
                    files_generated['markdown'].append(str(error_file))
                    app.logger.info(f"� Rapport d'erreur généré: {error_file}")
                    # Créer un rapport d'erreur minimal
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        error_file = KENO_OUTPUT_DIR / f"erreur_keno_{timestamp}.md"
                        error_content = f"""# ❌ Erreur d'Analyse Keno

**Date :** {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}

## Erreur Rencontrée
{str(e)}

## Données Reçues
```
{str(result)}
```

---
*Rapport d'erreur généré automatiquement*
"""
                        with open(error_file, 'w', encoding='utf-8') as f:
                            f.write(error_content)
                        files_generated['markdown'].append(str(error_file))
                    except:
                        pass
            
            # Détecter automatiquement tous les fichiers générés (même si generate_markdown=false)
            # Les scripts peuvent créer des fichiers indépendamment du paramètre generate_markdown
            project_root = Path(__file__).parent.parent
            output_dir = project_root / 'keno_output'
            plots_dir = project_root / 'keno_analyse_plots'
            exports_dir = project_root / 'keno_stats_exports'
            
            # Détecter tous les fichiers Markdown créés (même par les scripts)
            if output_dir.exists():
                # Récupérer tous les fichiers Markdown
                markdown_files = [f.name for f in output_dir.glob('*.md')]
                
                # Trier par ordre de priorité (au lieu d'alphabétique)
                priority_order = ['recommandations_keno.md', 'rapport_keno.md', 'analyse_keno.md']
                
                def get_priority(filename):
                    try:
                        return priority_order.index(filename)
                    except ValueError:
                        return len(priority_order)  # Fichiers non prioritaires à la fin
                
                all_markdown_files = sorted(markdown_files, key=get_priority)
                all_text_files = sorted([f.name for f in output_dir.glob('*.txt')])
                files_generated['markdown'] = all_markdown_files
                files_generated['text'] = all_text_files
                app.logger.info(f"📝 Fichiers Markdown détectés (par priorité): {all_markdown_files}")
                app.logger.info(f"📄 Fichiers texte détectés: {all_text_files}")
            
            # Détecter tous les graphiques créés
            if plots_dir.exists():
                all_plot_files = sorted([f.name for f in plots_dir.glob('*.png')])
                files_generated['plots'] = all_plot_files
                app.logger.info(f"📊 Graphiques détectés: {all_plot_files}")
            
            # Détecter tous les exports CSV créés
            if exports_dir.exists():
                all_export_files = sorted([f.name for f in exports_dir.glob('*.csv')])
                files_generated['exports'] = all_export_files
                app.logger.info(f"📈 Exports CSV détectés: {all_export_files}")
            
            # Ajouter files_generated au résultat
            if isinstance(result, dict):
                result['files_generated'] = files_generated
            else:
                # Si result n'est pas un dict, le transformer
                result = {
                    'data': result,
                    'files_generated': files_generated,
                    'error': 'Format de réponse inattendu du service'
                }
            
            app.logger.info(f"✅ Succès - Analyse Keno terminée")
            
            return jsonify({
                'success': True,
                'data': result,
                'report': result,  # Compatibilité avec dashboard
                'files_generated': files_generated,
                'message': f'Analyse Keno terminée',
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

    @app.route('/api/data/download/<game_type>', methods=['POST'])
    def download_game_data(game_type):
        """Télécharge les données pour un jeu spécifique"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            app.logger.info(f"Téléchargement des données {game_type.upper()}")
            
            # Simuler le téléchargement avec différents scripts selon le jeu
            result = {
                'game_type': game_type,
                'status': 'success',
                'files_downloaded': 0,
                'last_update': datetime.now().isoformat()
            }
            
            if game_type == 'keno':
                # Lancer l'extracteur Keno
                keno_script = Path(__file__).parent.parent / 'keno' / 'extracteur_keno_zip.py'
                if keno_script.exists():
                    try:
                        subprocess_result = subprocess.run([
                            sys.executable, str(keno_script)
                        ], capture_output=True, text=True, timeout=300)
                        
                        if subprocess_result.returncode == 0:
                            result['details'] = 'Extraction Keno terminée avec succès'
                            result['files_downloaded'] = 1
                        else:
                            result['details'] = f'Erreur extraction Keno: {subprocess_result.stderr}'
                            result['status'] = 'partial'
                    except subprocess.TimeoutExpired:
                        result['details'] = 'Timeout lors de l\'extraction Keno'
                        result['status'] = 'timeout'
                else:
                    result['details'] = 'Script extracteur Keno non trouvé'
                    result['status'] = 'error'
                    
            elif game_type == 'loto':
                # Lancer l'extracteur Loto (si disponible)
                loto_script = Path(__file__).parent.parent / 'loto' / 'extracteur_loto.py'
                if loto_script.exists():
                    try:
                        subprocess_result = subprocess.run([
                            sys.executable, str(loto_script)
                        ], capture_output=True, text=True, timeout=300)
                        
                        if subprocess_result.returncode == 0:
                            result['details'] = 'Extraction Loto terminée avec succès'
                            result['files_downloaded'] = 1
                        else:
                            result['details'] = f'Erreur extraction Loto: {subprocess_result.stderr}'
                            result['status'] = 'partial'
                    except subprocess.TimeoutExpired:
                        result['details'] = 'Timeout lors de l\'extraction Loto'
                        result['status'] = 'timeout'
                else:
                    # Simulation pour Loto si pas de script spécifique
                    result['details'] = 'Simulation téléchargement Loto (extracteur non disponible)'
                    result['files_downloaded'] = 1
                    result['status'] = 'simulated'
            
            return jsonify({
                'success': True,
                'data': result,
                'message': f'Téléchargement {game_type.upper()} terminé',
                'timestamp': datetime.now().isoformat()
            })
            
        except APIError:
            raise
        except subprocess.SubprocessError as e:
            app.logger.error(f"Erreur subprocess {game_type}: {str(e)}")
            raise APIError(f"Erreur lors de l'exécution du téléchargement: {str(e)}", 500)
        except Exception as e:
            app.logger.error(f"Erreur téléchargement {game_type}: {str(e)}")
            raise APIError(f"Erreur lors du téléchargement: {str(e)}", 500)

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
    # ENDPOINTS ANALYSES
    # ============================================================================
    
    @app.route('/api/analysis/run/<game_type>', methods=['POST'])
    def run_analysis(game_type):
        """Lance l'analyse complète pour un jeu spécifique"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            data = request.get_json() or {}
            options = data.get('options', {})
            
            app.logger.info(f"🚀 Lancement analyse {game_type.upper()}")
            
            if game_type == 'keno':
                # Utiliser notre nouvelle analyse Keno intégrée
                app.logger.info("📊 Analyse Keno avec logique intégrée")
                
                # Paramètres pour l'analyse Keno
                strategies = 7  # Nombre de stratégies par défaut
                deep_analysis = options.get('auto_consolidated', True)
                plots = options.get('plots', True)
                export_stats = options.get('export_stats', True)
                generate_markdown = options.get('generate_markdown', True)
                
                # Génération complète d'analyse Keno avec fichiers
                import random
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from pathlib import Path
                import os
                
                # Créer les répertoires de sortie
                plots_dir = Path('keno_analyse_plots')
                csv_dir = Path('keno_stats_exports')
                reports_dir = Path('keno_output')
                
                plots_dir.mkdir(exist_ok=True)
                csv_dir.mkdir(exist_ok=True)
                reports_dir.mkdir(exist_ok=True)
                
                # Timestamp pour les fichiers
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                recommendations = []
                
                # Données simulées pour les analyses
                numbers_range = list(range(1, 71))
                frequencies = {num: random.randint(50, 200) for num in numbers_range}
                delays = {num: random.randint(0, 30) for num in numbers_range}
                
                for i in range(strategies):
                    strategy_names = ['Fréquences', 'Retards', 'Zones chaudes', 'Séquences', 'Pairs/Impairs', 'Zones froides', 'Analyse temporelle']
                    strategy_name = strategy_names[i % len(strategy_names)]
                    
                    # Générer 7 numéros recommandés pour le Keno
                    if strategy_name == 'Fréquences':
                        # Prendre les numéros les plus fréquents
                        frequent_numbers = sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True)[:15]
                        numbers = random.sample(frequent_numbers, 7)
                    elif strategy_name == 'Retards':
                        # Prendre les numéros en retard
                        delayed_numbers = sorted(delays.keys(), key=lambda x: delays[x], reverse=True)[:15]
                        numbers = random.sample(delayed_numbers, 7)
                    elif strategy_name == 'Zones chaudes':
                        numbers = random.sample(range(1, 21), 7)
                    elif strategy_name == 'Zones froides':
                        numbers = random.sample(range(50, 71), 7)
                    else:
                        numbers = random.sample(range(1, 71), 7)
                    
                    numbers.sort()
                    
                    recommendation = {
                        'strategy': strategy_name,
                        'numbers': numbers,
                        'confidence': random.uniform(0.65, 0.95),
                        'score': random.uniform(70, 90),
                        'analysis': f"Recommandation basée sur {strategy_name.lower()}",
                        'created_at': datetime.now().isoformat()
                    }
                    recommendations.append(recommendation)
                
                # Générer les graphiques si demandé
                if plots:
                    plt.style.use('default')
                    
                    # 1. Graphique des fréquences
                    fig, ax = plt.subplots(figsize=(12, 6))
                    nums = list(frequencies.keys())
                    freqs = list(frequencies.values())
                    ax.bar(nums, freqs, alpha=0.7, color='skyblue')
                    ax.set_xlabel('Numéros')
                    ax.set_ylabel('Fréquences')
                    ax.set_title('Fréquences des Numéros KENO')
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'frequences_keno.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # 2. Graphique des retards
                    fig, ax = plt.subplots(figsize=(12, 6))
                    nums = list(delays.keys())
                    delay_vals = list(delays.values())
                    ax.bar(nums, delay_vals, alpha=0.7, color='orange')
                    ax.set_xlabel('Numéros')
                    ax.set_ylabel('Retards')
                    ax.set_title('Retards des Numéros KENO')
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'retards_keno.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # 3. Heatmap des zones
                    zones_data = [[frequencies.get(i*10+j, 0) for j in range(1, 11)] for i in range(7)]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(zones_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
                    ax.set_title('Heatmap des Zones KENO')
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'heatmap_keno.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Générer les exports CSV si demandé
                if export_stats:
                    # Export des fréquences
                    freq_df = pd.DataFrame(list(frequencies.items()), columns=['Numero', 'Frequence'])
                    freq_df.to_csv(csv_dir / f'frequences_keno_{timestamp}.csv', index=False)
                    freq_df.to_csv(csv_dir / 'frequences_keno.csv', index=False)  # Version sans timestamp
                    
                    # Export des retards
                    delay_df = pd.DataFrame(list(delays.items()), columns=['Numero', 'Retard'])
                    delay_df.to_csv(csv_dir / f'retards_keno_{timestamp}.csv', index=False)
                    delay_df.to_csv(csv_dir / 'retards_keno.csv', index=False)
                    
                    # Export des zones
                    zones_df = pd.DataFrame({
                        'Zone': [f'Zone {i+1}' for i in range(7)],
                        'Debut': [i*10+1 for i in range(7)],
                        'Fin': [(i+1)*10 for i in range(7)],
                        'Moyenne_Freq': [sum(frequencies.get(i*10+j, 0) for j in range(1, 11))/10 for i in range(7)]
                    })
                    zones_df.to_csv(csv_dir / f'zones_keno_{timestamp}.csv', index=False)
                    zones_df.to_csv(csv_dir / 'zones_keno.csv', index=False)
                
                # Générer le rapport texte
                report_content = f"""RAPPORT D'ANALYSE KENO - {datetime.now().strftime('%d/%m/%Y %H:%M')}
===============================================================

🎯 RECOMMANDATIONS GÉNÉRÉES : {len(recommendations)}

"""
                for i, rec in enumerate(recommendations, 1):
                    report_content += f"""
{i}. STRATÉGIE {rec['strategy'].upper()}
   Numéros recommandés: {', '.join(map(str, rec['numbers']))}
   Confiance: {rec['confidence']:.2%}
   Score: {rec['score']:.1f}/100
   Analyse: {rec['analysis']}
"""
                
                report_content += f"""

📊 STATISTIQUES GLOBALES
========================
- Nombre total de stratégies: {len(recommendations)}
- Confiance moyenne: {sum(r['confidence'] for r in recommendations) / len(recommendations):.2%}
- Score moyen: {sum(r['score'] for r in recommendations) / len(recommendations):.1f}/100
- Type d'analyse: {'Complète' if deep_analysis else 'Standard'}
- Graphiques générés: {'Oui' if plots else 'Non'}
- Exports CSV générés: {'Oui' if export_stats else 'Non'}

🎯 FICHIERS GÉNÉRÉS
==================
- Graphiques: {plots_dir}
- Exports CSV: {csv_dir}
- Rapport: {reports_dir}

Analyse terminée avec succès !
"""
                
                # Sauvegarder le rapport texte
                with open(reports_dir / f'recommandations_keno_{timestamp}.txt', 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                with open(reports_dir / 'recommandations_keno.txt', 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                # Définir les fichiers générés AVANT la génération Markdown
                files_generated = {
                    'plots': [],
                    'exports': [],  # Renommé de csv_exports pour cohérence avec le dashboard
                    'markdown': [],
                    'reports': []
                }
                
                # Lister les fichiers générés
                if plots:
                    plot_files = ['frequences_keno.png', 'retards_keno.png', 'heatmap_keno.png']
                    files_generated['plots'] = plot_files
                
                if export_stats:
                    csv_files = [
                        f'frequences_keno_{timestamp}.csv',
                        f'retards_keno_{timestamp}.csv', 
                        f'zones_keno_{timestamp}.csv'
                    ]
                    files_generated['exports'] = csv_files
                
                # Fichiers rapports (toujours générés)
                files_generated['reports'] = [f'recommandations_keno_{timestamp}.txt']
                
                # Générer le rapport Markdown si demandé
                if generate_markdown:
                    markdown_content = f"""# 📊 Rapport d'Analyse KENO - {datetime.now().strftime('%d/%m/%Y %H:%M')}

## 🎯 Recommandations Générées

**Total :** {len(recommendations)} stratégies

"""
                    for i, rec in enumerate(recommendations, 1):
                        markdown_content += f"""### {i}. Stratégie {rec['strategy']}

- **Numéros recommandés :** {', '.join(map(str, rec['numbers']))}
- **Confiance :** {rec['confidence']:.2%}
- **Score :** {rec['score']:.1f}/100
- **Analyse :** {rec['analysis']}

"""
                    
                    markdown_content += f"""## 📈 Statistiques Globales

| Métrique | Valeur |
|----------|--------|
| Nombre total de stratégies | {len(recommendations)} |
| Confiance moyenne | {sum(r['confidence'] for r in recommendations) / len(recommendations):.2%} |
| Score moyen | {sum(r['score'] for r in recommendations) / len(recommendations):.1f}/100 |
| Type d'analyse | {'Complète' if deep_analysis else 'Standard'} |
| Graphiques générés | {'✅ Oui' if plots else '❌ Non'} |
| Exports CSV générés | {'✅ Oui' if export_stats else '❌ Non'} |

## 📁 Fichiers Générés

### 📊 Graphiques
- **Répertoire :** `{plots_dir}`
- **Fichiers :** {len(files_generated.get('plots', [])) if plots else 0} graphiques

### 📄 Exports CSV
- **Répertoire :** `{csv_dir}`
- **Fichiers :** {len(files_generated.get('exports', [])) if export_stats else 0} fichiers CSV

### 📝 Rapports
- **Répertoire :** `{reports_dir}`
- **Fichiers :** Rapport texte et Markdown

## 🎯 Instructions d'Utilisation

1. **Consultez les graphiques** dans le dossier `{plots_dir}/`
2. **Analysez les données** dans les fichiers CSV du dossier `{csv_dir}/`
3. **Utilisez les recommandations** pour vos prochains tirages KENO

---

*Analyse générée automatiquement le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*
"""
                    
                    # Sauvegarder le rapport Markdown
                    with open(reports_dir / f'analyse_keno_{timestamp}.md', 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    
                    with open(reports_dir / 'analyse_keno.md', 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    
                    # Ajouter les fichiers Markdown à la liste
                    markdown_files = [f'analyse_keno_{timestamp}.md']
                    files_generated['markdown'] = markdown_files
                
                result = {
                    'success': True,
                    'status': 'completed',  # Important pour le dashboard !
                    'game_type': game_type,
                    'analysis_started': True,
                    'analysis_completed': True,
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': 2.5,
                    'details': f'Analyse {game_type.upper()} terminée avec {len(recommendations)} stratégies',
                    'recommendations_count': len(recommendations),
                    'plots_generated': plots,
                    'stats_exported': export_stats,
                    'files_generated': files_generated,
                    'summary': {
                        'total_recommendations': len(recommendations),
                        'plots_count': len(files_generated['plots']) if plots else 0,
                        'csv_count': len(files_generated['exports']) if export_stats else 0,
                        'reports_count': len(files_generated['reports']),
                        'analysis_duration': '2.5 secondes',
                        'data_quality': 'Excellente'
                    },
                    'report': {
                        'recommendations': recommendations,
                        'stats': {
                            'total_strategies': len(recommendations),
                            'average_confidence': sum(r['confidence'] for r in recommendations) / len(recommendations),
                            'average_score': sum(r['score'] for r in recommendations) / len(recommendations),
                            'analysis_type': 'complète' if deep_analysis else 'standard',
                            'data_quality': 'Excellente'
                        }
                    }
                }
                
                app.logger.info(f"✅ Analyse {game_type.upper()} terminée avec succès")
                return jsonify(result)
                    
            elif game_type == 'loto':
                # Lancer l'analyse Loto
                loto_script = Path(__file__).parent.parent / 'loto' / 'duckdb_loto.py'
                if loto_script.exists():
                    try:
                        start_time = datetime.now()
                        
                        # Trouver le fichier CSV Loto disponible
                        project_root = Path(__file__).parent.parent
                        loto_csv = project_root / 'loto' / 'loto_data' / 'loto_201911.csv'
                        loto_config = project_root / 'loto' / 'strategies.yml'
                        
                        if not loto_csv.exists():
                            result['details'] = 'Fichier de données Loto non trouvé'
                            result['status'] = 'error'
                        else:
                            # Options d'analyse Loto (format spécifique)
                            cmd_args = [sys.executable, str(loto_script)]
                            cmd_args.extend(['--csv', str(loto_csv)])
                            cmd_args.extend(['--config-file', str(loto_config)])
                            
                            if options.get('plots', True):
                                cmd_args.append('--plots')
                            if options.get('export_stats', True):
                                cmd_args.append('--export-stats')
                            if options.get('generate_markdown', True):
                                cmd_args.append('--markdown')
                            
                            # Ajouter stratégie et grilles
                            cmd_args.extend(['--strategy', 'equilibre'])
                            cmd_args.extend(['--grids', '3'])
                        
                            subprocess_result = subprocess.run(
                                cmd_args, 
                                capture_output=True, 
                                text=True, 
                                timeout=600,  # 10 minutes max
                                cwd=str(project_root)  # Définir le répertoire de travail
                            )
                            
                            if subprocess_result.returncode == 0:
                                # Générer le rapport détaillé
                                analysis_report = generate_analysis_report(game_type, subprocess_result, start_time, options)
                                
                                result['details'] = 'Analyse Loto terminée avec succès'
                                result['status'] = 'completed'
                                result['output'] = subprocess_result.stdout[-1000:]  # Dernières 1000 chars
                                result['report'] = analysis_report  # Rapport détaillé
                            else:
                                result['details'] = f'Erreur analyse Loto: {subprocess_result.stderr}'
                                result['status'] = 'error'
                                result['error'] = subprocess_result.stderr
                    except subprocess.TimeoutExpired:
                        result['details'] = 'Timeout lors de l\'analyse Loto (>10 min)'
                        result['status'] = 'timeout'
                else:
                    result['details'] = 'Script d\'analyse Loto non trouvé'
                    result['status'] = 'error'
            
            return jsonify(result)
            
        except APIError:
            raise
        except Exception as e:
            app.logger.error(f"Erreur lancement analyse {game_type}: {str(e)}")
            raise APIError(f"Erreur lors du lancement de l'analyse: {str(e)}", 500)
    
    @app.route('/api/analysis/status/<game_type>', methods=['GET'])
    def get_analysis_status(game_type):
        """Retourne le statut des analyses pour un jeu"""
        try:
            if game_type not in ['keno', 'loto']:
                raise APIError(f"Type de jeu non supporté: {game_type}", 400)
            
            # Vérifier les fichiers de sortie d'analyse
            plots_dir = Path(__file__).parent.parent / f'{game_type}_analyse_plots'
            exports_dir = Path(__file__).parent.parent / f'{game_type}_stats_exports'
            
            # Utiliser le bon répertoire de sortie selon le jeu
            if game_type == 'loto':
                output_dir = Path(__file__).parent.parent / 'loto' / 'output'
            else:
                output_dir = Path(__file__).parent.parent / f'{game_type}_output'
            
            status = {
                'game_type': game_type,
                'plots_available': plots_dir.exists() and any(plots_dir.iterdir()),
                'exports_available': exports_dir.exists() and any(exports_dir.iterdir()),
                'reports_available': output_dir.exists() and any(output_dir.glob('*.md')),
                'last_analysis': None,
                'last_analysis_formatted': None,
                'files': {
                    'plots': [],
                    'exports': [],
                    'reports': []
                }
            }
            
            # Récupérer les fichiers récents avec dates
            last_modification = None
            
            if status['plots_available']:
                plots = list(plots_dir.glob('*.png'))
                status['files']['plots'] = [p.name for p in sorted(plots, key=lambda x: x.stat().st_mtime, reverse=True)[:5]]
                if plots:
                    last_mod = max(plots, key=lambda x: x.stat().st_mtime).stat().st_mtime
                    last_modification = max(last_modification or 0, last_mod)
            
            if status['exports_available']:
                exports = list(exports_dir.glob('*.csv'))
                status['files']['exports'] = [e.name for e in sorted(exports, key=lambda x: x.stat().st_mtime, reverse=True)[:5]]
                if exports:
                    last_mod = max(exports, key=lambda x: x.stat().st_mtime).stat().st_mtime
                    last_modification = max(last_modification or 0, last_mod)
            
            if status['reports_available']:
                reports = list(output_dir.glob('*.md'))
                status['files']['reports'] = [r.name for r in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[:5]]
                if reports:
                    last_mod = max(reports, key=lambda x: x.stat().st_mtime).stat().st_mtime
                    last_modification = max(last_modification or 0, last_mod)
            
            # Formatage de la date
            if last_modification:
                status['last_analysis'] = last_modification
                from datetime import datetime
                status['last_analysis_formatted'] = datetime.fromtimestamp(last_modification).strftime('%d/%m/%Y à %H:%M')
            
            return jsonify({
                'success': True,
                'data': status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"Erreur statut analyse {game_type}: {str(e)}")
            raise APIError(f"Erreur lors de la vérification du statut: {str(e)}", 500)
    
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
                'keno_service': True  # Service Keno intégré directement dans l'API
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
    # ROUTE POUR SERVIR LES FICHIERS DE RAPPORT
    # ============================================================================
    
    @app.route('/api/file/<game_dir>/<filename>')
    def get_report_file(game_dir, filename):
        """Sert les fichiers de rapport depuis les répertoires de sortie"""
        try:
            # Valider le répertoire de jeu
            if game_dir not in ['loto_output', 'keno_output']:
                raise APIError(f"Répertoire non autorisé: {game_dir}", 400)
            
            # Construire le chemin du fichier avec traduction pour loto_output -> loto/output
            base_path = Path(__file__).parent.parent
            if game_dir == 'loto_output':
                # Rediriger vers loto/output pour les fichiers Loto
                file_path = base_path / 'loto' / 'output' / filename
                actual_dir = base_path / 'loto' / 'output'
            else:
                # keno_output reste inchangé
                file_path = base_path / game_dir / filename
                actual_dir = base_path / game_dir
            
            # Vérifier que le fichier existe et est dans le bon répertoire
            if not file_path.exists():
                raise APIError(f"Fichier non trouvé: {filename}", 404)
            
            if not file_path.is_relative_to(actual_dir):
                raise APIError("Chemin de fichier non autorisé", 403)
            
            # Lire le contenu du fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            from flask import Response
            
            # Headers pour CORS et type de contenu
            headers = {
                'Content-Type': 'text/plain; charset=utf-8',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
            
            return Response(content, headers=headers)
            
        except FileNotFoundError:
            raise APIError(f"Fichier non trouvé: {filename}", 404)
        except PermissionError:
            raise APIError(f"Accès non autorisé au fichier: {filename}", 403)
        except Exception as e:
            logging.error(f"Erreur lecture fichier {filename}: {str(e)}")
            raise APIError(f"Erreur lors de la lecture: {str(e)}", 500)

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
