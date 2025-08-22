#!/usr/bin/env python3
"""
Service de Gestion des Fichiers
===============================

Service pour gérer le téléchargement des fichiers générés,
les visualisations et l'analyse des stratégies.

Author: Système Loto/Keno API
Date: 22 août 2025
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import mimetypes

class FileService:
    """Service de gestion des fichiers générés"""
    
    def __init__(self, base_path: str = None):
        """
        Initialise le service de fichiers
        
        Args:
            base_path: Chemin de base du projet
        """
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent.parent
        
        # Définition des répertoires de sortie
        self.output_dirs = {
            'keno_csv': self.base_path / 'keno_stats_exports',
            'keno_plots': self.base_path / 'keno_analyse_plots', 
            'keno_output': self.base_path / 'keno_output',
            'loto_csv': self.base_path / 'loto_stats_exports',
            'loto_plots': self.base_path / 'loto_analyse_plots',
            'loto_output': self.base_path / 'output'
        }
        
        # Types de fichiers supportés
        self.supported_extensions = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.md': 'text/markdown',
            '.txt': 'text/plain'
        }
    
    def get_available_files(self, game_type: str = None) -> Dict[str, List[Dict]]:
        """
        Récupère la liste des fichiers disponibles
        
        Args:
            game_type: 'keno', 'loto' ou None pour tous
            
        Returns:
            Dict contenant les fichiers par catégorie
        """
        result = {}
        
        # Filtrer les répertoires selon le type de jeu
        dirs_to_scan = self.output_dirs.copy()
        if game_type == 'keno':
            dirs_to_scan = {k: v for k, v in dirs_to_scan.items() if 'keno' in k}
        elif game_type == 'loto':
            dirs_to_scan = {k: v for k, v in dirs_to_scan.items() if 'loto' in k}
        
        for dir_type, dir_path in dirs_to_scan.items():
            if dir_path.exists():
                files = []
                for file_path in dir_path.iterdir():
                    if file_path.is_file() and file_path.suffix in self.supported_extensions:
                        file_info = self._get_file_info(file_path)
                        files.append(file_info)
                
                # Trier par date de modification (plus récent en premier)
                files.sort(key=lambda x: x['modified'], reverse=True)
                result[dir_type] = files
            else:
                result[dir_type] = []
        
        return result
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Récupère les informations d'un fichier
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            Dict avec les informations du fichier
        """
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'path': str(file_path.relative_to(self.base_path)),
            'size': stat.st_size,
            'size_human': self._format_file_size(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': file_path.suffix,
            'mime_type': self.supported_extensions.get(file_path.suffix, 'application/octet-stream'),
            'category': self._get_file_category(file_path)
        }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Formate la taille de fichier en format lisible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _get_file_category(self, file_path: Path) -> str:
        """Détermine la catégorie d'un fichier"""
        name = file_path.name.lower()
        suffix = file_path.suffix.lower()
        
        if suffix in ['.png', '.jpg', '.jpeg']:
            return 'visualization'
        elif suffix == '.csv':
            return 'data'
        elif suffix == '.json':
            return 'analysis'
        elif suffix in ['.md', '.txt']:
            return 'report'
        else:
            return 'other'
    
    def get_file_content(self, file_path: str) -> Tuple[bytes, str, str]:
        """
        Récupère le contenu d'un fichier
        
        Args:
            file_path: Chemin relatif vers le fichier
            
        Returns:
            Tuple (contenu, mime_type, nom_fichier)
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        if not self._is_safe_path(full_path):
            raise PermissionError(f"Accès non autorisé au fichier: {file_path}")
        
        mime_type = self.supported_extensions.get(full_path.suffix, 'application/octet-stream')
        
        with open(full_path, 'rb') as f:
            content = f.read()
        
        return content, mime_type, full_path.name
    
    def _is_safe_path(self, file_path: Path) -> bool:
        """Vérifie que le chemin est sécurisé (dans les répertoires autorisés)"""
        try:
            file_path.resolve().relative_to(self.base_path.resolve())
            return True
        except ValueError:
            return False
    
    def analyze_strategies(self, game_type: str) -> Dict[str, Any]:
        """
        Analyse les stratégies disponibles et recommande la meilleure
        
        Args:
            game_type: 'keno' ou 'loto'
            
        Returns:
            Dict avec l'analyse des stratégies
        """
        if game_type == 'keno':
            return self._analyze_keno_strategies()
        elif game_type == 'loto':
            return self._analyze_loto_strategies()
        else:
            raise ValueError(f"Type de jeu non supporté: {game_type}")
    
    def _analyze_keno_strategies(self) -> Dict[str, Any]:
        """Analyse les stratégies Keno"""
        strategies = {}
        
        # Analyser le rapport complet JSON s'il existe
        report_path = self.output_dirs['keno_csv'] / 'rapport_complet.json'
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    strategies['from_report'] = report_data
            except Exception as e:
                print(f"Erreur lecture rapport: {e}")
        
        # Analyser les fichiers CSV pour extraire les performances
        csv_files = {
            'frequences': self.output_dirs['keno_csv'] / 'frequences_keno.csv',
            'retards': self.output_dirs['keno_csv'] / 'retards_keno.csv',
            'paires': self.output_dirs['keno_csv'] / 'paires_keno.csv',
            'zones': self.output_dirs['keno_csv'] / 'zones_keno.csv'
        }
        
        for strategy_name, csv_path in csv_files.items():
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    strategies[strategy_name] = self._evaluate_strategy_performance(df, strategy_name)
                except Exception as e:
                    print(f"Erreur analyse {strategy_name}: {e}")
        
        # Déterminer la meilleure stratégie
        best_strategy = self._determine_best_strategy(strategies)
        
        return {
            'strategies': strategies,
            'best_strategy': best_strategy,
            'analysis_date': datetime.now().isoformat(),
            'total_strategies': len(strategies)
        }
    
    def _analyze_loto_strategies(self) -> Dict[str, Any]:
        """Analyse les stratégies Loto"""
        # Implementation similaire pour Loto
        strategies = {}
        
        # Analyser les fichiers CSV Loto s'ils existent
        csv_files = {
            'frequences': self.output_dirs['loto_csv'] / 'frequences_loto.csv',
            'retards': self.output_dirs['loto_csv'] / 'retards_loto.csv',
            'sommes': self.output_dirs['loto_csv'] / 'sommes_loto.csv'
        }
        
        for strategy_name, csv_path in csv_files.items():
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    strategies[strategy_name] = self._evaluate_strategy_performance(df, strategy_name)
                except Exception as e:
                    print(f"Erreur analyse Loto {strategy_name}: {e}")
        
        # Déterminer la meilleure stratégie ou utiliser la stratégie par défaut
        if strategies:
            best_strategy = self._determine_best_strategy(strategies)
        else:
            best_strategy = {
                'name': 'equilibre',
                'score': 75.0,
                'reason': 'Stratégie par défaut - équilibre des numéros',
                'metrics': {}
            }
        
        return {
            'strategies': strategies,
            'best_strategy': best_strategy,
            'analysis_date': datetime.now().isoformat(),
            'total_strategies': len(strategies),
            'note': 'Analyse Loto avec stratégies disponibles'
        }
    
    def _evaluate_strategy_performance(self, df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        """Évalue la performance d'une stratégie basée sur les données CSV"""
        performance = {
            'name': strategy_name,
            'data_points': len(df),
            'score': 0.0,
            'metrics': {}
        }
        
        if strategy_name == 'frequences':
            # Pour les fréquences, rechercher les numéros les plus fréquents
            if 'frequence' in df.columns:
                performance['metrics']['avg_frequency'] = float(df['frequence'].mean())
                performance['metrics']['max_frequency'] = float(df['frequence'].max())
                performance['score'] = float(df['frequence'].mean() * 10)
        
        elif strategy_name == 'retards':
            # Pour les retards, rechercher les patterns
            if 'retard' in df.columns:
                performance['metrics']['avg_delay'] = float(df['retard'].mean())
                performance['metrics']['max_delay'] = float(df['retard'].max())
                # Score inversé pour les retards (moins c'est mieux)
                performance['score'] = float(max(0, 100 - df['retard'].mean()))
        
        elif strategy_name == 'paires':
            # Pour les paires, analyser la force des associations
            if len(df.columns) >= 2:
                performance['metrics']['pair_count'] = len(df)
                performance['score'] = float(len(df) * 2)
        
        elif strategy_name == 'zones':
            # Pour les zones, analyser l'équilibre
            if 'zone' in df.columns:
                zone_counts = df['zone'].value_counts()
                performance['metrics']['zone_balance'] = float(zone_counts.std())
                performance['score'] = float(100 - zone_counts.std())
        
        return performance
    
    def _determine_best_strategy(self, strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Détermine la meilleure stratégie basée sur les scores"""
        if not strategies:
            return {'name': 'none', 'score': 0, 'reason': 'Aucune stratégie analysée'}
        
        best_score = 0
        best_strategy_name = 'equilibre'
        best_metrics = {}
        
        for strategy_name, strategy_data in strategies.items():
            if isinstance(strategy_data, dict) and 'score' in strategy_data:
                if strategy_data['score'] > best_score:
                    best_score = strategy_data['score']
                    best_strategy_name = strategy_name
                    best_metrics = strategy_data.get('metrics', {})
        
        return {
            'name': best_strategy_name,
            'score': float(best_score),
            'metrics': best_metrics,
            'reason': f'Meilleur score de performance: {best_score:.2f}'
        }
    
    def get_strategy_recommendations(self, game_type: str) -> Dict[str, Any]:
        """
        Génère des recommandations basées sur l'analyse des stratégies
        
        Args:
            game_type: 'keno' ou 'loto'
            
        Returns:
            Dict avec les recommandations
        """
        analysis = self.analyze_strategies(game_type)
        best_strategy = analysis.get('best_strategy', {})
        
        recommendations = {
            'primary_strategy': best_strategy.get('name', 'equilibre'),
            'confidence': min(100, best_strategy.get('score', 0)),
            'recommendations': [],
            'analysis_summary': analysis
        }
        
        # Générer des recommandations spécifiques
        if best_strategy.get('name') == 'frequences':
            recommendations['recommendations'].append({
                'type': 'strategy',
                'message': 'Privilégier les numéros fréquents pour cette session'
            })
        elif best_strategy.get('name') == 'retards':
            recommendations['recommendations'].append({
                'type': 'strategy', 
                'message': 'Focus sur les numéros en retard pour un retour à la moyenne'
            })
        elif best_strategy.get('name') == 'paires':
            recommendations['recommendations'].append({
                'type': 'strategy',
                'message': 'Utiliser les paires historiquement performantes'
            })
        else:
            recommendations['recommendations'].append({
                'type': 'strategy',
                'message': 'Approche équilibrée recommandée'
            })
        
        return recommendations
