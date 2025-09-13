import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, mock_open
import tempfile
import json
from io import StringIO
from loto.loto_generator_advanced_Version2 import create_report, AdaptiveStrategy, N_SIMULATIONS, GLOBAL_SEED, N_CORES

class TestCreateReport:
    
    @pytest.fixture
    def sample_criteria(self):
        """Sample criteria data for testing"""
        return {
            'hot_numbers': [1, 5, 12, 23, 34, 15, 6, 3, 24, 30],
            'last_draw': [18, 49, 14, 24, 17],
            'pair_counts': pd.Series({
                (7, 11): 17,
                (8, 46): 16,
                (6, 28): 16,
                (12, 26): 16,
                (26, 42): 16
            }),
            'numbers_to_exclude': {3, 4, 6, 7, 10, 12, 13, 14, 15, 17, 18, 21, 22, 24, 26, 28, 30, 31, 33, 38, 42, 43, 49},
            'dynamic_weights': {
                'sum': np.float64(0.0),
                'std': np.float64(0.015),
                'pair_impair': np.float64(0.217),
                'decades_entropy': np.float64(0.096),
                'gaps': np.float64(0.029),
                'high_low': np.float64(0.211),
                'prime_ratio': np.float64(0.212),
                'position_variance': np.float64(0.22)
            },
            'freq': pd.Series(index=range(1, 50), data=np.random.randint(80, 120, 49))
        }
    
    @pytest.fixture
    def sample_best_grids(self):
        """Sample best grids data for testing"""
        return [
            {'grid': [9, 20, 23, 40, 44, 4], 'score': 0.8072},
            {'grid': [8, 20, 23, 40, 41, 1], 'score': 0.8068},
            {'grid': [2, 20, 25, 32, 41, 6], 'score': 0.8050},
            {'grid': [2, 20, 25, 32, 41, 5], 'score': 0.8050},
            {'grid': [5, 20, 25, 32, 44, 2], 'score': 0.8020}
        ]
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_adaptive_strategy_with_history(self):
        """Mock adaptive strategy with performance history"""
        strategy = Mock(spec=AdaptiveStrategy)
        strategy.performance_history = {
            'ml_scores': [0.75, 0.82, 0.78, 0.85, 0.79],
            'freq_scores': [0.65, 0.70, 0.68, 0.72, 0.69],
            'dates': ['2023-01-01T10:00:00', '2023-01-02T10:00:00', '2023-01-03T10:00:00']
        }
        strategy.ml_weight = 0.6
        return strategy
    
    @pytest.fixture
    def mock_adaptive_strategy_without_history(self):
        """Mock adaptive strategy without performance history"""
        strategy = Mock(spec=AdaptiveStrategy)
        strategy.performance_history = {'ml_scores': []}
        strategy.ml_weight = 0.6
        return strategy
    
    @pytest.fixture
    def sample_ml_top25_csv(self, temp_output_dir):
        """Create a sample ML top 25 CSV file"""
        ml_data = []
        for i in range(25):
            ml_data.append({
                'numero': i + 1,
                'score_composite': 0.95 - i * 0.02,
                'score_ml': 0.0354 - i * 0.001,
                'rang_ml': i + 1
            })
        
        ml_df = pd.DataFrame(ml_data)
        ml_csv_path = temp_output_dir / 'ml_top25_predictions.csv'
        ml_df.to_csv(ml_csv_path, index=False)
        return ml_csv_path
    
    @patch('loto.loto_generator_advanced_Version2.datetime')
    def test_create_report_basic_functionality(self, mock_datetime, sample_criteria, 
                                             sample_best_grids, temp_output_dir):
        """Test basic report creation functionality"""
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "13/09/2025 18:28"
        
        # Call the function
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 637.07)
        
        # Check if report file was created
        report_path = temp_output_dir / 'rapport_analyse.md'
        assert report_path.exists()
        
        # Read and verify content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify key sections are present
        assert "# Rapport d'Analyse Loto - 13/09/2025 18:28" in content
        assert "⚙️ Configuration d'Exécution" in content
        assert "🎯 Top 5 Grilles Recommandées" in content
        assert "637.07 secondes" in content
        assert f"{N_SIMULATIONS:,}" in content or f"{N_SIMULATIONS}" in content
        assert str(GLOBAL_SEED) in content
        assert str(N_CORES) in content
    
    def test_create_report_with_grids_formatting(self, sample_criteria, 
                                               sample_best_grids, temp_output_dir):
        """Test that grids are properly formatted in the report"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check grid formatting
        assert "**Grille 1**: `[9, 20, 23, 40, 44]` + Chance **4** (Score: 0.8072)" in content
        assert "**Grille 2**: `[8, 20, 23, 40, 41]` + Chance **1** (Score: 0.8068)" in content
        assert "**Grille 5**: `[5, 20, 25, 32, 44]` + Chance **2** (Score: 0.8020)" in content
    
    def test_create_report_with_adaptive_strategy_with_history(self, sample_criteria, 
                                                             sample_best_grids, temp_output_dir,
                                                             mock_adaptive_strategy_with_history):
        """Test report creation with adaptive strategy containing history"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0, 
                     mock_adaptive_strategy_with_history)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check adaptive strategy section
        assert "🤖 Stratégie Adaptative Machine Learning" in content
        assert "**Poids ML actuel**: `0.600` (60.0%)" in content
        assert "**Poids Fréquence actuel**: `0.400` (40.0%)" in content
        assert "**Précision ML récente**:" in content
        assert "**Évaluations effectuées**: `5`" in content
        assert "2023-01-03T10:00:00" in content
    
    def test_create_report_with_adaptive_strategy_without_history(self, sample_criteria, 
                                                                sample_best_grids, temp_output_dir,
                                                                mock_adaptive_strategy_without_history):
        """Test report creation with adaptive strategy without history"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0, 
                     mock_adaptive_strategy_without_history)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check adaptive strategy section for first use
        assert "🤖 Stratégie Adaptative Machine Learning" in content
        assert "**Poids ML initial**: `0.600` (60.0%)" in content
        assert "**État**: Première utilisation - collecte des données de performance en cours" in content
    
    def test_create_report_without_adaptive_strategy(self, sample_criteria, 
                                                   sample_best_grids, temp_output_dir):
        """Test report creation without adaptive strategy"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0, None)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should not contain adaptive strategy section
        assert "🤖 Stratégie Adaptative Machine Learning" not in content
    
    def test_create_report_with_ml_top25_file(self, sample_criteria, sample_best_grids, 
                                            temp_output_dir, sample_ml_top25_csv):
        """Test report creation when ML top 25 CSV file exists"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain ML section with top numbers
        assert "🤖 Top 10 des Numéros selon l'Intelligence Artificielle" in content
        assert "**1** (Score: 0.95" in content  # First number from sample data
        assert "📊 Fichier complet des 25 meilleurs disponible : `ml_top25_predictions.csv`" in content
        assert "📋 Analyse détaillée disponible : `ml_predictions_summary.md`" in content
    
    def test_create_report_without_ml_top25_file(self, sample_criteria, sample_best_grids, 
                                               temp_output_dir):
        """Test report creation when ML top 25 CSV file doesn't exist"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain ML section indicating non-availability
        assert "🤖 Prédictions ML" in content
        assert "*Non disponible - modèles ML non chargés*" in content
    
    def test_create_report_with_empty_grids(self, sample_criteria, temp_output_dir):
        """Test report creation with empty grids list"""
        empty_grids = []
        
        create_report(sample_criteria, empty_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        assert report_path.exists()
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should still contain basic structure but no grids
        assert "🎯 Top 5 Grilles Recommandées" in content
        assert "**Grille 1**:" not in content
    
    def test_create_report_with_empty_pair_counts(self, sample_criteria, 
                                                sample_best_grids, temp_output_dir):
        """Test report creation with empty pair counts"""
        criteria_copy = sample_criteria.copy()
        criteria_copy['pair_counts'] = pd.Series(dtype='int64')
        
        create_report(criteria_copy, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should show N/A for pairs
        assert "🔗 Top 5 Paires Fréquentes" in content
        assert "N/A" in content
    
    def test_create_report_excluded_numbers_formatting(self, sample_criteria, 
                                                     sample_best_grids, temp_output_dir):
        """Test that excluded numbers are properly formatted"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check excluded numbers formatting
        expected_excluded = "3, 4, 6, 7, 10, 12, 13, 14, 15, 17, 18, 21, 22, 24, 26, 28, 30, 31, 33, 38, 42, 43, 49"
        assert expected_excluded in content
    
    def test_create_report_statistical_summary(self, sample_criteria, 
                                             sample_best_grids, temp_output_dir):
        """Test statistical summary section"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check statistical summary
        assert "📊 Résumé de l'Analyse Statistique" in content
        assert "**Dernier tirage**: `18, 49, 14, 24, 17`" in content
        assert "**Numéros Hot (Top 10)**: `1, 5, 12, 23, 34, 15, 6, 3, 24, 30`" in content
        assert "**Poids de scoring dynamiques**:" in content
    
    def test_create_report_files_section(self, sample_criteria, 
                                       sample_best_grids, temp_output_dir):
        """Test that files section lists all expected files"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check files section
        assert "📈 Fichiers Générés" in content
        assert "numbers_analysis.csv" in content
        assert "grilles_conseillees.csv" in content
        assert "ml_top25_predictions.csv" in content
        assert "ml_full_predictions.csv" in content
        assert "ml_predictions_summary.md" in content
        assert "frequence_numeros.png" in content
        assert "autocorrelation_plot.png" in content
        assert "matrix_profile_plot.png" in content
    
    def test_create_report_visualizations_section(self, sample_criteria, 
                                                sample_best_grids, temp_output_dir):
        """Test visualizations section"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check visualizations section
        assert "📈 Visualisations Clés" in content
        assert "![Fréquence des numéros](frequence_numeros.png)" in content
        assert "![Autocorrélation des tirages](autocorrelation_plot.png)" in content
        assert "![Matrix Profile](matrix_profile_plot.png)" in content
    
    def test_create_report_encoding_handling(self, sample_criteria, 
                                           sample_best_grids, temp_output_dir):
        """Test that the report handles UTF-8 encoding correctly"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        
        # Read with explicit UTF-8 encoding
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for special characters that should be properly encoded
        assert "⚙️" in content
        assert "🎯" in content
        assert "🤖" in content
        assert "📊" in content
        assert "🔗" in content
        assert "📈" in content
        assert "🚫" in content
    
    @patch('builtins.print')
    def test_create_report_print_output(self, mock_print, sample_criteria, 
                                      sample_best_grids, temp_output_dir):
        """Test that the function prints expected messages"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        # Check that print was called with expected message
        mock_print.assert_any_call("Génération du rapport Markdown...")
        
        # Check final print message
        expected_path = temp_output_dir / 'rapport_analyse.md'
        mock_print.assert_any_call(f"   ✓ Rapport sauvegardé dans '{expected_path}'.")
    
    def test_create_report_edge_case_missing_keys(self, temp_output_dir):
        """Test report creation with minimal criteria (missing some keys)"""
        minimal_criteria = {
            'hot_numbers': [1, 2, 3, 15, 6, 3, 24, 30, 7, 22],
            'last_draw': [5, 10, 15, 20, 25],
            'dynamic_weights': {'sum': 0.5, 'std': 0.5},
            'numbers_to_exclude': {1, 2, 3},
            'pair_counts': pd.Series(dtype='int64')
        }
        
        minimal_grids = [
            {'grid': [1, 2, 3, 4, 5, 6], 'score': 0.5}
        ]
        
        # Should not raise exception
        create_report(minimal_criteria, minimal_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        assert report_path.exists()
    
    def test_create_report_with_corrupted_ml_csv(self, sample_criteria, sample_best_grids, 
                                               temp_output_dir):
        """Test report creation when ML CSV exists but is corrupted"""
        # Create a corrupted ML CSV file
        ml_csv_path = temp_output_dir / 'ml_top25_predictions.csv'
        with open(ml_csv_path, 'w') as f:
            f.write("corrupted,csv,content\nthis,is,not,valid")
        
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain error message about loading
        assert "🤖 Prédictions ML" in content
        assert "*Erreur lors du chargement:" in content
    
    def test_create_report_dynamic_weights_formatting(self, sample_criteria, 
                                                    sample_best_grids, temp_output_dir):
        """Test that dynamic weights are properly formatted with correct precision"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that weights are rounded to 3 decimal places
        assert "'sum': 0.0" in content
        assert "'std': 0.015" in content
        assert "'pair_impair': 0.217" in content
    
    def test_create_report_large_numbers_formatting(self, sample_criteria, 
                                                  sample_best_grids, temp_output_dir):
        """Test proper formatting of large numbers with thousand separators"""
        # Mock large simulation count
        with patch('loto.loto_generator_advanced_Version2.N_SIMULATIONS', 1000000):
            create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
            
            report_path = temp_output_dir / 'rapport_analyse.md'
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should contain formatted number
            assert "1,000,000" in content or "1000000" in content
    
    def test_create_report_execution_time_formatting(self, sample_criteria, 
                                                   sample_best_grids, temp_output_dir):
        """Test execution time formatting with different values"""
        test_times = [0.123, 12.456, 123.789, 1234.567]
        
        for exec_time in test_times:
            create_report(sample_criteria, sample_best_grids, temp_output_dir, exec_time)
            
            report_path = temp_output_dir / 'rapport_analyse.md'
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should contain formatted execution time
            expected_time = f"{exec_time:.2f} secondes"
            assert expected_time in content
    
    def test_create_report_file_permissions(self, sample_criteria, sample_best_grids):
        """Test that the report file is created with proper permissions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output_dir = Path(temp_dir)
            
            create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
            
            report_path = temp_output_dir / 'rapport_analyse.md'
            assert report_path.exists()
            assert report_path.is_file()
            assert report_path.stat().st_size > 0  # File should not be empty
    
    def test_create_report_content_structure(self, sample_criteria, sample_best_grids, 
                                          temp_output_dir):
        """Test that the report has the correct markdown structure"""
        create_report(sample_criteria, sample_best_grids, temp_output_dir, 100.0)
        
        report_path = temp_output_dir / 'rapport_analyse.md'
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check for main header
        assert any(line.startswith("# Rapport d'Analyse Loto") for line in lines)
        
        # Check for section headers
        expected_sections = [
            "## ⚙️ Configuration d'Exécution",
            "## 🎯 Top 5 Grilles Recommandées", 
            "## 🚫 Numéros Exclu",
            "## 📊 Résumé de l'Analyse Statistique",
            "## 📈 Fichiers Générés",
            "## 📈 Visualisations Clés"
        ]
        
        for section in expected_sections:
            assert any(section in line for line in lines), f"Section '{section}' not found"