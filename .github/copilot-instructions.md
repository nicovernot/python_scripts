# Loto/Keno Analysis System
Loto/Keno Analysis System is a Python-based statistical analysis and prediction system for French lottery games (Loto and Keno). It combines machine learning, statistical analysis, and visualization capabilities to analyze historical draw data and generate optimized game recommendations.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the information here.

## Working Effectively

### Bootstrap, Build and Test the Repository
- Install Python dependencies:
  - `pip install -r requirements.txt` -- NEVER CANCEL: Takes 45-90 seconds to complete. Set timeout to 180+ seconds.
  - Dependencies include: pandas, numpy, duckdb, matplotlib, seaborn, scikit-learn, xgboost, flask, beautifulsoup4
- Run essential tests to validate setup:
  - `python test/run_all_tests.py --essential` -- Takes 3-5 seconds. NEVER CANCEL. Set timeout to 30+ seconds.
- Run all tests to check system health:
  - `python test/run_all_tests.py` -- Takes 3-4 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
- Create required data directories:
  - `mkdir -p loto/loto_data keno/keno_data`

### Configuration Setup
- Copy environment configuration:
  - `cp .env.example .env` (if needed)
- Run interactive configuration (optional):
  - `python setup_config.py` -- Interactive setup takes 30-60 seconds. NEVER CANCEL.
- Validate configuration:
  - `python -c "from config_env import load_config, print_config_summary; load_config(); print_config_summary()"`

### Data Download (Network-Dependent)
- Download Loto historical data:
  - `python loto/result.py` -- NEVER CANCEL: Takes 60-300 seconds depending on network. Set timeout to 600+ seconds.
- Download Keno historical data:
  - `python keno/extracteur_keno_zip.py` -- NEVER CANCEL: Takes 120-600 seconds depending on network. Set timeout to 900+ seconds.
- Note: Downloads will fail in network-restricted environments. This is expected and the system can still be tested without live data.

### Run the Applications

#### CLI Menu Interface
- Interactive menu system:
  - `python cli_menu.py` -- Interactive interface, use Ctrl+C to exit
  - Alternative: `./lancer_menu.sh` (includes environment checks)

#### Web API Server
- Start Flask web server:
  - `python api/app.py` -- Runs on http://localhost:5000. NEVER CANCEL during testing.
  - Alternative: `./start_server.sh` (with error handling)
- API endpoints available at:
  - GET `/api/health` - Server health check
  - GET `/api/data/status` - Data availability status
  - POST `/api/loto/generate` - Generate Loto grids
  - POST `/api/keno/analyze` - Keno analysis

#### Individual Analysis Scripts
- Loto analysis:
  - `python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3 --plots --export-stats` -- NEVER CANCEL: Takes 10-60 seconds. Set timeout to 120+ seconds.
- Keno analysis:
  - `python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv --plots --export-stats` -- NEVER CANCEL: Takes 15-90 seconds. Set timeout to 180+ seconds.

## Validation

### Test Scenarios
- ALWAYS run essential tests before making changes:
  - `python test/run_all_tests.py --essential` (3-5 seconds)
- ALWAYS run fast tests after making changes:
  - `python test/run_all_tests.py --fast` (3-5 seconds, excludes performance tests)
- For comprehensive validation before commits:
  - `python test/run_all_tests.py` (3-4 seconds total)
- Test individual components:
  - `python test/test_loto.py --verbose` (for Loto-specific issues)
  - `python test/test_keno.py --verbose` (for Keno-specific issues)
  - `python test/test_performance.py` (10-15 seconds for performance metrics)

### Manual Validation Steps
- **Import Performance**: Validate core modules load properly:
  - `python -c "import pandas, numpy, duckdb, matplotlib, seaborn, sklearn, xgboost; print('All dependencies OK')"` (1-2 seconds)
- **Module Loading**: Test system modules:
  - `python -c "from loto import duckdb_loto; from keno import duckdb_keno; print('Modules loaded')"` (1-2 seconds)
- **Configuration**: Test environment setup:
  - `python config_env.py` (1-2 seconds) - should output configuration summary
- **Script Help**: Test script accessibility:
  - `python loto/duckdb_loto.py --help` (shows usage and options)
  - `python keno/duckdb_keno.py --help` (shows usage and options)
- **API Health**: Test web API if running:
  - Start: `python api/app.py` (runs on http://localhost:5000)
  - Test: `curl http://localhost:5000/api/health` (returns JSON health status)
  - Expected: `{"success": false, "data": {"status": "unhealthy"}}` (unhealthy due to missing data files)

### Expected Performance Metrics
- Module import times (measured):
  - pandas: ~0.3s, numpy: ~0.0s, duckdb: ~0.02s
  - matplotlib: ~0.25s, seaborn: ~0.5s, sklearn: ~0.06s
  - Total core imports: ~1.1s
  - Loto module: ~1.2s, Keno module: ~0.0s
- Memory usage: +0.3MB after all imports (reasonable)
- Test suite runtime: 3-4 seconds total

## Common Tasks

### Development Workflow
```bash
# 1. Setup and validate
python test/run_all_tests.py --essential
# 2. Make changes
# 3. Test changes
python test/run_all_tests.py --fast
# 4. Full validation before commit
python test/run_all_tests.py
```

### Debugging Issues
- Enable debug mode for Loto:
  ```bash
  python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 1 --debug --log-level DEBUG
  ```
- Enable debug mode for Keno:
  ```bash
  python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv --debug --verbose --log-file keno_debug.log
  ```
- Check system diagnostics:
  ```bash
  python tools/generate_diagnostic_report.py --include-logs --include-data-samples  # if available
  ```

### Performance Optimization
- Reduce memory usage:
  ```bash
  python loto/duckdb_loto.py --csv file.csv --grids 3 --no-ml
  ```
- Limit CPU usage:
  ```bash
  python loto/duckdb_loto.py --csv file.csv --grids 3 --threads 1
  ```
- Clean up disk space:
  ```bash
  python tools/cleanup.py --remove-cache --remove-old-exports  # if available
  ```

## Key System Components

### File Structure Reference
```
python_scripts/
├── loto/                    # Loto analysis system
│   ├── duckdb_loto.py      # Main Loto analyzer
│   ├── result.py           # Data downloader
│   ├── strategies.py       # Strategy definitions
│   ├── strategies.yml      # Configuration
│   └── loto_data/          # Downloaded data
├── keno/                    # Keno analysis system
│   ├── duckdb_keno.py      # Main Keno analyzer
│   ├── extracteur_keno_zip.py  # Data downloader
│   └── keno_data/          # Downloaded data
├── test/                    # Test suite
│   ├── run_all_tests.py    # Main test runner
│   ├── test_essential.py   # Dependency tests
│   ├── test_loto.py        # Loto tests
│   ├── test_keno.py        # Keno tests
│   └── test_performance.py # Performance tests
├── api/                     # Web API
│   ├── app.py              # Flask server
│   └── services/           # API services
├── cli_menu.py             # Interactive CLI
├── config_env.py           # Environment config
├── setup_config.py         # Setup wizard
└── requirements.txt        # Dependencies
```

### Configuration Files
- `.env` - Environment variables (copy from `.env.example`)
- `loto/strategies.yml` - Loto analysis strategies
- `requirements.txt` - Python dependencies

### Common Command Reference
```bash
# Quick system check
python test/run_all_tests.py --essential  # 3-5s

# Full development test
python test/run_all_tests.py --fast       # 3-5s

# Performance benchmark
python test/run_all_tests.py              # 3-4s

# Start interactive menu
python cli_menu.py

# Start web server
python api/app.py

# Download data (network required)
python loto/result.py                     # 60-300s
python keno/extracteur_keno_zip.py        # 120-600s

# Generate analysis
python loto/duckdb_loto.py --csv loto/loto_data/loto_201911.csv --grids 3  # 10-60s
python keno/duckdb_keno.py --csv keno/keno_data/keno_202010.csv            # 15-90s
```

### Timeout Guidelines
- **Package installation**: 180+ seconds
- **Essential tests**: 30+ seconds
- **Full tests**: 60+ seconds
- **Data downloads**: 600-900+ seconds
- **Analysis runs**: 120-180+ seconds
- **Module imports**: 5+ seconds
- **API operations**: 30+ seconds

### Known Limitations
- Network downloads fail in restricted environments (expected)
- Some tests require CSV data to pass completely
- ML features require adequate memory (2GB+ recommended)
- Performance varies with dataset size
- Interactive menus require manual input

### Troubleshooting
- **Missing dependencies**: Run `pip install -r requirements.txt` (45-90 seconds)
- **Import errors**: Check Python path with `python -c "import sys; print(sys.path)"`
- **Missing CSV files**: Run data download scripts or check network connectivity
  - Note: Downloads will fail in network-restricted environments (expected)
- **Permission errors**: Run `chmod +x *.py keno/*.py loto/*.py test/*.py`
- **Memory issues**: Use `--no-ml` flag or reduce batch sizes
- **Performance issues**: Use `--threads 1` to limit CPU usage
- **API not responding**: Check if port 5000 is available with `netstat -tan | grep 5000`
- **Test failures**: Ensure dependencies are installed and data directories exist