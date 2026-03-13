# ============================================================
# Makefile — Système Loto (Bayésien + XGBoost + PuLP)
# ============================================================
# Usage rapide :
#   make help          Affiche cette aide
#   make menu          Lance le menu interactif
#   make bayesian      Génère 6 grilles (modèle en cache)
#   make retrain       Régénère avec ré-entraînement XGBoost
#   make loto          Génère 3 grilles (générateur classique)
#   make data          Télécharge les dernières données FDJ
#   make test          Lance les tests
#   make clean         Supprime les fichiers temporaires
#   make install       Installe les dépendances Python
# ============================================================

PYTHON  := $(shell [ -f venv/bin/python ] && echo venv/bin/python || echo python3)
CSV     := loto/loto_data/loto_201911.csv
BAYESIAN:= loto/generateur_bayesien_loto.py
ADVANCED:= loto/loto_generator_advanced_Version2.py
CLASSIC := loto/duckdb_loto.py
STRATEGIES := loto/strategies.yml

.PHONY: help menu bayesian retrain loto loto-fast loto-full loto-plots \
        top25 data test install clean clean-all status

# ── Aide ────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  SYSTEME LOTO — Cibles disponibles"
	@echo "  ──────────────────────────────────────────────"
	@echo "  menu          Menu interactif complet"
	@echo ""
	@echo "  [Générateur Bayésien — recommandé]"
	@echo "  bayesian      6 grilles optimisées (modèle en cache)"
	@echo "  retrain       6 grilles + ré-entraînement XGBoost"
	@echo ""
	@echo "  [Générateur classique]"
	@echo "  loto          3 grilles (stratégie équilibrée)"
	@echo "  loto-fast     3 grilles rapides"
	@echo "  loto-full     5 grilles avec stats exportées"
	@echo "  loto-plots    3 grilles + visualisations"
	@echo ""
	@echo "  [Générateur avancé ML]"
	@echo "  advanced-quick    1 000 simulations"
	@echo "  advanced          10 000 simulations"
	@echo "  advanced-heavy    50 000 simulations"
	@echo ""
	@echo "  [Données & maintenance]"
	@echo "  top25         TOP 25 numéros équilibrés"
	@echo "  data          Télécharge données FDJ"
	@echo "  test          Lance les tests"
	@echo "  status        Vérifie l'environnement"
	@echo "  install       Installe les dépendances"
	@echo "  clean         Supprime fichiers temporaires"
	@echo "  clean-all     Supprime aussi les modèles et exports"
	@echo ""

# ── Menu interactif ─────────────────────────────────────────
menu:
	$(PYTHON) cli_menu.py

# ── Générateur Bayésien ─────────────────────────────────────
bayesian: $(CSV)
	$(PYTHON) $(BAYESIAN)

retrain: $(CSV)
	$(PYTHON) $(BAYESIAN) --retrain

# ── Générateur classique ─────────────────────────────────────
loto: $(CSV) $(STRATEGIES)
	$(PYTHON) $(CLASSIC) --csv $(CSV) --grids 3 --config $(STRATEGIES)

loto-fast: $(CSV) $(STRATEGIES)
	$(PYTHON) $(CLASSIC) --csv $(CSV) --grids 3 --config $(STRATEGIES)

loto-full: $(CSV) $(STRATEGIES)
	$(PYTHON) $(CLASSIC) --csv $(CSV) --grids 5 --export-stats --config $(STRATEGIES)

loto-plots: $(CSV) $(STRATEGIES)
	$(PYTHON) $(CLASSIC) --csv $(CSV) --grids 3 --plots --export-stats --config $(STRATEGIES)

# ── Générateur avancé ML ─────────────────────────────────────
advanced-quick: $(CSV)
	$(PYTHON) $(ADVANCED) --quick --silent

advanced: $(CSV)
	$(PYTHON) $(ADVANCED) --silent

advanced-heavy: $(CSV)
	$(PYTHON) $(ADVANCED) --intensive

# ── TOP 25 ───────────────────────────────────────────────────
top25: $(CSV) $(STRATEGIES)
	$(PYTHON) $(CLASSIC) --csv $(CSV) --export-stats --config-file $(STRATEGIES)

# ── Données ──────────────────────────────────────────────────
data:
	$(PYTHON) loto/result.py

# ── Tests ────────────────────────────────────────────────────
test:
	@echo "Lancement des tests..."
	$(PYTHON) -m pytest test/ -v 2>/dev/null || $(PYTHON) test/run_all_tests.py

# ── Statut ───────────────────────────────────────────────────
status:
	@echo ""
	@echo "  Environnement"
	@echo "  ─────────────────────────────────────────"
	@$(PYTHON) --version
	@echo "  Python : $(PYTHON)"
	@[ -f $(CSV) ] && echo "  CSV    : OK  ($(CSV))" || echo "  CSV    : MANQUANT ($(CSV))"
	@[ -f boost_models/xgb_bayesian_loto.json ] && \
		echo "  Modele : OK  (boost_models/xgb_bayesian_loto.json)" || \
		echo "  Modele : non entraine (lancez 'make retrain')"
	@[ -f loto_output/bayesian_grilles.json ] && \
		echo "  Output : OK  (loto_output/bayesian_grilles.json)" || \
		echo "  Output : aucune grille generee"
	@echo ""
	@$(PYTHON) -c "import duckdb, xgboost, pulp, sklearn, scipy; \
		print('  Deps   : duckdb=' + duckdb.__version__ + \
		      '  xgboost=' + xgboost.__version__ + \
		      '  pulp=' + pulp.__version__)" 2>/dev/null || \
		echo "  Deps   : certaines dependances manquantes (make install)"
	@echo ""

# ── Installation ─────────────────────────────────────────────
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ── Nettoyage ────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -maxdepth 1 -name "temp_*.py" -delete
	rm -f loto/data/*.parquet
	@echo "Nettoyage termine."

clean-all: clean
	rm -f boost_models/xgb_bayesian_loto.json
	rm -f boost_models/scaler_bayesian_loto.joblib
	rm -f loto_output/bayesian_grilles.json
	rm -rf loto_analyse_plots/*.png
	rm -rf loto_stats_exports/*.csv
	@echo "Nettoyage complet termine."

# ── Raccourcis utiles ────────────────────────────────────────
.DEFAULT_GOAL := help
