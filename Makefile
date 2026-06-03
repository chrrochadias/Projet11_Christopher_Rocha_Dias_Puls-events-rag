# ============================================================
# Puls-Events RAG — Makefile
# ============================================================
# Entrypoint canonique du projet. Lance `make help` pour la liste.
# ============================================================

# Variables
PYTHON := python
PIP := pip
VENV := venv
SRC_DIR := src
TESTS_DIR := tests
SCRIPTS_DIR := scripts
DATA_DIR := data

# Cible par défaut : afficher l'aide
.DEFAULT_GOAL := help

# ============================================================
# Configuration de l'environnement
# ============================================================

.PHONY: install
install: ## Installer les dépendances dans le venv courant
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dépendances installées"

.PHONY: venv
venv: ## Créer un environnement virtuel Python
	$(PYTHON) -m venv $(VENV)
	@echo "✅ venv créé. Active-le avec : source $(VENV)/bin/activate"

.PHONY: check-env
check-env: ## Vérifier que les variables .env sont bien définies
	@$(PYTHON) -c "from src.config import MISTRAL_API_KEY, OPENAGENDA_PUBLIC_KEY; print('✅ Variables .env OK')"

# ============================================================
# Pipeline de données
# ============================================================

.PHONY: ingest
ingest: ## Ingérer les événements depuis OpenAgenda (sans rebuild de l'index)
	$(PYTHON) -m $(SRC_DIR).data_ingestion

.PHONY: build
build: ## Construire la base vectorielle FAISS (build incrémental)
	$(PYTHON) $(SCRIPTS_DIR)/build_index.py

.PHONY: rebuild
rebuild: ## Reconstruire complètement la base (refetch OpenAgenda + index)
	$(PYTHON) $(SCRIPTS_DIR)/build_index.py --refresh

# ============================================================
# Tests
# ============================================================

.PHONY: test
test: ## Lancer TOUS les tests (unitaires + intégration)
	pytest $(TESTS_DIR)/ -v

.PHONY: test-unit
test-unit: ## Lancer uniquement les tests unitaires (rapides, sans FAISS)
	pytest $(TESTS_DIR)/test_preprocessing.py -v

.PHONY: test-integration
test-integration: ## Lancer les tests d'intégration (nécessite l'index FAISS construit)
	pytest $(TESTS_DIR)/test_freshness.py $(TESTS_DIR)/test_geography.py -v

.PHONY: test-coverage
test-coverage: ## Lancer les tests avec rapport de couverture
	pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=term-missing

# ============================================================
# Démonstration et évaluation
# ============================================================

.PHONY: demo
demo: ## Lancer une démonstration interactive du RAG
	$(PYTHON) -m $(SRC_DIR).rag_chain

.PHONY: ask
ask: ## Poser une question au RAG : make ask Q="ta question ici"
	@$(PYTHON) -m $(SRC_DIR).rag_chain "$(Q)"

.PHONY: evaluate
evaluate: ## Lancer l'évaluation RAGAS sur le jeu Q/R annoté (10-15 min)
	$(PYTHON) $(SCRIPTS_DIR)/run_evaluation.py

.PHONY: annotate
annotate: ## Assistant interactif d'annotation des Q/R
	$(PYTHON) $(SCRIPTS_DIR)/annotation_helper.py

# ============================================================
# Pipeline complet (entrée-sortie)
# ============================================================

.PHONY: pipeline
pipeline: rebuild test demo ## Pipeline complet : rebuild → tests → démo (entrée-sortie)
	@echo ""
	@echo "✅ Pipeline complet exécuté avec succès"

.PHONY: ci
ci: test ## Cible CI : tests automatisés (à brancher sur GitHub Actions)
	@echo "✅ CI OK"

# ============================================================
# Maintenance
# ============================================================

.PHONY: clean
clean: ## Nettoyer les fichiers temporaires Python
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Nettoyage terminé"

.PHONY: clean-index
clean-index: ## Supprimer l'index FAISS (force la reconstruction au prochain build)
	rm -f $(DATA_DIR)/faiss_index/index.faiss $(DATA_DIR)/faiss_index/index.pkl
	@echo "✅ Index FAISS supprimé"

.PHONY: clean-all
clean-all: clean clean-index ## Nettoyage complet (cache Python + index FAISS)

# ============================================================
# Aide
# ============================================================

.PHONY: help
help: ## Afficher cette aide
	@echo ""
	@echo "Puls-Events RAG — Commandes disponibles"
	@echo "========================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Exemples :"
	@echo "  make install         # Installer les dépendances"
	@echo "  make build           # Construire la base vectorielle"
	@echo "  make test            # Lancer tous les tests"
	@echo "  make ask Q=\"...\"     # Poser une question au RAG"
	@echo "  make evaluate        # Évaluer le RAG via RAGAS"
	@echo ""