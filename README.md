# Puls-Events RAG — POC

> Proof of Concept d'un système de **Retrieval-Augmented Generation** pour la recommandation d'événements culturels, basé sur LangChain, Mistral et FAISS.

---

## 🎯 Contexte

Puls-Events est une plateforme web de découverte d'événements culturels en temps réel. Elle agrège des sources publiques comme [OpenAgenda](https://openagenda.com/) pour proposer aux utilisateurs des événements adaptés à leurs préférences, filtrables par lieu et par période.

Ce POC démontre la faisabilité d'un **chatbot intelligent** capable de fournir des recommandations personnalisées à partir d'un corpus d'événements, en s'appuyant sur :

- **LangChain** — framework d'orchestration des composants LLM
- **Mistral AI** — modèle d'embedding (`mistral-embed`) + modèle générateur (`mistral-small-latest`)
- **FAISS** — base vectorielle locale pour la recherche sémantique
- **RAGAS** — framework d'évaluation des systèmes RAG

---

## 🎯 Objectifs

Ce POC répond à trois objectifs structurants :

1. **Démontrer la faisabilité technique** d'un système RAG complet (retrieval + augmentation + génération) sur des données réelles d'événements culturels publics.
2. **Valider la qualité métier** des réponses générées via une démarche d'évaluation rigoureuse (jeu de test annoté + métriques RAGAS).
3. **Identifier les limites architecturales** du système naïf et proposer des recommandations chiffrées pour la version production.

Le POC est conçu pour être :
- **Reproductible** : `git clone` + `.env` + 3 commandes → système opérationnel
- **Agnostique de la source** : changer la valeur `AGENDA_UID` permet de cibler n'importe quel agenda OpenAgenda
- **Testable** : tests unitaires couvrant le préprocessing et le contrat de données (fraîcheur < 1 an + périmètre géographique)

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  OpenAgenda  │────▶│ Préprocessing│────▶│   Embeddings │
│  (source)    │     │ + Filtres    │     │  (Mistral)   │
└──────────────┘     │ qualité/géo  │     └──────┬───────┘
                     └──────────────┘            │
                                                 ▼
                                          ┌──────────────┐
                                          │ FAISS Index  │
                                          │   (local)    │
                                          └──────┬───────┘
                                                 │
   Question utilisateur ──▶ Embedding ──▶ Top-K retrieval
                                                 │
                                                 ▼
                                       Prompt augmenté + Mistral
                                                 ▼
                                          Réponse + sources
```

---

## 📂 Structure du projet

```
puls-events-rag/
├── src/                            # Modules réutilisables
│   ├── config.py                   # Configuration centralisée (.env, constantes)
│   ├── data_ingestion.py           # Fetch OpenAgenda + filtrage qualité
│   ├── preprocessing.py            # Composition texte embeddable + métadonnées
│   ├── vector_store.py             # Gestion index FAISS (build/load/cache)
│   ├── rag_chain.py                # Chaîne RAG complète (LCEL)
│   └── evaluation.py               # Pipeline d'évaluation RAGAS
├── scripts/                        # Points d'entrée CLI
│   ├── build_index.py              # Reconstruit la base vectorielle
│   ├── run_evaluation.py           # Lance l'évaluation RAGAS
│   └── annotation_helper.py        # Assistant d'annotation des Q/R
├── tests/                          # Tests automatisés (pytest)
│   ├── test_preprocessing.py       # Tests unitaires fonctions pures
│   ├── test_freshness.py           # Test fraîcheur des données (< 1 an)
│   ├── test_geography.py           # Test périmètre géographique (Paris)
│   └── conftest.py                 # Configuration pytest
├── data/
│   ├── raw/                        # Événements OpenAgenda bruts (JSON)
│   ├── processed/                  # Données nettoyées
│   ├── faiss_index/                # Index vectoriel sérialisé (généré)
│   └── evaluation/                 # Jeu Q/R annoté + rapports RAGAS
├── notebooks/
│   └── 01_exploration_openagenda.ipynb   # EDA initiale
├── docs/                           # Rapport technique + présentation
├── requirements.txt                # Dépendances Python (pip)
├── .env.example                    # Template variables d'environnement
├── .gitignore
└── README.md
```

---

## 🚀 Installation et reproduction

### Prérequis

- **Python 3.10+** (testé sur Python 3.12)
- Une **clé API Mistral** ([console.mistral.ai](https://console.mistral.ai/api-keys/))
- Une **clé publique OpenAgenda** ([openagenda.com](https://openagenda.com/))

### Installation pas à pas

```bash
# 1. Cloner le dépôt
git clone chrrochadias/Projet11_Christopher_Rocha_Dias_Puls-events-rag
cd Projet11_Christopher_Rocha_Dias_Puls-events-rag

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurer les secrets
cp .env.example .env
# Éditer .env et renseigner :
#   - MISTRAL_API_KEY=...
#   - OPENAGENDA_PUBLIC_KEY=...

# 5. Vérifier l'installation
python -c "import langchain, faiss; from langchain_mistralai import ChatMistralAI; print('Stack OK')"
```

### Pipeline complet de mise en service

```bash
# Étape 1 : Ingérer les événements + reconstruire l'index vectoriel
python scripts/build_index.py

# Étape 2 : Tester le système en posant une question
python -m src.rag_chain "Y a-t-il des conférences sur la spiritualité ?"

# Étape 3 : Lancer les tests automatisés
pytest tests/ -v

# Étape 4 (optionnel) : Lancer l'évaluation RAGAS
python scripts/run_evaluation.py
```

---

## 🔧 Utilisation détaillée

### Reconstruire la base vectorielle

```bash
# Build avec cache (réutilise data/raw/)
python scripts/build_index.py

# Build complet avec re-fetch OpenAgenda
python scripts/build_index.py --refresh

# Limiter le nombre d'événements
python scripts/build_index.py --n 100
```

### Interroger le RAG

```bash
# Question par défaut
python -m src.rag_chain

# Question personnalisée
python -m src.rag_chain "Quels concerts de musique classique sont prévus ?"
```

### Lancer les tests

```bash
# Tous les tests
pytest tests/ -v

# Tests unitaires uniquement (rapides)
pytest tests/test_preprocessing.py -v

# Tests de contrat de données (nécessite l'index construit)
pytest tests/test_freshness.py tests/test_geography.py -v
```

### Évaluer la qualité du RAG

```bash
# Évaluation complète avec RAGAS (10-15 min)
python scripts/run_evaluation.py

# Avec top-K personnalisé
python scripts/run_evaluation.py --k 6
```

---

## 📊 Configuration du POC

| Paramètre | Valeur | Localisation |
|-----------|--------|--------------|
| Agenda source | Diocèse de Paris (uid=82290100) | `src/config.py` |
| Périmètre géographique | Paris (75xxx) | `src/config.py` |
| Fraîcheur des données | < 365 jours | `src/config.py` |
| Modèle d'embedding | `mistral-embed` (1024 dims) | `src/config.py` |
| Modèle générateur | `mistral-small-latest` (température 0) | `src/config.py` |
| Vector store | FAISS IndexFlatL2 (local) | `src/vector_store.py` |
| Top-K retrieval | 4 | `src/rag_chain.py` |
| Chunking | Aucun (1 événement = 1 document) | `src/preprocessing.py` |
| Troncature max | 4000 caractères | `src/config.py` |

---

## 🧪 Évaluation

Le système est évalué via [RAGAS](https://docs.ragas.io/) sur **15 questions annotées** couvrant 7 catégories : `thematic_search`, `format_specific`, `audience_specific`, `location_specific`, `specific_event`, `out_of_scope`, `ambiguous`.

**4 métriques** sont mesurées :
- **Faithfulness** : non-hallucination
- **Answer Relevancy** : pertinence de la réponse à la question
- **Context Precision** : qualité du retrieval (les bons docs en haut)
- **Context Recall** : couverture du retrieval

Les résultats détaillés sont disponibles dans `docs/rapport_technique.docx`.

---

## 📝 Livrables

- [x] Environnement reproductible (`requirements.txt`, `.env.example`)
- [x] Pipeline d'ingestion + préprocessing (`src/`, `scripts/build_index.py`)
- [x] Système RAG complet (`src/rag_chain.py`)
- [x] Tests automatisés (`tests/` avec 34 tests)
- [x] Évaluation RAGAS (`src/evaluation.py`, jeu Q/R annoté)
- [x] Rapport technique (`docs/rapport_technique.docx`)
- [x] Présentation (`docs/presentation.pptx`)

---

## 🛠️ Stack technique

| Couche | Outil | Version |
|--------|-------|---------|
| Langage | Python | 3.10+ |
| Framework LLM | LangChain | 0.3+ |
| LLM & embeddings | Mistral AI | API |
| Vector store | FAISS (CPU) | 1.8+ |
| Évaluation | RAGAS | 0.2 |
| Tests | pytest | 8.0+ |
| Gestion deps | pip + venv | stdlib |

---

## 👤 Auteur

Christopher Rocha Dias — Formation Data Engineering OpenClassrooms — Projet 11

---

## 📄 Licence

Ce projet est un livrable pédagogique dans le cadre de la formation OpenClassrooms.