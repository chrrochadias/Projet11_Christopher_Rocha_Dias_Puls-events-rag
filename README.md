# Puls-Events RAG — POC

> Proof of Concept d'un système de **Retrieval-Augmented Generation** pour la recommandation d'événements culturels, basé sur LangChain, Mistral et FAISS.

---

## 🎯 Contexte

Puls-Events est une plateforme de découverte d'événements culturels en temps réel. Ce POC démontre la faisabilité d'un chatbot intelligent capable de fournir des recommandations personnalisées à partir d'un corpus d'événements issus d'**OpenAgenda**, en s'appuyant sur :

- **LangChain** — orchestration des composants LLM
- **Mistral AI** — modèle génératif (via API)
- **FAISS** — base vectorielle locale pour la recherche sémantique

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  OpenAgenda  │────▶│ Préprocessing│────▶│   Embeddings │
│  (source)    │     │ + Chunking   │     │  (Mistral)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ FAISS Index  │
                                          │   (local)    │
                                          └──────┬───────┘
                                                 │
   Question utilisateur ──▶ Embedding ──▶ Top-K retrieval
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │   Mistral    │
                                          │  (génération)│
                                          └──────┬───────┘
                                                 ▼
                                          Réponse augmentée
```

---

## 🚀 Installation

### Prérequis

- **Python 3.10+**
- Une **clé API Mistral** ([console.mistral.ai](https://console.mistral.ai/api-keys/))

### Étapes

```bash
# 1. Cloner le repo
git clone <url-du-repo>
cd puls-events-rag

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurer les secrets
cp .env.example .env
# Éditer .env et remplir MISTRAL_API_KEY

# 5. Vérifier l'installation
python -c "import langchain, faiss; from langchain_mistralai import ChatMistralAI; print('OK')"
```

---

## 📂 Structure du projet

```
puls-events-rag/
├── src/                        # Modules réutilisables
│   ├── config.py               # Constantes (région, dates, modèles)
│   ├── data_ingestion.py       # Récupération OpenAgenda
│   ├── preprocessing.py        # Nettoyage + chunking
│   ├── vector_store.py         # Gestion FAISS
│   ├── rag_chain.py            # La chaîne LangChain
│   └── evaluation.py           # Métriques d'évaluation
├── scripts/                    # Points d'entrée exécutables
│   ├── build_index.py          # Reconstruit la base vectorielle
│   └── run_demo.py             # Lance la démo live
├── tests/                      # Tests unitaires (pytest)
│   ├── test_freshness.py       # Événements < 1 an
│   └── test_geography.py       # Périmètre géographique
├── data/
│   ├── raw/                    # Données OpenAgenda brutes
│   ├── processed/              # Données nettoyées
│   └── faiss_index/            # Index vectoriel sérialisé
├── notebooks/                  # Exploration
├── docs/                       # Rapport + présentation
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Utilisation

### Reconstruire la base vectorielle

```bash
python scripts/build_index.py
```

Ce script télécharge les événements OpenAgenda du périmètre configuré, applique le préprocessing, génère les embeddings via Mistral et sauvegarde l'index FAISS sur disque.

### Lancer la démo

```bash
python scripts/run_demo.py
```

### Lancer les tests

```bash
pytest tests/ -v
```

---

## 📊 Périmètre du POC

| Paramètre | Valeur |
|-----------|--------|
| Région géographique | _(à définir)_ |
| Fraîcheur des données | < 365 jours |
| Source | OpenAgenda |
| Volume estimé | _(à mesurer)_ |

---

## 📐 Choix techniques

_(Section à compléter au fil du build — sera la base du rapport technique.)_

| Composant | Choix | Justification |
|-----------|-------|---------------|
| Vector store | FAISS (IndexFlatL2) | Local, gratuit, parfait pour POC |
| Embeddings | Mistral Embed | Cohérence avec le LLM, qualité francophone |
| LLM | Mistral (API) | Imposé par le cahier des charges, qualité FR |
| Framework | LangChain | Orchestration standard du marché |
| Chunking | _(à définir)_ | _(à justifier)_ |

---

## 🧪 Évaluation

Un jeu de données de questions/réponses annotées est utilisé pour mesurer la qualité du système. Voir `src/evaluation.py` et `docs/rapport_technique.docx`.

---

## 📝 Livrables

- [x] Environnement reproductible (`requirements.txt`, `.env.example`)
- [ ] Pipeline RAG complet (`src/` + `scripts/`)
- [ ] Tests unitaires (`tests/`)
- [ ] Rapport technique (`docs/rapport_technique.docx`)
- [ ] Présentation (`docs/presentation.pptx`)
- [ ] Démo live (`scripts/run_demo.py`)
- [ ] Jeu de Q/R annoté pour évaluation

---

## 👤 Auteur

Chris — Formation Data Engineering OpenClassrooms — Projet 11
