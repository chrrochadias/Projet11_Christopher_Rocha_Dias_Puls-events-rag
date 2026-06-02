"""
Configuration centralisée du POC Puls-Events RAG.

Single source of truth pour :
- les variables d'environnement (secrets)
- les constantes métier (agenda cible, fraîcheur, modèles)
- les chemins du projet
"""
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

# Chargement automatique du .env, peu importe d'où le script est lancé
load_dotenv(find_dotenv(usecwd=True))

# ============================================================
# Secrets (depuis .env)
# ============================================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAGENDA_PUBLIC_KEY = os.getenv("OPENAGENDA_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY manquante dans .env")
if not OPENAGENDA_PUBLIC_KEY:
    raise ValueError("OPENAGENDA_PUBLIC_KEY manquante dans .env")

# ============================================================
# Chemins du projet (relatifs à la racine du repo)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "faiss_index"

# ============================================================
# Paramètres OpenAgenda
# ============================================================
OPENAGENDA_BASE_URL = "https://api.openagenda.com/v2"
AGENDA_UID = 82290100              # Diocèse de Paris
MAX_EVENT_AGE_DAYS = 365           # Brief Jérémy : événements < 1 an
TARGET_CITY = "Paris"              # Pour les tests géographiques

# ============================================================
# Paramètres de préprocessing
# ============================================================
MAX_DESCRIPTION_CHARS = 4000       # Troncature des outliers

# ============================================================
# Paramètres modèles Mistral
# ============================================================
MISTRAL_EMBED_MODEL = "mistral-embed"     # Modèle d'embedding
MISTRAL_LLM_MODEL = "mistral-small-latest"  # Modèle générateur (POC)
EMBEDDING_DIM = 1024                       # Dimension de mistral-embed