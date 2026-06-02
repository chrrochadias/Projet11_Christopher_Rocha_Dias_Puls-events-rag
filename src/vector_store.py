"""
Gestion de la base vectorielle FAISS pour le POC Puls-Events RAG.

Fonctions :
- build_documents()         : convertit les events en Document LangChain
- build_vector_store()      : crée un nouvel index FAISS
- load_vector_store()       : charge un index existant
- get_or_build_vector_store(): pattern cache-or-build
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

from src.config import (
    MISTRAL_API_KEY,
    MISTRAL_EMBED_MODEL,
    INDEX_DIR,
    RAW_DIR,
    AGENDA_UID,
)
from src.preprocessing import build_embeddable_text, extract_metadata


def build_documents(events: list[dict]) -> list[Document]:
    """
    Convertit une liste d'événements OpenAgenda en Document LangChain.

    Chaque Document contient :
    - page_content : le texte qui sera embeddé (title + description)
    - metadata     : les champs structurés (uid, dates, localisation)

    Args:
        events: liste de dicts événements OpenAgenda

    Returns:
        Liste de Document LangChain prêts à être indexés
    """
    documents = []
    for event in events:
        text = build_embeddable_text(event)
        if not text.strip():
            continue  # Sécurité supplémentaire, ne devrait pas arriver après filtrage
        metadata = extract_metadata(event)
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_vector_store(
    events: list[dict],
    save_to_disk: bool = True,
) -> FAISS:
    """
    Construit un index FAISS depuis une liste d'événements.

    Le processus :
    1. Convertit chaque event en Document LangChain
    2. Embed tous les page_content via mistral-embed (batché automatiquement)
    3. Stocke vecteurs + métadonnées dans un index FAISS local

    Args:
        events: liste d'événements OpenAgenda (déjà filtrés)
        save_to_disk: si True, sérialise l'index dans INDEX_DIR

    Returns:
        Instance FAISS prête à être interrogée
    """
    print(f"🔨 Construction de l'index vectoriel pour {len(events)} événements...")

    documents = build_documents(events)
    print(f"   → {len(documents)} documents préparés")

    embeddings = MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=MISTRAL_API_KEY,
    )

    print(f"   → Vectorisation via {MISTRAL_EMBED_MODEL}...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"   ✅ Index FAISS construit ({vectorstore.index.ntotal} vecteurs)")

    if save_to_disk:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        print(f"   💾 Index sauvegardé : {INDEX_DIR}")

    return vectorstore


def load_vector_store() -> FAISS:
    """
    Charge un index FAISS existant depuis disque.

    Note de sécurité : `allow_dangerous_deserialization=True` est nécessaire
    car FAISS sérialise les métadonnées en pickle. C'est acceptable ici
    car nous chargeons un index que nous avons nous-mêmes produit.

    Returns:
        Instance FAISS chargée depuis disque

    Raises:
        FileNotFoundError: si aucun index n'existe à INDEX_DIR
    """
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"Aucun index FAISS trouvé à {INDEX_DIR}. "
            f"Lance d'abord scripts/build_index.py"
        )

    embeddings = MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=MISTRAL_API_KEY,
    )
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"📂 Index FAISS chargé depuis {INDEX_DIR} ({vectorstore.index.ntotal} vecteurs)")
    return vectorstore


def get_or_build_vector_store(force_rebuild: bool = False) -> FAISS:
    """
    Pattern cache-or-build : charge l'index s'il existe, sinon le reconstruit.

    Args:
        force_rebuild: si True, ignore le cache et reconstruit l'index

    Returns:
        Instance FAISS prête à interroger
    """
    if not force_rebuild and (INDEX_DIR / "index.faiss").exists():
        return load_vector_store()

    # Pas d'index → on charge les events depuis data/raw/ et on construit
    raw_path = RAW_DIR / f"agenda_{AGENDA_UID}_events.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Pas d'événements bruts à {raw_path}. "
            f"Lance d'abord `python -m src.data_ingestion`"
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    return build_vector_store(events, save_to_disk=True)