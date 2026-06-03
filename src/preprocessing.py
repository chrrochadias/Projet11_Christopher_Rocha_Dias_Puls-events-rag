"""
Préprocessing des événements OpenAgenda pour le pipeline RAG.

Fonctions :
- extract_text() : helper pour les champs multilingues OpenAgenda
- is_event_embeddable() : filtre de qualité
- build_embeddable_text() : composition du texte à embedder
- extract_metadata() : extraction des métadonnées de filtrage
"""
from typing import Optional
from src.config import MAX_DESCRIPTION_CHARS


def extract_text(field) -> str:
    """
    Extrait le texte FR d'un champ multilingue OpenAgenda.

    OpenAgenda retourne ses textes sous forme de dicts :
        {"fr": "...", "en": "...", "de": "..."}

    On prend systématiquement la version FR (notre périmètre est parisien).
    """
    if isinstance(field, dict):
        return field.get("fr", "") or ""
    return field or ""


def is_event_embeddable(event: dict) -> bool:
    """
    Détermine si un événement contient du texte exploitable pour embedding.

    Un event est rejeté s'il n'a NI titre NI description NI longDescription.
    """
    title = extract_text(event.get("title")).strip()
    desc = extract_text(event.get("description")).strip()
    long_desc = extract_text(event.get("longDescription")).strip()
    return bool(title or desc or long_desc)


def build_embeddable_text(event: dict) -> str:
    """
    Compose le texte qui sera embeddé pour un événement.

    Stratégie :
    - Toujours préfixer par le titre (signal sémantique fort, court)
    - Privilégier longDescription (plus riche)
    - Fallback sur description si longDescription absente
    - Troncature à MAX_DESCRIPTION_CHARS pour gérer les outliers
    """
    title = extract_text(event.get("title")).strip()
    long_desc = extract_text(event.get("longDescription")).strip()
    desc = extract_text(event.get("description")).strip()

    # Choix du corps : longDescription en priorité
    body = long_desc if long_desc else desc

    # Troncature si dépassement
    if len(body) > MAX_DESCRIPTION_CHARS:
        body = body[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "…"

    # Composition finale
    if title and body:
        return f"{title}\n\n{body}"
    return title or body  # Au pire on a au moins l'un des deux

def is_in_target_geography(event: dict, target_city: str = "Paris") -> bool:
    """
    Vérifie qu'un événement se déroule dans la ville cible.

    Critères :
    - location.city == target_city (insensible à la casse)
    - OU location.postalCode commence par '75' (Paris intra-muros)

    Un event sans location est REJETÉ (impossible de valider le périmètre).
    """
    location = event.get("location") or {}
    if not isinstance(location, dict):
        return False

    city = (location.get("city") or "").strip().lower()
    postal_code = str(location.get("postalCode") or "").strip()

    matches_city = city == target_city.lower()
    matches_postal = postal_code.startswith("75")

    return matches_city or matches_postal

def extract_metadata(event: dict) -> dict:
    """
    Extrait les métadonnées de filtrage pour un événement.

    Ces données sont stockées AVEC le vecteur dans FAISS mais
    ne sont PAS embeddées : elles servent au filtrage déterministe
    (date, ville) et à la restitution dans la réponse.
    """
    location = event.get("location") or {}
    last_timing = event.get("lastTiming") or {}
    first_timing = event.get("firstTiming") or {}

    return {
        "uid": event.get("uid"),
        "title": extract_text(event.get("title")),
        "city": location.get("city") if isinstance(location, dict) else None,
        "postal_code": location.get("postalCode") if isinstance(location, dict) else None,
        "latitude": location.get("latitude") if isinstance(location, dict) else None,
        "longitude": location.get("longitude") if isinstance(location, dict) else None,
        "first_timing_begin": first_timing.get("begin") if isinstance(first_timing, dict) else None,
        "last_timing_end": last_timing.get("end") if isinstance(last_timing, dict) else None,
        "slug": event.get("slug"),
    }