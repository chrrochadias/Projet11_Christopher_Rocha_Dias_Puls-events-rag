"""
Ingestion des événements depuis l'API OpenAgenda.

Fonctions :
- fetch_events() : récupère N événements via pagination
- ingest_agenda() : pipeline complet d'ingestion + filtrage qualité
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from typing import Optional

import requests

from src.config import (
    OPENAGENDA_BASE_URL,
    OPENAGENDA_PUBLIC_KEY,
    AGENDA_UID,
    RAW_DIR,
)
from src.preprocessing import is_event_embeddable


def fetch_events(
        agenda_uid: int = AGENDA_UID,
        n: int = 500,
        upcoming_only: bool = True,
        sleep_between_pages: float = 0.2,
) -> list[dict]:
    """
    Récupère n événements via pagination cursor de l'API OpenAgenda.

    Args:
        agenda_uid: UID de l'agenda OpenAgenda
        n: nombre maximum d'événements à récupérer
        upcoming_only: si True, filtre les events à venir (timings[gte] = now)
        sleep_between_pages: délai entre les pages (politesse API)

    Returns:
        Liste de dicts événements (champ `detailed=1`)

    Raises:
        requests.HTTPError: si l'API retourne une erreur non transitoire
    """
    collected: list[dict] = []
    after: Optional[list] = None
    page_size = 100  # max OpenAgenda
    now_iso = datetime.now(timezone.utc).isoformat()

    while len(collected) < n:
        params = {
            "key": OPENAGENDA_PUBLIC_KEY,
            "size": min(page_size, n - len(collected)),
            "detailed": 1,
        }
        if upcoming_only:
            params["timings[gte]"] = now_iso
        if after is not None:
            # Format validé : requests sérialise une liste en params répétés
            params["after"] = after

        r = requests.get(
            f"{OPENAGENDA_BASE_URL}/agendas/{agenda_uid}/events",
            params=params,
            timeout=30,
        )
        if not r.ok:
            print(f"❌ {r.status_code} | {r.text[:300]}")
            r.raise_for_status()

        data = r.json()
        batch = data.get("events", [])
        if not batch:
            break

        collected.extend(batch)
        after = data.get("after")
        if not after:
            break

        time.sleep(sleep_between_pages)

    return collected[:n]


def ingest_agenda(
        agenda_uid: int = AGENDA_UID,
        n: int = 500,
        save_raw: bool = True,
) -> list[dict]:
    """
    Pipeline complet d'ingestion : fetch + filtre qualité + sauvegarde optionnelle.

    Args:
        agenda_uid: UID de l'agenda
        n: nombre maximum d'événements
        save_raw: si True, sauvegarde le JSON brut dans data/raw/

    Returns:
        Liste des événements valides (filtrés)
    """
    print(f"📥 Fetching events from agenda {agenda_uid}...")
    raw_events = fetch_events(agenda_uid=agenda_uid, n=n, upcoming_only=True)
    print(f"   → {len(raw_events)} événements bruts récupérés")

    valid_events = [e for e in raw_events if is_event_embeddable(e)]
    rejected = len(raw_events) - len(valid_events)
    print(f"   → {len(valid_events)} événements valides ({rejected} rejetés par filtre qualité)")

    if save_raw:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RAW_DIR / f"agenda_{agenda_uid}_events.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(valid_events, f, ensure_ascii=False, indent=2)
        print(f"   💾 Sauvegardé : {output_path}")

    return valid_events


if __name__ == "__main__":
    # Permet de lancer le module directement :
    # python -m src.data_ingestion
    events = ingest_agenda()
    print(f"\n✅ Pipeline d'ingestion terminé : {len(events)} événements prêts.")