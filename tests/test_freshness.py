"""
Tests de contrat de données : fraîcheur des événements dans la base vectorielle.

Exigence du brief Jérémy :
    "Prévois un fichier .py pour tester que les données intégrées dans
     la base vectorielle correspondent bien à des évènements de
     moins d'un an dans la région géographique que tu auras sélectionnée."

Ce fichier couvre la dimension TEMPORELLE.

Lancement : pytest tests/test_freshness.py -v
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.config import MAX_EVENT_AGE_DAYS
from src.vector_store import load_vector_store


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module")
def vectorstore():
    """Charge l'index FAISS une seule fois pour tous les tests du module."""
    return load_vector_store()


@pytest.fixture(scope="module")
def all_documents(vectorstore):
    """Récupère tous les documents stockés dans l'index FAISS."""
    # FAISS stocke les Documents dans son docstore interne
    docstore = vectorstore.docstore._dict
    return list(docstore.values())


@pytest.fixture(scope="module")
def now():
    """Référence temporelle UTC pour les comparaisons."""
    return datetime.now(timezone.utc)


@pytest.fixture(scope="module")
def max_age_threshold(now):
    """Date limite : tout événement avec end < threshold est trop vieux."""
    return now - timedelta(days=MAX_EVENT_AGE_DAYS)


# ============================================================
# Helpers
# ============================================================
def parse_iso_date(date_str: str) -> datetime:
    """Parse une date ISO 8601 en datetime UTC-aware."""
    # Gère les formats avec timezone (+02:00) ou Z
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


# ============================================================
# Tests
# ============================================================
class TestFreshness:
    """Validation que tous les events stockés respectent la contrainte < 1 an."""

    def test_index_is_not_empty(self, all_documents):
        """Sanity check : l'index contient au moins 1 document."""
        assert len(all_documents) > 0, "L'index FAISS est vide"

    def test_all_documents_have_last_timing_end(self, all_documents):
        """Chaque document doit avoir une date de fin (pour pouvoir vérifier la fraîcheur)."""
        documents_without_date = [
            doc for doc in all_documents
            if not doc.metadata.get("last_timing_end")
        ]
        assert len(documents_without_date) == 0, (
            f"{len(documents_without_date)} documents sans last_timing_end : "
            f"{[d.metadata.get('uid') for d in documents_without_date[:5]]}"
        )

    def test_all_events_are_recent(self, all_documents, max_age_threshold):
        """
        ⭐ TEST PRINCIPAL : tous les events doivent avoir une date de fin
        postérieure à (maintenant - 365 jours).

        Cette assertion est la traduction directe de l'exigence du brief.
        """
        stale_events = []
        for doc in all_documents:
            end_str = doc.metadata.get("last_timing_end")
            if not end_str:
                continue  # déjà testé ci-dessus
            end_date = parse_iso_date(end_str)
            if end_date < max_age_threshold:
                stale_events.append({
                    "uid": doc.metadata.get("uid"),
                    "title": doc.metadata.get("title"),
                    "end_date": end_date.isoformat(),
                })

        assert len(stale_events) == 0, (
            f"{len(stale_events)} événements trop anciens (> {MAX_EVENT_AGE_DAYS} jours) :\n"
            + "\n".join([f"  - {e['title']} ({e['end_date']})" for e in stale_events[:5]])
        )

    def test_dates_are_parsable(self, all_documents):
        """Toutes les dates stockées doivent être au format ISO 8601 valide."""
        unparsable = []
        for doc in all_documents:
            end_str = doc.metadata.get("last_timing_end")
            if not end_str:
                continue
            try:
                parse_iso_date(end_str)
            except (ValueError, TypeError):
                unparsable.append({
                    "uid": doc.metadata.get("uid"),
                    "raw": end_str,
                })

        assert len(unparsable) == 0, (
            f"{len(unparsable)} dates non-parsables : {unparsable[:3]}"
        )
