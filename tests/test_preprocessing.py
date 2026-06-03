"""
Tests unitaires des fonctions de préprocessing.

Ces tests sont des tests unitaires "purs" : ils testent des fonctions
sans I/O ni appel réseau, avec des données contrôlées en entrée.

Lancement : pytest tests/test_preprocessing.py -v
"""
import pytest

from src.preprocessing import (
    extract_text,
    is_event_embeddable,
    build_embeddable_text,
    extract_metadata,
)


# ============================================================
# Tests de extract_text()
# ============================================================
class TestExtractText:
    """Helper d'extraction du texte FR depuis les champs multilingues."""

    @pytest.mark.parametrize("input_field,expected", [
        ({"fr": "Bonjour", "en": "Hello"}, "Bonjour"),
        ({"fr": "Concert"}, "Concert"),
        ({"en": "Only English"}, ""),                  # pas de FR → vide
        ({}, ""),                                       # dict vide
        (None, ""),                                     # None
        ("string brute", "string brute"),               # string directe
        ({"fr": None}, ""),                             # FR explicitement null
    ])
    def test_extract_text_handles_all_inputs(self, input_field, expected):
        assert extract_text(input_field) == expected


# ============================================================
# Tests de is_event_embeddable()
# ============================================================
class TestIsEventEmbeddable:
    """Filtre de qualité : un event sans aucun texte exploitable est rejeté."""

    def test_event_with_full_content_is_embeddable(self):
        event = {
            "title": {"fr": "Concert de jazz"},
            "description": {"fr": "Une soirée musicale"},
            "longDescription": {"fr": "Description complète."},
        }
        assert is_event_embeddable(event) is True

    def test_event_with_only_title_is_embeddable(self):
        event = {"title": {"fr": "Atelier"}}
        assert is_event_embeddable(event) is True

    def test_event_with_only_description_is_embeddable(self):
        event = {"description": {"fr": "Quelque chose"}}
        assert is_event_embeddable(event) is True

    def test_completely_empty_event_is_rejected(self):
        event = {}
        assert is_event_embeddable(event) is False

    def test_event_with_only_whitespace_is_rejected(self):
        event = {
            "title": {"fr": "   "},
            "description": {"fr": "\n\t"},
            "longDescription": {"fr": ""},
        }
        assert is_event_embeddable(event) is False

    def test_event_with_null_fields_is_rejected(self):
        event = {"title": None, "description": None, "longDescription": None}
        assert is_event_embeddable(event) is False


# ============================================================
# Tests de build_embeddable_text()
# ============================================================
class TestBuildEmbeddableText:
    """Composition du texte qui sera embeddé."""

    def test_combines_title_and_long_description(self):
        event = {
            "title": {"fr": "Concert"},
            "longDescription": {"fr": "Une belle soirée musicale."},
        }
        result = build_embeddable_text(event)
        assert "Concert" in result
        assert "Une belle soirée musicale." in result
        assert "\n\n" in result  # séparateur entre titre et corps

    def test_falls_back_to_description_if_no_long_description(self):
        event = {
            "title": {"fr": "Concert"},
            "description": {"fr": "Description courte"},
        }
        result = build_embeddable_text(event)
        assert "Description courte" in result

    def test_long_description_takes_priority_over_description(self):
        event = {
            "title": {"fr": "Concert"},
            "description": {"fr": "Courte"},
            "longDescription": {"fr": "Longue version détaillée"},
        }
        result = build_embeddable_text(event)
        assert "Longue version" in result
        assert "Courte" not in result

    def test_truncates_overlong_descriptions(self):
        # Description de 10 000 caractères
        very_long = "mot " * 2500  # ~10 000 chars
        event = {
            "title": {"fr": "Titre"},
            "longDescription": {"fr": very_long},
        }
        result = build_embeddable_text(event)
        # Doit être tronqué autour de 4000 chars (+ titre + séparateur + "…")
        assert len(result) < 4100
        assert result.endswith("…")

    def test_truncation_is_word_aware(self):
        """La troncature ne doit pas couper un mot au milieu."""
        very_long = "antiquité " * 500  # ~5000 chars
        event = {
            "title": {"fr": "T"},
            "longDescription": {"fr": very_long},
        }
        result = build_embeddable_text(event)
        # La fin (avant le …) ne doit pas couper "antiquité"
        body_without_ellipsis = result.rstrip("…").rstrip()
        assert not body_without_ellipsis.endswith("antiqu")


# ============================================================
# Tests de extract_metadata()
# ============================================================
class TestExtractMetadata:
    """Extraction des métadonnées structurées pour filtrage."""

    def test_extracts_all_fields_when_present(self):
        event = {
            "uid": 12345,
            "title": {"fr": "Concert"},
            "slug": "concert-jazz",
            "location": {
                "city": "Paris",
                "postalCode": "75015",
                "latitude": 48.84,
                "longitude": 2.30,
            },
            "firstTiming": {"begin": "2026-06-01T20:00:00Z"},
            "lastTiming": {"end": "2026-06-01T22:00:00Z"},
        }
        meta = extract_metadata(event)
        assert meta["uid"] == 12345
        assert meta["title"] == "Concert"
        assert meta["city"] == "Paris"
        assert meta["postal_code"] == "75015"
        assert meta["latitude"] == 48.84
        assert meta["last_timing_end"] == "2026-06-01T22:00:00Z"

    def test_handles_missing_location(self):
        event = {"uid": 1, "title": {"fr": "Test"}}
        meta = extract_metadata(event)
        assert meta["city"] is None
        assert meta["postal_code"] is None
        assert meta["latitude"] is None

    def test_handles_missing_timing(self):
        event = {"uid": 1, "title": {"fr": "Test"}}
        meta = extract_metadata(event)
        assert meta["first_timing_begin"] is None
        assert meta["last_timing_end"] is None
