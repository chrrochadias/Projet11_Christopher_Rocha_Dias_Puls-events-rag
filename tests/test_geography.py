"""
Tests de contrat de données : périmètre géographique des événements.

Exigence du brief Jérémy :
    "tester que les données intégrées dans la base vectorielle correspondent
     bien à des évènements [...] dans la région géographique que tu auras
     sélectionnée."

Ce fichier couvre la dimension GÉOGRAPHIQUE.

Périmètre choisi pour le POC : Paris (75xxx) — agenda du Diocèse de Paris.

Lancement : pytest tests/test_geography.py -v
"""
import pytest

from src.config import TARGET_CITY
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
    docstore = vectorstore.docstore._dict
    return list(docstore.values())


# ============================================================
# Tests
# ============================================================
class TestGeography:
    """Validation que tous les events stockés sont bien dans le périmètre Paris."""

    def test_index_is_not_empty(self, all_documents):
        assert len(all_documents) > 0, "L'index FAISS est vide"

    def test_majority_of_events_have_city_metadata(self, all_documents):
        """
        Au moins 90% des events doivent avoir un champ 'city' renseigné.
        On tolère un léger taux de manquant pour les events sans location.
        """
        with_city = sum(1 for doc in all_documents if doc.metadata.get("city"))
        coverage = with_city / len(all_documents)
        assert coverage >= 0.90, (
            f"Couverture du champ 'city' insuffisante : {coverage:.1%} "
            f"({with_city}/{len(all_documents)})"
        )

    def test_all_events_are_in_target_city(self, all_documents):
        """
        ⭐ TEST PRINCIPAL : tous les events avec ville renseignée doivent être
        dans la ville cible (Paris pour ce POC).

        Cas dérogatoires (city = None) sont traités séparément.
        """
        out_of_scope = []
        for doc in all_documents:
            city = doc.metadata.get("city")
            if city is None:
                continue  # tolérance, voir test ci-dessus
            if city.strip().lower() != TARGET_CITY.lower():
                out_of_scope.append({
                    "uid": doc.metadata.get("uid"),
                    "title": doc.metadata.get("title"),
                    "city": city,
                })

        assert len(out_of_scope) == 0, (
            f"{len(out_of_scope)} événements hors du périmètre '{TARGET_CITY}' :\n"
            + "\n".join([f"  - {e['title']} ({e['city']})" for e in out_of_scope[:5]])
        )

    def test_postal_codes_are_parisian(self, all_documents):
        """
        Vérification complémentaire : les codes postaux des events parisiens
        doivent commencer par 75 (Paris intra-muros).
        """
        wrong_postal_codes = []
        for doc in all_documents:
            postal_code = doc.metadata.get("postal_code")
            if not postal_code:
                continue
            postal_code = str(postal_code).strip()
            if not postal_code.startswith("75"):
                wrong_postal_codes.append({
                    "uid": doc.metadata.get("uid"),
                    "title": doc.metadata.get("title"),
                    "postal_code": postal_code,
                })

        assert len(wrong_postal_codes) == 0, (
            f"{len(wrong_postal_codes)} événements avec code postal hors 75xxx :\n"
            + "\n".join([f"  - {e['title']} ({e['postal_code']})" for e in wrong_postal_codes[:5]])
        )
