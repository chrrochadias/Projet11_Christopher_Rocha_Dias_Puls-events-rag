"""
Configuration pytest globale et fixtures partagées.

Ce fichier est automatiquement chargé par pytest et permet de :
- Définir des markers personnalisés (slow, integration)
- Partager des fixtures entre fichiers de test
"""
import pytest

def pytest_configure(config):
    """Enregistre les markers personnalisés pour éviter les warnings pytest."""
    config.addinivalue_line(
        "markers",
        "integration: tests d'intégration nécessitant l'index FAISS construit"
    )
    config.addinivalue_line(
        "markers",
        "slow: tests lents (appels API, vectorisation)"
    )