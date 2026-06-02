"""
Script CLI : reconstruction complète de la base vectorielle.

Exécution :
    python scripts/build_index.py             # build incrémental (réutilise raw/)
    python scripts/build_index.py --refresh   # refetch + rebuild complet

Répond à l'exigence du brief : "Le système doit être en mesure de
reconstruire la base vectorielle sur demande."
"""
import argparse
import sys
from pathlib import Path

# Permet d'exécuter le script depuis n'importe où en ajoutant la racine au path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import ingest_agenda
from src.vector_store import build_vector_store
from src.config import AGENDA_UID, RAW_DIR
import json


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruit la base vectorielle Puls-Events"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch les events depuis OpenAgenda avant de reconstruire l'index",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Nombre maximum d'événements à récupérer (défaut: 500)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Build de la base vectorielle Puls-Events")
    print("=" * 60)

    raw_path = RAW_DIR / f"agenda_{AGENDA_UID}_events.json"

    if args.refresh or not raw_path.exists():
        # Étape 1 : ingestion depuis OpenAgenda
        events = ingest_agenda(n=args.n, save_raw=True)
    else:
        # Réutilise le cache disque
        print(f"📂 Chargement du cache : {raw_path}")
        with open(raw_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        print(f"   → {len(events)} événements chargés depuis le cache")

    # Étape 2 : construction de l'index vectoriel
    build_vector_store(events, save_to_disk=True)

    print("\n" + "=" * 60)
    print("✅ Build terminé avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()