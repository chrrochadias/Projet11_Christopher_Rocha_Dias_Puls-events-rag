"""
Script CLI : évaluation du système RAG sur le jeu de Q/R annoté.

Exécution :
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --k 6        # ajuster top-k retrieval
"""
import argparse
import sys
from pathlib import Path

# Permet d'exécuter le script depuis n'importe où
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Évalue le système RAG Puls-Events sur le jeu de Q/R annoté"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Nombre de documents retournés par le retrieval (défaut: 4)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Évaluation RAGAS du système Puls-Events RAG")
    print("=" * 60)
    print(f"Paramètres : k={args.k}")

    report_path = run_full_evaluation(k=args.k)

    print("\n" + "=" * 60)
    print("✅ Évaluation terminée avec succès")
    print(f"📄 Rapport complet : {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
