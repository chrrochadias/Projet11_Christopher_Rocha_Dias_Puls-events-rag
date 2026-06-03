"""
Assistant interactif pour affiner les ground_truths du jeu Q/R.

Pour chaque question :
1. Affiche la question + ground_truth actuelle
2. Lance le retriever avec k=8 et affiche les events candidats
3. Tu notes les uids pertinents + tu rédiges la nouvelle ground_truth
4. Le script sauvegarde au fur et à mesure

Lancement :
    python scripts/annotation_helper.py
    python scripts/annotation_helper.py --start q05    # reprendre à une question précise
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import build_rag_chain

QA_PATH = PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.json"
BACKUP_PATH = PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.backup.json"


def show_candidates(question: str, retriever, k: int = 8):
    """Affiche les top-k events candidats pour aider l'annotation."""
    docs = retriever.invoke(question)
    print(f"\n📚 Top-{len(docs)} candidats du retriever :\n")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        title = meta.get("title", "?")
        uid = meta.get("uid", "?")
        end = meta.get("last_timing_end", "?")[:10]
        city = meta.get("city", "?")
        # Aperçu des 200 premiers chars de la description
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] uid={uid}  📅{end}  📍{city}")
        print(f"      🏷️  {title}")
        print(f"      📝 {preview}...\n")


def annotate_question(q: dict, retriever) -> dict:
    """Mode interactif d'annotation pour une question."""
    print("\n" + "=" * 70)
    print(f"❓ {q['id']} [{q.get('category', '?')}] — {q['question']}")
    print("=" * 70)
    print(f"\n📋 Ground truth actuelle :\n   {q['ground_truth']}\n")

    show_candidates(q["question"], retriever)

    print("─" * 70)
    print("📝 ANNOTATION")
    print("─" * 70)
    print("Tape les uids des events pertinents (séparés par virgule), ou ENTER pour passer")
    uids_input = input("UIDs pertinents : ").strip()

    if not uids_input:
        print("⏭️  Question passée, ground_truth inchangée")
        return q

    try:
        new_uids = [int(u.strip()) for u in uids_input.split(",") if u.strip()]
    except ValueError:
        print("⚠️ Format invalide, question ignorée")
        return q

    print("\nTape la NOUVELLE ground_truth (sur une ligne, ENTER pour valider) :")
    new_gt = input("> ").strip()

    if new_gt:
        q["ground_truth"] = new_gt
    q["expected_source_uids"] = new_uids
    print(f"\n✅ Annotation enregistrée ({len(new_uids)} uids)")
    return q


def main():
    parser = argparse.ArgumentParser(description="Assistant d'annotation des ground_truths")
    parser.add_argument("--start", type=str, default=None,
                        help="ID de question pour reprendre (ex: q05)")
    parser.add_argument("--k", type=int, default=8,
                        help="Nombre de candidats à afficher (défaut: 8)")
    args = parser.parse_args()

    # Backup avant modification
    with open(QA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    BACKUP_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Backup créé : {BACKUP_PATH}\n")

    # Construit le retriever (sans le LLM, on n'en a pas besoin pour l'annotation)
    print("🔨 Chargement du retriever FAISS...")
    _, retriever = build_rag_chain(k=args.k)
    print("✅ Retriever prêt\n")

    questions = data["questions"]
    start_idx = 0
    if args.start:
        for i, q in enumerate(questions):
            if q["id"] == args.start:
                start_idx = i
                break

    for i in range(start_idx, len(questions)):
        try:
            questions[i] = annotate_question(questions[i], retriever)
        except KeyboardInterrupt:
            print("\n\n⚠️ Interruption, sauvegarde en cours...")
            break

        # Sauvegarde après CHAQUE annotation (résilience)
        data["questions"] = questions
        with open(QA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("✅ Annotation terminée. Dataset mis à jour.")
    print(f"   Backup disponible : {BACKUP_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()