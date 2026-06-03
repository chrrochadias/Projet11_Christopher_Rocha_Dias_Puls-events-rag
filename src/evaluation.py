"""
Évaluation du système RAG via RAGAS.

Pipeline :
1. Charge le jeu de Q/R annoté (data/evaluation/qa_dataset.json)
2. Pour chaque question, exécute la chaîne RAG (retrieval + génération)
3. Calcule 4 métriques RAGAS sur les résultats
4. Produit un rapport synthétique (JSON + console)

Métriques RAGAS calculées :
- faithfulness       : la réponse est-elle fidèle au contexte récupéré ?
- answer_relevancy   : la réponse répond-elle à la question ?
- context_precision  : les bons documents sont-ils en haut du retrieval ?
- context_recall     : tous les docs pertinents sont-ils dans le contexte ?

NB : RAGAS utilise un LLM "juge" pour évaluer. On configure Mistral comme juge
     pour rester cohérent avec le reste de la stack.
"""
import json
from pathlib import Path
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from src.config import (
    PROJECT_ROOT,
    MISTRAL_API_KEY,
    MISTRAL_LLM_MODEL,
    MISTRAL_EMBED_MODEL,
)
from src.rag_chain import build_rag_chain, format_docs


# ============================================================
# Chemins
# ============================================================
QA_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"


# ============================================================
# Chargement du jeu de Q/R
# ============================================================
def load_qa_dataset(path: Path = QA_DATASET_PATH) -> list[dict]:
    """Charge le jeu de questions annotées."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ============================================================
# Génération des prédictions
# ============================================================
def generate_predictions(questions: list[dict], k: int = 4) -> list[dict]:
    """
    Exécute le RAG sur chaque question annotée.

    Pour chaque question, capture :
    - la réponse générée
    - les documents contexte récupérés

    Args:
        questions: liste de questions annotées
        k: nombre de documents à récupérer

    Returns:
        Liste de dicts au format attendu par RAGAS
    """
    chain, retriever = build_rag_chain(k=k)
    predictions = []

    print(f"\n🤖 Génération des réponses sur {len(questions)} questions...")
    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        print(f"   [{i}/{len(questions)}] {question_text[:60]}...")

        # Retrieval explicite pour capturer les contextes
        retrieved_docs = retriever.invoke(question_text)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Génération
        answer = chain.invoke({"question": question_text})

        predictions.append({
            "question": question_text,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q["ground_truth"],
            "id": q["id"],
            "category": q.get("category", "unknown"),
        })

    return predictions


# ============================================================
# Évaluation RAGAS
# ============================================================
def run_ragas_evaluation(predictions: list[dict]) -> dict:
    """..."""
    dataset_dict = {
        "question": [p["question"] for p in predictions],
        "answer": [p["answer"] for p in predictions],
        "contexts": [p["contexts"] for p in predictions],
        "ground_truth": [p["ground_truth"] for p in predictions],
    }
    ragas_dataset = Dataset.from_dict(dataset_dict)

    judge_llm = LangchainLLMWrapper(ChatMistralAI(
        model=MISTRAL_LLM_MODEL,
        temperature=0.0,
        api_key=MISTRAL_API_KEY,
        max_retries=5,         # ← retries Mistral natifs
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=MISTRAL_API_KEY,
        max_retries=5,
    ))

    # RunConfig pour throttle l'évaluation RAGAS
    from ragas.run_config import RunConfig
    run_config = RunConfig(
        timeout=60,
        max_retries=5,
        max_wait=60,
        max_workers=2,         # ← réduit le parallélisme (défaut 16) → limite les 429
    )

    print("\n📊 Lancement de l'évaluation RAGAS (peut prendre 3-7 minutes)...")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,  # ← appliqué
    )

    return result


# ============================================================
# Reporting
# ============================================================
def save_evaluation_report(predictions: list[dict], ragas_result, output_dir: Path = RESULTS_DIR):
    """Sauvegarde le rapport d'évaluation en JSON pour traçabilité."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"evaluation_report_{timestamp}.json"

    scores_df = ragas_result.to_pandas()
    metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Score global (moyennes en ignorant NaN)
    aggregate = {
        col: float(scores_df[col].mean(skipna=True))
        for col in metric_columns
        if col in scores_df.columns
    }

    # Détail par question (conversion sécurisée)
    per_question_scores = scores_df.fillna("N/A").to_dict(orient="records")

    report = {
        "timestamp": timestamp,
        "n_questions": len(predictions),
        "aggregate_scores": aggregate,
        "per_question_scores": per_question_scores,
        "predictions": [
            {
                "id": p["id"],
                "category": p["category"],
                "question": p["question"],
                "answer": p["answer"],
                "ground_truth": p["ground_truth"],
            }
            for p in predictions
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Rapport sauvegardé : {output_path}")
    return output_path

def print_summary(ragas_result, predictions: list[dict]):
    """Affiche un résumé console des résultats."""
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("=" * 60)
    print(f"\n📋 {len(predictions)} questions évaluées\n")

    # Convertit l'objet RAGAS en dict via to_pandas() (API stable)
    try:
        scores_df = ragas_result.to_pandas()
        # Les colonnes sont les métriques + 'user_input', 'response', etc.
        metric_columns = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]
        aggregate = {}
        for col in metric_columns:
            if col in scores_df.columns:
                # Moyenne en ignorant les NaN
                aggregate[col] = scores_df[col].mean(skipna=True)
    except Exception as e:
        print(f"⚠️ Erreur extraction scores : {e}")
        return

    print("Scores agrégés (moyennes) :")
    metric_labels = {
        "faithfulness": "Faithfulness     (non-hallucination)",
        "answer_relevancy": "Answer Relevancy (pertinence réponse)",
        "context_precision": "Context Precision (qualité retrieval)",
        "context_recall": "Context Recall   (couverture retrieval)",
    }
    for metric_key, label in metric_labels.items():
        score = aggregate.get(metric_key)
        if score is None or (isinstance(score, float) and score != score):  # NaN check
            print(f"   ⚪ {label:42s} : N/A")
        else:
            emoji = "🟢" if score >= 0.75 else ("🟡" if score >= 0.5 else "🔴")
            print(f"   {emoji} {label:42s} : {score:.3f}")

    # Comptage des NaN (utile pour debugging)
    n_nan_total = sum(scores_df[col].isna().sum() for col in metric_columns if col in scores_df.columns)
    if n_nan_total > 0:
        print(f"\n   ⚠️ {n_nan_total} score(s) NaN détecté(s) — probablement dûs aux rate limits Mistral")


# ============================================================
# Pipeline complet
# ============================================================
def run_full_evaluation(k: int = 4) -> Path:
    """
    Pipeline d'évaluation complet :
    1. Charge le jeu Q/R
    2. Génère les prédictions
    3. Lance RAGAS
    4. Affiche + sauvegarde le rapport
    """
    questions = load_qa_dataset()
    predictions = generate_predictions(questions, k=k)
    result = run_ragas_evaluation(predictions)
    print_summary(result, predictions)
    report_path = save_evaluation_report(predictions, result)
    return report_path


if __name__ == "__main__":
    run_full_evaluation()
