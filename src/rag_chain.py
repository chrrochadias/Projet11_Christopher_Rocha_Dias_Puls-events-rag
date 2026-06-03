"""
Chaîne RAG complète : retrieval + augmentation + génération via Mistral.

Architecture :
    Question
      → Retriever (FAISS, top-k similarity)
      → Prompt augmenté (contexte + question)
      → LLM Mistral
      → Réponse en langage naturel + sources

Fonctions :
- build_rag_chain()     : construit la chaîne LCEL composable
- answer_question()     : interface haut niveau (question → réponse + sources)
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from src.config import (
    MISTRAL_API_KEY,
    MISTRAL_LLM_MODEL,
)
from src.vector_store import get_or_build_vector_store


# ============================================================
# Prompt système
# ============================================================
SYSTEM_PROMPT = """\
Tu es un assistant culturel pour la plateforme Puls-Events, spécialisée dans \
la recommandation d'événements culturels.

Ton rôle est d'aider les utilisateurs à découvrir des événements pertinents \
en t'appuyant exclusivement sur le contexte d'événements fourni ci-dessous.

Règles strictes :
1. Réponds UNIQUEMENT en t'appuyant sur les événements fournis dans le contexte.
2. Si le contexte ne contient pas d'événement pertinent pour répondre à la \
question, dis-le honnêtement : "Je n'ai pas trouvé d'événement correspondant \
à votre demande dans la base actuelle."
3. N'invente JAMAIS d'événement, de date, de lieu ou de description.
4. Pour chaque événement recommandé, mentionne son titre, sa date et son lieu.
5. Réponds de manière concise, claire et chaleureuse en français.

Contexte (événements pertinents) :
{context}
"""

USER_PROMPT = "{question}"


def format_docs(docs: list[Document]) -> str:
    """
    Formate une liste de Documents en un bloc texte injectable dans le prompt.

    Chaque événement est présenté avec ses métadonnées clés (titre, date, lieu)
    suivies de sa description complète.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        block = (
            f"--- Événement {i} ---\n"
            f"Titre : {meta.get('title', 'Sans titre')}\n"
            f"Date  : {meta.get('first_timing_begin', 'Non précisée')} "
            f"→ {meta.get('last_timing_end', 'Non précisée')}\n"
            f"Lieu  : {meta.get('city', 'Non précisé')} "
            f"({meta.get('postal_code', '')})\n"
            f"Description :\n{doc.page_content}\n"
        )
        formatted.append(block)
    return "\n".join(formatted)


def build_rag_chain(k: int = 4, temperature: float = 0.0):
    """
    Construit la chaîne RAG complète en LCEL.

    Args:
        k: nombre de documents à récupérer (top-k retrieval)
        temperature: créativité du LLM (0 = factuel, 1 = créatif)

    Returns:
        Tuple (chain, retriever) — chain pour la génération,
        retriever pour pouvoir accéder aux sources séparément
    """
    vectorstore = get_or_build_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    llm = ChatMistralAI(
        model=MISTRAL_LLM_MODEL,
        temperature=temperature,
        api_key=MISTRAL_API_KEY,
    )

    # LCEL : composition fonctionnelle de la chaîne
    # 1. Le retriever récupère les docs pertinents
    # 2. format_docs les transforme en contexte texte
    # 3. RunnablePassthrough.assign() injecte ce contexte dans le dict d'entrée
    # 4. Le prompt formate le tout en messages
    # 5. Le LLM génère la réponse
    # 6. StrOutputParser extrait le texte de la réponse
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def answer_question(question: str, k: int = 4) -> dict:
    """
    Interface haut niveau : pose une question et reçoit réponse + sources.

    Args:
        question: la question de l'utilisateur en langage naturel
        k: nombre de sources à utiliser pour le retrieval

    Returns:
        dict avec :
        - 'question' : la question d'entrée
        - 'answer'   : la réponse générée par le LLM
        - 'sources'  : liste des Documents utilisés comme contexte
    """
    chain, retriever = build_rag_chain(k=k)

    # On récupère les sources AVANT pour pouvoir les retourner
    sources = retriever.invoke(question)

    # Puis on génère la réponse
    answer = chain.invoke({"question": question})

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    # Démo en ligne de commande
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Y a-t-il des conférences sur la spiritualité prévues ce mois-ci ?"

    print(f"\n❓ Question : {question}\n")
    print("⏳ Génération de la réponse...\n")

    result = answer_question(question)

    print("=" * 60)
    print("🤖 RÉPONSE")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print(f"📚 SOURCES UTILISÉES ({len(result['sources'])})")
    print("=" * 60)
    for i, doc in enumerate(result["sources"], 1):
        print(f"\n[{i}] {doc.metadata.get('title')}")
        print(f"    📅 {doc.metadata.get('last_timing_end')}")
        print(f"    📍 {doc.metadata.get('city')}")