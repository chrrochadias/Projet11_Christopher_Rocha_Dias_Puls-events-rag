"""
Interface Streamlit pour le système RAG Puls-Events.

Lancement :
    streamlit run app.py

Architecture :
- Cache du retriever et de la chain via @st.cache_resource (chargement unique)
- Historique de conversation maintenu dans st.session_state
- Affichage des sources utilisées en expander pour la traçabilité
"""
import streamlit as st

from src.rag_chain import build_rag_chain


# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Puls-Events RAG",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# Chargement de la chaîne RAG (mis en cache)
# ============================================================
@st.cache_resource(show_spinner="🔨 Chargement de la base vectorielle FAISS...")
def load_chain():
    """
    Charge la chaîne RAG une seule fois pour toute la session Streamlit.

    Le décorateur @st.cache_resource garantit que cette fonction n'est
    exécutée qu'une seule fois — les requêtes utilisateurs ultérieures
    réutilisent le même retriever et la même chain en mémoire.
    """
    return build_rag_chain(k=4)


# ============================================================
# Initialisation de l'historique de conversation
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Sidebar — Informations contextuelles
# ============================================================
with st.sidebar:
    st.title("🎭 Puls-Events RAG")
    st.markdown("---")

    st.markdown("### À propos")
    st.markdown(
        "**Assistant intelligent** de recommandation d'événements culturels "
        "basé sur une architecture RAG (Retrieval-Augmented Generation)."
    )

    st.markdown("### Stack technique")
    st.markdown(
        "- **LangChain** — orchestration\n"
        "- **Mistral AI** — embeddings & LLM\n"
        "- **FAISS** — base vectorielle\n"
        "- **Streamlit** — interface"
    )

    st.markdown("### Source de données")
    st.markdown(
        "Agenda du **Diocèse de Paris** "
        "(via OpenAgenda API)"
    )

    st.markdown("---")

    st.markdown("### 💡 Exemples de questions")
    example_questions = [
        "Quels concerts de musique classique sont prévus ?",
        "Y a-t-il des conférences sur la spiritualité ?",
        "Quels événements pour les familles ?",
        "Que proposez-vous autour de l'écologie ?",
        "Y a-t-il des retraites ou pèlerinages ?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")

    if st.button("🗑️  Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# En-tête principal
# ============================================================
st.title("🎭 Puls-Events")
st.caption("Assistant intelligent pour découvrir des événements culturels à Paris")


# ============================================================
# Affichage de l'historique de conversation
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Affichage des sources si présentes (messages assistant)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 Sources utilisées ({len(message['sources'])})"):
                for i, src in enumerate(message["sources"], 1):
                    meta = src.get("metadata", {})
                    st.markdown(
                        f"**[{i}] {meta.get('title', 'Sans titre')}**  \n"
                        f"📅 {meta.get('last_timing_end', 'Date non précisée')[:10]}  \n"
                        f"📍 {meta.get('city', '?')} ({meta.get('postal_code', '')})"
                    )
                    if i < len(message["sources"]):
                        st.markdown("---")


# ============================================================
# Récupération de la question utilisateur
# ============================================================
# Soit via le chat_input, soit via un bouton d'exemple cliqué
user_question = st.chat_input("Posez votre question sur les événements...")

if not user_question and "pending_question" in st.session_state:
    user_question = st.session_state.pending_question
    del st.session_state.pending_question


# ============================================================
# Traitement de la question
# ============================================================
if user_question:
    # 1. Afficher immédiatement la question utilisateur
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Générer la réponse RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans la base et génération..."):
            chain, retriever = load_chain()
            # Retrieval explicite pour capturer les sources
            retrieved_docs = retriever.invoke(user_question)
            # Génération de la réponse
            answer = chain.invoke({"question": user_question})

        # 3. Afficher la réponse
        st.markdown(answer)

        # 4. Afficher les sources en expander
        sources_data = [
            {"metadata": doc.metadata, "content": doc.page_content}
            for doc in retrieved_docs
        ]
        with st.expander(f"📚 Sources utilisées ({len(sources_data)})"):
            for i, src in enumerate(sources_data, 1):
                meta = src["metadata"]
                st.markdown(
                    f"**[{i}] {meta.get('title', 'Sans titre')}**  \n"
                    f"📅 {meta.get('last_timing_end', 'Date non précisée')[:10]}  \n"
                    f"📍 {meta.get('city', '?')} ({meta.get('postal_code', '')})"
                )
                if i < len(sources_data):
                    st.markdown("---")

    # 5. Sauvegarder dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_data,
    })