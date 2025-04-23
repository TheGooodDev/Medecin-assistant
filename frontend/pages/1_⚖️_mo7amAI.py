import sys
import os
import traceback
import streamlit as st

# --- Project imports ---------------------------------------------------------
# 👇 Add repo root to PYTHONPATH so that `app.*` imports resolve when running
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_community.document_loaders import PyPDFLoader  # noqa: F401 – kept for future upload feature
from app import config
from app.utils.utils import load_api_key
from app.utils.utils_streamlit import display_model_config
from app.rag_engine import RAGPipeline, FAISSRetriever, OpenAILLM

# ---------------------------------------------------------------------------
# 🎨 Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mo7ami Diali – Assistant Juridique IA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 🏠 HERO SECTION
# ---------------------------------------------------------------------------
st.title("⚖️ Mo7ami Diali – Votre assistant juridique intelligent")

st.markdown(
    """
    Saisissez **toute question de droit** – en français ou en arabe – et recevez **en quelques secondes** une
    réponse structurée **appuyée sur les textes de loi cités**. Idéal pour les **professionnels**, les **étudiants**
    ou toute personne ayant besoin d’un éclairage rapide et fiable.
    """
)

# # 👉 Call‑to‑action to focus the input (purely cosmetic on first load)
# if st.button("✨ Poser ma première question"):
#     st.experimental_set_query_params(focus="input")

# # ---------------------------------------------------------------------------
# # 🖊️ QUESTION INPUT
# # ---------------------------------------------------------------------------
# placeholder_examples = [
#     "Quels sont les droits d’une femme mariée selon le Code de la famille ?",
#     "Comment résilier un bail commercial de manière anticipée ?",
#     "Quelles sont les conditions d’un recours fiscal en appel ?",
# ]

# rotate placeholder each refresh using modulo of run count in session state
# run_count = st.session_state.get("run_count", 0)
# st.session_state["run_count"] = run_count + 1
# placeholder = placeholder_examples[run_count % len(placeholder_examples)]

st.markdown("""

Bienvenue sur **Mo7ami Diali**, votre assistant juridique intelligent.  
Grâce à l’IA, le savoir juridique devient **accessible**, **instantané** et **personnalisé**.

> 🧑‍⚖️ *L’avocat ne se trouve plus derrière un bureau : il est entre vos mains.*

---

## 💡 Que pouvez-vous faire ici ?
- 🔎 Poser une question juridique en langage naturel
- 📚 Obtenir des réponses étayées par les textes de loi indexés (Code pénal, Moudawana, etc.)
- 📄 Identifier les articles de référence utilisés pour chaque réponse

---

## 🧠 Une nouvelle façon de pratiquer le droit
Vous êtes étudiant, juriste, ou avocat ?
Mo7ami Diali vous aide à **trouver les bons textes**, **plus vite**, **plus sûrement**, **avec la clarté d’une IA entraînée sur votre base documentaire**.

---
""")


question = st.text_input(
    "📮 Formulez votre question juridique :",
    placeholder="Ex: Quels sont les droits d’une femme mariée selon le Code de la famille ?",
)

# ---------------------------------------------------------------------------
# 🔧 SIDEBAR – MODEL & ADVANCED OPTIONS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres du modèle")

    # Wrap existing component in an expander to hide complexity from casual users
    with st.expander("Options avancées", expanded=False):
        model, temperature, k = display_model_config("global")

    # 🔑 Optional user API key (overrides .env key)
    user_api_key = st.text_input(
        "🔑 Ta clé OpenAI (optionnelle)",
        type="password",
        placeholder="sk-...",
        help="Si renseignée, elle sera utilisée à la place de la clé stockée dans l’environnement.",
    )

    # 🌐 Interface language – groundwork for future i18n (not yet wired to the backend)
    lang = st.selectbox("🌍 Langue de l’interface", ["Français", "العربية", "English"], index=0)

    st.markdown("---")
    st.markdown(
        "<sub>Les **réponses fournies** ne constituent **pas un avis juridique** et ne remplacent pas la consultation d’un professionnel qualifié.</sub>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 🚀 ASK THE QUESTION
# ---------------------------------------------------------------------------
if st.button("📨 Obtenir une réponse", type="primary") and question:
    with st.spinner("🤖 Ton avocat numérique réfléchit..."):
        try:
            # 1️⃣ Load the key (user‑provided or default)
            load_api_key(user_api_key)

            # 2️⃣ Build pipeline components
            llm = OpenAILLM(model_name=model, temperature=temperature, user_api_key=user_api_key)
            retriever = FAISSRetriever(persist_path=config.VECTORSTORE_PATH)
            pipeline = RAGPipeline(retriever=retriever, llm=llm)

            # 3️⃣ Ask the pipeline
            result = pipeline.ask(question, k=k)

            # -----------------------------------------------
            # ⬅️ Answer | ➡️ Sources – two‑columns layout
            # -----------------------------------------------
            col_answer, col_sources = st.columns([2, 1], gap="large")

            with col_answer:
                st.success("## 🧠 Réponse générée par l’IA")
                st.markdown(result["result"], unsafe_allow_html=True)

            with col_sources:
                st.markdown("## 📂 Sources juridiques consultées")
                if result.get("source_documents"):
                    for i, doc in enumerate(result["source_documents"], start=1):
                        title = doc.metadata.get("source", "Document inconnu")
                        page = doc.metadata.get("page", "?")
                        content = doc.page_content[:600] + "…"
                        with st.expander(f"📄 {title} (page {page})"):
                            st.markdown(content)
                else:
                    st.info("Aucune source documentaire n’a été retournée.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
            st.code(traceback.format_exc(), language="python")

# ---------------------------------------------------------------------------
# 🦺 PRIVACY & SECURITY NOTICE
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    ### 🛡️ Confidentialité
    Vos requêtes sont **traitées localement** ; **aucun document** n’est envoyé à des serveurs tiers.
    """
)

# ---------------------------------------------------------------------------
# 📜 FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding-top:2rem;">
        <sub>⚖️ <b>Mo7ami Diali</b> – Plateforme d’assistance juridique IA • Développée avec 💡 par
        <a href="https://github.com/charifel" target="_blank">Charif EL JAZOULI</a> •
        <a href="https://github.com/ton-lien-github" target="_blank">Voir le projet sur GitHub</a></sub>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# END OF FILE ✨
# ---------------------------------------------------------------------------
