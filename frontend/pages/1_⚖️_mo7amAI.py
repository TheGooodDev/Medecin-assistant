import sys
import os
# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from langchain_community.document_loaders import PyPDFLoader
import tempfile
import streamlit as st
import traceback
from app import config
from app.utils.utils import load_api_key  
from app.utils.utils_streamlit import display_model_config
from app.rag_engine import RAGPipeline, FAISSRetriever, TemporaryFAISSRetriever,OpenAILLM


# 🎨 Configuration de la page
st.set_page_config(
    page_title="Mo7ami Diali - Assistant Juridique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🌟 Titre principal
st.title("⚖️ Mo7ami Diali – Votre assistant juridique intelligent")
st.markdown("""

Bienvenue sur **Mo7ami Diali**, un assistant virtuel conçu pour les **professionnels du droit**, les **étudiants en droit**, et toute personne souhaitant **interroger rapidement un corpus juridique**.

Grâce à l’IA, le savoir juridique devient **accessible**, **instantané** et **personnalisé**.

---

### 🔍 **Ce que vous pouvez faire ici**
- ✍️ Poser des questions juridiques complexes de façon simple
- 🧠 Obtenir des réponses augmentées par les textes de loi, avec sources claires

---

### 🛡️ **Respect de la confidentialité**
Toutes vos requêtes sont traitées **localement**, et les documents restent **sous votre contrôle**.

---

### ⚙️ **Comment ça marche ?**
1. Sélectionnez le modèle (GPT) et le niveau de température (précision vs créativité)
2. Posez votre question
3. Obtenez une réponse accompagnée des articles de loi utilisés

---
""")

st.markdown("""
Pose ta question ci-dessous ✉️
Ton mo7ami te répondra en quelques secondes.
""")

# 🧠 Barre latérale : Clé API et paramètres
with st.sidebar:
    # ⚙️ Paramètres du modèle (directement visibles)
    model, temperature, k = display_model_config("global")

     # 🔑 Clé OpenAI personnalisée
    user_api_key = st.text_input(
        "🔑 Ta clé OpenAI (optionnelle)",
        type="password",
        placeholder="sk-...",
        # help="Elle sera utilisée à la place de celle du .env si renseignée.",
    )


# 🖋️ Entrée utilisateur : Question
question = st.text_input(
    "📮 Formulez votre question juridique :",
    placeholder="Ex: Quels sont les droits d’une femme mariée selon le Code de la famille ?"
)



# 🚀 Bouton d'envoi de la question
if st.button("📨 Obtenir une réponse") and question:
    with st.spinner("🤖 Ton avocat réfléchit..."):
        try:
            # load the key 
            load_api_key(user_api_key)

            # Initialise les composants
            llm = OpenAILLM(model_name=model, temperature=temperature, user_api_key=user_api_key)
            retriever = FAISSRetriever(persist_path=config.VECTORSTORE_PATH)
            pipeline = RAGPipeline(retriever=retriever, llm=llm)

            # Pose la question
            result = pipeline.ask(question, k=k)

            # 🔸 Layout en deux colonnes
            col_left, col_right = st.columns([2, 1])

            # 🔸 Colonne gauche : réponse
            with col_left:
                st.success("## 🧠 Réponse générée par l'IA")
                st.markdown(result["result"])

            # 🔸 Colonne droite : documents sources
            with col_right:
                st.markdown("""
                ## 📂 Sources juridiques consultées
                """)
                if result.get("source_documents"):
                    for i, doc in enumerate(result["source_documents"], start=1):
                        title = doc.metadata.get("source", "Document inconnu")
                        page = doc.metadata.get("page", "?")
                        content = doc.page_content[:500] + "..."

                        with st.expander(f"📄 {title} (page {page})"):
                            st.markdown(content)
                else:
                    st.info("Aucune source documentaire n'a été retournée.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
            st.code(traceback.format_exc(), language="python")

# 🎡 Footer
st.markdown(
    "<div style='text-align: center; padding-top: 2rem;'>"
    "<sub>⚖️ <b>Mo7ami Diali</b> – Plateforme d'assistance juridique intelligente • "
    "Développée avec 💡 par <b>Charif EL JAZOULI</b> • "
    "<a href='https://github.com/ton-lien-github' target='_blank'>Voir le projet sur GitHub</a></sub>"
    "</div>",
    unsafe_allow_html=True
)