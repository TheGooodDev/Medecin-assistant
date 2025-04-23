# === Configuration initiale ===
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_community.document_loaders import PyPDFLoader
import tempfile
import streamlit as st
import traceback
from app import config
from app.utils.utils import load_api_key
from app.utils.utils_streamlit import display_model_config
from app.rag_engine import RAGPipeline, TemporaryFAISSRetriever, OpenAILLM

# === 🎨 Configuration visuelle ===
st.set_page_config(
    page_title="Analyse Documentaire Juridique",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === 🧑‍⚖️ En-tête principal ===
st.markdown("""
# 📚 Analyse Juridique Temporaire

Importez un document juridique (PDF), posez votre question, et recevez une réponse basée sur son contenu. 
Ce service respecte la confidentialité de vos fichiers, qui ne sont **pas sauvegardés**.
""")

# === 🔐 Clé API utilisateur (optionnel) ===
with st.sidebar:
    model, temperature, k = display_model_config("global")
    user_api_key = st.text_input("🔑 Votre clé OpenAI (facultatif)", type="password", placeholder="sk-...")

# === 📁 Upload de documents PDF ===
st.warning("⚠️ Ce document ne sera **pas sauvegardé**. Il est utilisé uniquement pendant cette session.")
uploaded_files = st.file_uploader("📎 Téléversez un ou plusieurs PDF juridiques :", type=["pdf"], accept_multiple_files=True)

session_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        session_docs.extend(docs)

        os.remove(tmp_path)

    st.success(f"✅ {len(session_docs)} page(s) chargée(s) depuis les documents uploadés.")

# === ❓ Interaction utilisateur ===
if session_docs:
    st.divider()
    st.markdown("## ❓ Posez votre question juridique")

    question = st.text_input("🧾 Votre question sur le document :")

    if st.button("💬 Interroger le document") and question:
        with st.spinner("💡 Analyse en cours..."):
            try:
                load_api_key(user_api_key)
                llm = OpenAILLM(model_name=model, temperature=temperature, user_api_key=user_api_key)
                retriever = TemporaryFAISSRetriever(docs=session_docs)
                pipeline = RAGPipeline(retriever=retriever, llm=llm)
                result = pipeline.ask(question, k=k)

                st.markdown("### 🧠 Réponse générée")
                st.success(result["result"])

                st.markdown("### 📂 Sources extraites")
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Document inconnu")
                    st.markdown(f"- `{source}`")

            except Exception as e:
                st.error(f"❌ Erreur pendant l'exécution : {e}")
                st.code(traceback.format_exc(), language="python")

# === 🖋️ Footer professionnel ===
st.markdown(
    """
    <div style='text-align: center; padding-top: 2rem;'>
    <sub>⚖️ <b>Mo7ami Diali</b> – Assistant temporaire pour l’analyse juridique • 
    Développé avec ❤️ par <b>Charif EL JAZOULI</b> • 
    <a href='https://github.com/ton-lien-github' target='_blank'>GitHub</a></sub>
    </div>
    """,
    unsafe_allow_html=True
)
