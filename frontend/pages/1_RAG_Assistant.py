# frontend/streamlit_app.py
import sys
import os

# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ✅ Les imports de ton application doivent venir APRÈS
import streamlit as st
from app.rag_engine import ask_question
from app import config
from app.utils import load_api_key  # ✅ Nouvelle version qui gère clé API utilisateur
import traceback


# 🎨 Configuration de la page
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🌟 Titre principal
st.title("🔎 RAG Chatbot - Question / Réponse augmentée")
st.markdown("""
Pose ta question ci-dessous ✉️
Le modèle te répondra à partir des documents PDF que tu as indexés.
""")

# 🔑 Champ pour que l'utilisateur saisisse sa propre clé API (optionnel)
user_api_key = st.sidebar.text_input(
    "🔑 Ta clé OpenAI (optionnelle)",
    type="password",
    placeholder="sk-...",
    help="Si vide, la clé par défaut sera utilisée (sécurisée)."
)

# 🖋️ Entrée utilisateur : Question
question = st.text_input(
    "❓ Ta question :",
    placeholder="Ex: Ask me ! The World is yours."
)

# ⚙️ Paramètres avancés (modèle, température, top-k)
with st.expander("⚙️ Paramètres du modèle"):
    col1, col2 = st.columns(2)
    with col1:
        model = st.text_input("Modèle OpenAI", value=config.DEFAULT_MODEL)
        temperature = st.slider("Température", 0.0, 1.0, float(config.DEFAULT_TEMPERATURE), 0.05)
    with col2:
        k = st.slider("Top K documents", 1, 10, int(config.DEFAULT_K))

# 🚀 Bouton d'envoi de la question
if st.button("📤 Poser la question") and question:
    with st.spinner("🤖 Le modèle réfléchit..."):
        try:
            # ✅ Charge la bonne clé API (user ou fallback)
            load_api_key(user_api_key)

            result = ask_question(
                question=question,
                model_name=model,
                temperature=temperature,
                k=k
                )

            # 🔸 Layout en deux colonnes
            col_left, col_right = st.columns([2, 1])

            # 🔸 Colonne gauche : réponse
            with col_left:
                st.success("✅ Réponse :")
                st.markdown(result["result"])

            # 🔸 Colonne droite : documents sources
            with col_right:
                st.markdown("""
                #### 📄 Sources documentaires utilisées
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
    "<sub>🚀 Créé avec 💪 par <b>Charif EL JAZOULI</b> • "
    "<a href='https://github.com/ton-lien-github' target='_blank'>GitHub</a></sub>"
    "</div>",
    unsafe_allow_html=True
)