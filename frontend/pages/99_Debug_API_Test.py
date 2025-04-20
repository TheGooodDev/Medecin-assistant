import streamlit as st
import traceback
import os

st.set_page_config(page_title="🔧 Debug API Key", page_icon="🛠️")

st.title("🧪 Debug API Key Loading")

try:
    from app.utils import load_api_key
    st.success("✅ Import réussi de `load_api_key()`")

    # Champ pour test
    user_api_key = st.text_input("🔑 Clé OpenAI test", type="password")

    if st.button("🚀 Tester le chargement"):
        load_api_key(user_api_key=user_api_key)
        key_env = os.getenv("OPENAI_API_KEY", "[non définie]")
        st.code(f"Clé en mémoire (env) : {key_env[:10]}...")

except Exception as e:
    st.error("❌ Erreur au chargement de la page !")
    st.code(traceback.format_exc())
