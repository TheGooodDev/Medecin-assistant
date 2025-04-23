# 🔁 Patch pour forcer le redeploiement sur Streamlit

import os
import streamlit as st
from dotenv import load_dotenv

def load_api_key(user_api_key: str = None):
    load_dotenv()
    import streamlit as st
    import os

    if user_api_key:
        cleaned_key = user_api_key.strip()
        os.environ["OPENAI_API_KEY"] = cleaned_key
        st.session_state["OPENAI_API_KEY_USED"] = cleaned_key
        st.session_state["OPENAI_API_SOURCE"] = "user"
        st.success("✅ Clé API personnalisée chargée.")
    else:
        api_key = os.getenv("OPENAI_API_KEY").strip()
        if not api_key:
            # st.error("❌ Aucune clé API OpenAI fournie.")
            raise ValueError("OPENAI_API_KEY est manquante.")
        os.environ["OPENAI_API_KEY"] = api_key
        # st.session_state["OPENAI_API_KEY_USED"] = api_key
        # st.session_state["OPENAI_API_SOURCE"] = "default"
