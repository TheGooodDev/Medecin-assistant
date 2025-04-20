# 🔁 Patch pour forcer le redeploiement sur Streamlit

from dotenv import load_dotenv
import os

import os
import streamlit as st

def load_api_key(user_api_key: str = None):
    print("Chargement API Key...")  # 👈 juste pour forcer le changement
    """
    Charge la clé API OpenAI :
    - si l'utilisateur en fournit une (depuis l'interface Streamlit), on l'utilise
    - sinon, on prend celle du .env ou st.secrets
    """
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        st.success("✅ Clé API personnalisée chargée avec succès.")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Aucune clé API OpenAI fournie.")
            raise ValueError("OPENAI_API_KEY est manquante.")
        os.environ["OPENAI_API_KEY"] = api_key
