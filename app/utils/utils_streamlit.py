import streamlit as st
from app import config

def display_model_config(section_name: str):
    """
    Affiche dans la sidebar les paramètres du modèle (modèle, température, top-k).
    Retourne les valeurs choisies par l'utilisateur.
    """
    st.sidebar.markdown("## ⚙️ Paramètres du modèle")

    model = st.text_input(
        "Modèle LLM",
        value=config.DEFAULT_MODEL,
        key=f"model_{section_name}"
    )

    temperature = st.slider(
        "Température",
        min_value=0.0,
        max_value=1.0,
        value=float(config.DEFAULT_TEMPERATURE),
        step=0.05,
        key=f"temperature_{section_name}"
    )

    k = st.slider(
        "Top K documents",
        min_value=1,
        max_value=10,
        value=int(config.DEFAULT_K),
        key=f"k_{section_name}"
    )

    return model, temperature, k    
