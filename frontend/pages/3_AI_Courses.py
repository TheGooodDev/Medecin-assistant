import sys
import os

# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import os
import io

st.set_page_config(page_title="Cours IA", page_icon="📚")

st.title("📚 Mes Cours en Intelligence Artificielle")

# 📁 Dossier des fichiers PDF
pdf_folder = "frontend/assets/courses"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

for pdf_file in sorted(pdf_files):
    st.markdown(f"### 📄 {pdf_file}")

    try:
        doc = fitz.open(os.path.join(pdf_folder, pdf_file))
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom 2x
        img_data = Image.open(io.BytesIO(pix.tobytes("png")))

        st.image(img_data, caption=pdf_file, width=300)  # 👈 Réduction ici

        with open(os.path.join(pdf_folder, pdf_file), "rb") as f:
            st.download_button(
                label="📥 Télécharger le cours",
                data=f,
                file_name=pdf_file,
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"❌ Erreur pour le fichier {pdf_file} : {e}")
