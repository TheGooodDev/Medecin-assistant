import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="centered")

# 📁 Helper pour accéder aux images dans frontend/assets/
def load_image(filename):
    return Image.open(os.path.join("frontend", "assets", filename))

# 🌙 Dark mode CSS
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        h1, h2, h3 {
            color: #FAFAFA;
        }
        .logo-badge {
            text-align: center;
        }
        .logo-badge img {
            max-height: 60px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# 📸 Photo de profil centrée
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(load_image("charif.JPG"), caption="Charif EL JAZOULI", width=220)

# 🧑 Présentation
st.markdown("""
<h1 style='text-align: center;'>Hi, I'm Charif EL JAZOULI</h1>
<h3 style='text-align: center;'>👨‍💼 Manager Data Scientist @ Sia Partners</h3>
<h4 style='text-align: center;'>🎓 Lecturer @ ESIEE, IFP School, Ynov, Gustave Eiffel</h4>

<p style='text-align: center;'>🚀 Passionate about <b>AI</b>, <b>LLMs</b>, <b>RAG</b>, and all things <b>Data</b>.</p>
""", unsafe_allow_html=True)

# # 🏢 Logos affiliations
# st.markdown("""---\n#### 🏫 Affiliations""")
# aff_cols = st.columns(4)
# logos = [
#     ("ifp.png", "IFP School"),
#     ("esiee.png", "ESIEE Paris"),
#     ("gustave_eifelle.jpg", "Université Gustave Eiffel"),
#     ("ynov.png", "Ynov Campus"),
# ]

# for col, (logo, label) in zip(aff_cols, logos):
#     with col:
#         st.image(load_image(logo), caption=label, use_column_width=True)

# 📄 Liens importants
st.markdown("""---""")
st.markdown("""
### 🔗 Let's connect

- 📄 **[Download my CV](https://tonlien.cv)**
- 💼 **[LinkedIn](https://www.linkedin.com/in/charif-el-jazouli)**
- 💻 **[GitHub](https://github.com/Charifjaz)**  
""")

# 🧭 Navigation
st.markdown("""---""")
st.markdown("""
### 🚀 Explore my work

- 💬 RAG-powered Assistant  
- 📈 Time Series Playground  
- 📚 Teaching Materials & Courses  
""")
