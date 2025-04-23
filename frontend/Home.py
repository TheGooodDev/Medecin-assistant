"""
Professional Streamlit homepage for Charif EL JAZOULI
No portrait photo, compact decorative GIF, modern dark theme
Save this file as streamlit_app.py and run with `streamlit run streamlit_app.py`
Assets expected in frontend/assets/
    • rotation_polygon.gif  (optional – can comment out)

2025‑04‑21
"""

import streamlit as st
import pathlib, base64, textwrap
from PIL import Image

# --------------------------------------------------
# ⚙️  Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Charif EL JAZOULI | Data & AI",
    page_icon="🏠",
    layout="wide",
)

# --------------------------------------------------
# 📁 Helper – asset loading
# --------------------------------------------------
ROOT_ASSETS = pathlib.Path("frontend") / "assets"


def b64_asset(filename: str) -> str:
    """Return base64 encoded data for images/GIFs stored in assets."""
    path = ROOT_ASSETS / filename
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# --------------------------------------------------
# 🎨  Custom CSS – minimalist dark theme & buttons
# --------------------------------------------------
CSS = """
<style>
body, .stApp {background:#0E1117; color:#FAFAFA; font-family: "Segoe UI", sans-serif;}

/* hero heading */
.hero-title {font-size:3rem; font-weight:800; margin:0;}
.hero-sub  {font-size:1.25rem; font-weight:400; margin-top:0.2rem; color:#a1a1a1;}

/* buttons */
.btn {
    display:inline-block; padding:0.55rem 1.1rem; background:#1b4f72; color:#FFFFFF;
    border-radius:6px; text-decoration:none; font-weight:600; margin-right:0.5rem;
    transition:background 0.2s;
}
.btn:hover {background:#2e86c1;}

/* tags */
.tag {
    display:inline-block; background:#1F2937; color:#E5E7EB; padding:0.25rem 0.55rem;
    border-radius:4px; font-size:0.8rem; margin:0.2rem 0.25rem 0;
}

/* GIF */
.gif-wrap img {width:180px; max-width:100%; height:auto; display:block; margin:0 auto;}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------
# 🏠  HERO SECTION
# --------------------------------------------------
hero_left, hero_right = st.columns([2, 1])

with hero_left:
    st.markdown(
        """
        <p class="hero-title">Charif EL JAZOULI</p>
        <p class="hero-sub">Manager Data Scientist @ Sia Partners · Lecturer · AI & LLMs enthusiast</p>
        """,
        unsafe_allow_html=True,
    )
    # CTA buttons
    st.markdown(
        """
        <a class="btn" href="https://github.com/Charifjaz" target="_blank">GitHub</a>
        <a class="btn" href="https://www.linkedin.com/in/charif-el-jazouli" target="_blank">LinkedIn</a>
        <a class="btn" href="https://tonlien.cv" target="_blank">Download CV</a>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    # Display portrait instead of GIF
    photo_b64 = b64_asset("../assets/images/charif.JPG")
    if photo_b64:
        st.markdown(
            f"<div class='gif-wrap'><img src='data:image/jpeg;base64,{photo_b64}' alt='Charif EL JAZOULI' style='border-radius:8px; width:180px;' /></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# --------------------------------------------------
# 🔍  ABOUT / EXPERTISE
# --------------------------------------------------
about_cols = st.columns([3, 2])

with about_cols[0]:
    st.subheader("About me ▸")
    st.markdown(
        textwrap.dedent(
            """
            Seasoned data‑science leader with **8+ years** of experience delivering business value through
            statistical modelling, machine‑learning & MLOps. As a lecturer I translate complex concepts
            into actionable insights for engineers and decision‑makers.
            Currently exploring **Generative AI** – **RAG pipelines**, **LLM evaluation**, and scalable
            deployments on the cloud.
            """
        )
    )

    st.markdown("**Core skills**:")
    for tag in [
        "Python",
        "Machine Learning",
        "Generative AI",
        "RAG",
        "LLM Ops",
        "Time‑Series",
        "MLOps",
        "Cloud ( Azure | AWS )",
        "Teaching",
    ]:
        st.markdown(f"<span class='tag'>{tag}</span>", unsafe_allow_html=True)

with about_cols[1]:
    st.subheader("At a glance ▸")
    st.metric(label="Years in Data", value="8+")
    st.metric(label="Projects Led", value="15 +")
    st.metric(label="Students taught / year", value="120 +")

st.markdown("---")

# --------------------------------------------------
# 📌  LATEST PROJECTS (placeholder)
# --------------------------------------------------
proj1, proj2 = st.columns(2)

with proj1:
    st.markdown("#### 🧑‍💬 RAG‑powered Assistant")
    st.write(
        "Full‑stack demo of retrieval‑augmented generation with evaluation dashboard."
    )
with proj2:
    st.markdown("#### 📈 Time‑Series Playground")
    st.write(
        "Interactive sandbox to benchmark forecasting algorithms on energy datasets."
    )

st.markdown("---")

# --------------------------------------------------
# 📫  CONTACT
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center;'>Feel free to reach out on <a href='mailto:charif@example.com'>email</a> or connect on <a href='https://www.linkedin.com/in/charif-el-jazouli' target='_blank'>LinkedIn</a>.</p>",
    unsafe_allow_html=True,
)
