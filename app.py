import os
import pickle
import streamlit as st
import PyPDF2
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

# Initialisation OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialisation du modèle de vecteur
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


st.title("🧠 Assistant Médical IA")
st.write("Posez une question médicale basée sur vos fichiers PDF ou sur la base documentaire par défaut.")

# ========== FONCTIONS ==========

def extract_text_from_pdfs(files):
    """Extrait le texte de fichiers PDF."""
    full_text = ""
    for file in files:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
    return full_text

def load_documents(folder_path="docs"):
    """Charge tous les fichiers texte du dossier docs/"""
    docs, metadata = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                docs.append(text)
                metadata.append({"source": filename, "text": text})
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), 'rb') as f:
                text = extract_text_from_pdfs([f])
                docs.append(text)
                metadata.append({"source": filename, "text": text})
                
    return docs, metadata

def build_faiss_index(documents, metadata, index_path="faiss_index"):
    if not documents:
        raise ValueError("Aucun document n'a été chargé pour l'indexation.")

    embeddings = model.encode(documents, show_progress_bar=True)

    if len(embeddings) == 0 or len(embeddings[0]) == 0:
        raise ValueError("Échec de la vectorisation : embeddings vides.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, f"{index_path}.index")
    with open(f"{index_path}.meta", "wb") as f:
        pickle.dump(metadata, f)

def search_faiss(query, index_path="faiss_index", top_k=3):
    """Recherche les passages les plus proches dans l'index FAISS."""
    index = faiss.read_index(f"{index_path}.index")
    with open(f"{index_path}.meta", "rb") as f:
        metadata = pickle.load(f)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return "\n\n".join([metadata[i]["text"] for i in indices[0]])

# ========== INTERFACE ==========

uploaded_files = st.file_uploader("📄 Téléversez vos fichiers PDF", type="pdf", accept_multiple_files=True)
question = st.text_input("💬 Posez votre question médicale ici :")

if st.button("🧠 Obtenir une réponse") and question:
    with st.spinner("⏳ Analyse et génération de réponse..."):

        # Si PDF, extrait du contenu
        if uploaded_files:
            context_text = extract_text_from_pdfs(uploaded_files)[:12000]
        else:
            # Si pas d'index, on le construit
            if not os.path.exists("faiss_index.index"):
                docs, meta = load_documents("docs")
                build_faiss_index(docs, meta)
            context_text = search_faiss(question)

        # Appel API OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un assistant IA médical. Tu réponds uniquement à partir du contenu fourni."},
                {"role": "user", "content": f"Voici des documents médicaux :\n{context_text}\n\nQuestion : {question}"}
            ],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content
        st.markdown("### ✅ Réponse :")
        st.write(answer)
