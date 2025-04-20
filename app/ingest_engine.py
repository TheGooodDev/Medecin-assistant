"""
Module d'indexation incrémentale pour un projet RAG.

Ce script :
- scanne tous les fichiers PDF dans le dossier `data/`
- ignore ceux déjà indexés (via un fichier `indexed_files.json`)
- construit ou met à jour un vecteurstore FAISS
"""

import os
import glob
import json
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.utils import load_api_key
from app.config import DATA_FOLDER, VECTORSTORE_PATH, INDEX_TRACKING_FILE


def get_all_pdfs(folder_path: str) -> List[str]:
    """
    Liste tous les fichiers PDF dans un dossier.

    Args:
        folder_path (str): Chemin vers le dossier contenant les PDF.

    Returns:
        List[str]: Liste complète des chemins vers les fichiers PDF.
    """
    return glob.glob(os.path.join(folder_path, "*.pdf"))


def load_indexed_files() -> List[str]:
    """
    Charge la liste des fichiers déjà indexés depuis un fichier JSON.

    Returns:
        List[str]: Liste des noms de fichiers déjà indexés.
    """
    if os.path.exists(INDEX_TRACKING_FILE):
        with open(INDEX_TRACKING_FILE, "r") as f:
            return json.load(f)
    return []


def save_indexed_files(file_list: List[str]):
    """
    Enregistre la liste des fichiers indexés dans un fichier JSON.

    Args:
        file_list (List[str]): Liste mise à jour des noms de fichiers indexés.
    """
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)  # Cree le dossier s'il n'existe pas
    with open(INDEX_TRACKING_FILE, "w") as f:
        json.dump(file_list, f, indent=2)


def load_documents(file_paths: List[str]) -> List:
    """
    Charge tous les documents PDF en objets LangChain.

    Args:
        file_paths (List[str]): Liste des chemins de fichiers à charger.

    Returns:
        List[Document]: Liste de documents extraits.
    """
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # On ajoute le nom du fichier comme source dans les métadonnées
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)
    return all_docs


def build_incremental_vectorstore(
    data_folder: str, persist_path: str = VECTORSTORE_PATH
) -> None:
    """
    Construit ou met à jour un vecteurstore FAISS de façon incrémentale.

    Seuls les fichiers PDF non encore indexés seront traités.

    Args:
        data_folder (str): Dossier contenant les fichiers PDF.
        persist_path (str): Chemin du vecteurstore à créer ou mettre à jour.
    """
    load_api_key()  # Charge la clé OpenAI depuis le .env

    # Liste des fichiers actuels et de ceux déjà indexés
    all_files = get_all_pdfs(data_folder)
    already_indexed = load_indexed_files()

    # Filtrage : on ne garde que les nouveaux fichiers
    new_files = [f for f in all_files if os.path.basename(f) not in already_indexed]

    if not new_files:
        print("✅ Aucun nouveau fichier à indexer.")
        return

    print(f"📄 Nouveaux fichiers détectés : {len(new_files)}")
    for f in new_files:
        print(" •", f)

    # Chargement et découpage des nouveaux documents
    documents = load_documents(new_files)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Création des embeddings à partir des chunks
    embeddings = OpenAIEmbeddings()

    # Chargement ou création du vecteurstore FAISS
    if os.path.exists(os.path.join(persist_path, "index.faiss")):
        vectordb = FAISS.load_local(
            persist_path, embeddings, allow_dangerous_deserialization=True
        )
        vectordb.add_documents(chunks)  # ajout des nouveaux chunks
    else:
        vectordb = FAISS.from_documents(chunks, embeddings)

    # Sauvegarde du vecteurstore mis à jour
    vectordb.save_local(persist_path)

    # Mise à jour de la liste des fichiers indexés
    updated_indexed = list(
        set(already_indexed + [os.path.basename(f) for f in new_files])
    )
    save_indexed_files(updated_indexed)

    print(f"✅ Index mis à jour avec {len(new_files)} nouveau(x) fichier(s).")


if __name__ == "__main__":
    build_incremental_vectorstore(DATA_FOLDER)
