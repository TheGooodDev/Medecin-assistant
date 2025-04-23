# ingestion_engine.py (refactorisé en classe)
import sys
import os
# 🔧 Ajout du dossier parent pour les imports depuis app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils.utils import load_api_key
from config import DATA_FOLDER_AVOCAT, VECTORSTORE_PATH, INDEX_TRACKING_FILE
from loguru import logger


class RAGIngestEngine:
    def __init__(self, data_folder: str = DATA_FOLDER_AVOCAT, persist_path: str = VECTORSTORE_PATH):
        self.data_folder = data_folder
        self.persist_path = persist_path
        load_api_key()  # Initialise la clé OpenAI (ou autre provider)

    def get_all_pdfs(self) -> List[str]:
        return glob.glob(os.path.join(self.data_folder, "*.pdf"))

    def load_indexed_files(self) -> List[str]:
        if os.path.exists(INDEX_TRACKING_FILE):
            with open(INDEX_TRACKING_FILE, "r") as f:
                return json.load(f)
        return []

    def save_indexed_files(self, file_list: List[str]):
        os.makedirs(self.persist_path, exist_ok=True)
        with open(INDEX_TRACKING_FILE, "w") as f:
            json.dump(file_list, f, indent=2)

    def load_documents(self, file_paths: List[str]) -> List:
        all_docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(path)
            all_docs.extend(docs)
        return all_docs

    def build(self):
        all_files = self.get_all_pdfs()
        already_indexed = self.load_indexed_files()
        new_files = [f for f in all_files if os.path.basename(f) not in already_indexed]

        if not new_files:
            logger.info("✅ Aucun nouveau fichier à indexer.")
            return

        logger.info(f"📄 Nouveaux fichiers à indexer : {len(new_files)}")
        for f in new_files:
            logger.info(f" • {f}")

        documents = self.load_documents(new_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        if os.path.exists(os.path.join(self.persist_path, "index.faiss")):
            vectordb = FAISS.load_local(
                self.persist_path, embeddings, allow_dangerous_deserialization=True
            )
            vectordb.add_documents(chunks)
        else:
            vectordb = FAISS.from_documents(chunks, embeddings)


        vectordb.save_local(self.persist_path)

        updated_indexed = list(set(already_indexed + [os.path.basename(f) for f in new_files]))
        self.save_indexed_files(updated_indexed)

        logger.success(f"✅ Index mis à jour avec {len(new_files)} fichier(s).")


if __name__ == "__main__":
    engine = RAGIngestEngine()
    engine.build()
