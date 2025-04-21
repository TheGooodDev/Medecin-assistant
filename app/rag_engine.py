# app/rag_components.py

import sys
import os
# 🔧 Ajout du dossier parent pour les imports depuis app/
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# === 0. Import Packages ===
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# === 1. Base Retriever ===
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int):
        pass


# === 2. FAISS Retriever ===
class FAISSRetriever(BaseRetriever):
    def __init__(self, persist_path: str = "vectorstore"):
        self.persist_path = persist_path
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = FAISS.load_local(
            persist_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int):
        return self.vectordb.as_retriever(search_kwargs={"k": k})
    

# === 2. TEMP FAISS Retriever ===
class TemporaryFAISSRetriever(BaseRetriever):
    def __init__(self, docs, chunk_size=500, chunk_overlap=50):
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        self.vectordb = FAISS.from_documents(chunks, embeddings)

    def retrieve(self, query: str, k: int):
        return self.vectordb.as_retriever(search_kwargs={"k": k})

# === 3. Base LLM ===
class BaseLLM(ABC):
    @abstractmethod
    def answer(self, question: str, documents: list):
        pass


# === 4. OpenAI LLM ===
class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float, user_api_key: str = None):
        # load_api_key(user_api_key)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def answer(self, question: str, retriever):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
        )
        return qa_chain.invoke({"query": question})


# === 5. RAG Pipeline ===
class RAGPipeline:
    def __init__(self, retriever: BaseRetriever, llm: BaseLLM):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, k: int = 3):
        retriever = self.retriever.retrieve(question, k)
        result = self.llm.answer(question, retriever)
        return {
            "result": result["result"],
            "source_documents":  result.get("source_documents", [])
        }
